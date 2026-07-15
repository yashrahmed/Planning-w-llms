"""Run the iterative FloorplanQA Ollama tool-design experiment."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluate_jsonl import (
    DEFAULT_OLLAMA_MODEL,
    Example,
    answers_match,
    extract_model_answer,
    iter_examples,
    sample_examples,
    write_evaluation_report,
)
from .floorplan_tools import (
    TOOLSET_CHANGES,
    FloorplanToolRuntime,
    tool_names,
    tools_for_iteration,
)
from .generate_questions import (
    DEFAULT_LAYOUT_DIR,
    DEFAULT_OUTPUT_DIR,
    GENERATION_REPORT_FILENAME,
    TASKS,
    generate_record,
    layout_paths,
    load_layout,
    write_records,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL = PACKAGE_ROOT / "datasets" / "ollama-eval-pools" / "seed-0" / "questions.jsonl"
DEFAULT_EXPERIMENT_DIR = PACKAGE_ROOT / "experiments" / "tool-loop"
DEFAULT_TRAINING_PATH = DEFAULT_OUTPUT_DIR / "questions.jsonl"
BASELINE_GLOB = "qwen3.5-4b-seed-0*.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run successive FloorplanQA tool-calling evaluations."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--training-path", type=Path, default=DEFAULT_TRAINING_PATH)
    parser.add_argument("--training-count", type=int, default=50)
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=1,
        help=(
            "First toolset version to evaluate "
            f"(1-{max(TOOLSET_CHANGES)})."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    return parser.parse_args()


def existing_jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as input_file:
        return sum(1 for line in input_file if line.strip())


def ensure_training_examples(
    path: Path,
    count: int,
    seed: int,
    layout_dir: Path,
) -> dict[str, Any]:
    """Generate exactly ``count`` examples only when fewer currently exist."""
    existing = existing_jsonl_count(path)
    if existing >= count:
        return {
            "action": "kept_existing",
            "path": str(path),
            "requested": count,
            "records": existing,
            "seed": seed,
        }

    records: list[dict[str, Any]] = []
    used_layouts: list[str] = []
    failures: list[str] = []
    for source_path in layout_paths(layout_dir, seed):
        try:
            context = load_layout(source_path)
            layout_records = [
                generate_record(context, layout_dir, task, seed) for task in TASKS
            ]
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{source_path}: {error}")
            continue
        records.extend(layout_records)
        used_layouts.append(str(source_path.relative_to(layout_dir)))
        if len(records) >= count:
            break
    if len(records) < count:
        raise RuntimeError(f"only generated {len(records)} of {count} requested examples")

    records = records[:count]
    write_records(records, path)
    task_counts = Counter(str(record["task"]) for record in records)
    report = {
        "action": "generated",
        "path": str(path),
        "requested": count,
        "records": len(records),
        "seed": seed,
        "layout_selection": "sha256-uniform-all-layouts",
        "task_selection": "all-eight-per-layout_then_deterministic_prefix",
        "used_layouts": used_layouts,
        "task_counts": dict(sorted(task_counts.items())),
        "failures": failures,
    }
    report_path = path.parent / GENERATION_REPORT_FILENAME
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def compact_question(example: Example) -> str:
    user_content = next(
        message["content"]
        for message in example.messages
        if message["role"] == "user"
    )
    instruction = user_content.split("\n\nRoom layout:", 1)[0]
    formats = {
        "pair_distance": "a decimal distance in meters rounded to three places",
        "free_space": "a decimal area in square meters rounded to three places",
        "view_angle": "a decimal angle in degrees rounded to three places",
        "repositioning": "a decimal distance in meters rounded to three places",
        "max_box": "a decimal area in square meters rounded to three places",
        "placement": "exactly True or False",
        "shortest_path": "a compact JSON list of [x,y] waypoints",
        "visibility": "a compact JSON list of entity labels",
    }
    return (
        f"{instruction}.\n"
        "The layout is loaded behind the tools; do not ask for or manually reconstruct its JSON. "
        f"The answer must be {formats.get(example.task, 'the requested value')}.\n"
        "End with exactly: *Final answer*: <answer>"
    )


def parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("tool arguments must be a JSON object")


def call_ollama(
    url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    seed: int,
    max_tokens: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "seed": seed,
                "num_ctx": 16384,
                "num_predict": max_tokens,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
    except urllib.error.URLError as error:
        raise RuntimeError(f"could not reach Ollama at {url}") from error
    if not isinstance(result, dict) or not isinstance(result.get("message"), dict):
        raise RuntimeError(f"unexpected Ollama response: {result}")
    return result


def run_agent(
    example: Example,
    iteration: int,
    model: str,
    url: str,
    seed: int,
    max_tokens: int,
    layout_dir: Path,
) -> dict[str, Any]:
    tools = tools_for_iteration(iteration)
    runtime = FloorplanToolRuntime(example, layout_dir, seed)
    if iteration >= 6:
        system_prompt = (
            "You are a FloorplanQA tool agent. Call get_final_answer exactly once. Then copy "
            "its final_answer value byte-for-byte into one line formatted as "
            "*Final answer*: <answer>. Never omit, shorten, round, or reformat array elements."
        )
    else:
        system_prompt = (
            "You are a FloorplanQA tool agent. Call the single most relevant geometry tool "
            "immediately; never do polygon arithmetic yourself. After a tool returns, copy the "
            "appropriate exact value (prefer its final_answer field) and emit one short final line. "
            "You may call another tool only when the first tool cannot answer the question."
        )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": compact_question(example)},
    ]
    remaining_tokens = max_tokens
    trace: list[dict[str, Any]] = []
    last_content = ""
    model_calls = 0
    done_reasons: list[str | None] = []
    forced_final = False

    for _ in range(8):
        if remaining_tokens <= 0:
            break
        result = call_ollama(
            url, model, messages, tools, seed, remaining_tokens
        )
        model_calls += 1
        generated = max(0, int(result.get("eval_count") or 0))
        remaining_tokens = max(0, remaining_tokens - generated)
        done_reasons.append(result.get("done_reason"))
        assistant_message = dict(result["message"])
        messages.append(assistant_message)
        content = str(assistant_message.get("content") or "")
        if content:
            last_content = content
        tool_calls = assistant_message.get("tool_calls") or []
        if tool_calls:
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "")
                try:
                    arguments = parse_tool_arguments(function.get("arguments") or {})
                    output = runtime.execute(name, arguments)
                    tool_error = None
                except Exception as error:
                    arguments = (
                        function.get("arguments")
                        if isinstance(function.get("arguments"), dict)
                        else {}
                    )
                    output = {"error": f"{type(error).__name__}: {error}"}
                    tool_error = str(output["error"])
                trace.append(
                    {
                        "tool": name,
                        "arguments": arguments,
                        "output": output,
                        "error": tool_error,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": json.dumps(output, separators=(",", ":")),
                    }
                )
            continue

        if extract_model_answer(content) is not None:
            break
        if not forced_final and remaining_tokens > 0:
            messages.append(
                {
                    "role": "user",
                    "content": "Return the tool's exact result now on one line as *Final answer*: <answer>.",
                }
            )
            forced_final = True
            continue
        break

    return {
        "response": last_content,
        "tool_trace": trace,
        "model_calls": model_calls,
        "generated_tokens": max_tokens - remaining_tokens,
        "done_reasons": done_reasons,
        "token_budget": max_tokens,
    }


def build_report(
    *,
    iteration: int,
    model: str,
    input_path: Path,
    input_count: int,
    examples: list[Example],
    results: list[dict[str, Any]],
    seed: int,
    max_tokens: int,
    started_at: str,
    status: str,
) -> dict[str, Any]:
    correct = sum(bool(result["correct"]) for result in results)
    return {
        "schema_version": 1,
        "experiment": "floorplan-qa-tool-loop",
        "iteration": iteration,
        "status": status,
        "started_at": started_at,
        "updated_at": utc_now(),
        "completed_at": utc_now() if status == "complete" else None,
        "model": model,
        "input": str(input_path),
        "configuration": {
            "seed": seed,
            "sample_size": len(examples),
            "input_examples": input_count,
            "sampling": "uniform_without_replacement",
            "thinking": False,
            "temperature": 0,
            "max_generated_tokens_per_question_across_agent_turns": max_tokens,
        },
        "toolset": {
            "version": iteration,
            "change": TOOLSET_CHANGES[iteration],
            "tools": tool_names(iteration),
        },
        "selected_ids": [example.example_id for example in examples],
        "selected_distribution": dict(
            sorted(Counter(example.task for example in examples).items())
        ),
        "summary": {
            "completed": len(results),
            "total": len(examples),
            "correct": correct,
            "incorrect": len(results) - correct,
            "score": correct / len(results) if results else 0.0,
            "formatting_failures": sum(
                result["parsed_answer"] is None and result["error"] is None
                for result in results
            ),
            "runtime_seconds": round(
                sum(float(result["duration_seconds"]) for result in results), 6
            ),
            "generated_tokens": sum(int(result["generated_tokens"]) for result in results),
        },
        "results": results,
    }


def compile_feedback(report: dict[str, Any]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in report["results"]:
        by_task[str(result["task"])].append(result)
    task_feedback = {}
    failed_tasks = []
    for task, results in sorted(by_task.items()):
        correct = sum(bool(result["correct"]) for result in results)
        if correct != len(results):
            failed_tasks.append(task)
        task_feedback[task] = {
            "correct": correct,
            "total": len(results),
            "score": correct / len(results),
            "formatting_failures": sum(result["parsed_answer"] is None for result in results),
            "tool_errors": sum(
                trace["error"] is not None
                for result in results
                for trace in result["tool_trace"]
            ),
        }
    tool_usage = Counter(
        trace["tool"]
        for result in report["results"]
        for trace in result["tool_trace"]
    )
    iteration = int(report["iteration"])
    next_iteration = (
        iteration + 1 if iteration < max(TOOLSET_CHANGES) else None
    )
    return {
        "iteration": iteration,
        "score": report["summary"]["score"],
        "all_questions_correct": report["summary"]["correct"] == report["summary"]["total"],
        "successes": {
            "correct": report["summary"]["correct"],
            "tasks_with_perfect_score": [
                task
                for task, values in task_feedback.items()
                if values["correct"] == values["total"]
            ],
        },
        "failures": {
            "incorrect": report["summary"]["incorrect"],
            "failed_tasks": failed_tasks,
            "formatting_failures": report["summary"]["formatting_failures"],
            "runtime_errors": sum(result["error"] is not None for result in report["results"]),
        },
        "by_task": task_feedback,
        "tool_usage": dict(sorted(tool_usage.items())),
        "performance": {
            "runtime_seconds": report["summary"]["runtime_seconds"],
            "average_seconds_per_question": (
                report["summary"]["runtime_seconds"] / report["summary"]["completed"]
            ),
            "generated_tokens": report["summary"]["generated_tokens"],
        },
        "next_design": (
            None
            if next_iteration is None or report["summary"]["incorrect"] == 0
            else {
                "iteration": next_iteration,
                "change": TOOLSET_CHANGES[next_iteration],
                "rationale": (
                    f"Address remaining failures in {', '.join(failed_tasks) or 'answer formatting'} "
                    "while retaining successful tools."
                ),
            }
        ),
    }


def run_iteration(
    *,
    iteration: int,
    examples: list[Example],
    input_count: int,
    input_path: Path,
    output_path: Path,
    layout_dir: Path,
    model: str,
    url: str,
    seed: int,
    max_tokens: int,
) -> dict[str, Any]:
    started_at = utc_now()
    results: list[dict[str, Any]] = []
    write_evaluation_report(
        output_path,
        build_report(
            iteration=iteration,
            model=model,
            input_path=input_path,
            input_count=input_count,
            examples=examples,
            results=results,
            seed=seed,
            max_tokens=max_tokens,
            started_at=started_at,
            status="running",
        ),
    )
    for index, example in enumerate(examples, start=1):
        print(
            f"Iteration {iteration}, question {index}/{len(examples)}: "
            f"{example.example_id}",
            flush=True,
        )
        started = time.perf_counter()
        error: str | None = None
        try:
            agent = run_agent(
                example,
                iteration,
                model,
                url,
                seed,
                max_tokens,
                layout_dir,
            )
            response = str(agent["response"])
            parsed_answer = extract_model_answer(response)
            correct = answers_match(
                parsed_answer,
                example.expected,
                task=example.task,
                reference_answer=example.reference_answer,
                example=example,
                layout_dir=layout_dir,
            )
        except Exception as caught_error:
            response = ""
            parsed_answer = None
            correct = False
            error = f"{type(caught_error).__name__}: {caught_error}"
            agent = {
                "tool_trace": [],
                "model_calls": 0,
                "generated_tokens": 0,
                "done_reasons": [],
                "token_budget": max_tokens,
            }
        duration = time.perf_counter() - started
        result = {
            "index": index,
            "line_number": example.line_number,
            "id": example.example_id,
            "task": example.task,
            "source_layout": example.source_layout,
            "expected": example.expected,
            "reference_answer": example.reference_answer,
            "parsed_answer": parsed_answer,
            "correct": correct,
            "response": response,
            "error": error,
            "duration_seconds": round(duration, 6),
            **agent,
        }
        results.append(result)
        running_report = build_report(
            iteration=iteration,
            model=model,
            input_path=input_path,
            input_count=input_count,
            examples=examples,
            results=results,
            seed=seed,
            max_tokens=max_tokens,
            started_at=started_at,
            status="running",
        )
        write_evaluation_report(output_path, running_report)
        print(
            f"  {'CORRECT' if correct else 'INCORRECT'} in {duration:.2f}s; "
            f"tokens={agent['generated_tokens']}; tools="
            f"{[trace['tool'] for trace in agent['tool_trace']]}",
            flush=True,
        )
    final_report = build_report(
        iteration=iteration,
        model=model,
        input_path=input_path,
        input_count=input_count,
        examples=examples,
        results=results,
        seed=seed,
        max_tokens=max_tokens,
        started_at=started_at,
        status="complete",
    )
    write_evaluation_report(output_path, final_report)
    return final_report


def baseline_summaries(evaluation_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(evaluation_dir.glob(BASELINE_GLOB)):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summaries.append(
            {
                "file": str(path),
                "status": report.get("status"),
                "configuration": report.get("configuration"),
                "summary": report.get("summary"),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    if args.training_count < 1 or args.sample_size < 1 or args.max_tokens < 1:
        raise ValueError("counts and token budget must be positive")
    if not 1 <= args.max_iterations <= max(TOOLSET_CHANGES):
        raise ValueError(
            f"--max-iterations must be between 1 and {max(TOOLSET_CHANGES)}"
        )
    if not 1 <= args.start_iteration <= args.max_iterations:
        raise ValueError(
            "--start-iteration must be between 1 and --max-iterations"
        )

    input_path = args.input.expanduser().resolve()
    layout_dir = args.layout_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    training_path = args.training_path.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training = ensure_training_examples(
        training_path, args.training_count, args.training_seed, layout_dir
    )
    print(f"Training examples: {training['records']} ({training['action']})", flush=True)

    input_examples = list(iter_examples(input_path))
    examples = sample_examples(input_examples, args.sample_size, args.seed)
    selected_ids = [example.example_id for example in examples]
    completed_reports = []
    for previous_iteration in range(1, args.start_iteration):
        previous_path = output_dir / f"iteration-{previous_iteration}-results.json"
        if not previous_path.is_file():
            continue
        previous_report = json.loads(previous_path.read_text(encoding="utf-8"))
        if (
            previous_report.get("status") == "complete"
            and previous_report.get("selected_ids") == selected_ids
        ):
            completed_reports.append(previous_report)
            print(
                f"Reusing completed iteration {previous_iteration}: "
                f"{previous_report['summary']['correct']}/{args.sample_size} correct",
                flush=True,
            )
    for iteration in range(args.start_iteration, args.max_iterations + 1):
        output_path = output_dir / f"iteration-{iteration}-results.json"
        feedback_path = output_dir / f"iteration-{iteration}-feedback.json"
        report = run_iteration(
            iteration=iteration,
            examples=examples,
            input_count=len(input_examples),
            input_path=input_path,
            output_path=output_path,
            layout_dir=layout_dir,
            model=args.model,
            url=args.ollama_url,
            seed=args.seed,
            max_tokens=args.max_tokens,
        )
        feedback = compile_feedback(report)
        write_evaluation_report(feedback_path, feedback)
        completed_reports.append(report)
        print(
            f"Iteration {iteration}: {report['summary']['correct']}/{args.sample_size} correct",
            flush=True,
        )
        if report["summary"]["correct"] == args.sample_size:
            break

    latest = completed_reports[-1]
    summary = {
        "schema_version": 1,
        "experiment": "floorplan-qa-tool-loop",
        "created_at": utc_now(),
        "training": training,
        "baseline_runs_from_2026_07_12": baseline_summaries(
            PACKAGE_ROOT / "datasets" / "evaluations"
        ),
        "stopping_condition": (
            f"all_{args.sample_size}_correct"
            if latest["summary"]["correct"] == args.sample_size
            else "five_iterations_completed"
        ),
        "iterations_completed": len(completed_reports),
        "starting_toolset_version": min(
            int(report["iteration"]) for report in completed_reports
        ),
        "scores": [
            {
                "iteration": report["iteration"],
                "correct": report["summary"]["correct"],
                "total": report["summary"]["total"],
                "score": report["summary"]["score"],
                "runtime_seconds": report["summary"]["runtime_seconds"],
            }
            for report in completed_reports
        ],
        "latest_toolset": latest["toolset"],
        "toolset_evolution": [
            {
                "iteration": report["iteration"],
                "change": report["toolset"]["change"],
                "tools": report["toolset"]["tools"],
            }
            for report in completed_reports
        ],
        "fixed_sample_ids": latest["selected_ids"],
        "max_generated_tokens_per_question": args.max_tokens,
    }
    summary_path = output_dir / "summary.json"
    write_evaluation_report(summary_path, summary)
    print(f"Experiment summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
