"""Evaluate FloorplanQA examples with explicit-file model-facing tools."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .evaluate_jsonl import (
    DEFAULT_OLLAMA_MODEL,
    Example,
    answers_match,
    extract_model_answer,
    iter_examples,
    utc_now,
    write_evaluation_report,
)
from .floorplan_tools import TOOLS, FloorplanToolRuntime


SYSTEM_PROMPT = (
    "You are a FloorplanQA agent. Solve each question using the available "
    "floorplan tools whenever they provide relevant geometric evidence. The room "
    "layout is not embedded in the prompt: pass the exact file identifier written "
    "in the question to any tool that requires it. Choose tool names and arguments "
    "only from the user-visible question and tool schemas. Do not invent tool "
    "results. When a tool directly returns the quantity requested by the question, "
    "and the tool does not report an error, your next response must be the final "
    "answer. Do not call another tool to verify, recompute, reformat, or inspect "
    "that result. Never repeat a tool call with the same arguments. In particular, "
    "do not use the calculator on coordinates, paths, entity lists, distances, "
    "angles, or Boolean values already returned by another tool. End with the "
    "exact final-answer format requested by the user."
)

AGENT_DATA_BOUNDARY = {
    "model_input": [
        "record system and user messages with reference assistant messages removed",
        "agent system prompt",
        "advertised tool schemas",
        "tool results requested by the model",
    ],
    "tool_runtime_input": [
        "model-selected tool name",
        "model-supplied tool arguments",
        "allowed directory containing the referenced layout files",
    ],
    "scorer_only": [
        "task",
        "expected answer",
        "reference answer",
        "task parameters",
        "source layout provenance",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an explicit-file FloorplanQA tool agent through Ollama."
    )
    parser.add_argument("jsonl_path", type=Path)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N examples in input-file order.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help=(
            "Evaluate only this question ID; repeat for multiple IDs. Selected "
            "questions retain input-file order."
        ),
    )
    parser.add_argument(
        "--ollama-retries",
        type=int,
        default=2,
        help="Retries after transient Ollama connection failures (default: 2).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument(
        "--layout-dir",
        type=Path,
        help="Directory containing the layout files; defaults to the JSONL directory.",
    )
    return parser.parse_args()


def parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("tool arguments must be a JSON object")


def agent_messages(example: Example) -> list[dict[str, Any]]:
    original_system = [
        message["content"]
        for message in example.messages
        if message["role"] == "system"
    ]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "\n\n".join([SYSTEM_PROMPT, *original_system]),
        }
    ]
    messages.extend(
        dict(message) for message in example.messages if message["role"] == "user"
    )
    return messages


def build_ollama_payload(
    messages: list[dict[str, Any]],
    model: str,
    seed: int,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": 16384,
            "num_predict": max_tokens,
        },
    }


def call_ollama(
    url: str,
    messages: list[dict[str, Any]],
    model: str,
    seed: int,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    payload = json.dumps(
        build_ollama_payload(messages, model, seed, max_tokens)
    ).encode("utf-8")
    result: Any = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                result = json.load(response)
            break
        except urllib.error.URLError as error:
            if attempt >= retries:
                raise RuntimeError(
                    f"could not reach Ollama at {url} after {retries + 1} attempts; "
                    "start it with 'ollama serve'"
                ) from error
            time.sleep(min(2**attempt, 4))
    if not isinstance(result, dict) or not isinstance(result.get("message"), dict):
        raise RuntimeError(f"unexpected Ollama response: {result}")
    return result


def run_agent(
    example: Example,
    runtime: FloorplanToolRuntime,
    model: str,
    url: str,
    seed: int,
    max_tokens: int,
    max_turns: int,
    ollama_retries: int,
) -> dict[str, Any]:
    messages = agent_messages(example)
    remaining_tokens = max_tokens
    trace: list[dict[str, Any]] = []
    last_content = ""
    model_calls = 0
    forced_final = False
    done_reasons: list[str | None] = []

    for _ in range(max_turns):
        if remaining_tokens <= 0:
            break
        result = call_ollama(
            url,
            messages,
            model,
            seed,
            remaining_tokens,
            ollama_retries,
        )
        model_calls += 1
        generated_tokens = max(0, int(result.get("eval_count") or 0))
        remaining_tokens = max(0, remaining_tokens - generated_tokens)
        done_reasons.append(
            str(result["done_reason"]) if result.get("done_reason") is not None else None
        )

        assistant_message = dict(result["message"])
        messages.append(assistant_message)
        content = str(assistant_message.get("content") or "")
        if content.strip():
            last_content = content

        tool_calls = assistant_message.get("tool_calls") or []
        if tool_calls:
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                tool_name = str(function.get("name") or "")
                raw_arguments = function.get("arguments") or {}
                try:
                    arguments = parse_tool_arguments(raw_arguments)
                    output = runtime.execute(tool_name, arguments)
                    tool_error = None
                except Exception as error:
                    arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
                    tool_error = f"{type(error).__name__}: {error}"
                    output = f"Tool error: {tool_error}"
                trace.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "output": output,
                        "error": tool_error,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": output,
                    }
                )
            continue

        if extract_model_answer(content) is not None:
            break
        if not forced_final and remaining_tokens > 0:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Provide the final answer now in the exact format requested "
                        "by the original question."
                    ),
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


def task_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["task"])].append(result)
    return {
        task: {
            "correct": sum(bool(result["correct"]) for result in task_results),
            "total": len(task_results),
            "formatting_failures": sum(
                result["parsed_answer"] is None and result["error"] is None
                for result in task_results
            ),
            "runtime_errors": sum(
                result["error"] is not None for result in task_results
            ),
        }
        for task, task_results in sorted(grouped.items())
    }


def build_report(
    *,
    input_path: Path,
    input_count: int,
    examples: list[Example],
    results: list[dict[str, Any]],
    model: str,
    seed: int,
    max_tokens: int,
    max_turns: int,
    ollama_retries: int,
    layout_dir: Path,
    started_at: str,
    status: str,
) -> dict[str, Any]:
    correct = sum(bool(result["correct"]) for result in results)
    tool_usage = Counter(
        trace["tool"]
        for result in results
        for trace in result["tool_trace"]
    )
    return {
        "schema_version": 1,
        "experiment": "floorplan-qa-explicit-file-tools",
        "status": status,
        "started_at": started_at,
        "updated_at": utc_now(),
        "completed_at": utc_now() if status == "complete" else None,
        "model": model,
        "input": str(input_path),
        "configuration": {
            "seed": seed,
            "temperature": 0,
            "thinking": False,
            "sample_size": len(examples),
            "input_examples": input_count,
            "selection": (
                "all_input_examples_in_file_order"
                if len(examples) == input_count
                else (
                    "first_n_in_file_order"
                    if all(
                        example.line_number == index
                        for index, example in enumerate(examples, start=1)
                    )
                    else "selected_question_ids_in_file_order"
                )
            ),
            "max_generated_tokens_per_question_across_agent_turns": max_tokens,
            "max_agent_turns": max_turns,
            "ollama_retries": ollama_retries,
            "layout_dir": str(layout_dir),
            "agent_data_boundary": AGENT_DATA_BOUNDARY,
        },
        "toolset": {
            "tools": [tool["function"]["name"] for tool in TOOLS],
            "definitions": TOOLS,
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
            "runtime_errors": sum(result["error"] is not None for result in results),
            "runtime_seconds": round(
                sum(float(result["duration_seconds"]) for result in results), 6
            ),
            "generated_tokens": sum(
                int(result["generated_tokens"]) for result in results
            ),
            "model_calls": sum(int(result["model_calls"]) for result in results),
            "tool_calls": sum(len(result["tool_trace"]) for result in results),
        },
        "by_task": task_summary(results),
        "tool_usage": dict(sorted(tool_usage.items())),
        "failed_ids": [
            result["id"] for result in results if not bool(result["correct"])
        ],
        "results": results,
    }


def main() -> None:
    args = parse_args()
    if (
        args.max_tokens < 1
        or args.max_turns < 1
        or args.ollama_retries < 0
        or (args.limit is not None and args.limit < 1)
    ):
        raise ValueError("token and turn limits must be positive; retries cannot be negative")
    if args.limit is not None and args.question_id:
        raise ValueError("--limit and --question-id cannot be used together")
    if len(args.question_id) != len(set(args.question_id)):
        raise ValueError("--question-id values must be unique")

    input_path = args.jsonl_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {input_path}")
    output_path = args.output_json.expanduser().resolve()
    if output_path.suffix.casefold() != ".json":
        raise ValueError("--output-json must end in .json")
    layout_dir = (
        args.layout_dir.expanduser().resolve()
        if args.layout_dir is not None
        else input_path.parent
    )
    if not layout_dir.is_dir():
        raise NotADirectoryError(f"layout directory not found: {layout_dir}")

    input_examples = list(iter_examples(input_path))
    if not input_examples:
        raise ValueError(f"JSONL file contains no examples: {input_path}")
    if args.limit is not None and args.limit > len(input_examples):
        raise ValueError(
            f"--limit {args.limit} exceeds the {len(input_examples)} input examples"
        )
    if args.question_id:
        selected_ids = set(args.question_id)
        available_ids = {example.example_id for example in input_examples}
        missing_ids = selected_ids - available_ids
        if missing_ids:
            raise ValueError(
                "unknown --question-id value(s): " + ", ".join(sorted(missing_ids))
            )
        examples = [
            example for example in input_examples if example.example_id in selected_ids
        ]
    elif args.limit is not None:
        examples = input_examples[: args.limit]
    else:
        examples = input_examples
    runtime = FloorplanToolRuntime(layout_dir)
    started_at = utc_now()
    results: list[dict[str, Any]] = []

    print(
        f"Model: {args.model}\nInput: {input_path}\nExamples: {len(examples)}\n"
        f"Layout directory: {layout_dir}\nSeed: {args.seed}\n"
        f"Token budget per question: {args.max_tokens}\nOutput: {output_path}",
        flush=True,
    )
    write_evaluation_report(
        output_path,
        build_report(
            input_path=input_path,
            input_count=len(input_examples),
            examples=examples,
            results=results,
            model=args.model,
            seed=args.seed,
            max_tokens=args.max_tokens,
            max_turns=args.max_turns,
            ollama_retries=args.ollama_retries,
            layout_dir=layout_dir,
            started_at=started_at,
            status="running",
        ),
    )

    for index, example in enumerate(examples, start=1):
        print(
            f"Question {index}/{len(examples)}: {example.example_id}",
            flush=True,
        )
        started = time.perf_counter()
        error: str | None = None
        try:
            agent = run_agent(
                example,
                runtime,
                args.model,
                args.ollama_url,
                args.seed,
                args.max_tokens,
                args.max_turns,
                args.ollama_retries,
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
            error = f"{type(caught_error).__name__}: {caught_error}"
            response = ""
            parsed_answer = None
            correct = False
            agent = {
                "tool_trace": [],
                "model_calls": 0,
                "generated_tokens": 0,
                "done_reasons": [],
                "token_budget": args.max_tokens,
            }
        duration = time.perf_counter() - started
        result = {
            "index": index,
            "line_number": example.line_number,
            "id": example.example_id,
            "task": example.task,
            "layout_file": example.layout_file,
            "source_layout": example.source_layout,
            "expected": example.expected,
            "reference_answer": example.reference_answer,
            "parsed_answer": parsed_answer,
            "correct": correct,
            "formatting_failure": parsed_answer is None and error is None,
            "response": response,
            "error": error,
            "duration_seconds": round(duration, 6),
            **agent,
        }
        results.append(result)
        report = build_report(
            input_path=input_path,
            input_count=len(input_examples),
            examples=examples,
            results=results,
            model=args.model,
            seed=args.seed,
            max_tokens=args.max_tokens,
            max_turns=args.max_turns,
            ollama_retries=args.ollama_retries,
            layout_dir=layout_dir,
            started_at=started_at,
            status="running",
        )
        write_evaluation_report(output_path, report)
        print(
            f"  {'CORRECT' if correct else 'INCORRECT'}; {duration:.2f}s; "
            f"tokens={agent['generated_tokens']}; "
            f"tools={[trace['tool'] for trace in agent['tool_trace']]}",
            flush=True,
        )

    report = build_report(
        input_path=input_path,
        input_count=len(input_examples),
        examples=examples,
        results=results,
        model=args.model,
        seed=args.seed,
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
        ollama_retries=args.ollama_retries,
        layout_dir=layout_dir,
        started_at=started_at,
        status="complete",
    )
    write_evaluation_report(output_path, report)
    print(
        f"Final score: {report['summary']['correct']}/{len(examples)} "
        f"({report['summary']['score']:.1%})\nJSON: {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
