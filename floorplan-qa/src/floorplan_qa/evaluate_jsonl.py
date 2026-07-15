"""Evaluate FloorplanQA JSONL examples with a local model backend."""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from shapely.geometry import LineString
from shapely.validation import make_valid

from .generate_questions import DEFAULT_LAYOUT_DIR
from .geometry import (
    entity_centroid,
    label,
    load_layout,
    navigation_geometry,
)

DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_HF_MODEL = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
FINAL_ANSWER_PATTERN = re.compile(
    r"(?:\*{0,2})final\s+answer(?:\*{0,2})\s*:\s*(.+)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Example:
    line_number: int
    example_id: str
    messages: list[dict[str, str]]
    expected: str
    task: str
    reference_answer: Any
    source_layout: str | None
    parameters: dict[str, Any]
    layout_file: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local model over a QA JSONL file and score its answers."
    )
    parser.add_argument("backend", choices=("ollama", "huggingface"))
    parser.add_argument("jsonl_path", type=Path)
    parser.add_argument("--model", help="Override the backend's default model.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Uniformly sample this many examples before evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for example sampling and model generation (default: 0).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Checkpoint structured evaluation results to this JSON file.",
    )
    parser.add_argument(
        "--ollama-url", default="http://127.0.0.1:11434/api/chat"
    )
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    return parser.parse_args()


def extract_input_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Return input-only messages without leaking an assistant/reference answer."""
    messages = record.get("messages")
    if isinstance(messages, list):
        extracted = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role in {"system", "user"} and isinstance(content, str):
                extracted.append({"role": role, "content": content})
        if any(message["role"] == "user" for message in extracted):
            return extracted

    for key in ("question", "input", "prompt"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return [{"role": "user", "content": value}]

    raise ValueError("no user input found (expected messages, question, input, or prompt)")


def extract_expected_answer(record: dict[str, Any]) -> str:
    answer = record.get("answer")
    if answer is None:
        raise ValueError("missing answer field")
    return str(answer).strip()


def iter_examples(path: Path) -> Iterator[Example]:
    with path.open(encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            yield Example(
                line_number=line_number,
                example_id=str(record.get("id", f"line-{line_number}")),
                messages=extract_input_messages(record),
                expected=extract_expected_answer(record),
                task=str(record.get("task", "")),
                reference_answer=record.get("reference_answer", record.get("answer")),
                source_layout=(
                    str(record["source_layout"])
                    if record.get("source_layout") is not None
                    else None
                ),
                parameters=(
                    record["parameters"]
                    if isinstance(record.get("parameters"), dict)
                    else {}
                ),
                layout_file=(
                    str(record["layout_file"])
                    if record.get("layout_file") is not None
                    else None
                ),
            )


def sample_examples(
    examples: list[Example], sample_size: int | None, seed: int
) -> list[Example]:
    if sample_size is None:
        return examples
    if sample_size < 1:
        raise ValueError("--sample-size must be positive")
    if sample_size > len(examples):
        raise ValueError(
            f"--sample-size {sample_size} exceeds the {len(examples)} input examples"
        )
    return random.Random(seed).sample(examples, sample_size)


def extract_model_answer(response: str) -> str | None:
    matches = FINAL_ANSWER_PATTERN.findall(response)
    if not matches:
        return None
    answer = matches[-1].strip()
    return re.sub(r"[*_`]+$", "", answer).strip()


def decimal_value(value: str) -> Decimal | None:
    match = re.fullmatch(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"(?:\s*(?:m|meters?|m²|square\s+meters?))?\s*",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        return Decimal(match.group(1))
    except InvalidOperation:
        return None


def parse_sequence(value: str) -> list[Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            if value.strip().startswith("[") and value.strip().endswith("]"):
                inner = value.strip()[1:-1].strip()
                return (
                    []
                    if not inner
                    else [item.strip(" '\"`") for item in inner.split(",")]
                )
            return None
    return parsed if isinstance(parsed, list) else None


def discrete_frechet(
    first: list[tuple[float, float]], second: list[tuple[float, float]]
) -> float:
    if not first or not second:
        return math.inf
    values = [[math.inf for _ in second] for _ in first]
    for first_index, first_point in enumerate(first):
        for second_index, second_point in enumerate(second):
            distance = math.dist(first_point, second_point)
            if first_index == 0 and second_index == 0:
                values[first_index][second_index] = distance
            elif first_index == 0:
                values[first_index][second_index] = max(
                    values[first_index][second_index - 1], distance
                )
            elif second_index == 0:
                values[first_index][second_index] = max(
                    values[first_index - 1][second_index], distance
                )
            else:
                values[first_index][second_index] = max(
                    min(
                        values[first_index - 1][second_index],
                        values[first_index - 1][second_index - 1],
                        values[first_index][second_index - 1],
                    ),
                    distance,
                )
    return values[-1][-1]


def parse_path(value: str | Any) -> list[tuple[float, float]] | None:
    parsed = parse_sequence(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        return None
    try:
        path = [tuple(float(coordinate) for coordinate in point) for point in parsed]
    except (TypeError, ValueError):
        return None
    return path if all(len(point) == 2 for point in path) else None


def candidate_path_is_valid(
    example: Example, path: list[tuple[float, float]], layout_dir: Path
) -> bool:
    layout_reference = example.layout_file or example.source_layout
    if layout_reference is None or len(path) < 2:
        return False
    context = load_layout(layout_dir / layout_reference)
    entities = {label(entity): entity for entity in context.entities}
    try:
        first = entities[example.parameters["object_1"]]
        second = entities[example.parameters["object_2"]]
        clearance = float(example.parameters["clearance"])
    except (KeyError, TypeError, ValueError):
        return False
    if math.dist(path[0], entity_centroid(first)) > 0.01:
        return False
    if math.dist(path[-1], entity_centroid(second)) > 0.01:
        return False
    navigable, start_space, goal_space = navigation_geometry(
        context, first, second, clearance
    )
    tolerance = 0.002
    if len(path) == 2:
        return make_valid(start_space.union(goal_space)).buffer(tolerance).covers(
            LineString(path)
        )
    segments = [
        LineString([path[index], path[index + 1]])
        for index in range(len(path) - 1)
    ]
    return bool(
        start_space.buffer(tolerance).covers(segments[0])
        and goal_space.buffer(tolerance).covers(segments[-1])
        and all(
            navigable.buffer(tolerance).covers(segment)
            for segment in segments[1:-1]
        )
    )


def answers_match(
    actual: str | None,
    expected: str,
    task: str = "",
    reference_answer: Any = None,
    example: Example | None = None,
    layout_dir: Path | None = None,
) -> bool:
    if actual is None:
        return False

    actual_number = decimal_value(actual)
    expected_number = decimal_value(
        str(reference_answer if reference_answer is not None else expected)
    )
    if actual_number is not None and expected_number is not None:
        if task in {"pair_distance", "view_angle", "repositioning", "max_box"}:
            tolerance = Decimal("0.02")
        elif task == "free_space":
            tolerance = Decimal("0.05")
        else:
            decimal_places = max(0, -expected_number.as_tuple().exponent)
            quantum = Decimal(1).scaleb(-decimal_places)
            return actual_number.quantize(quantum) == expected_number.quantize(quantum)
        absolute_error = abs(actual_number - expected_number)
        if abs(expected_number) <= Decimal("0.001"):
            return absolute_error <= Decimal("0.001")
        return absolute_error / abs(expected_number) <= tolerance

    if task == "visibility":
        actual_values = parse_sequence(actual)
        expected_values = (
            reference_answer
            if isinstance(reference_answer, list)
            else parse_sequence(expected)
        )
        return bool(
            actual_values is not None
            and expected_values is not None
            and {str(value).casefold() for value in actual_values}
            == {str(value).casefold() for value in expected_values}
        )

    if task == "shortest_path" and example is not None and layout_dir is not None:
        actual_path = parse_path(actual)
        expected_path = parse_path(reference_answer)
        return bool(
            actual_path is not None
            and expected_path is not None
            and candidate_path_is_valid(example, actual_path, layout_dir)
            and discrete_frechet(actual_path, expected_path) <= 0.6
        )

    def normalize(value: str) -> str:
        return " ".join(value.casefold().strip().split())

    return normalize(actual) == normalize(expected)


def build_ollama_payload(
    example: Example,
    model: str,
    thinking: bool,
    seed: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Build the V1 request with the complete raw prompt and no tools."""
    return {
        "model": model,
        "messages": example.messages,
        "stream": False,
        "think": thinking,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": 16384,
            "num_predict": max_tokens,
        },
    }


def make_ollama_runner(args: argparse.Namespace) -> Callable[[Example], str]:
    model = args.model or DEFAULT_OLLAMA_MODEL

    def run(example: Example) -> str:
        payload = json.dumps(
            build_ollama_payload(
                example,
                model=model,
                thinking=args.thinking,
                seed=args.seed,
                max_tokens=args.max_tokens,
            )
        ).encode("utf-8")
        request = urllib.request.Request(
            args.ollama_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                result = json.load(response)
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"could not reach Ollama at {args.ollama_url}; "
                "start it with 'ollama serve'"
            ) from error
        try:
            return str(result["message"]["content"])
        except (KeyError, TypeError) as error:
            raise RuntimeError(f"unexpected Ollama response: {result}") from error

    print(f"Backend: Ollama\nModel: {model}")
    return run


def make_huggingface_runner(args: argparse.Namespace) -> Callable[[Example], str]:
    if sys.platform != "darwin":
        raise RuntimeError("the Hugging Face runner uses MLX and requires macOS")

    try:
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as error:
        raise RuntimeError("MLX-LM is not installed; run 'uv sync' first") from error

    model_id = args.model or DEFAULT_HF_MODEL
    print(f"Backend: Hugging Face via MLX-LM\nModel: {model_id}")
    print("Loading model from the Hugging Face cache/Hub ...", flush=True)
    model, tokenizer = load(model_id)

    def run(example: Example) -> str:
        mx.random.seed(args.seed)
        prompt = tokenizer.apply_chat_template(
            example.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.thinking,
        )
        return generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(temp=0.0),
            verbose=False,
            prefill_step_size=1024,
        )

    return run


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_model(args: argparse.Namespace) -> str:
    if args.model:
        return str(args.model)
    return DEFAULT_OLLAMA_MODEL if args.backend == "ollama" else DEFAULT_HF_MODEL


def build_evaluation_report(
    args: argparse.Namespace,
    input_path: Path,
    input_count: int,
    examples: list[Example],
    results: list[dict[str, Any]],
    started_at: str,
    status: str,
) -> dict[str, Any]:
    completed = len(results)
    correct = sum(bool(result["correct"]) for result in results)
    formatting_failures = sum(
        result["parsed_answer"] is None and result["error"] is None
        for result in results
    )
    task_counts = Counter(example.task for example in examples)
    source_counts = Counter(
        (
            Path(example.source_layout).parent.name
            if example.source_layout is not None
            else "unknown"
        )
        for example in examples
    )
    return {
        "schema_version": 1,
        "status": status,
        "started_at": started_at,
        "updated_at": utc_now(),
        "completed_at": utc_now() if status == "complete" else None,
        "backend": args.backend,
        "model": selected_model(args),
        "input": str(input_path),
        "configuration": {
            "seed": args.seed,
            "temperature": 0,
            "thinking": bool(args.thinking),
            "max_tokens_per_question": args.max_tokens,
            "sample_size": len(examples),
            "input_examples": input_count,
            "sampling": "uniform_without_replacement",
        },
        "selected_distribution": {
            "tasks": dict(sorted(task_counts.items())),
            "layout_sources": dict(sorted(source_counts.items())),
        },
        "summary": {
            "completed": completed,
            "total": len(examples),
            "correct": correct,
            "incorrect": completed - correct,
            "score": correct / completed if completed else 0.0,
            "formatting_failures": formatting_failures,
            "runtime_seconds": sum(
                float(result["duration_seconds"]) for result in results
            ),
        },
        "results": results,
    }


def write_evaluation_report(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def print_result(
    index: int,
    total: int,
    example: Example,
    response: str,
    parsed_answer: str | None,
    correct: bool,
    duration_seconds: float,
    error: str | None = None,
) -> None:
    width = 78
    print("\n" + "=" * width)
    print(f"Example {index}/{total}: {example.example_id} (JSONL line {example.line_number})")
    print("-" * width)
    print(error if error is not None else (response.rstrip() or "<empty response>"))
    print("-" * width)
    print(f"Expected: {example.expected}")
    print(f"Parsed:   {parsed_answer if parsed_answer is not None else '<not found>'}")
    print(f"Result:   {'CORRECT' if correct else 'INCORRECT'}")
    print(f"Duration: {duration_seconds:.2f}s")


def main() -> None:
    args = parse_args()
    path = args.jsonl_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")

    input_examples = list(iter_examples(path))
    if not input_examples:
        raise ValueError(f"JSONL file contains no examples: {path}")
    examples = sample_examples(input_examples, args.sample_size, args.seed)
    output_path = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else None
    )
    if output_path is not None and output_path.suffix.casefold() != ".json":
        raise ValueError("--output-json must end in .json")

    runner = (
        make_ollama_runner(args)
        if args.backend == "ollama"
        else make_huggingface_runner(args)
    )
    print(
        f"Input: {path}\nInput examples: {len(input_examples)}\n"
        f"Selected examples: {len(examples)}\nSeed: {args.seed}",
        flush=True,
    )
    if output_path is not None:
        print(f"JSON report: {output_path}", flush=True)

    started_at = utc_now()
    results: list[dict[str, Any]] = []
    if output_path is not None:
        write_evaluation_report(
            output_path,
            build_evaluation_report(
                args,
                path,
                len(input_examples),
                examples,
                results,
                started_at,
                "running",
            ),
        )
    for index, example in enumerate(examples, start=1):
        print(
            f"\nRunning example {index}/{len(examples)}: {example.example_id} ...",
            flush=True,
        )
        started = time.perf_counter()
        error: str | None = None
        try:
            response = runner(example)
            parsed_answer = extract_model_answer(response)
            correct = answers_match(
                parsed_answer,
                example.expected,
                task=example.task,
                reference_answer=example.reference_answer,
                example=example,
                layout_dir=args.layout_dir.expanduser().resolve(),
            )
        except Exception as caught_error:
            response = ""
            parsed_answer = None
            correct = False
            error = f"{type(caught_error).__name__}: {caught_error}"
        duration_seconds = time.perf_counter() - started
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
            "formatting_failure": parsed_answer is None and error is None,
            "response": response,
            "error": error,
            "duration_seconds": round(duration_seconds, 6),
        }
        results.append(result)
        if output_path is not None:
            write_evaluation_report(
                output_path,
                build_evaluation_report(
                    args,
                    path,
                    len(input_examples),
                    examples,
                    results,
                    started_at,
                    "running",
                ),
            )
        print_result(
            index,
            len(examples),
            example,
            response,
            parsed_answer,
            correct,
            duration_seconds,
            error,
        )

    correct_count = sum(bool(result["correct"]) for result in results)
    score = correct_count / len(examples)
    if output_path is not None:
        write_evaluation_report(
            output_path,
            build_evaluation_report(
                args,
                path,
                len(input_examples),
                examples,
                results,
                started_at,
                "complete",
            ),
        )
    print("\n" + "=" * 78)
    print("FINAL SCORE")
    print(f"Correct: {correct_count}/{len(examples)}")
    print(f"Score:   {score:.1%}")
    if output_path is not None:
        print(f"JSON:    {output_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
