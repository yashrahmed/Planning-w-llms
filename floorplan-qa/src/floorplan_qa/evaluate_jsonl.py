"""Evaluate FloorplanQA JSONL examples with a local model backend."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from shapely.geometry import LineString
from shapely.validation import make_valid

from .generate_questions import (
    DEFAULT_LAYOUT_DIR,
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
            )


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
    if example.source_layout is None or len(path) < 2:
        return False
    context = load_layout(layout_dir / example.source_layout)
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


def make_ollama_runner(args: argparse.Namespace) -> Callable[[Example], str]:
    model = args.model or DEFAULT_OLLAMA_MODEL

    def run(example: Example) -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": example.messages,
                "stream": False,
                "think": args.thinking,
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": 16384,
                    "num_predict": args.max_tokens,
                },
            }
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
        mx.random.seed(42)
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


def print_result(
    index: int,
    total: int,
    example: Example,
    response: str,
    parsed_answer: str | None,
    correct: bool,
) -> None:
    width = 78
    print("\n" + "=" * width)
    print(f"Example {index}/{total}: {example.example_id} (JSONL line {example.line_number})")
    print("-" * width)
    print(response.rstrip() or "<empty response>")
    print("-" * width)
    print(f"Expected: {example.expected}")
    print(f"Parsed:   {parsed_answer if parsed_answer is not None else '<not found>'}")
    print(f"Result:   {'CORRECT' if correct else 'INCORRECT'}")


def main() -> None:
    args = parse_args()
    path = args.jsonl_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")

    examples = list(iter_examples(path))
    if not examples:
        raise ValueError(f"JSONL file contains no examples: {path}")

    runner = (
        make_ollama_runner(args)
        if args.backend == "ollama"
        else make_huggingface_runner(args)
    )
    print(f"Input: {path}\nExamples: {len(examples)}", flush=True)

    correct_count = 0
    for index, example in enumerate(examples, start=1):
        print(
            f"\nRunning example {index}/{len(examples)}: {example.example_id} ...",
            flush=True,
        )
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
        correct_count += int(correct)
        print_result(index, len(examples), example, response, parsed_answer, correct)

    score = correct_count / len(examples)
    print("\n" + "=" * 78)
    print("FINAL SCORE")
    print(f"Correct: {correct_count}/{len(examples)}")
    print(f"Score:   {score:.1%}")
    print("=" * 78)


if __name__ == "__main__":
    main()
