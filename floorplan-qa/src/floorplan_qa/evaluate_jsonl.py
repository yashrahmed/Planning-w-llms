"""Evaluate FloorplanQA JSONL examples with a local model backend."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

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


def answers_match(actual: str | None, expected: str) -> bool:
    if actual is None:
        return False

    actual_number = decimal_value(actual)
    expected_number = decimal_value(expected)
    if actual_number is not None and expected_number is not None:
        decimal_places = max(0, -expected_number.as_tuple().exponent)
        quantum = Decimal(1).scaleb(-decimal_places)
        return actual_number.quantize(quantum) == expected_number.quantize(quantum)

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
        correct = answers_match(parsed_answer, example.expected)
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
