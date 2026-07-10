"""Generate deterministic training questions from downloaded FloorplanQA layouts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT_DIR = PACKAGE_ROOT / "datasets" / "FloorplanQA-Layouts" / "layouts"
DEFAULT_TOOLING_DIR = PACKAGE_ROOT / "fpqa-tooling"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "dataset" / "train-qa"
OUTPUT_FILENAME = "questions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic pair-distance QA training examples."
    )
    parser.add_argument(
        "--num-examples",
        type=positive_integer,
        required=True,
        help="Number of QA examples to generate.",
    )
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument("--tooling-dir", type=Path, default=DEFAULT_TOOLING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def natural_sort_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def iter_layouts_round_robin(layout_dir: Path) -> Iterator[Path]:
    room_dirs = sorted(path for path in layout_dir.iterdir() if path.is_dir())
    queues = [
        iter(sorted(room_dir.glob("*.json"), key=natural_sort_key))
        for room_dir in room_dirs
    ]

    while queues:
        remaining: list[Iterator[Path]] = []
        for queue in queues:
            try:
                yield next(queue)
                remaining.append(queue)
            except StopIteration:
                continue
        queues = remaining


def load_upstream_questions(tooling_dir: Path) -> ModuleType:
    questions_path = tooling_dir / "src" / "evaluation" / "questions.py"
    if not questions_path.is_file():
        raise FileNotFoundError(
            f"FloorplanQA prompt definitions not found at {questions_path}. "
            "Run 'uv run checkout-fpqa-tooling' first."
        )

    spec = importlib.util.spec_from_file_location(
        "floorplanqa_upstream_questions", questions_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load prompt definitions from {questions_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def polygon_centroid(points: list[dict[str, float]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("An object must contain at least two points")
    if len(points) == 2:
        return (
            (float(points[0]["x"]) + float(points[1]["x"])) / 2.0,
            (float(points[0]["y"]) + float(points[1]["y"])) / 2.0,
        )

    coordinates = [(float(point["x"]), float(point["y"])) for point in points]
    area_twice = 0.0
    x_sum = 0.0
    y_sum = 0.0
    for index, (x1, y1) in enumerate(coordinates):
        x2, y2 = coordinates[(index + 1) % len(coordinates)]
        cross = x1 * y2 - x2 * y1
        area_twice += cross
        x_sum += (x1 + x2) * cross
        y_sum += (y1 + y2) * cross

    if math.isclose(area_twice, 0.0, abs_tol=1e-12):
        return (
            sum(x for x, _ in coordinates) / len(coordinates),
            sum(y for _, y in coordinates) / len(coordinates),
        )

    return (x_sum / (3.0 * area_twice), y_sum / (3.0 * area_twice))


def objects_and_openings(layout: dict[str, Any]) -> list[dict[str, Any]]:
    openings = layout.get("openings") or {}
    return [
        *(layout.get("objects") or []),
        *(openings.get("windows") or []),
        *(openings.get("doors") or []),
    ]


def select_object_pair(
    layout: dict[str, Any], source_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    objects = objects_and_openings(layout)
    if len(objects) < 2:
        raise ValueError(f"Layout has fewer than two objects: {source_path}")

    identity = f"{layout.get('room_type')}:{layout.get('layout_id')}:{source_path.name}"
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    selected = random.Random(seed).sample(objects, 2)
    return selected[0], selected[1]


def generate_example(
    source_path: Path,
    layout_dir: Path,
    prompt_template: str,
    system_prompt: str,
) -> dict[str, Any]:
    layout = json.loads(source_path.read_text(encoding="utf-8"))
    object_1, object_2 = select_object_pair(layout, source_path)
    center_1 = polygon_centroid(object_1["points"])
    center_2 = polygon_centroid(object_2["points"])
    answer = math.dist(center_1, center_2)
    room_type = str(layout.get("room_type") or source_path.parent.name)
    layout_id = layout.get("layout_id", source_path.stem)
    source_group = source_path.parent.name

    question = prompt_template.format(
        room_type=room_type,
        format="JSON",
        room=json.dumps(layout, ensure_ascii=False),
        obj1=object_1.get("label", "unknown"),
        obj2=object_2.get("label", "unknown"),
    ).strip()
    formatted_answer = f"{answer:.3f}"

    return {
        "id": f"pair-distance-{source_group}-{layout_id}",
        "task": "pair_distance",
        "layout_id": layout_id,
        "room_type": room_type,
        "source_layout": str(source_path.relative_to(layout_dir)),
        "object_1": object_1.get("label", "unknown"),
        "object_2": object_2.get("label", "unknown"),
        "question": question,
        "answer": formatted_answer,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": f"*Final answer*: {formatted_answer}",
            },
        ],
    }


def main() -> None:
    args = parse_args()
    layout_dir = args.layout_dir.expanduser().resolve()
    tooling_dir = args.tooling_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not layout_dir.is_dir():
        raise FileNotFoundError(
            f"FloorplanQA layouts not found at {layout_dir}. "
            "Run 'uv run download-floorplan-qa' first."
        )

    upstream_questions = load_upstream_questions(tooling_dir)
    prompt_template = upstream_questions.PROMPTS["pair_distance"]
    system_prompt = upstream_questions.SYSTEM_PROMPTS["pair_distance"]

    examples: list[dict[str, Any]] = []
    for source_path in iter_layouts_round_robin(layout_dir):
        try:
            examples.append(
                generate_example(
                    source_path, layout_dir, prompt_template, system_prompt
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            print(f"Skipping {source_path}: {error}")
            continue
        if len(examples) == args.num_examples:
            break

    if len(examples) != args.num_examples:
        raise RuntimeError(
            f"Requested {args.num_examples} examples, but only generated {len(examples)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    with output_path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            json.dump(example, output_file, ensure_ascii=False)
            output_file.write("\n")

    print(f"Generated {len(examples)} QA examples at {output_path}")


if __name__ == "__main__":
    main()
