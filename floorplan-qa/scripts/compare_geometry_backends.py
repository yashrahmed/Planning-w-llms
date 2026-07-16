#!/usr/bin/env python3
"""Snapshot placement and repositioning tool behavior for backend comparisons."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from floorplan_qa.floorplan_tools import FloorplanToolRuntime


DISTANCE_PATTERN = re.compile(r" can move ([0-9]+(?:\.[0-9]+)?) meters ")


def placement_status(output: str) -> bool | None:
    if "The answer is True." in output:
        return True
    if "The answer is False." in output:
        return False
    return None


def snapshot(questions_path: Path, layout_dir: Path) -> dict[str, Any]:
    runtime = FloorplanToolRuntime(layout_dir)
    records = [
        json.loads(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        record
        for record in records
        if record.get("task") in {"placement", "repositioning"}
    ]
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for record in selected:
        task = str(record["task"])
        parameters = record["parameters"]
        if task == "placement":
            output = runtime.find_space_with_size(
                float(parameters["object_width"]),
                float(parameters["object_depth"]),
                str(record["layout_file"]),
            )
            actual = placement_status(output)
            expected = bool(record["reference_answer"])
            results.append(
                {
                    "id": record["id"],
                    "task": task,
                    "expected": expected,
                    "actual": actual,
                    "matches": actual is expected,
                    "output": output,
                }
            )
            continue

        output = runtime.test_movement(
            str(parameters["object_to_move"]),
            str(parameters["direction"]),
            str(record["layout_file"]),
        )
        match = DISTANCE_PATTERN.search(output)
        actual_distance = float(match.group(1)) if match else None
        expected_distance = round(float(record["reference_answer"]), 3)
        results.append(
            {
                "id": record["id"],
                "task": task,
                "expected": expected_distance,
                "actual": actual_distance,
                "matches": actual_distance == expected_distance,
                "output": output,
            }
        )

    placement = [item for item in results if item["task"] == "placement"]
    repositioning = [item for item in results if item["task"] == "repositioning"]
    return {
        "questions": str(questions_path),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": {
            "total": len(results),
            "placement": {
                "total": len(placement),
                "matches": sum(bool(item["matches"]) for item in placement),
                "true": sum(item["actual"] is True for item in placement),
                "false": sum(item["actual"] is False for item in placement),
                "inconclusive": sum(item["actual"] is None for item in placement),
            },
            "repositioning": {
                "total": len(repositioning),
                "matches": sum(bool(item["matches"]) for item in repositioning),
            },
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("datasets/train-qa/questions.jsonl"),
    )
    parser.add_argument(
        "--layout-dir",
        type=Path,
        default=Path("datasets/train-qa"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = snapshot(arguments.questions, arguments.layout_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"elapsed_seconds={report['elapsed_seconds']:.6f}")


if __name__ == "__main__":
    main()
