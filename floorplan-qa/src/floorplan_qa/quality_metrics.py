"""Measure structural and geometric quality of generated FloorplanQA records."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from shapely.affinity import translate
from shapely.geometry import LineString
from shapely.ops import unary_union
from shapely.validation import make_valid

from .generate_questions import (
    DEFAULT_LAYOUT_DIR,
    GENERATION_REPORT_FILENAME,
    TASKS,
    generate_record,
)
from .geometry import (
    DEFAULT_GRID_RESOLUTION,
    GEOMETRY_TOLERANCE,
    MAX_BOX_RELATIVE_TOLERANCE,
    centered_rectangle,
    entity_centroid,
    entity_polygon,
    geometry_polygons,
    is_ceiling_fixture,
    is_soft_covering,
    label,
    load_layout,
    navigation_geometry,
)

RATE_THRESHOLDS = {
    "complete_layout_yield": 0.95,
    "layout_completeness_rate": 1.0,
    "schema_validity_rate": 1.0,
    "prompt_validity_rate": 1.0,
    "reference_reproducibility_rate": 1.0,
    "geometry_witness_validity_rate": 1.0,
    "placement_certification_rate": 1.0,
    "path_validity_rate": 1.0,
    "max_box_convergence_rate": 1.0,
    "deterministic_match_rate": 1.0,
}

REQUIRED_FIELDS = {
    "id",
    "task",
    "layout_id",
    "room_type",
    "split",
    "layout_file",
    "source_layout",
    "parameters",
    "question",
    "answer",
    "reference_answer",
    "provenance",
    "messages",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic FloorplanQA question-generation quality."
    )
    parser.add_argument("jsonl_path", type=Path)
    parser.add_argument("--compare", type=Path, help="Second generation for byte comparison.")
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(value)
    return records


def rate(passed: int, total: int) -> float:
    return passed / total if total else 0.0


def record_schema_valid(record: dict[str, Any]) -> bool:
    if not REQUIRED_FIELDS.issubset(record):
        return False
    if record["task"] not in TASKS:
        return False
    if not isinstance(record["parameters"], dict):
        return False
    if record["split"] not in ("train", "test", "val"):
        return False
    source_layout = record.get("source_layout")
    if not isinstance(source_layout, str):
        return False
    expected_layout_file = (
        f"{Path(source_layout).parent.name}-{record['layout_id']}-"
        f"{record['split']}.json"
    )
    if record["layout_file"] != expected_layout_file:
        return False
    provenance = record["provenance"]
    if not isinstance(provenance, dict):
        return False
    if provenance.get("task_selection") != "all-eight-per-layout":
        return False
    if provenance.get("compatibility_mode") != "paper":
        return False
    messages = record["messages"]
    return bool(
        isinstance(messages, list)
        and [message.get("role") for message in messages]
        == ["system", "user", "assistant"]
    )


def record_prompt_valid(record: dict[str, Any]) -> bool:
    question = record.get("question")
    messages = record.get("messages")
    if not isinstance(question, str) or not isinstance(messages, list):
        return False
    layout_reference = (
        f"Room layout can be found in file : {record.get('layout_file')}"
    )
    return bool(
        layout_reference in question
        and "Room layout:\n{" not in question
        and "*Final answer*: <answer>" in question
        and messages[1].get("content") == question
        and messages[2].get("content")
        == f"*Final answer*: {record.get('answer')}"
    )


def free_space_for_blocking_tasks(context: Any) -> Any:
    blockers = [
        entity_polygon(entity)
        for entity in context.entities
        if not is_soft_covering(entity) and not is_ceiling_fixture(entity)
    ]
    if not blockers:
        return context.room
    return make_valid(context.room.difference(unary_union(blockers)))


def max_box_witness_valid(record: dict[str, Any], context: Any) -> bool:
    witness = record["parameters"].get("witness") or {}
    try:
        center = tuple(float(value) for value in witness["center"])
        width = float(witness["width"])
        depth = float(witness["depth"])
        angle = math.radians(float(witness["rotation_degrees"]))
    except (KeyError, TypeError, ValueError):
        return False
    rectangle = centered_rectangle(center, width, depth, angle)
    free_space = free_space_for_blocking_tasks(context)
    reference = float(record["reference_answer"])
    return bool(
        free_space.buffer(1e-6).covers(rectangle)
        and abs(rectangle.area - reference) <= max(1e-5, reference * 1e-5)
    )


def placement_certified(record: dict[str, Any], context: Any) -> bool:
    parameters = record["parameters"]
    width = float(parameters["object_width"])
    depth = float(parameters["object_depth"])
    free_space = free_space_for_blocking_tasks(context)
    if bool(record["reference_answer"]):
        witness = parameters.get("witness") or {}
        try:
            center = tuple(float(value) for value in witness["center"])
            angle = math.radians(float(witness["rotation_degrees"]))
        except (KeyError, TypeError, ValueError):
            return False
        return free_space.buffer(1e-6).covers(
            centered_rectangle(center, width, depth, angle)
        )

    certificate = parameters.get("false_certificate") or {}
    if certificate.get("type") != "free_component_area_upper_bound":
        return False
    maximum_component_area = max(
        (float(part.area) for part in geometry_polygons(free_space)), default=0.0
    )
    return maximum_component_area + GEOMETRY_TOLERANCE < width * depth


def entities_by_label(context: Any) -> dict[str, dict[str, Any]]:
    return {label(entity): entity for entity in context.entities}


def path_valid(record: dict[str, Any], context: Any) -> bool:
    parameters = record["parameters"]
    entities = entities_by_label(context)
    try:
        first = entities[parameters["object_1"]]
        second = entities[parameters["object_2"]]
        clearance = float(parameters["clearance"])
        path = [tuple(float(value) for value in point) for point in record["reference_answer"]]
    except (KeyError, TypeError, ValueError):
        return False
    if len(path) < 2:
        return False
    if math.dist(path[0], entity_centroid(first)) > 0.002:
        return False
    if math.dist(path[-1], entity_centroid(second)) > 0.002:
        return False
    navigable, start_space, goal_space = navigation_geometry(
        context, first, second, clearance
    )
    tolerance = 0.002
    if len(path) == 2:
        return make_valid(start_space.union(goal_space)).buffer(tolerance).covers(
            LineString(path)
        )
    segments = [LineString([path[index], path[index + 1]]) for index in range(len(path) - 1)]
    if not start_space.buffer(tolerance).covers(segments[0]):
        return False
    if not goal_space.buffer(tolerance).covers(segments[-1]):
        return False
    return all(
        navigable.buffer(tolerance).covers(segment) for segment in segments[1:-1]
    )


def reposition_witness_valid(record: dict[str, Any], context: Any) -> bool:
    parameters = record["parameters"]
    entity = next(
        (
            candidate
            for candidate in context.objects
            if label(candidate) == parameters.get("object_to_move")
        ),
        None,
    )
    if entity is None:
        return False
    directions = {
        "up": (0.0, 1.0),
        "down": (0.0, -1.0),
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
    }
    direction = directions.get(parameters.get("direction"))
    if direction is None:
        return False
    distance = float(record["reference_answer"])
    moved = translate(
        entity_polygon(entity),
        xoff=direction[0] * distance,
        yoff=direction[1] * distance,
    )
    obstacles = [
        entity_polygon(candidate)
        for candidate in context.objects
        if candidate is not entity
        and not is_soft_covering(candidate)
        and not is_ceiling_fixture(candidate)
    ]
    obstacle_union = unary_union(obstacles) if obstacles else None
    return bool(
        moved.difference(context.room).area <= GEOMETRY_TOLERANCE
        and (
            obstacle_union is None
            or moved.intersection(obstacle_union).area <= GEOMETRY_TOLERANCE
        )
    )


def assess(
    records: list[dict[str, Any]], layout_dir: Path, compare_equal: bool | None
) -> tuple[dict[str, float], list[str]]:
    failures: list[str] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record.get("source_layout", ""), str(record.get("layout_id")))].append(record)

    complete = 0
    for key, group in grouped.items():
        tasks = Counter(record.get("task") for record in group)
        valid = len(group) == len(TASKS) and all(tasks[task] == 1 for task in TASKS)
        complete += int(valid)
        if not valid:
            failures.append(f"incomplete layout {key}: {dict(tasks)}")

    schema_passes = sum(record_schema_valid(record) for record in records)
    prompt_passes = sum(record_prompt_valid(record) for record in records)

    contexts: dict[str, Any] = {}
    reproducible = 0
    witness_passes = 0
    witness_total = 0
    placement_passes = 0
    placement_total = 0
    path_passes = 0
    path_total = 0
    max_box_converged = 0
    max_box_total = 0
    for record in records:
        relative = str(record["source_layout"])
        if relative not in contexts:
            contexts[relative] = load_layout(layout_dir / relative)
        context = contexts[relative]
        provenance = record["provenance"]
        seed = int(provenance["global_seed"])
        grid_resolution = float(
            record.get("parameters", {}).get(
                "grid_resolution", DEFAULT_GRID_RESOLUTION
            )
        )
        regenerated = generate_record(
            context,
            layout_dir,
            record["task"],
            seed,
            grid_resolution=grid_resolution,
            split=str(record["split"]),
        )
        deterministic_fields = (
            "parameters",
            "answer",
            "reference_answer",
            "question",
            "layout_file",
        )
        same = all(regenerated[field] == record[field] for field in deterministic_fields)
        reproducible += int(same)
        if not same:
            failures.append(f"non-reproducible record {record['id']}")

        valid_witness: bool | None = None
        if record["task"] == "max_box":
            valid_witness = max_box_witness_valid(record, context)
            max_box_total += 1
            solver = provenance.get("solver") or {}
            converged = bool(solver.get("converged")) and float(
                solver.get("relative_improvement_window", math.inf)
            ) <= MAX_BOX_RELATIVE_TOLERANCE
            max_box_converged += int(converged)
        elif record["task"] == "placement":
            valid_witness = placement_certified(record, context)
            placement_total += 1
            placement_passes += int(valid_witness)
        elif record["task"] == "shortest_path":
            valid_witness = path_valid(record, context)
            path_total += 1
            path_passes += int(valid_witness)
        elif record["task"] == "repositioning":
            valid_witness = reposition_witness_valid(record, context)
        if valid_witness is not None:
            witness_total += 1
            witness_passes += int(valid_witness)
            if not valid_witness:
                failures.append(f"invalid geometry witness {record['id']}")

    metrics = {
        "layout_completeness_rate": rate(complete, len(grouped)),
        "schema_validity_rate": rate(schema_passes, len(records)),
        "prompt_validity_rate": rate(prompt_passes, len(records)),
        "reference_reproducibility_rate": rate(reproducible, len(records)),
        "geometry_witness_validity_rate": rate(witness_passes, witness_total),
        "placement_certification_rate": rate(placement_passes, placement_total),
        "path_validity_rate": rate(path_passes, path_total),
        "max_box_convergence_rate": rate(max_box_converged, max_box_total),
    }
    if compare_equal is not None:
        metrics["deterministic_match_rate"] = 1.0 if compare_equal else 0.0
    placement_records = [record for record in records if record["task"] == "placement"]
    metrics["placement_true_rate"] = rate(
        sum(bool(record["reference_answer"]) for record in placement_records),
        len(placement_records),
    )
    return metrics, failures


def main() -> None:
    args = parse_args()
    input_path = args.jsonl_path.expanduser().resolve()
    layout_dir = args.layout_dir.expanduser().resolve()
    records = read_jsonl(input_path)
    compare_equal = None
    if args.compare is not None:
        compare_equal = input_path.read_bytes() == args.compare.expanduser().resolve().read_bytes()
    metrics, failures = assess(records, layout_dir, compare_equal)
    generation_report_path = input_path.parent / GENERATION_REPORT_FILENAME
    if generation_report_path.is_file():
        generation_report = json.loads(generation_report_path.read_text(encoding="utf-8"))
        metrics["complete_layout_yield"] = float(
            generation_report["complete_layout_yield"]
        )
    unique_layouts = {
        (record["source_layout"], record["layout_id"]): record
        for record in records
    }
    observed_sources = Counter(
        Path(record["source_layout"]).parent.name
        for record in unique_layouts.values()
    )
    population_sources = {
        directory.name: len(list(directory.glob("*.json")))
        for directory in layout_dir.iterdir()
        if directory.is_dir()
    }
    observed_total = sum(observed_sources.values())
    population_total = sum(population_sources.values())
    source_rates = {
        source: {
            "observed_count": observed_sources.get(source, 0),
            "observed_rate": rate(observed_sources.get(source, 0), observed_total),
            "population_rate": rate(count, population_total),
        }
        for source, count in sorted(population_sources.items())
    }
    metrics["layout_source_max_absolute_deviation"] = max(
        (
            abs(values["observed_rate"] - values["population_rate"])
            for values in source_rates.values()
        ),
        default=0.0,
    )
    report = {
        "records": len(records),
        "layouts": len({(record["source_layout"], record["layout_id"]) for record in records}),
        "metrics": metrics,
        "thresholds": RATE_THRESHOLDS,
        "advisory": {"placement_true_rate_preferred_range": [0.35, 0.65]},
        "distributions": {
            "tasks": dict(sorted(Counter(record["task"] for record in records).items())),
            "layout_sources": source_rates,
        },
        "failures": failures,
    }
    print(f"Records: {report['records']}  Layouts: {report['layouts']}")
    for name, value in metrics.items():
        threshold = RATE_THRESHOLDS.get(name)
        status = "PASS" if threshold is None or value >= threshold else "FAIL"
        suffix = f" (threshold {threshold:.0%})" if threshold is not None else " (advisory)"
        print(f"{status:4} {name}: {value:.1%}{suffix}")
    print("Observed layout-source distribution:")
    for source, values in source_rates.items():
        print(
            f"  {source}: {values['observed_count']} "
            f"({values['observed_rate']:.1%}; corpus {values['population_rate']:.1%})"
        )
    if failures:
        print("Failures:")
        for failure in failures[:20]:
            print(f"  - {failure}")
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    hard_failures = [
        name
        for name, threshold in RATE_THRESHOLDS.items()
        if name in metrics and metrics[name] < threshold
    ]
    if hard_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
