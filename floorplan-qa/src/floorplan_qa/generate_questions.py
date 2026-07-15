"""Generate seeded FloorplanQA examples with deterministic geometry solvers."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from itertools import combinations
from pathlib import Path
from typing import Any

from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.validation import make_valid

from .geometry import (
    DEFAULT_GRID_RESOLUTION,
    GEOMETRY_TOLERANCE,
    REPOSITION_TOLERANCE,
    LayoutContext,
    TaskResult,
    centered_rectangle,
    edge_angles,
    entity_centroid,
    entity_polygon,
    format_number,
    geometry_polygons,
    intersecting_entities,
    is_ceiling_fixture,
    is_soft_covering,
    label,
    load_layout,
    max_box_task,
    maximum_slide_distance,
    polygon_from_points,
    sample_points_in_geometry,
    shortest_grid_path,
    stable_rng,
    union_polygons,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT_DIR = PACKAGE_ROOT / "datasets" / "FloorplanQA-Layouts" / "layouts"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "datasets" / "train-qa"
OUTPUT_FILENAME = "questions.jsonl"
GENERATION_REPORT_FILENAME = "generation-report.json"
DATASET_SPLITS = ("train", "test", "val")
SOLVER_VERSION = "paper-v2"
TASKS = (
    "pair_distance",
    "free_space",
    "view_angle",
    "repositioning",
    "max_box",
    "placement",
    "shortest_path",
    "visibility",
)

MOVABLE_LABELS = {
    "kitchen": ("stove", "fridge", "sink", "dishwasher", "table", "chair"),
    "living_room": (
        "sofa",
        "loveseat",
        "armchair",
        "coffee table",
        "side table",
        "tv stand",
        "bookshelf",
        "plant",
    ),
    "bedroom": (
        "bed",
        "dresser",
        "wardrobe",
        "desk",
        "chair",
        "bookshelf",
        "ottoman",
        "plant",
    ),
}
PATH_LABELS = {
    "kitchen": ("stove", "fridge", "sink", "dishwasher", "door", "window", "table"),
    "living_room": (
        "door",
        "window",
        "sofa",
        "loveseat",
        "armchair",
        "coffee table",
        "side table",
        "tv stand",
        "television",
        "bookshelf",
        "fireplace",
    ),
    "bedroom": ("door", "window", "bed", "dresser", "wardrobe", "desk", "bookshelf", "chair"),
}
PLACEMENT_CATALOG = {
    "kitchen": (
        ("compact kitchen cart", 0.8, 0.5),
        ("dining table", 1.8, 0.9),
        ("large kitchen island", 2.4, 1.2),
        ("commercial prep table", 3.0, 1.0),
    ),
    "living_room": (
        ("side table", 0.8, 0.8),
        ("antique storage chest", 2.5, 1.0),
        ("large sectional sofa", 3.8, 2.5),
        ("game table", 3.0, 1.7),
    ),
    "bedroom": (
        ("bedside table", 0.6, 0.5),
        ("desk table", 2.0, 1.0),
        ("wardrobe", 3.0, 0.8),
        ("large bed", 2.5, 2.2),
    ),
    "hssd": (
        ("small cabinet", 0.8, 0.5),
        ("desk table", 2.0, 1.0),
        ("antique storage chest", 2.5, 1.0),
        ("large storage unit", 3.5, 1.2),
    ),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all eight deterministic FloorplanQA tasks for each selected layout."
        )
    )
    parser.add_argument(
        "--num-layouts",
        type=positive_integer,
        required=True,
        help="Number of layouts to select (each emits eight QA records).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global seed controlling layout and task selection (default: 0).",
    )
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split",
        choices=DATASET_SPLITS,
        default="train",
        help="Dataset split included in emitted layout filenames (default: train).",
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=DEFAULT_GRID_RESOLUTION,
        help="Grid spacing in meters for paper-style shortest-path A* (default: 0.10).",
    )
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

def layout_paths(layout_dir: Path, seed: int) -> list[Path]:
    """Return a seeded uniform shuffle over every released layout."""
    paths: list[Path] = []
    for room_dir in sorted(path for path in layout_dir.iterdir() if path.is_dir()):
        paths.extend(sorted(room_dir.glob("*.json"), key=natural_sort_key))
    stable_rng(seed, "layout-order").shuffle(paths)
    return paths

def select_pair(
    entities: list[dict[str, Any]], rng: random.Random
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(entities) < 2:
        raise ValueError("fewer than two eligible entities")
    first, second = rng.sample(entities, 2)
    return first, second

def pair_distance_task(context: LayoutContext, rng: random.Random) -> TaskResult:
    first, second = select_pair(context.entities, rng)
    answer = math.dist(entity_centroid(first), entity_centroid(second))
    return TaskResult(
        parameters={"object_1": label(first), "object_2": label(second)},
        answer_value=answer,
        answer_text=format_number(answer),
        instruction=(
            f"calculate the Euclidean distance in meters between the polygon "
            f"centroids of '{label(first)}' and '{label(second)}'"
        ),
        output_description="a float in meters rounded to three decimal places",
        solver_metadata={"algorithm": "area_weighted_centroid_euclidean"},
    )

def free_space_task(context: LayoutContext, _: random.Random) -> TaskResult:
    occupied = [
        entity_polygon(entity)
        for entity in context.objects
        if not is_ceiling_fixture(entity)
        and polygon_from_points(entity.get("points") or []) is not None
    ]
    occupied_union = union_polygons(occupied)
    free = (
        context.room
        if occupied_union is None
        else context.room.difference(context.room.intersection(occupied_union))
    )
    answer = float(free.area)
    return TaskResult(
        parameters={},
        answer_value=answer,
        answer_text=format_number(answer),
        instruction=(
            "calculate the total non-occupied floor area in square meters; "
            "union overlapping object polygons, ignore doors, windows, and "
            "ceiling-only fixtures, but count floor coverings as occupied"
        ),
        output_description="a float in square meters rounded to three decimal places",
        solver_metadata={"algorithm": "room_difference_occupied_union"},
    )

def view_angle_task(context: LayoutContext, rng: random.Random) -> TaskResult:
    first, second = select_pair(context.entities, rng)
    start = entity_centroid(first)
    end = entity_centroid(second)
    dx, dy = end[0] - start[0], end[1] - start[1]
    magnitude = math.hypot(dx, dy)
    if magnitude <= 1e-12:
        raise ValueError("selected entities have coincident centroids")
    cosine = max(-1.0, min(1.0, dy / magnitude))
    answer = math.degrees(math.acos(cosine))
    return TaskResult(
        parameters={"object_1": label(first), "object_2": label(second)},
        answer_value=answer,
        answer_text=format_number(answer),
        instruction=(
            f"compute the smallest absolute angle in degrees between the vector "
            f"from the centroid of '{label(first)}' to the centroid of "
            f"'{label(second)}' and global north (0, 1)"
        ),
        output_description="an angle from 0 to 180 degrees rounded to three decimal places",
        solver_metadata={"algorithm": "normalized_dot_acos"},
    )

def candidate_movable_objects(context: LayoutContext) -> list[dict[str, Any]]:
    candidates = [
        entity
        for entity in context.objects
        if not is_soft_covering(entity)
        and not is_ceiling_fixture(entity)
        and polygon_from_points(entity.get("points") or []) is not None
    ]
    preferred_tokens = MOVABLE_LABELS.get(context.source_group, ())
    preferred = [
        entity
        for entity in candidates
        if any(token in label(entity).lower() for token in preferred_tokens)
    ]
    return preferred or candidates

def repositioning_task(context: LayoutContext, rng: random.Random) -> TaskResult:
    movable = candidate_movable_objects(context)
    if not movable:
        raise ValueError("layout has no movable objects")
    rng.shuffle(movable)
    directions = [
        ("up", (0.0, 1.0)),
        ("down", (0.0, -1.0)),
        ("left", (-1.0, 0.0)),
        ("right", (1.0, 0.0)),
    ]
    rng.shuffle(directions)

    selected = movable[0]
    selected_direction = directions[0]
    answer = 0.0
    for moving_entity in movable:
        moving_polygon = entity_polygon(moving_entity)
        obstacles = [
            entity_polygon(entity)
            for entity in context.objects
            if entity is not moving_entity
            and not is_soft_covering(entity)
            and not is_ceiling_fixture(entity)
            and polygon_from_points(entity.get("points") or []) is not None
        ]
        for direction in directions:
            distance = maximum_slide_distance(
                moving_polygon, context.room, obstacles, direction[1]
            )
            selected = moving_entity
            selected_direction = direction
            answer = distance
            if distance >= 0.01:
                break
        if answer >= 0.01:
            break

    return TaskResult(
        parameters={
            "object_to_move": label(selected),
            "direction": selected_direction[0],
        },
        answer_value=answer,
        answer_text=format_number(answer),
        instruction=(
            f"calculate how far '{label(selected)}' can move in the "
            f"'{selected_direction[0]}' direction before touching another "
            "blocking object or the room boundary"
        ),
        output_description="a distance in meters rounded to three decimal places",
        solver_metadata={
            "algorithm": "continuous_swept_volume_bisection",
            "distance_tolerance_m": REPOSITION_TOLERANCE,
        },
    )

def configuration_space_region(
    free_space: Any, width: float, depth: float, angle: float
) -> Any:
    """Return the corner-constraint center region for a rotated rectangle."""
    cosine, sine = math.cos(angle), math.sin(angle)
    offsets = []
    for x, y in (
        (-width / 2.0, -depth / 2.0),
        (width / 2.0, -depth / 2.0),
        (width / 2.0, depth / 2.0),
        (-width / 2.0, depth / 2.0),
    ):
        offsets.append((x * cosine - y * sine, x * sine + y * cosine))
    region = None
    for offset_x, offset_y in offsets:
        translated = translate(free_space, xoff=-offset_x, yoff=-offset_y)
        region = translated if region is None else region.intersection(translated)
        if region.is_empty:
            break
    return make_valid(region) if region is not None else Polygon()

def placement_witness(
    free_space: Any,
    width: float,
    depth: float,
    rng: random.Random,
) -> dict[str, Any] | None:
    angles = set(edge_angles(free_space))
    angles.update(round(index * math.pi / 48, 10) for index in range(48))
    for angle in sorted(angles):
        region = configuration_space_region(free_space, width, depth, angle)
        if region.is_empty:
            continue
        for center in sample_points_in_geometry(region, rng, 16):
            rectangle = centered_rectangle(center, width, depth, angle)
            if free_space.buffer(GEOMETRY_TOLERANCE).covers(rectangle):
                return {
                    "center": [round(center[0], 8), round(center[1], 8)],
                    "rotation_degrees": round(math.degrees(angle), 8),
                }
    return None

def placement_false_certificate(
    free_space: Any, width: float, depth: float
) -> dict[str, Any] | None:
    component_areas = [float(part.area) for part in geometry_polygons(free_space)]
    maximum_component_area = max(component_areas, default=0.0)
    query_area = width * depth
    if maximum_component_area + GEOMETRY_TOLERANCE < query_area:
        return {
            "type": "free_component_area_upper_bound",
            "maximum_component_area_m2": maximum_component_area,
            "query_rectangle_area_m2": query_area,
        }
    return None

def placement_task(context: LayoutContext, rng: random.Random) -> TaskResult:
    catalog = PLACEMENT_CATALOG.get(
        context.source_group, PLACEMENT_CATALOG["hssd"]
    )
    blockers = [
        entity_polygon(entity)
        for entity in context.entities
        if not is_soft_covering(entity) and not is_ceiling_fixture(entity)
    ]
    blocker_union = union_polygons(blockers)
    free_space = (
        context.room
        if blocker_union is None
        else make_valid(context.room.difference(blocker_union))
    )

    candidates = list(catalog)
    rng.shuffle(candidates)
    selected: tuple[str, float, float] | None = None
    fit = False
    witness: dict[str, Any] | None = None
    certificate: dict[str, Any] | None = None
    selection_attempt = 0
    for candidate_index, (object_name, width, depth) in enumerate(candidates, start=1):
        candidate_witness = (
            None
            if free_space.is_empty
            else placement_witness(free_space, width, depth, rng)
        )
        candidate_certificate = placement_false_certificate(free_space, width, depth)
        candidate_fit = candidate_witness is not None
        certified = candidate_fit or candidate_certificate is not None
        if certified:
            selected = (object_name, width, depth)
            fit = candidate_fit
            witness = candidate_witness
            certificate = candidate_certificate
            selection_attempt = candidate_index
            break
    if selected is None:
        raise ValueError("could not find a witnessed or certified placement case")
    object_name, width, depth = selected

    return TaskResult(
        parameters={
            "object_name": object_name,
            "object_width": width,
            "object_depth": depth,
            "witness": witness,
            "false_certificate": certificate,
            "uniform_catalog_selection_attempt": selection_attempt,
        },
        answer_value=fit,
        answer_text="True" if fit else "False",
        instruction=(
            f"determine whether the rectangle '{object_name}' with width "
            f"{width:.3f} m and depth {depth:.3f} m can fit fully inside the "
            "room at any rotation without overlapping blocking objects or openings"
        ),
        output_description="exactly True or False",
        solver_metadata={
            "algorithm": "configuration_space_with_exact_witness",
            "rotation_step_degrees": 3.75,
            "answer_certified": witness is not None or certificate is not None,
            "parameter_selection": "uniform_catalog_order_with_certified_rejection",
        },
    )

def eligible_path_entities(context: LayoutContext) -> list[dict[str, Any]]:
    entities = [
        entity
        for entity in context.entities
        if not is_soft_covering(entity) and not is_ceiling_fixture(entity)
    ]
    preferred_tokens = PATH_LABELS.get(context.source_group, ())
    preferred = [
        entity
        for entity in entities
        if any(token in label(entity).lower() for token in preferred_tokens)
    ]
    return preferred if len(preferred) >= 2 else entities

def shortest_path_task(
    context: LayoutContext,
    rng: random.Random,
    resolution: float = DEFAULT_GRID_RESOLUTION,
) -> TaskResult:
    entities = eligible_path_entities(context)
    pairs = list(combinations(entities, 2))
    rng.shuffle(pairs)
    clearance = 0.15
    selected_pair = None
    path: list[tuple[float, float]] = []
    metadata: dict[str, Any] = {}
    for first, second in pairs[:50]:
        candidate, candidate_metadata = shortest_grid_path(
            context,
            first,
            second,
            clearance=clearance,
            resolution=resolution,
        )
        if len(candidate) >= 2:
            selected_pair = (first, second)
            path = candidate
            metadata = candidate_metadata
            break
    if selected_pair is None:
        raise ValueError("could not find a connected entity pair")

    rounded_path = [[round(x, 3), round(y, 3)] for x, y in path]
    answer_text = json.dumps(rounded_path, separators=(",", ":"))
    return TaskResult(
        parameters={
            "object_1": label(selected_pair[0]),
            "object_2": label(selected_pair[1]),
            "clearance": clearance,
            "grid_resolution": resolution,
        },
        answer_value=rounded_path,
        answer_text=answer_text,
        instruction=(
            f"determine a shortest valid waypoint path from the centroid of "
            f"'{label(selected_pair[0])}' to the centroid of "
            f"'{label(selected_pair[1])}' while maintaining {clearance:.2f} m "
            "clearance from all other blocking objects"
        ),
        output_description="a JSON list of [x, y] waypoints rounded to three decimals",
        solver_metadata=metadata,
    )

def visibility_task(context: LayoutContext, rng: random.Random) -> TaskResult:
    pairs = list(combinations(context.entities, 2))
    rng.shuffle(pairs)
    first, second = pairs[0]
    hits: list[dict[str, Any]] = []
    for candidate_first, candidate_second in pairs[:100]:
        candidate_hits = intersecting_entities(
            context.entities, candidate_first, candidate_second
        )
        first, second, hits = candidate_first, candidate_second, candidate_hits
        if hits:
            break
    answer = [label(entity) for entity in hits]
    answer_text = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    return TaskResult(
        parameters={"object_1": label(first), "object_2": label(second)},
        answer_value=answer,
        answer_text=answer_text,
        instruction=(
            f"find every entity polygon intersecting the line segment from the "
            f"centroid of '{label(first)}' to the centroid of '{label(second)}'; "
            "exclude the starting and ending entities"
        ),
        output_description="a JSON list of intersecting entity labels in traversal order",
        solver_metadata={"algorithm": "actual_polygon_segment_intersection"},
    )

TASK_GENERATORS = {
    "pair_distance": pair_distance_task,
    "free_space": free_space_task,
    "view_angle": view_angle_task,
    "repositioning": repositioning_task,
    "max_box": max_box_task,
    "placement": placement_task,
    "shortest_path": shortest_path_task,
    "visibility": visibility_task,
}

def layout_filename(context: LayoutContext, split: str) -> str:
    if split not in DATASET_SPLITS:
        raise ValueError(f"unsupported dataset split: {split}")
    return f"{context.source_group}-{context.layout_id}-{split}.json"

def build_question(result: TaskResult, room_layout_file: str) -> str:
    return (
        f"Given the layout of the room, {result.instruction}.\n\n"
        f"Room layout can be found in file : {room_layout_file}\n\n"
        "Briefly show the geometric steps used. If required data is invalid or "
        "missing, return '*Final answer*: ERROR'. Otherwise put the answer on "
        "the last line exactly as:\n"
        f"*Final answer*: <answer>\n"
        f"Where <answer> is {result.output_description}."
    )

def generate_record(
    context: LayoutContext,
    layout_dir: Path,
    task: str,
    seed: int,
    grid_resolution: float = DEFAULT_GRID_RESOLUTION,
    split: str = "train",
) -> dict[str, Any]:
    rng = stable_rng(seed, context.source_group, context.layout_id, task)
    result = (
        shortest_path_task(context, rng, resolution=grid_resolution)
        if task == "shortest_path"
        else TASK_GENERATORS[task](context, rng)
    )
    room_layout_file = layout_filename(context, split)
    question = build_question(result, room_layout_file)
    system_prompt = (
        "Use exact polygon geometry where possible. Always provide a final answer "
        "and do not return ERROR merely because the computation is difficult."
    )
    return {
        "id": f"{task.replace('_', '-')}-{context.source_group}-{context.layout_id}",
        "task": task,
        "layout_id": context.layout_id,
        "room_type": context.room_type,
        "split": split,
        "layout_file": room_layout_file,
        "source_layout": str(context.source_path.relative_to(layout_dir)),
        "parameters": result.parameters,
        "question": question,
        "answer": result.answer_text,
        "reference_answer": result.answer_value,
        "provenance": {
            "global_seed": seed,
            "solver_version": SOLVER_VERSION,
            "compatibility_mode": "paper",
            "prompt_version": "fixed-template-v5-concise-layout-file",
            "layout_selection": "sha256-uniform-all-layouts",
            "task_selection": "all-eight-per-layout",
            "solver": result.solver_metadata,
            "validation": context.validation,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": f"*Final answer*: {result.answer_text}",
            },
        ],
    }

def write_records(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            json.dump(record, output_file, ensure_ascii=False)
            output_file.write("\n")

def write_layout_files(
    contexts: list[LayoutContext], output_dir: Path, split: str
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames: list[str] = []
    for context in contexts:
        filename = layout_filename(context, split)
        (output_dir / filename).write_text(
            json.dumps(context.layout, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        filenames.append(filename)
    return filenames

def main() -> None:
    args = parse_args()
    source_dir = args.layout_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"FloorplanQA layouts not found at {source_dir}. "
            "Run 'uv run download-floorplan-qa' first."
        )
    if args.grid_resolution <= 0.0:
        raise ValueError("--grid-resolution must be positive")

    paths = layout_paths(source_dir, args.seed)
    records: list[dict[str, Any]] = []
    contexts: list[LayoutContext] = []
    failures: list[str] = []
    emitted_layouts = 0
    for source_path in paths:
        if emitted_layouts >= args.num_layouts:
            break
        try:
            context = load_layout(source_path)
            layout_records = [
                generate_record(
                    context,
                    source_dir,
                    task,
                    args.seed,
                    grid_resolution=args.grid_resolution,
                    split=args.split,
                )
                for task in TASKS
            ]
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{source_path}: {error}")
            continue
        records.extend(layout_records)
        contexts.append(context)
        emitted_layouts += 1

    if emitted_layouts != args.num_layouts:
        details = "\n".join(failures[-10:])
        raise RuntimeError(
            f"Requested {args.num_layouts} layouts, generated {emitted_layouts}.\n"
            f"Recent failures:\n{details}"
        )

    output_path = output_dir / OUTPUT_FILENAME
    emitted_layout_files = write_layout_files(contexts, output_dir, args.split)
    write_records(records, output_path)
    counts = {task: sum(record["task"] == task for record in records) for task in TASKS}
    print(
        f"Generated {len(records)} QA examples from {emitted_layouts} layouts "
        f"at {output_path}"
    )
    print(f"Seed: {args.seed}")
    print("Task counts:")
    for task, count in counts.items():
        print(f"  {task}: {count}")
    source_counts: dict[str, int] = {}
    for record in records[:: len(TASKS)]:
        source = Path(record["source_layout"]).parent.name
        source_counts[source] = source_counts.get(source, 0) + 1
    print("Layout source counts:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")
    considered_layouts = emitted_layouts + len(failures)
    generation_report = {
        "seed": args.seed,
        "split": args.split,
        "requested_layouts": args.num_layouts,
        "emitted_layouts": emitted_layouts,
        "records": len(records),
        "considered_layouts": considered_layouts,
        "skipped_layouts": len(failures),
        "complete_layout_yield": emitted_layouts / considered_layouts,
        "task_counts": counts,
        "layout_source_counts": source_counts,
        "layout_selection": "sha256-uniform-all-layouts",
        "layout_files": emitted_layout_files,
        "failures": failures,
    }
    report_path = output_dir / GENERATION_REPORT_FILENAME
    report_path.write_text(
        json.dumps(generation_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Complete layout yield: {generation_report['complete_layout_yield']:.1%} "
        f"({emitted_layouts}/{considered_layouts})"
    )
    print(f"Generation report: {report_path}")
    if failures:
        print(f"Skipped layouts: {len(failures)}")

if __name__ == "__main__":
    main()
