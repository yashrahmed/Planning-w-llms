"""Generate seeded FloorplanQA examples with deterministic geometry solvers."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import random
import re
from dataclasses import dataclass, field
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

from shapely.affinity import translate
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, polygonize, unary_union
from shapely.validation import make_valid

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT_DIR = PACKAGE_ROOT / "datasets" / "FloorplanQA-Layouts" / "layouts"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "datasets" / "train-qa"
OUTPUT_FILENAME = "questions.jsonl"
GENERATION_REPORT_FILENAME = "generation-report.json"
DATASET_SPLITS = ("train", "test", "val")
SOLVER_VERSION = "paper-v2"
GEOMETRY_TOLERANCE = 1e-7
REPOSITION_TOLERANCE = 1e-5
MAX_BOX_RELATIVE_TOLERANCE = 0.02
DEFAULT_GRID_RESOLUTION = 0.10
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

SOFT_COVERING_PATTERN = re.compile(r"rug|carpet|mat|doormat|runner", re.I)
CEILING_FIXTURE_PATTERN = re.compile(r"light|chandelier|fan|pendant", re.I)
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


@dataclass(frozen=True)
class LayoutContext:
    source_path: Path
    source_group: str
    layout: dict[str, Any]
    layout_id: str
    room_type: str
    room: Polygon
    objects: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    validation: dict[str, list[str]]


@dataclass(frozen=True)
class TaskResult:
    parameters: dict[str, Any]
    answer_value: Any
    answer_text: str
    instruction: str
    output_description: str
    solver_metadata: dict[str, Any] = field(default_factory=dict)


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


def stable_rng(seed: int, *parts: object) -> random.Random:
    identity = ":".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


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


def label(entity: dict[str, Any]) -> str:
    return str(entity.get("label") or entity.get("name") or "").strip()


def is_soft_covering(entity: dict[str, Any]) -> bool:
    return bool(SOFT_COVERING_PATTERN.search(label(entity)))


def is_ceiling_fixture(entity: dict[str, Any]) -> bool:
    return bool(CEILING_FIXTURE_PATTERN.search(label(entity)))


def geometry_polygons(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    polygons: list[Polygon] = []
    for part in getattr(geometry, "geoms", []):
        polygons.extend(geometry_polygons(part))
    return polygons


def polygon_from_points(points: list[dict[str, Any]]) -> Polygon | None:
    if len(points) < 3:
        return None
    polygon = make_valid(
        Polygon([(float(point["x"]), float(point["y"])) for point in points])
    )
    polygons = [part for part in geometry_polygons(polygon) if part.area > 1e-10]
    return max(polygons, key=lambda part: part.area) if polygons else None


def build_room_polygon(layout: dict[str, Any]) -> Polygon:
    boundary = layout.get("room_boundary") or []
    polygon = polygon_from_points(boundary)
    if polygon is not None:
        return polygon

    lines = []
    for wall in layout.get("walls") or (layout.get("room") or {}).get("walls") or []:
        start = wall.get("start")
        end = wall.get("end")
        if start and end:
            lines.append(
                LineString(
                    [
                        (float(start["x"]), float(start["y"])),
                        (float(end["x"]), float(end["y"])),
                    ]
                )
            )
    faces = list(polygonize(linemerge(unary_union(lines)))) if lines else []
    if not faces:
        raise ValueError("could not reconstruct a room polygon")
    return max(faces, key=lambda face: face.area)


def objects_and_openings(layout: dict[str, Any]) -> list[dict[str, Any]]:
    openings = layout.get("openings") or {}
    return [
        *(layout.get("objects") or []),
        *(openings.get("windows") or []),
        *(openings.get("doors") or []),
    ]


def validate_layout_geometry(
    layout: dict[str, Any], room: Polygon, entities: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """Audit the fixed released corpus without recreating its unpublished filter."""
    errors: list[str] = []
    warnings: list[str] = []
    if room.is_empty or room.area <= GEOMETRY_TOLERANCE:
        errors.append("room boundary is empty or degenerate")

    declared = objects_and_openings(layout)
    invalid_labels = [
        label(entity) or "<unlabeled>"
        for entity in declared
        if polygon_from_points(entity.get("points") or []) is None
    ]
    if invalid_labels:
        errors.append(f"invalid entity polygons: {', '.join(invalid_labels[:5])}")

    labels = [label(entity).casefold() for entity in entities]
    duplicates = sorted({name for name in labels if labels.count(name) > 1})
    if duplicates:
        errors.append(f"ambiguous duplicate labels: {', '.join(duplicates[:5])}")

    openings = layout.get("openings") or {}
    opening_ids = {
        id(entity)
        for entity in [
            *(openings.get("doors") or []),
            *(openings.get("windows") or []),
        ]
    }
    for entity in entities:
        polygon = entity_polygon(entity)
        outside_area = polygon.difference(room.buffer(1e-5)).area
        if outside_area > 1e-5:
            message = f"{label(entity)!r} extends {outside_area:.6g} m2 outside the room"
            if id(entity) in opening_ids:
                warnings.append(message)
            else:
                errors.append(message)

    for opening in [*(openings.get("doors") or []), *(openings.get("windows") or [])]:
        polygon = polygon_from_points(opening.get("points") or [])
        if polygon is not None and polygon.distance(room.boundary) > 0.05:
            warnings.append(f"opening {label(opening)!r} is not attached to the boundary")

    blockers = [
        entity
        for entity in (layout.get("objects") or [])
        if not is_soft_covering(entity)
        and not is_ceiling_fixture(entity)
        and polygon_from_points(entity.get("points") or []) is not None
    ]
    overlap_count = 0
    for first, second in combinations(blockers, 2):
        if entity_polygon(first).intersection(entity_polygon(second)).area > 1e-5:
            overlap_count += 1
    if overlap_count:
        warnings.append(f"{overlap_count} blocking-object overlap(s) detected")
    return {"errors": errors, "warnings": warnings}


def load_layout(source_path: Path) -> LayoutContext:
    layout = json.loads(source_path.read_text(encoding="utf-8"))
    room = build_room_polygon(layout)
    objects = list(layout.get("objects") or [])
    entities = [
        entity
        for entity in objects_and_openings(layout)
        if label(entity) and polygon_from_points(entity.get("points") or []) is not None
    ]
    if len(entities) < 2:
        raise ValueError("layout has fewer than two polygonal entities")
    validation = validate_layout_geometry(layout, room, entities)
    if validation["errors"]:
        raise ValueError("; ".join(validation["errors"]))
    return LayoutContext(
        source_path=source_path,
        source_group=source_path.parent.name,
        layout=layout,
        layout_id=str(layout.get("layout_id", source_path.stem)),
        room_type=str(layout.get("room_type") or source_path.parent.name),
        room=room,
        objects=objects,
        entities=entities,
        validation=validation,
    )


def entity_polygon(entity: dict[str, Any]) -> Polygon:
    polygon = polygon_from_points(entity.get("points") or [])
    if polygon is None:
        raise ValueError(f"invalid polygon for entity {label(entity)!r}")
    return polygon


def polygon_centroid(points: list[dict[str, float]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("an entity must contain at least two points")
    if len(points) == 2:
        return (
            (float(points[0]["x"]) + float(points[1]["x"])) / 2.0,
            (float(points[0]["y"]) + float(points[1]["y"])) / 2.0,
        )
    polygon = polygon_from_points(points)
    if polygon is None:
        raise ValueError("could not calculate polygon centroid")
    return (float(polygon.centroid.x), float(polygon.centroid.y))


def entity_centroid(entity: dict[str, Any]) -> tuple[float, float]:
    return polygon_centroid(entity.get("points") or [])


def select_pair(
    entities: list[dict[str, Any]], rng: random.Random
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(entities) < 2:
        raise ValueError("fewer than two eligible entities")
    first, second = rng.sample(entities, 2)
    return first, second


def union_polygons(polygons: list[Polygon]) -> Any:
    return unary_union(polygons) if polygons else None


def format_number(value: float) -> str:
    return f"{value:.3f}"


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


def maximum_slide_distance(
    moving: Polygon,
    room: Polygon,
    obstacles: list[Polygon],
    direction: tuple[float, float],
    tolerance: float = REPOSITION_TOLERANCE,
) -> float:
    """Continuously bracket the first collision using a monotone swept volume."""
    obstacle_union = union_polygons(obstacles)
    if not room.covers(moving):
        return 0.0
    if obstacle_union is not None and moving.intersection(obstacle_union).area > 1e-8:
        return 0.0

    def swept_volume(distance: float) -> Any:
        moved = translate(
            moving,
            xoff=direction[0] * distance,
            yoff=direction[1] * distance,
        )
        pieces: list[Any] = [moving, moved]
        for first, second in pairwise(list(moving.exterior.coords)):
            shifted_first = (
                first[0] + direction[0] * distance,
                first[1] + direction[1] * distance,
            )
            shifted_second = (
                second[0] + direction[0] * distance,
                second[1] + direction[1] * distance,
            )
            pieces.append(Polygon([first, second, shifted_second, shifted_first]))
        return unary_union(pieces)

    def collision_by(distance: float) -> bool:
        swept = swept_volume(distance)
        if swept.difference(room).area > GEOMETRY_TOLERANCE:
            return True
        return bool(
            obstacle_union is not None
            and swept.intersection(obstacle_union).area > GEOMETRY_TOLERANCE
        )

    min_x, min_y, max_x, max_y = room.bounds
    low = 0.0
    high = math.hypot(max_x - min_x, max_y - min_y) + max(
        max_x - min_x, max_y - min_y
    )
    if not collision_by(high):
        raise ValueError("could not bracket repositioning collision")
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if collision_by(middle):
            high = middle
        else:
            low = middle
    return low


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
            "blocking object or the room boundary; rugs and ceiling-only "
            "fixtures are nonblocking"
        ),
        output_description="a distance in meters rounded to three decimal places",
        solver_metadata={
            "algorithm": "continuous_swept_volume_bisection",
            "distance_tolerance_m": REPOSITION_TOLERANCE,
        },
    )


def rectangle_from_extents(
    center: tuple[float, float],
    extents: list[float],
    angle: float,
) -> Polygon:
    left, right, down, up = extents
    cosine, sine = math.cos(angle), math.sin(angle)
    local = [(-left, -down), (right, -down), (right, up), (-left, up)]
    points = [
        (
            center[0] + x * cosine - y * sine,
            center[1] + x * sine + y * cosine,
        )
        for x, y in local
    ]
    return Polygon(points)


def sample_points_in_geometry(
    geometry: Any, rng: random.Random, count: int
) -> list[tuple[float, float]]:
    points = [
        (float(polygon.representative_point().x), float(polygon.representative_point().y))
        for polygon in geometry_polygons(geometry)
    ]
    min_x, min_y, max_x, max_y = geometry.bounds
    attempts = 0
    while len(points) < count and attempts < count * 100:
        point = (rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
        if geometry.covers(Point(point)):
            points.append(point)
        attempts += 1
    return points


def grow_rectangle(
    free_space: Any,
    center: tuple[float, float],
    angle: float,
    cap: float,
) -> tuple[Polygon, float, list[float]]:
    extents = [0.01, 0.01, 0.01, 0.01]
    if not free_space.covers(rectangle_from_extents(center, extents, angle)):
        return Polygon(), 0.0, extents

    for _ in range(3):
        previous_area = (extents[0] + extents[1]) * (extents[2] + extents[3])
        for side in range(4):
            low = extents[side]
            high = cap
            for _ in range(24):
                middle = (low + high) / 2.0
                candidate_extents = list(extents)
                candidate_extents[side] = middle
                if free_space.covers(
                    rectangle_from_extents(center, candidate_extents, angle)
                ):
                    low = middle
                else:
                    high = middle
            extents[side] = low
        area = (extents[0] + extents[1]) * (extents[2] + extents[3])
        if area - previous_area < 1e-6:
            break

    rectangle = rectangle_from_extents(center, extents, angle)
    return rectangle, float(rectangle.area), extents


def edge_angles(geometry: Any) -> list[float]:
    values = {round(index * math.pi / 12, 10) for index in range(12)}
    for polygon in geometry_polygons(geometry):
        for first, second in pairwise(list(polygon.exterior.coords)):
            dx, dy = second[0] - first[0], second[1] - first[1]
            if math.hypot(dx, dy) > GEOMETRY_TOLERANCE:
                values.add(round(math.atan2(dy, dx) % math.pi, 10))
    return sorted(values)


def rectangle_parameters_from_growth(
    center: tuple[float, float], extents: list[float], angle: float
) -> list[float]:
    left, right, down, up = extents
    local_x = (right - left) / 2.0
    local_y = (up - down) / 2.0
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        center[0] + local_x * cosine - local_y * sine,
        center[1] + local_x * sine + local_y * cosine,
        left + right,
        down + up,
        angle % math.pi,
    ]


def rectangle_from_parameters(parameters: list[float]) -> Polygon:
    x, y, width, depth, angle = parameters
    return centered_rectangle((x, y), width, depth, angle)


def optimize_maximum_rectangle(
    free_space: Any, room: Polygon, rng: random.Random
) -> tuple[list[float], dict[str, Any]]:
    """Deterministic global search with exact feasibility and local refinement."""
    min_x, min_y, max_x, max_y = room.bounds
    cap = math.hypot(max_x - min_x, max_y - min_y)
    angles = edge_angles(free_space)
    centers = sample_points_in_geometry(free_space, rng, 10)
    seeds: list[tuple[float, list[float]]] = []
    for center in centers:
        for angle in angles[:20]:
            _, area, extents = grow_rectangle(free_space, center, angle, cap)
            if area > GEOMETRY_TOLERANCE:
                seeds.append(
                    (area, rectangle_parameters_from_growth(center, extents, angle))
                )
    if not seeds:
        return [0.0, 0.0, 0.0, 0.0, 0.0], {
            "algorithm": "deterministic_differential_evolution",
            "converged": True,
            "iterations": 0,
            "relative_improvement_window": 0.0,
        }

    bounds = [
        (min_x, max_x),
        (min_y, max_y),
        (0.001, cap),
        (0.001, cap),
        (0.0, math.pi),
    ]

    def normalize(candidate: list[float]) -> list[float]:
        normalized = []
        for value, (lower, upper) in zip(candidate, bounds, strict=True):
            normalized.append(max(lower, min(upper, value)))
        normalized[4] %= math.pi
        return normalized

    def fitness(candidate: list[float]) -> float:
        if candidate[2] <= 0.0 or candidate[3] <= 0.0:
            return 0.0
        rectangle = rectangle_from_parameters(candidate)
        return (
            float(rectangle.area)
            if free_space.buffer(GEOMETRY_TOLERANCE).covers(rectangle)
            else 0.0
        )

    seeds.sort(key=lambda item: item[0], reverse=True)
    population_size = 36
    population = [parameters for _, parameters in seeds[:population_size]]
    while len(population) < population_size:
        center = centers[len(population) % len(centers)]
        population.append(
            [center[0], center[1], 0.02, 0.02, rng.uniform(0.0, math.pi)]
        )
    scores = [fitness(candidate) for candidate in population]
    history = [max(scores)]
    converged = False
    relative_improvement = math.inf
    iterations = 0
    for generation in range(1, 61):
        for index, target in enumerate(population):
            choices = [item for item in range(population_size) if item != index]
            first, second, third = rng.sample(choices, 3)
            mutant = [
                population[first][dimension]
                + 0.72
                * (
                    population[second][dimension]
                    - population[third][dimension]
                )
                for dimension in range(5)
            ]
            mutant = normalize(mutant)
            forced = rng.randrange(5)
            trial = [
                mutant[dimension]
                if dimension == forced or rng.random() < 0.82
                else target[dimension]
                for dimension in range(5)
            ]
            trial_score = fitness(trial)
            if trial_score >= scores[index]:
                population[index] = trial
                scores[index] = trial_score
        history.append(max(scores))
        iterations = generation
        if generation >= 20:
            old = history[-11]
            current = history[-1]
            relative_improvement = (current - old) / max(current, GEOMETRY_TOLERANCE)
            if relative_improvement <= MAX_BOX_RELATIVE_TOLERANCE:
                converged = True
                break

    best_index = max(range(population_size), key=scores.__getitem__)
    best = list(population[best_index])
    best_score = scores[best_index]
    ranges = [max_x - min_x, max_y - min_y, cap, cap, math.pi]
    for scale in (0.05, 0.02, 0.01, 0.005, 0.002):
        improved = True
        while improved:
            improved = False
            for dimension in range(5):
                for sign in (-1.0, 1.0):
                    candidate = list(best)
                    candidate[dimension] += sign * ranges[dimension] * scale
                    candidate = normalize(candidate)
                    score = fitness(candidate)
                    if score > best_score + GEOMETRY_TOLERANCE:
                        best, best_score = candidate, score
                        improved = True

    metadata = {
        "algorithm": "deterministic_differential_evolution",
        "population": population_size,
        "iterations": iterations,
        "converged": converged,
        "relative_tolerance": MAX_BOX_RELATIVE_TOLERANCE,
        "relative_improvement_window": round(relative_improvement, 8),
        "angle_candidates": len(angles),
        "best_valid_lower_bound_m2": best_score,
    }
    return best, metadata


def max_box_task(context: LayoutContext, rng: random.Random) -> TaskResult:
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
    if free_space.is_empty:
        answer = 0.0
        best = [0.0, 0.0, 0.0, 0.0, 0.0]
        metadata = {
            "algorithm": "deterministic_differential_evolution",
            "converged": True,
            "iterations": 0,
            "relative_tolerance": MAX_BOX_RELATIVE_TOLERANCE,
            "relative_improvement_window": 0.0,
            "best_valid_lower_bound_m2": 0.0,
        }
    else:
        best, metadata = optimize_maximum_rectangle(free_space, context.room, rng)
        answer = float(best[2] * best[3])

    return TaskResult(
        parameters={
            "witness": {
                "center": [round(best[0], 8), round(best[1], 8)],
                "width": round(best[2], 8),
                "depth": round(best[3], 8),
                "rotation_degrees": round(math.degrees(best[4]), 8),
            },
        },
        answer_value=answer,
        answer_text=format_number(answer),
        instruction=(
            "calculate the area of the largest rectangle that can fit at any "
            "rotation without overlapping blocking objects or openings; rugs "
            "and ceiling-only fixtures are nonblocking"
        ),
        output_description="an area in square meters rounded to three decimal places",
        solver_metadata=metadata,
    )


def centered_rectangle(
    center: tuple[float, float], width: float, depth: float, angle: float
) -> Polygon:
    return rectangle_from_extents(
        center, [width / 2.0, width / 2.0, depth / 2.0, depth / 2.0], angle
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
            "room at any rotation without overlapping blocking objects or "
            "openings; rugs and ceiling-only fixtures are nonblocking"
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


def navigation_geometry(
    context: LayoutContext,
    start_entity: dict[str, Any],
    goal_entity: dict[str, Any],
    clearance: float,
) -> tuple[Any, Any, Any]:
    blocking_polygons = [
        entity_polygon(entity).buffer(clearance, join_style=2)
        for entity in context.entities
        if entity is not start_entity
        and entity is not goal_entity
        and not is_soft_covering(entity)
        and not is_ceiling_fixture(entity)
    ]
    blockers = union_polygons(blocking_polygons)
    navigable = context.room.buffer(-clearance, join_style=2)
    if blockers is not None:
        navigable = make_valid(navigable.difference(blockers))
    start_space = make_valid(
        navigable.union(entity_polygon(start_entity).intersection(context.room))
    )
    goal_space = make_valid(
        navigable.union(entity_polygon(goal_entity).intersection(context.room))
    )
    return navigable, start_space, goal_space


def simplify_collinear_path(
    path: list[tuple[float, float]], tolerance: float = 1e-9
) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    for index in range(1, len(path) - 1):
        first = simplified[-1]
        middle = path[index]
        last = path[index + 1]
        cross = (middle[0] - first[0]) * (last[1] - middle[1]) - (
            middle[1] - first[1]
        ) * (last[0] - middle[0])
        if abs(cross) > tolerance:
            simplified.append(middle)
    simplified.append(path[-1])
    return simplified


def shortest_grid_path(
    context: LayoutContext,
    start_entity: dict[str, Any],
    goal_entity: dict[str, Any],
    clearance: float,
    resolution: float = DEFAULT_GRID_RESOLUTION,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    start = entity_centroid(start_entity)
    goal = entity_centroid(goal_entity)
    navigable, start_space, goal_space = navigation_geometry(
        context, start_entity, goal_entity, clearance
    )
    if navigable.is_empty:
        return [], {}

    direct = LineString([start, goal])
    if make_valid(start_space.union(goal_space)).buffer(GEOMETRY_TOLERANCE).covers(
        direct
    ):
        return [start, goal], {
            "algorithm": "grid_astar",
            "grid_resolution_m": resolution,
            "connectivity": 8,
            "raw_grid_nodes": 0,
        }

    min_x, min_y, max_x, max_y = context.room.bounds
    x_start = math.floor(min_x / resolution)
    x_end = math.ceil(max_x / resolution)
    y_start = math.floor(min_y / resolution)
    y_end = math.ceil(max_y / resolution)
    nodes: dict[tuple[int, int], tuple[float, float]] = {}
    padded_navigable = navigable.buffer(GEOMETRY_TOLERANCE)
    for x_index in range(x_start, x_end + 1):
        for y_index in range(y_start, y_end + 1):
            coordinate = (
                round(x_index * resolution, 10),
                round(y_index * resolution, 10),
            )
            if padded_navigable.covers(Point(coordinate)):
                nodes[(x_index, y_index)] = coordinate
    if not nodes:
        return [], {}

    def connector_nodes(
        endpoint: tuple[float, float], connection_space: Any
    ) -> dict[tuple[int, int], float]:
        nearest = sorted(
            nodes.items(), key=lambda item: math.dist(endpoint, item[1])
        )[:96]
        connected: dict[tuple[int, int], float] = {}
        for node, coordinate in nearest:
            segment = LineString([endpoint, coordinate])
            if connection_space.buffer(GEOMETRY_TOLERANCE).covers(segment):
                connected[node] = float(segment.length)
                if len(connected) >= 12:
                    break
        return connected

    start_connections = connector_nodes(start, start_space)
    goal_connections = connector_nodes(goal, goal_space)
    if not start_connections or not goal_connections:
        return [], {}

    directions = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    distances: dict[tuple[int, int], float] = {}
    previous: dict[tuple[int, int], tuple[int, int] | None] = {}
    queue: list[tuple[float, float, tuple[int, int]]] = []
    for node, cost in start_connections.items():
        distances[node] = cost
        previous[node] = None
        heapq.heappush(queue, (cost + math.dist(nodes[node], goal), cost, node))

    reached: tuple[int, int] | None = None
    while queue:
        _, distance, node = heapq.heappop(queue)
        if distance > distances.get(node, math.inf) + GEOMETRY_TOLERANCE:
            continue
        if node in goal_connections:
            reached = node
            break
        first = nodes[node]
        for dx, dy in directions:
            neighbor = (node[0] + dx, node[1] + dy)
            if neighbor not in nodes:
                continue
            second = nodes[neighbor]
            segment = LineString([first, second])
            if not padded_navigable.covers(segment):
                continue
            candidate = distance + float(segment.length)
            if candidate + GEOMETRY_TOLERANCE < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                previous[neighbor] = node
                priority = candidate + math.dist(second, goal)
                heapq.heappush(queue, (priority, candidate, neighbor))

    if reached is None:
        return [], {}
    grid_path: list[tuple[float, float]] = []
    current: tuple[int, int] | None = reached
    while current is not None:
        grid_path.append(nodes[current])
        current = previous[current]
    grid_path.reverse()
    path = simplify_collinear_path([start, *grid_path, goal])
    return path, {
        "algorithm": "grid_astar",
        "grid_resolution_m": resolution,
        "connectivity": 8,
        "raw_grid_nodes": len(grid_path),
    }


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


def intersecting_entities(
    entities: list[dict[str, Any]],
    start_entity: dict[str, Any],
    goal_entity: dict[str, Any],
) -> list[dict[str, Any]]:
    start = entity_centroid(start_entity)
    goal = entity_centroid(goal_entity)
    segment = LineString([start, goal])
    hits: list[tuple[float, dict[str, Any]]] = []
    for entity in entities:
        if entity is start_entity or entity is goal_entity:
            continue
        intersection = segment.intersection(entity_polygon(entity))
        if intersection.is_empty:
            continue
        if intersection.geom_type == "Point" and (
            intersection.equals(Point(start)) or intersection.equals(Point(goal))
        ):
            continue
        location = float(segment.project(intersection.representative_point()))
        hits.append((location, entity))
    hits.sort(key=lambda item: (item[0], label(item[1]).casefold()))
    return [entity for _, entity in hits]


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
            "prompt_version": "fixed-template-v4-generic-layout-file",
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
