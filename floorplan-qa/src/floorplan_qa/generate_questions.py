"""Generate seeded FloorplanQA examples with deterministic geometry solvers."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import random
import re
from dataclasses import dataclass
from itertools import combinations
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


@dataclass(frozen=True)
class TaskResult:
    parameters: dict[str, Any]
    answer_value: Any
    answer_text: str
    instruction: str
    output_description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a seeded, balanced mix of FloorplanQA task types."
    )
    parser.add_argument(
        "--num-examples",
        type=positive_integer,
        required=True,
        help="Number of QA examples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global seed controlling layout and task selection (default: 0).",
    )
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    paths: list[Path] = []
    for room_dir in sorted(path for path in layout_dir.iterdir() if path.is_dir()):
        paths.extend(sorted(room_dir.glob("*.json"), key=natural_sort_key))
    stable_rng(seed, "layout-order").shuffle(paths)
    return paths


def task_schedule(count: int, seed: int) -> list[str]:
    schedule: list[str] = []
    block_index = 0
    while len(schedule) < count:
        block = list(TASKS)
        stable_rng(seed, "task-block", block_index).shuffle(block)
        schedule.extend(block)
        block_index += 1
    return schedule[:count]


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
    return LayoutContext(
        source_path=source_path,
        source_group=source_path.parent.name,
        layout=layout,
        layout_id=str(layout.get("layout_id", source_path.stem)),
        room_type=str(layout.get("room_type") or source_path.parent.name),
        room=room,
        objects=objects,
        entities=entities,
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
    step: float = 0.01,
) -> float:
    obstacle_union = union_polygons(obstacles)
    if not room.covers(moving):
        return 0.0
    if obstacle_union is not None and moving.intersection(obstacle_union).area > 1e-8:
        return 0.0

    min_x, min_y, max_x, max_y = room.bounds
    max_distance = math.hypot(max_x - min_x, max_y - min_y)
    last_valid = 0.0
    for step_index in range(1, math.ceil(max_distance / step) + 1):
        distance = step_index * step
        moved = translate(
            moving,
            xoff=direction[0] * distance,
            yoff=direction[1] * distance,
        )
        if not room.covers(moved):
            break
        if obstacle_union is not None and moved.intersection(obstacle_union).area > 1e-8:
            break
        last_valid = distance
    return last_valid


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
) -> tuple[Polygon, float]:
    extents = [0.01, 0.01, 0.01, 0.01]
    if not free_space.covers(rectangle_from_extents(center, extents, angle)):
        return Polygon(), 0.0

    for _ in range(4):
        previous_area = (extents[0] + extents[1]) * (extents[2] + extents[3])
        for side in range(4):
            low = extents[side]
            high = cap
            for _ in range(32):
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
    return rectangle, float(rectangle.area)


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
        best_angle = 0.0
    else:
        min_x, min_y, max_x, max_y = context.room.bounds
        cap = math.hypot(max_x - min_x, max_y - min_y)
        centers = sample_points_in_geometry(free_space, rng, 24)
        angles = [index * math.pi / 12 for index in range(12)]
        best_area = 0.0
        best_angle = 0.0
        for center in centers:
            for angle in angles:
                _, area = grow_rectangle(free_space, center, angle, cap)
                if area > best_area:
                    best_area = area
                    best_angle = angle
        answer = best_area

    return TaskResult(
        parameters={
            "angle_samples": 12,
            "center_samples": 24,
            "best_angle_degrees": round(math.degrees(best_angle), 3),
        },
        answer_value=answer,
        answer_text=format_number(answer),
        instruction=(
            "calculate the area of the largest rectangle that can fit at any "
            "rotation without overlapping blocking objects or openings; rugs "
            "and ceiling-only fixtures are nonblocking"
        ),
        output_description="an area in square meters rounded to three decimal places",
    )


def centered_rectangle(
    center: tuple[float, float], width: float, depth: float, angle: float
) -> Polygon:
    return rectangle_from_extents(
        center, [width / 2.0, width / 2.0, depth / 2.0, depth / 2.0], angle
    )


def placement_task(context: LayoutContext, rng: random.Random) -> TaskResult:
    catalog = PLACEMENT_CATALOG.get(
        context.source_group, PLACEMENT_CATALOG["hssd"]
    )
    object_name, width, depth = rng.choice(catalog)
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

    fit = False
    if not free_space.is_empty:
        centers = sample_points_in_geometry(free_space, rng, 160)
        angles = [index * math.pi / 24 for index in range(24)]
        for center in centers:
            if any(
                free_space.covers(centered_rectangle(center, width, depth, angle))
                for angle in angles
            ):
                fit = True
                break

    return TaskResult(
        parameters={
            "object_name": object_name,
            "object_width": width,
            "object_depth": depth,
            "angle_samples": 24,
            "center_samples": 160,
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


def visibility_nodes(geometry: Any) -> list[tuple[float, float]]:
    nodes: list[tuple[float, float]] = []
    for polygon in geometry_polygons(geometry):
        nodes.extend(
            (round(float(x), 6), round(float(y), 6))
            for x, y in list(polygon.exterior.coords)[:-1]
        )
    return nodes


def shortest_visibility_path(
    context: LayoutContext,
    start_entity: dict[str, Any],
    goal_entity: dict[str, Any],
    clearance: float,
) -> list[tuple[float, float]]:
    start = entity_centroid(start_entity)
    goal = entity_centroid(goal_entity)
    blocking_polygons = [
        entity_polygon(entity).buffer(clearance, join_style=2)
        for entity in context.entities
        if entity is not start_entity
        and entity is not goal_entity
        and not is_soft_covering(entity)
        and not is_ceiling_fixture(entity)
    ]
    blockers = union_polygons(blocking_polygons)
    blocker_interior = (
        blockers.buffer(-1e-7) if blockers is not None and not blockers.is_empty else None
    )

    nodes = [start, goal]
    if blockers is not None:
        nodes.extend(visibility_nodes(blockers))
    nodes.extend(visibility_nodes(context.room))
    nodes = list(dict.fromkeys(nodes))
    graph: dict[tuple[float, float], list[tuple[float, tuple[float, float]]]] = {
        node: [] for node in nodes
    }

    for first_index, first in enumerate(nodes):
        for second in nodes[first_index + 1 :]:
            segment = LineString([first, second])
            if not context.room.covers(segment):
                continue
            if (
                blocker_interior is not None
                and not blocker_interior.is_empty
                and not segment.disjoint(blocker_interior)
            ):
                continue
            distance = float(segment.length)
            graph[first].append((distance, second))
            graph[second].append((distance, first))

    distances = {node: math.inf for node in nodes}
    previous: dict[tuple[float, float], tuple[float, float] | None] = {
        node: None for node in nodes
    }
    distances[start] = 0.0
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node]:
            continue
        if node == goal:
            break
        for weight, neighbor in graph[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))

    if not math.isfinite(distances[goal]):
        return []
    path = []
    current: tuple[float, float] | None = goal
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    return path


def shortest_path_task(context: LayoutContext, rng: random.Random) -> TaskResult:
    entities = eligible_path_entities(context)
    pairs = list(combinations(entities, 2))
    rng.shuffle(pairs)
    clearance = 0.15
    selected_pair = None
    path: list[tuple[float, float]] = []
    for first, second in pairs[:20]:
        candidate = shortest_visibility_path(
            context, first, second, clearance=clearance
        )
        if len(candidate) >= 2:
            selected_pair = (first, second)
            path = candidate
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
            "algorithm": "visibility_graph_dijkstra",
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


def build_question(context: LayoutContext, result: TaskResult) -> str:
    room_json = json.dumps(
        context.layout, ensure_ascii=False, separators=(",", ":")
    )
    return (
        f"Given the {context.room_type} layout below in JSON, {result.instruction}.\n\n"
        f"Room layout:\n{room_json}\n\n"
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
) -> dict[str, Any]:
    rng = stable_rng(seed, context.source_group, context.layout_id, task)
    result = TASK_GENERATORS[task](context, rng)
    question = build_question(context, result)
    system_prompt = (
        "Use exact polygon geometry where possible. Always provide a final answer "
        "and do not return ERROR merely because the computation is difficult."
    )
    return {
        "id": f"{task.replace('_', '-')}-{context.source_group}-{context.layout_id}",
        "task": task,
        "layout_id": context.layout_id,
        "room_type": context.room_type,
        "source_layout": str(context.source_path.relative_to(layout_dir)),
        "parameters": result.parameters,
        "question": question,
        "answer": result.answer_text,
        "reference_answer": result.answer_value,
        "provenance": {
            "global_seed": seed,
            "solver_version": "experimental-v1",
            "task_selection": "seeded-balanced-blocks",
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


def main() -> None:
    args = parse_args()
    source_dir = args.layout_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"FloorplanQA layouts not found at {source_dir}. "
            "Run 'uv run download-floorplan-qa' first."
        )

    paths = layout_paths(source_dir, args.seed)
    schedule = task_schedule(args.num_examples, args.seed)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    path_index = 0
    while len(records) < args.num_examples and path_index < len(paths):
        source_path = paths[path_index]
        path_index += 1
        task = schedule[len(records)]
        try:
            context = load_layout(source_path)
            records.append(
                generate_record(context, source_dir, task, args.seed)
            )
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{source_path}: {task}: {error}")

    if len(records) != args.num_examples:
        details = "\n".join(failures[-10:])
        raise RuntimeError(
            f"Requested {args.num_examples} examples, generated {len(records)}.\n"
            f"Recent failures:\n{details}"
        )

    output_path = output_dir / OUTPUT_FILENAME
    write_records(records, output_path)
    counts = {task: sum(record["task"] == task for record in records) for task in TASKS}
    print(f"Generated {len(records)} QA examples at {output_path}")
    print(f"Seed: {args.seed}")
    print("Task counts:")
    for task, count in counts.items():
        print(f"  {task}: {count}")
    if failures:
        print(f"Skipped layouts: {len(failures)}")


if __name__ == "__main__":
    main()
