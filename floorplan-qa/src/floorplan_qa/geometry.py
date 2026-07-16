"""Shared geometry primitives and solvers for FloorplanQA."""

from __future__ import annotations

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
from shapely.ops import linemerge, polygonize, triangulate, unary_union
from shapely.validation import make_valid

GEOMETRY_TOLERANCE = 1e-7

REPOSITION_TOLERANCE = 1e-5

MAX_BOX_RELATIVE_TOLERANCE = 0.02

DEFAULT_GRID_RESOLUTION = 0.10

SOFT_COVERING_PATTERN = re.compile(r"rug|carpet|mat|doormat|runner", re.I)

CEILING_FIXTURE_PATTERN = re.compile(r"light|chandelier|fan|pendant", re.I)

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

def stable_rng(seed: int, *parts: object) -> random.Random:
    identity = ":".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))

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

def union_polygons(polygons: list[Polygon]) -> Any:
    return unary_union(polygons) if polygons else None

def format_number(value: float) -> str:
    return f"{value:.3f}"

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

def halton_value(index: int, base: int) -> float:
    """Return one deterministic low-discrepancy coordinate in [0, 1)."""
    result = 0.0
    fraction = 1.0
    while index:
        fraction /= base
        index, remainder = divmod(index, base)
        result += remainder * fraction
    return result


def deterministic_points_in_geometry(
    geometry: Any, count: int
) -> list[tuple[float, float]]:
    """Return stable interior candidates without an RNG or layout identity.

    Representative points cover every connected polygon component first.
    Triangulation representatives add deterministic coverage of non-convex
    regions, and a Halton sequence fills any remaining budget.
    """
    if geometry is None or geometry.is_empty or count <= 0:
        return []

    points: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()

    def add(point: Point) -> None:
        coordinate = (float(point.x), float(point.y))
        key = (round(coordinate[0], 12), round(coordinate[1], 12))
        if key not in seen and geometry.covers(point):
            seen.add(key)
            points.append(coordinate)

    polygons = sorted(geometry_polygons(geometry), key=lambda part: part.bounds)
    for polygon in polygons:
        add(polygon.representative_point())
        add(polygon.centroid)
    for polygon in polygons:
        for triangle in triangulate(polygon):
            clipped = triangle.intersection(polygon)
            for part in geometry_polygons(clipped):
                if part.area > GEOMETRY_TOLERANCE:
                    add(part.representative_point())
                    if len(points) >= count:
                        return points[:count]

    min_x, min_y, max_x, max_y = geometry.bounds
    index = 1
    attempts = 0
    while len(points) < count and attempts < count * 100:
        candidate = Point(
            min_x + halton_value(index, 2) * (max_x - min_x),
            min_y + halton_value(index, 3) * (max_y - min_y),
        )
        add(candidate)
        index += 1
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
    free_space: Any, room: Polygon
) -> tuple[list[float], dict[str, Any]]:
    """Run a seed-independent numerical search with exact witness validation."""
    min_x, min_y, max_x, max_y = room.bounds
    cap = math.hypot(max_x - min_x, max_y - min_y)
    angles = edge_angles(free_space)
    centers = deterministic_points_in_geometry(free_space, 12)
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
            "algorithm": "deterministic_candidate_refinement",
            "converged": True,
            "global_optimum_certified": False,
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
        angle = angles[len(population) % len(angles)]
        population.append([center[0], center[1], 0.02, 0.02, angle])
    scores = [fitness(candidate) for candidate in population]
    history = [max(scores)]
    converged = False
    relative_improvement = math.inf
    iterations = 0
    for generation in range(1, 61):
        for index, target in enumerate(population):
            choices = [item for item in range(population_size) if item != index]
            offset = (generation * 7 + index * 11) % len(choices)
            ordered = choices[offset:] + choices[:offset]
            first, second, third = ordered[:3]
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
            retained_dimension = (index + generation) % 5
            trial = [
                target[dimension]
                if dimension == retained_dimension
                else mutant[dimension]
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
        "algorithm": "deterministic_candidate_refinement",
        "population": population_size,
        "iterations": iterations,
        "converged": converged,
        "global_optimum_certified": False,
        "relative_tolerance": MAX_BOX_RELATIVE_TOLERANCE,
        "relative_improvement_window": round(relative_improvement, 8),
        "angle_candidates": len(angles),
        "best_valid_lower_bound_m2": best_score,
    }
    return best, metadata

def max_box_task(context: LayoutContext) -> TaskResult:
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
            "algorithm": "deterministic_candidate_refinement",
            "converged": True,
            "global_optimum_certified": True,
            "iterations": 0,
            "relative_tolerance": MAX_BOX_RELATIVE_TOLERANCE,
            "relative_improvement_window": 0.0,
            "best_valid_lower_bound_m2": 0.0,
        }
    else:
        best, metadata = optimize_maximum_rectangle(free_space, context.room)
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
            "rotation without overlapping blocking objects or openings"
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

def placement_free_space(context: LayoutContext) -> Any:
    blockers = [
        entity_polygon(entity)
        for entity in context.entities
        if not is_soft_covering(entity) and not is_ceiling_fixture(entity)
    ]
    blocker_union = union_polygons(blockers)
    return (
        context.room
        if blocker_union is None
        else make_valid(context.room.difference(blocker_union))
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
) -> dict[str, Any] | None:
    angles = set(edge_angles(free_space))
    angles.update(round(index * math.pi / 48, 10) for index in range(48))
    for angle in sorted(angles):
        region = configuration_space_region(free_space, width, depth, angle)
        if region.is_empty:
            continue
        for center in deterministic_points_in_geometry(region, 32):
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
