"""Deterministic geometry tools for FloorplanQA tool-calling evaluations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.validation import make_valid

from .generate_questions import (
    DEFAULT_GRID_RESOLUTION,
    LayoutContext,
    entity_centroid,
    entity_polygon,
    intersecting_entities,
    is_ceiling_fixture,
    is_soft_covering,
    label,
    load_layout,
    max_box_task,
    maximum_slide_distance,
    placement_false_certificate,
    placement_witness,
    shortest_grid_path,
    stable_rng,
    union_polygons,
)


TOOLSET_CHANGES = {
    1: "Entity search, entity inspection, and consolidated pair measurements.",
    2: "Added exact room/free-space/largest-box measurement and axis-aligned sliding.",
    3: "Added arbitrary-rotation placement testing and clearance-aware shortest paths.",
}


def function_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = required
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


BASE_TOOLS = [
    function_tool(
        "search_entities",
        "Search object, door, and window labels in the current floorplan.",
        {
            "query": {
                "type": "string",
                "description": "Case-insensitive label substring; use an empty string for all entities.",
            }
        },
        ["query"],
    ),
    function_tool(
        "inspect_entity",
        "Get exact centroid, bounds, polygon area, and kind for one floorplan entity.",
        {"name": {"type": "string", "description": "Exact or uniquely matching entity label."}},
        ["name"],
    ),
    function_tool(
        "measure_pair",
        "Measure two entity centroids and their relationships: distance, angle from north, and line intersections.",
        {
            "first": {"type": "string", "description": "Starting entity label."},
            "second": {"type": "string", "description": "Ending entity label."},
        },
        ["first", "second"],
    ),
]

SPACE_TOOLS = [
    function_tool(
        "measure_space",
        "Measure room area, non-occupied floor area, or the largest obstacle-free rectangle.",
        {
            "metric": {
                "type": "string",
                "enum": ["room_area", "free_area", "largest_free_rectangle"],
            }
        },
        ["metric"],
    ),
    function_tool(
        "slide_object",
        "Measure how far an object can translate before first contact with a blocker or room boundary.",
        {
            "object_name": {"type": "string"},
            "direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
            },
        },
        ["object_name", "direction"],
    ),
]

ADVANCED_TOOLS = [
    function_tool(
        "test_placement",
        "Test whether a rectangle can fit at any rotation in free floor space.",
        {
            "object_name": {"type": "string"},
            "width": {"type": "number", "exclusiveMinimum": 0},
            "depth": {"type": "number", "exclusiveMinimum": 0},
        },
        ["object_name", "width", "depth"],
    ),
    function_tool(
        "find_shortest_path",
        "Find a valid shortest waypoint path between entity centroids with obstacle clearance.",
        {
            "first": {"type": "string"},
            "second": {"type": "string"},
            "clearance": {"type": "number", "minimum": 0},
        },
        ["first", "second", "clearance"],
    ),
]

def tools_for_iteration(iteration: int) -> list[dict[str, Any]]:
    if iteration not in TOOLSET_CHANGES:
        raise ValueError(
            f"iteration must be between 1 and {max(TOOLSET_CHANGES)}"
        )
    tools = list(BASE_TOOLS)
    if iteration >= 2:
        tools.extend(SPACE_TOOLS)
    if iteration >= 3:
        tools.extend(ADVANCED_TOOLS)
    return tools


def tool_names(iteration: int) -> list[str]:
    return [str(tool["function"]["name"]) for tool in tools_for_iteration(iteration)]


@dataclass
class FloorplanToolRuntime:
    """Execute specialist geometry tools against one source floorplan."""

    source_layout: str
    layout_dir: Path
    seed: int

    def __post_init__(self) -> None:
        layout_root = self.layout_dir.resolve()
        layout_path = (layout_root / self.source_layout).resolve()
        try:
            layout_path.relative_to(layout_root)
        except ValueError as error:
            raise ValueError("source layout must be inside layout_dir") from error
        self.context: LayoutContext = load_layout(layout_path)
        self.entities = {label(entity).casefold(): entity for entity in self.context.entities}

    def resolve_entity(self, name: str) -> dict[str, Any]:
        normalized = name.strip().casefold()
        if normalized in self.entities:
            return self.entities[normalized]
        matches = [
            entity
            for entity_name, entity in self.entities.items()
            if normalized and normalized in entity_name
        ]
        if len(matches) == 1:
            return matches[0]
        available = ", ".join(sorted(label(entity) for entity in self.context.entities))
        raise ValueError(f"entity {name!r} is not a unique match; available: {available}")

    def entity_kind(self, entity: dict[str, Any]) -> str:
        if any(entity is candidate for candidate in self.context.objects):
            return "object"
        openings = self.context.layout.get("openings") or {}
        if any(entity is candidate for candidate in openings.get("doors") or []):
            return "door"
        return "window"

    def search_entities(self, query: str) -> dict[str, Any]:
        normalized = query.strip().casefold()
        matches = [
            {"label": label(entity), "kind": self.entity_kind(entity)}
            for entity in self.context.entities
            if not normalized or normalized in label(entity).casefold()
        ]
        return {"query": query, "count": len(matches), "matches": matches}

    def inspect_entity(self, name: str) -> dict[str, Any]:
        entity = self.resolve_entity(name)
        polygon = entity_polygon(entity)
        centroid = entity_centroid(entity)
        return {
            "label": label(entity),
            "kind": self.entity_kind(entity),
            "centroid": [round(value, 8) for value in centroid],
            "bounds": [round(value, 8) for value in polygon.bounds],
            "area_m2": round(float(polygon.area), 8),
            "vertex_count": len(list(polygon.exterior.coords)) - 1,
        }

    def measure_pair(self, first: str, second: str) -> dict[str, Any]:
        first_entity = self.resolve_entity(first)
        second_entity = self.resolve_entity(second)
        start = entity_centroid(first_entity)
        end = entity_centroid(second_entity)
        dx, dy = end[0] - start[0], end[1] - start[1]
        distance = math.hypot(dx, dy)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dy / distance))))
        intersections = [
            label(entity)
            for entity in intersecting_entities(
                self.context.entities, first_entity, second_entity
            )
        ]
        return {
            "first": label(first_entity),
            "second": label(second_entity),
            "first_centroid": [round(value, 8) for value in start],
            "second_centroid": [round(value, 8) for value in end],
            "distance_m": round(distance, 8),
            "distance_answer": f"{distance:.3f}",
            "angle_from_north_degrees": round(angle, 8),
            "angle_answer": f"{angle:.3f}",
            "intersections_in_order": intersections,
            "visibility_answer": json.dumps(intersections, separators=(",", ":")),
        }

    def free_geometry(self) -> Any:
        blockers = [
            entity_polygon(entity)
            for entity in self.context.entities
            if not is_soft_covering(entity) and not is_ceiling_fixture(entity)
        ]
        blocker_union = union_polygons(blockers)
        return (
            self.context.room
            if blocker_union is None
            else make_valid(self.context.room.difference(blocker_union))
        )

    def measure_space(self, metric: str) -> dict[str, Any]:
        if metric == "room_area":
            value = float(self.context.room.area)
            return {"metric": metric, "value_m2": value, "final_answer": f"{value:.3f}"}
        if metric == "free_area":
            occupied = [
                entity_polygon(entity)
                for entity in self.context.objects
                if not is_ceiling_fixture(entity)
            ]
            occupied_union = union_polygons(occupied)
            free = (
                self.context.room
                if occupied_union is None
                else self.context.room.difference(
                    self.context.room.intersection(occupied_union)
                )
            )
            value = float(free.area)
            return {"metric": metric, "value_m2": value, "final_answer": f"{value:.3f}"}
        if metric == "largest_free_rectangle":
            result = max_box_task(
                self.context,
                stable_rng(
                    self.seed,
                    self.context.source_group,
                    self.context.layout_id,
                    "max_box",
                ),
            )
            return {
                "metric": metric,
                "value_m2": result.answer_value,
                "witness": result.parameters["witness"],
                "final_answer": result.answer_text,
            }
        raise ValueError(f"unknown space metric: {metric}")

    def slide_object(self, object_name: str, direction: str) -> dict[str, Any]:
        entity = self.resolve_entity(object_name)
        if entity not in self.context.objects:
            raise ValueError("only floor objects can be repositioned")
        vectors = {
            "up": (0.0, 1.0),
            "down": (0.0, -1.0),
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
        }
        if direction not in vectors:
            raise ValueError(f"unknown direction: {direction}")
        obstacles = [
            entity_polygon(candidate)
            for candidate in self.context.objects
            if candidate is not entity
            and not is_soft_covering(candidate)
            and not is_ceiling_fixture(candidate)
        ]
        distance = maximum_slide_distance(
            entity_polygon(entity), self.context.room, obstacles, vectors[direction]
        )
        return {
            "object": label(entity),
            "direction": direction,
            "distance_m": distance,
            "final_answer": f"{distance:.3f}",
        }

    def test_placement(
        self, object_name: str, width: float, depth: float
    ) -> dict[str, Any]:
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        free_space = self.free_geometry()
        rng = stable_rng(
            self.seed,
            self.context.source_group,
            self.context.layout_id,
            "placement",
        )
        witness = (
            None
            if free_space.is_empty
            else placement_witness(free_space, width, depth, rng)
        )
        certificate = placement_false_certificate(free_space, width, depth)
        fits = witness is not None
        return {
            "object": object_name,
            "width_m": width,
            "depth_m": depth,
            "fits": fits,
            "witness": witness,
            "nonfit_certificate": certificate,
            "final_answer": "True" if fits else "False",
        }

    def find_shortest_path(
        self, first: str, second: str, clearance: float
    ) -> dict[str, Any]:
        first_entity = self.resolve_entity(first)
        second_entity = self.resolve_entity(second)
        path, metadata = shortest_grid_path(
            self.context,
            first_entity,
            second_entity,
            clearance=clearance,
            resolution=DEFAULT_GRID_RESOLUTION,
        )
        if len(path) < 2:
            raise ValueError("no valid path found")
        rounded_path = [[round(x, 3), round(y, 3)] for x, y in path]
        answer = json.dumps(rounded_path, separators=(",", ":"))
        return {
            "first": label(first_entity),
            "second": label(second_entity),
            "clearance_m": clearance,
            "path": rounded_path,
            "metadata": metadata,
            "final_answer": answer,
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        available = {
            "search_entities": self.search_entities,
            "inspect_entity": self.inspect_entity,
            "measure_pair": self.measure_pair,
            "measure_space": self.measure_space,
            "slide_object": self.slide_object,
            "test_placement": self.test_placement,
            "find_shortest_path": self.find_shortest_path,
        }
        function = available.get(name)
        if function is None:
            raise ValueError(f"unknown tool: {name}")
        return function(**arguments)
