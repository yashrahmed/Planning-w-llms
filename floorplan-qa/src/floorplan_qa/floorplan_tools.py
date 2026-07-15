"""Model-facing tools for inspecting FloorplanQA layout files."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .geometry import (
    build_room_polygon,
    entity_centroid,
    entity_polygon,
    intersecting_entities,
    is_ceiling_fixture,
    is_soft_covering,
    load_layout,
    max_box_task,
    maximum_slide_distance,
    shortest_grid_path,
    stable_rng,
    union_polygons,
)


INSPECT_ROOM_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_room",
        "description": (
            "Returns a compact natural-language inventory from a room-layout "
            "JSON file, listing objects, doors, and windows by type with their "
            "exact IDs. Does not return geometry, measurements, walls, or raw "
            "JSON."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": (
                        "The exact JSON filename given in the question after "
                        "'Room layout can be found in file :'."
                    ),
                }
            },
            "required": ["file_id"],
            "additionalProperties": False,
        },
    },
}

PAIR_DISTANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "pair_distance",
        "description": (
            "Returns the Euclidean distance in meters between the polygon "
            "centroids of two entities in a room-layout JSON file. Entity IDs "
            "may refer to objects, doors, or windows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_id_1": {
                    "type": "string",
                    "description": "The exact ID of the first object, door, or window.",
                },
                "object_id_2": {
                    "type": "string",
                    "description": "The exact ID of the second object, door, or window.",
                },
                "file_id": {
                    "type": "string",
                    "description": (
                        "The exact JSON filename given in the question after "
                        "'Room layout can be found in file :'."
                    ),
                },
            },
            "required": ["object_id_1", "object_id_2", "file_id"],
            "additionalProperties": False,
        },
    },
}

VIEW_ANGLE_TOOL = {
    "type": "function",
    "function": {
        "name": "view_angle",
        "description": (
            "Returns the unsigned angle in degrees between north (the positive "
            "y-axis) and the vector from the first entity's polygon centroid to "
            "the second entity's polygon centroid. The result is between 0 and "
            "180 degrees. Entity IDs may refer to objects, doors, or windows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_id_1": {
                    "type": "string",
                    "description": "The exact ID of the starting object, door, or window.",
                },
                "object_id_2": {
                    "type": "string",
                    "description": "The exact ID of the ending object, door, or window.",
                },
                "file_id": {
                    "type": "string",
                    "description": (
                        "The exact JSON filename given in the question after "
                        "'Room layout can be found in file :'."
                    ),
                },
            },
            "required": ["object_id_1", "object_id_2", "file_id"],
            "additionalProperties": False,
        },
    },
}

INSPECT_ENTITY_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_entity",
        "description": (
            "Returns an entity's kind, polygon centroid, axis-aligned minimum "
            "and maximum coordinates, polygon area, and vertex count. The "
            "entity ID may refer to an object, door, or window."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The exact ID of an object, door, or window.",
                },
                "file_id": {
                    "type": "string",
                    "description": (
                        "The exact JSON filename given in the question after "
                        "'Room layout can be found in file :'."
                    ),
                },
            },
            "required": ["object_id", "file_id"],
            "additionalProperties": False,
        },
    },
}

RAY_TRACE_TOOL = {
    "type": "function",
    "function": {
        "name": "ray_trace",
        "description": (
            "Traces the finite line segment from the first entity's polygon "
            "centroid to the second entity's polygon centroid and returns the "
            "IDs of every other entity polygon it intersects, ordered from the "
            "first entity to the second. Entity IDs may refer to objects, doors, "
            "or windows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_id_1": {
                    "type": "string",
                    "description": "The exact ID of the starting object, door, or window.",
                },
                "object_id_2": {
                    "type": "string",
                    "description": "The exact ID of the ending object, door, or window.",
                },
                "file_id": {
                    "type": "string",
                    "description": (
                        "The exact JSON filename given in the question after "
                        "'Room layout can be found in file :'."
                    ),
                },
            },
            "required": ["object_id_1", "object_id_2", "file_id"],
            "additionalProperties": False,
        },
    },
}

LARGEST_EMPTY_AREA_TOOL = {
    "type": "function",
    "function": {
        "name": "largest_empty_area",
        "description": (
            "Returns the width, length, and area of the largest rectangle "
            "that fits fully inside the room at any rotation without overlapping "
            "blocking objects or openings. Rugs and ceiling-only fixtures are "
            "treated as nonblocking. Dimensions are in meters and area is in "
            "square meters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": (
                        "The exact JSON filename given in the question after "
                        "'Room layout can be found in file :'."
                    ),
                }
            },
            "required": ["file_id"],
            "additionalProperties": False,
        },
    },
}

OCCUPIED_FLOOR_AREA_TOOL = {
    "type": "function",
    "function": {
        "name": "occupied_floor_area",
        "description": (
            "Returns the total floor area covered by the geometric union of room "
            "object polygons, counting overlaps only once. Rugs and other floor "
            "coverings count as occupied; doors, windows, and ceiling-only "
            "fixtures do not. The result is in square meters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": (
                        "The exact JSON filename given in the question after "
                        "'Room layout can be found in file :'."
                    ),
                }
            },
            "required": ["file_id"],
            "additionalProperties": False,
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Applies one arithmetic operation to two numeric operands and returns "
            "the result rounded to three decimal places. Supported operations are "
            "add, sub, mul, and div. Division by zero returns Error."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operand_1": {
                    "type": "number",
                    "description": "The left-hand numeric operand.",
                },
                "operand_2": {
                    "type": "number",
                    "description": "The right-hand numeric operand.",
                },
                "operator": {
                    "type": "string",
                    "enum": ["add", "sub", "mul", "div"],
                    "description": "The arithmetic operation to apply.",
                },
            },
            "required": ["operand_1", "operand_2", "operator"],
            "additionalProperties": False,
        },
    },
}

SHORTEST_PATH_TOOL = {
    "type": "function",
    "function": {
        "name": "shortest_path",
        "description": (
            "Returns an ordered shortest waypoint path from the first entity's "
            "polygon centroid to the second entity's polygon centroid while "
            "maintaining 0.15 meters of clearance from other blocking entities. "
            "Rugs and ceiling-only fixtures are nonblocking. Entity IDs may "
            "refer to objects, doors, or windows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_id_1": {
                    "type": "string",
                    "description": "The exact ID of the starting object, door, or window.",
                },
                "object_id_2": {
                    "type": "string",
                    "description": "The exact ID of the ending object, door, or window.",
                },
                "file_id": {
                    "type": "string",
                    "description": (
                        "The exact JSON filename given in the question after "
                        "'Room layout can be found in file :'."
                    ),
                },
            },
            "required": ["object_id_1", "object_id_2", "file_id"],
            "additionalProperties": False,
        },
    },
}

TEST_MOVEMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "test_movement",
        "description": (
            "Returns the maximum distance in meters that an object can translate "
            "up, down, left, or right before its polygon first touches another "
            "blocking object or the room boundary. Rugs and ceiling-only fixtures "
            "are nonblocking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The exact ID of the object to move.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "The direction in which to translate the object.",
                },
                "file_id": {
                    "type": "string",
                    "description": (
                        "The exact JSON filename given in the question after "
                        "'Room layout can be found in file :'."
                    ),
                },
            },
            "required": ["object_id", "direction", "file_id"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [
    INSPECT_ROOM_TOOL,
    PAIR_DISTANCE_TOOL,
    VIEW_ANGLE_TOOL,
    INSPECT_ENTITY_TOOL,
    RAY_TRACE_TOOL,
    LARGEST_EMPTY_AREA_TOOL,
    OCCUPIED_FLOOR_AREA_TOOL,
    CALCULATOR_TOOL,
    SHORTEST_PATH_TOOL,
    TEST_MOVEMENT_TOOL,
]

NUMBERED_ID_SUFFIX = re.compile(r"_\d+$")


def entity_type(entity_id: str) -> str:
    """Infer the shared type from IDs such as ``chair_1`` and ``chair_2``."""
    return NUMBERED_ID_SUFFIX.sub("", entity_id)


def pluralize(noun: str, count: int) -> str:
    """Pluralize the final word of a compact entity-type label."""
    if count == 1:
        return noun

    words = noun.split()
    final = words[-1]
    if len(final) > 1 and final.endswith("y") and final[-2] not in "aeiou":
        final = f"{final[:-1]}ies"
    elif final.endswith(("s", "x", "z", "ch", "sh")):
        final = f"{final}es"
    else:
        final = f"{final}s"
    return " ".join([*words[:-1], final])


def format_entity_group(kind: str, entity_ids: list[str]) -> str:
    count = len(entity_ids)
    display_kind = entity_type(kind).replace("_", " ")
    identifier_label = "ID" if count == 1 else "IDs"
    identifiers = ", ".join(entity_ids)
    return (
        f"- {count} {pluralize(display_kind, count)} with "
        f"{identifier_label} [{identifiers}]."
    )


def require_entity_ids(entities: Any, section: str) -> list[str]:
    if not isinstance(entities, list):
        raise ValueError(f"layout field {section!r} must be a list")

    entity_ids: list[str] = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ValueError(f"{section}[{index}] must be an object")
        entity_id = entity.get("label")
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ValueError(f"{section}[{index}] has no non-empty label")
        entity_ids.append(entity_id.strip())
    return entity_ids


def grouped_object_ids(entity_ids: list[str]) -> dict[str, list[str]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for entity_id in entity_ids:
        grouped[entity_type(entity_id)].append(entity_id)
    return dict(grouped)


@dataclass(frozen=True)
class FloorplanToolRuntime:
    """Run model-requested tools against files in one allowed directory."""

    layout_dir: Path

    def resolve_layout_file(self, file_id: str) -> Path:
        if not isinstance(file_id, str) or not file_id.strip():
            raise ValueError("file_id must be a non-empty string")

        normalized = file_id.strip()
        requested = Path(normalized)
        if requested.name != normalized or requested.suffix.casefold() != ".json":
            raise ValueError("file_id must be a JSON filename without a directory")

        layout_root = self.layout_dir.expanduser().resolve()
        layout_path = (layout_root / normalized).resolve()
        try:
            layout_path.relative_to(layout_root)
        except ValueError as error:
            raise ValueError("file_id must resolve inside the layout directory") from error
        if not layout_path.is_file():
            raise FileNotFoundError(f"room layout file not found: {normalized}")
        return layout_path

    def read_layout(self, file_id: str) -> dict[str, Any]:
        layout_path = self.resolve_layout_file(file_id)
        try:
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"room layout file is not valid JSON: {file_id}") from error
        if not isinstance(layout, dict):
            raise ValueError("room layout must contain a JSON object")
        return layout

    def inspect_room(self, file_id: str) -> str:
        layout = self.read_layout(file_id)
        object_ids = require_entity_ids(layout.get("objects", []), "objects")
        openings = layout.get("openings", {})
        if not isinstance(openings, dict):
            raise ValueError("layout field 'openings' must be an object")
        door_ids = require_entity_ids(openings.get("doors", []), "openings.doors")
        window_ids = require_entity_ids(
            openings.get("windows", []), "openings.windows"
        )

        room_type = layout.get("room_type")
        room_description = (
            str(room_type).replace("_", " ")
            if isinstance(room_type, (str, int, float)) and str(room_type).strip()
            else "unspecified"
        )
        room_area = float(build_room_polygon(layout).area)
        lines = [
            f"Room type: {room_description}.",
            f"Room floor area: {room_area:.3f} square meters.",
            f"{len(object_ids)} objects:",
        ]
        for kind, ids in sorted(grouped_object_ids(object_ids).items()):
            lines.append(format_entity_group(kind, ids))

        lines.append(f"{len(door_ids) + len(window_ids)} openings:")
        if door_ids:
            lines.append(format_entity_group("door", door_ids))
        else:
            lines.append("- 0 doors.")
        if window_ids:
            lines.append(format_entity_group("window", window_ids))
        else:
            lines.append("- 0 windows.")
        return "\n".join(lines)

    @staticmethod
    def entities_by_id(layout: dict[str, Any]) -> dict[str, dict[str, Any]]:
        objects = layout.get("objects", [])
        openings = layout.get("openings", {})
        if not isinstance(openings, dict):
            raise ValueError("layout field 'openings' must be an object")
        sections = {
            "objects": objects,
            "openings.doors": openings.get("doors", []),
            "openings.windows": openings.get("windows", []),
        }

        entities: dict[str, dict[str, Any]] = {}
        for section, values in sections.items():
            entity_ids = require_entity_ids(values, section)
            for entity_id, entity in zip(entity_ids, values, strict=True):
                if entity_id in entities:
                    raise ValueError(f"duplicate entity ID in room layout: {entity_id}")
                entities[entity_id] = entity
        return entities

    @staticmethod
    def resolve_entity(
        entities: dict[str, dict[str, Any]], object_id: str
    ) -> dict[str, Any]:
        if not isinstance(object_id, str) or not object_id.strip():
            raise ValueError("object ID must be a non-empty string")
        normalized = object_id.strip()
        if normalized not in entities:
            available = ", ".join(sorted(entities))
            raise ValueError(
                f"unknown entity ID {normalized!r}; available IDs: [{available}]"
            )
        return entities[normalized]

    @staticmethod
    def entity_kind(layout: dict[str, Any], entity: dict[str, Any]) -> str:
        if any(entity is candidate for candidate in layout.get("objects", [])):
            return "object"
        openings = layout.get("openings", {})
        if any(entity is candidate for candidate in openings.get("doors", [])):
            return "door"
        if any(entity is candidate for candidate in openings.get("windows", [])):
            return "window"
        raise ValueError("entity is not declared in the room layout")

    def pair_distance(
        self, object_id_1: str, object_id_2: str, file_id: str
    ) -> str:
        layout = self.read_layout(file_id)
        entities = self.entities_by_id(layout)
        first = self.resolve_entity(entities, object_id_1)
        second = self.resolve_entity(entities, object_id_2)
        distance = math.dist(entity_centroid(first), entity_centroid(second))
        return (
            f"The Euclidean distance between the polygon centroids of "
            f"'{object_id_1.strip()}' and '{object_id_2.strip()}' is "
            f"{distance:.3f} meters."
        )

    def view_angle(
        self, object_id_1: str, object_id_2: str, file_id: str
    ) -> str:
        layout = self.read_layout(file_id)
        entities = self.entities_by_id(layout)
        first = self.resolve_entity(entities, object_id_1)
        second = self.resolve_entity(entities, object_id_2)
        start = entity_centroid(first)
        end = entity_centroid(second)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        magnitude = math.hypot(dx, dy)
        if magnitude <= 1e-12:
            raise ValueError("view angle is undefined for coincident centroids")
        cosine = max(-1.0, min(1.0, dy / magnitude))
        angle = math.degrees(math.acos(cosine))
        return (
            f"The view angle from the polygon centroid of "
            f"'{object_id_1.strip()}' to '{object_id_2.strip()}' is "
            f"{angle:.3f} degrees from north (the positive y-axis)."
        )

    def inspect_entity(self, object_id: str, file_id: str) -> str:
        layout = self.read_layout(file_id)
        entities = self.entities_by_id(layout)
        entity = self.resolve_entity(entities, object_id)
        polygon = entity_polygon(entity)
        centroid_x, centroid_y = entity_centroid(entity)
        min_x, min_y, max_x, max_y = polygon.bounds
        kind = self.entity_kind(layout, entity)
        article = "an" if kind == "object" else "a"
        vertex_count = len(list(polygon.exterior.coords)) - 1
        return (
            f"Entity '{object_id.strip()}' is {article} {kind}. "
            f"Its polygon centroid is x={centroid_x:.3f}, y={centroid_y:.3f} "
            "meters. Its axis-aligned bounding box is "
            f"min_x={min_x:.3f}, min_y={min_y:.3f}, max_x={max_x:.3f}, "
            f"and max_y={max_y:.3f} meters. Its polygon area is "
            f"{polygon.area:.3f} square meters and it has {vertex_count} vertices."
        )

    def ray_trace(
        self, object_id_1: str, object_id_2: str, file_id: str
    ) -> str:
        layout = self.read_layout(file_id)
        entities = self.entities_by_id(layout)
        first = self.resolve_entity(entities, object_id_1)
        second = self.resolve_entity(entities, object_id_2)
        hits = intersecting_entities(list(entities.values()), first, second)
        hit_ids = [str(entity["label"]).strip() for entity in hits]
        formatted_hits = ", ".join(hit_ids)
        if not hit_ids:
            return (
                f"The centroid-to-centroid segment from '{object_id_1.strip()}' "
                f"to '{object_id_2.strip()}' intersects no other entities: []."
            )
        return (
            f"The centroid-to-centroid segment from '{object_id_1.strip()}' to "
            f"'{object_id_2.strip()}' intersects {len(hit_ids)} other "
            f"{'entity' if len(hit_ids) == 1 else 'entities'} in traversal "
            f"order: [{formatted_hits}]."
        )

    def largest_empty_area(self, file_id: str) -> str:
        layout_path = self.resolve_layout_file(file_id)
        context = load_layout(layout_path)
        filename_parts = Path(file_id).stem.rsplit("-", 2)
        source_group = (
            filename_parts[0] if len(filename_parts) == 3 else context.source_group
        )
        context = replace(context, source_group=source_group)
        result = max_box_task(
            context,
            stable_rng(0, context.source_group, context.layout_id, "max_box"),
        )
        witness = result.parameters["witness"]
        width = float(witness["width"])
        length = float(witness["depth"])
        area = float(result.answer_value)
        return (
            f"The largest empty rectangle is {width:.3f} meters wide and "
            f"{length:.3f} meters long, with an area of {area:.3f} square meters."
        )

    def occupied_floor_area(self, file_id: str) -> str:
        context = load_layout(self.resolve_layout_file(file_id))
        occupied_polygons = [
            entity_polygon(entity)
            for entity in context.objects
            if not is_ceiling_fixture(entity)
        ]
        occupied_union = union_polygons(occupied_polygons)
        occupied_area = (
            0.0
            if occupied_union is None
            else float(context.room.intersection(occupied_union).area)
        )
        return f"The occupied floor area is {occupied_area:.3f} square meters."

    @staticmethod
    def calculator(operand_1: float, operand_2: float, operator: str) -> str:
        if isinstance(operand_1, bool) or not isinstance(operand_1, (int, float)):
            raise ValueError("operand_1 must be a number")
        if isinstance(operand_2, bool) or not isinstance(operand_2, (int, float)):
            raise ValueError("operand_2 must be a number")
        if operator not in {"add", "sub", "mul", "div"}:
            raise ValueError(f"unknown calculator operator: {operator}")

        left = float(operand_1)
        right = float(operand_2)
        if operator == "add":
            result = left + right
        elif operator == "sub":
            result = left - right
        elif operator == "mul":
            result = left * right
        else:
            if right == 0.0:
                return "Error"
            result = left / right
        return f"The result is {result:.3f}."

    def shortest_path(
        self, object_id_1: str, object_id_2: str, file_id: str
    ) -> str:
        context = load_layout(self.resolve_layout_file(file_id))
        entities = self.entities_by_id(context.layout)
        first = self.resolve_entity(entities, object_id_1)
        second = self.resolve_entity(entities, object_id_2)
        path, _ = shortest_grid_path(
            context,
            first,
            second,
            clearance=0.15,
        )
        if not path:
            return (
                f"No valid waypoint path was found from '{object_id_1.strip()}' "
                f"to '{object_id_2.strip()}' with 0.15 meters of clearance."
            )
        formatted_path = "[" + ",".join(
            f"[{point[0]:.3f},{point[1]:.3f}]" for point in path
        ) + "]"
        return (
            f"The shortest valid waypoint path from '{object_id_1.strip()}' to "
            f"'{object_id_2.strip()}' with 0.15 meters of clearance is "
            f"{formatted_path}."
        )

    def test_movement(self, object_id: str, direction: str, file_id: str) -> str:
        context = load_layout(self.resolve_layout_file(file_id))
        entities = self.entities_by_id(context.layout)
        moving_entity = self.resolve_entity(entities, object_id)
        if not any(moving_entity is candidate for candidate in context.objects):
            raise ValueError("only room objects can be moved")
        direction_vectors = {
            "up": (0.0, 1.0),
            "down": (0.0, -1.0),
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
        }
        if direction not in direction_vectors:
            raise ValueError(f"unknown movement direction: {direction}")
        obstacles = [
            entity_polygon(entity)
            for entity in context.objects
            if entity is not moving_entity
            and not is_soft_covering(entity)
            and not is_ceiling_fixture(entity)
        ]
        distance = maximum_slide_distance(
            entity_polygon(moving_entity),
            context.room,
            obstacles,
            direction_vectors[direction],
        )
        return (
            f"Object '{object_id.strip()}' can move {distance:.3f} meters "
            f"{direction} before touching a blocking object or the room boundary."
        )

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")

        required_by_tool = {
            "inspect_room": {"file_id"},
            "pair_distance": {"object_id_1", "object_id_2", "file_id"},
            "view_angle": {"object_id_1", "object_id_2", "file_id"},
            "inspect_entity": {"object_id", "file_id"},
            "ray_trace": {"object_id_1", "object_id_2", "file_id"},
            "largest_empty_area": {"file_id"},
            "occupied_floor_area": {"file_id"},
            "calculator": {"operand_1", "operand_2", "operator"},
            "shortest_path": {"object_id_1", "object_id_2", "file_id"},
            "test_movement": {"object_id", "direction", "file_id"},
        }
        if name not in required_by_tool:
            raise ValueError(f"unknown tool: {name}")
        required = required_by_tool[name]
        unexpected = set(arguments) - required
        if unexpected:
            raise ValueError(
                f"unexpected {name} argument(s): {', '.join(sorted(unexpected))}"
            )
        missing = required - set(arguments)
        if missing:
            raise ValueError(f"{name} requires {', '.join(sorted(missing))}")

        if name == "inspect_room":
            return self.inspect_room(arguments["file_id"])
        if name == "pair_distance":
            return self.pair_distance(
                arguments["object_id_1"],
                arguments["object_id_2"],
                arguments["file_id"],
            )
        if name == "view_angle":
            return self.view_angle(
                arguments["object_id_1"],
                arguments["object_id_2"],
                arguments["file_id"],
            )
        if name == "inspect_entity":
            return self.inspect_entity(arguments["object_id"], arguments["file_id"])
        if name == "ray_trace":
            return self.ray_trace(
                arguments["object_id_1"],
                arguments["object_id_2"],
                arguments["file_id"],
            )
        if name == "largest_empty_area":
            return self.largest_empty_area(arguments["file_id"])
        if name == "occupied_floor_area":
            return self.occupied_floor_area(arguments["file_id"])
        if name == "calculator":
            return self.calculator(
                arguments["operand_1"],
                arguments["operand_2"],
                arguments["operator"],
            )
        if name == "shortest_path":
            return self.shortest_path(
                arguments["object_id_1"],
                arguments["object_id_2"],
                arguments["file_id"],
            )
        return self.test_movement(
            arguments["object_id"],
            arguments["direction"],
            arguments["file_id"],
        )
