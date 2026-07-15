from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from floorplan_qa.floorplan_tools import (
    CALCULATOR_TOOL,
    INSPECT_ENTITY_TOOL,
    INSPECT_ROOM_TOOL,
    LARGEST_EMPTY_AREA_TOOL,
    OCCUPIED_FLOOR_AREA_TOOL,
    PAIR_DISTANCE_TOOL,
    RAY_TRACE_TOOL,
    SHORTEST_PATH_TOOL,
    TEST_MOVEMENT_TOOL,
    TOOLS,
    VIEW_ANGLE_TOOL,
    FloorplanToolRuntime,
)


def rectangle(label: str, x1: float, y1: float, x2: float, y2: float) -> dict:
    return {
        "label": label,
        "points": [
            {"x": x1, "y": y1},
            {"x": x2, "y": y1},
            {"x": x2, "y": y2},
            {"x": x1, "y": y2},
        ],
    }


class FloorplanToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.layout_dir = Path(self.temporary_directory.name)
        self.file_id = "living_room-7-train.json"
        layout = {
            "layout_id": 7,
            "room_type": "living_room",
            "room_boundary": [
                {"x": 0.0, "y": 0.0},
                {"x": 5.0, "y": 0.0},
                {"x": 5.0, "y": 5.0},
                {"x": 0.0, "y": 5.0},
                {"x": 0.0, "y": 0.0},
            ],
            "objects": [
                rectangle("chair_1", 0.0, 0.0, 1.0, 1.0),
                rectangle("table", 1.0, 1.0, 2.0, 2.0),
                rectangle("chair_2", 3.0, 4.0, 4.0, 5.0),
                rectangle("storage_caddy_1", 2.0, 2.0, 3.0, 3.0),
                rectangle("storage_caddy_2", 3.0, 2.0, 4.0, 3.0),
            ],
            "openings": {
                "doors": [rectangle("door", 0.0, 4.0, 1.0, 5.0)],
                "windows": [
                    rectangle("window_1", 4.0, 1.0, 5.0, 2.0),
                    rectangle("window_2", 4.0, 2.0, 5.0, 3.0),
                ],
            },
        }
        (self.layout_dir / self.file_id).write_text(
            json.dumps(layout), encoding="utf-8"
        )
        self.runtime = FloorplanToolRuntime(self.layout_dir)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_inspect_room_returns_grouped_natural_language_inventory(self) -> None:
        self.assertEqual(
            self.runtime.inspect_room(self.file_id),
            "\n".join(
                [
                    "Room type: living room.",
                    "Room floor area: 25.000 square meters.",
                    "5 objects:",
                    "- 2 chairs with IDs [chair_1, chair_2].",
                    "- 2 storage caddies with IDs [storage_caddy_1, storage_caddy_2].",
                    "- 1 table with ID [table].",
                    "3 openings:",
                    "- 1 door with ID [door].",
                    "- 2 windows with IDs [window_1, window_2].",
                ]
            ),
        )

    def test_inspect_room_reports_absent_opening_types(self) -> None:
        file_id = "bedroom-8-val.json"
        layout = {
            "layout_id": 8,
            "room_type": "bedroom",
            "room_boundary": [
                {"x": 0.0, "y": 0.0},
                {"x": 5.0, "y": 0.0},
                {"x": 5.0, "y": 5.0},
                {"x": 0.0, "y": 5.0},
                {"x": 0.0, "y": 0.0},
            ],
            "objects": [],
            "openings": {"doors": [], "windows": []},
        }
        (self.layout_dir / file_id).write_text(json.dumps(layout), encoding="utf-8")

        self.assertEqual(
            self.runtime.inspect_room(file_id),
            "\n".join(
                [
                    "Room type: bedroom.",
                    "Room floor area: 25.000 square meters.",
                    "0 objects:",
                    "0 openings:",
                    "- 0 doors.",
                    "- 0 windows.",
                ]
            ),
        )

    def test_file_id_cannot_escape_layout_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "without a directory"):
            self.runtime.inspect_room("../living_room-7-train.json")

    def test_execute_accepts_only_the_advertised_argument(self) -> None:
        self.assertEqual(
            self.runtime.execute("inspect_room", {"file_id": self.file_id}),
            self.runtime.inspect_room(self.file_id),
        )
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.runtime.execute(
                "inspect_room", {"file_id": self.file_id, "source_layout": "secret"}
            )

    def test_pair_distance_uses_polygon_centroids(self) -> None:
        self.assertEqual(
            self.runtime.pair_distance("chair_1", "chair_2", self.file_id),
            "The Euclidean distance between the polygon centroids of "
            "'chair_1' and 'chair_2' is 5.000 meters.",
        )

    def test_pair_distance_accepts_an_opening_id(self) -> None:
        self.assertEqual(
            self.runtime.execute(
                "pair_distance",
                {
                    "object_id_1": "chair_1",
                    "object_id_2": "door",
                    "file_id": self.file_id,
                },
            ),
            "The Euclidean distance between the polygon centroids of "
            "'chair_1' and 'door' is 4.000 meters.",
        )

    def test_pair_distance_requires_exact_known_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown entity ID"):
            self.runtime.pair_distance("chair", "table", self.file_id)

    def test_view_angle_is_measured_from_north(self) -> None:
        self.assertEqual(
            self.runtime.execute(
                "view_angle",
                {
                    "object_id_1": "chair_1",
                    "object_id_2": "chair_2",
                    "file_id": self.file_id,
                },
            ),
            "The view angle from the polygon centroid of 'chair_1' to "
            "'chair_2' is 36.870 degrees from north (the positive y-axis).",
        )
        self.assertEqual(
            self.runtime.view_angle("chair_2", "chair_1", self.file_id),
            "The view angle from the polygon centroid of 'chair_2' to "
            "'chair_1' is 143.130 degrees from north (the positive y-axis).",
        )

    def test_inspect_entity_returns_the_complete_geometry_summary(self) -> None:
        self.assertEqual(
            self.runtime.execute(
                "inspect_entity",
                {"object_id": "table", "file_id": self.file_id},
            ),
            "Entity 'table' is an object. Its polygon centroid is x=1.500, "
            "y=1.500 meters. Its axis-aligned bounding box is min_x=1.000, "
            "min_y=1.000, max_x=2.000, and max_y=2.000 meters. Its polygon "
            "area is 1.000 square meters and it has 4 vertices.",
        )

    def test_inspect_entity_identifies_opening_kind(self) -> None:
        self.assertIn(
            "Entity 'door' is a door.",
            self.runtime.inspect_entity("door", self.file_id),
        )

    def test_ray_trace_returns_intersections_in_traversal_order(self) -> None:
        self.assertEqual(
            self.runtime.execute(
                "ray_trace",
                {
                    "object_id_1": "chair_1",
                    "object_id_2": "chair_2",
                    "file_id": self.file_id,
                },
            ),
            "The centroid-to-centroid segment from 'chair_1' to 'chair_2' "
            "intersects 2 other entities in traversal order: "
            "[table, storage_caddy_1].",
        )

    def test_ray_trace_reports_an_empty_intersection_list(self) -> None:
        self.assertEqual(
            self.runtime.ray_trace("chair_1", "door", self.file_id),
            "The centroid-to-centroid segment from 'chair_1' to 'door' "
            "intersects no other entities: [].",
        )

    def test_largest_empty_area_returns_dimensions_and_area(self) -> None:
        task_result = SimpleNamespace(
            parameters={"witness": {"width": 4.325, "depth": 1.6}},
            answer_value=6.92,
        )
        with patch(
            "floorplan_qa.floorplan_tools.max_box_task",
            return_value=task_result,
        ):
            output = self.runtime.execute(
                "largest_empty_area",
                {"file_id": self.file_id},
            )

        self.assertEqual(
            output,
            "The largest empty rectangle is 4.325 meters wide and 1.600 meters "
            "long, with an area of 6.920 square meters.",
        )

    def test_occupied_floor_area_unions_objects_and_applies_task_filters(self) -> None:
        file_id = "area-room-9-train.json"
        layout = {
            "layout_id": 9,
            "room_type": "bedroom",
            "room_boundary": [
                {"x": 0.0, "y": 0.0},
                {"x": 5.0, "y": 0.0},
                {"x": 5.0, "y": 5.0},
                {"x": 0.0, "y": 5.0},
                {"x": 0.0, "y": 0.0},
            ],
            "objects": [
                rectangle("rug", 0.0, 0.0, 2.0, 2.0),
                rectangle("table", 1.0, 1.0, 3.0, 3.0),
                rectangle("ceiling light", 3.0, 3.0, 5.0, 5.0),
            ],
            "openings": {
                "doors": [rectangle("door", 0.0, 4.0, 1.0, 5.0)],
                "windows": [],
            },
        }
        (self.layout_dir / file_id).write_text(json.dumps(layout), encoding="utf-8")

        self.assertEqual(
            self.runtime.execute("occupied_floor_area", {"file_id": file_id}),
            "The occupied floor area is 7.000 square meters.",
        )

    def test_calculator_supports_all_operations_and_zero_division(self) -> None:
        self.assertEqual(self.runtime.calculator(7, 2, "add"), "The result is 9.000.")
        self.assertEqual(self.runtime.calculator(7, 2, "sub"), "The result is 5.000.")
        self.assertEqual(self.runtime.calculator(7, 2, "mul"), "The result is 14.000.")
        self.assertEqual(self.runtime.calculator(7, 2, "div"), "The result is 3.500.")
        self.assertEqual(self.runtime.calculator(7, 0, "div"), "Error")

    def test_shortest_path_returns_ordered_three_decimal_waypoints(self) -> None:
        with patch(
            "floorplan_qa.floorplan_tools.shortest_grid_path",
            return_value=([(0.5, 0.5), (3.5, 4.5)], {}),
        ) as path_solver:
            output = self.runtime.execute(
                "shortest_path",
                {
                    "object_id_1": "chair_1",
                    "object_id_2": "chair_2",
                    "file_id": self.file_id,
                },
            )

        self.assertEqual(
            output,
            "The shortest valid waypoint path from 'chair_1' to 'chair_2' with "
            "0.15 meters of clearance is [[0.500,0.500],[3.500,4.500]].",
        )
        self.assertEqual(path_solver.call_args.kwargs["clearance"], 0.15)

    def test_movement_returns_first_collision_distance(self) -> None:
        with patch(
            "floorplan_qa.floorplan_tools.maximum_slide_distance",
            return_value=1.2,
        ) as movement_solver:
            output = self.runtime.execute(
                "test_movement",
                {
                    "object_id": "chair_1",
                    "direction": "up",
                    "file_id": self.file_id,
                },
            )

        self.assertEqual(
            output,
            "Object 'chair_1' can move 1.200 meters up before touching a "
            "blocking object or the room boundary.",
        )
        self.assertEqual(movement_solver.call_args.args[3], (0.0, 1.0))

    def test_tool_schema_exposes_only_file_id(self) -> None:
        function = INSPECT_ROOM_TOOL["function"]
        self.assertEqual(function["name"], "inspect_room")
        self.assertEqual(function["parameters"]["required"], ["file_id"])
        self.assertEqual(
            set(function["parameters"]["properties"]),
            {"file_id"},
        )
        self.assertFalse(function["parameters"]["additionalProperties"])

    def test_pair_distance_schema_requires_both_ids_and_file_id(self) -> None:
        function = PAIR_DISTANCE_TOOL["function"]
        self.assertEqual(function["name"], "pair_distance")
        self.assertEqual(
            function["parameters"]["required"],
            ["object_id_1", "object_id_2", "file_id"],
        )
        self.assertEqual(
            set(function["parameters"]["properties"]),
            {"object_id_1", "object_id_2", "file_id"},
        )
        self.assertFalse(function["parameters"]["additionalProperties"])

    def test_view_angle_schema_requires_both_ids_and_file_id(self) -> None:
        function = VIEW_ANGLE_TOOL["function"]
        self.assertEqual(function["name"], "view_angle")
        self.assertEqual(
            function["parameters"]["required"],
            ["object_id_1", "object_id_2", "file_id"],
        )
        self.assertEqual(
            set(function["parameters"]["properties"]),
            {"object_id_1", "object_id_2", "file_id"},
        )
        self.assertFalse(function["parameters"]["additionalProperties"])

    def test_inspect_entity_schema_requires_one_id_and_file_id(self) -> None:
        function = INSPECT_ENTITY_TOOL["function"]
        self.assertEqual(function["name"], "inspect_entity")
        self.assertEqual(
            function["parameters"]["required"],
            ["object_id", "file_id"],
        )
        self.assertEqual(
            set(function["parameters"]["properties"]),
            {"object_id", "file_id"},
        )
        self.assertFalse(function["parameters"]["additionalProperties"])

    def test_ray_trace_schema_requires_both_ids_and_file_id(self) -> None:
        function = RAY_TRACE_TOOL["function"]
        self.assertEqual(function["name"], "ray_trace")
        self.assertEqual(
            function["parameters"]["required"],
            ["object_id_1", "object_id_2", "file_id"],
        )
        self.assertEqual(
            set(function["parameters"]["properties"]),
            {"object_id_1", "object_id_2", "file_id"},
        )
        self.assertFalse(function["parameters"]["additionalProperties"])

    def test_largest_empty_area_schema_requires_only_file_id(self) -> None:
        function = LARGEST_EMPTY_AREA_TOOL["function"]
        self.assertEqual(function["name"], "largest_empty_area")
        self.assertEqual(function["parameters"]["required"], ["file_id"])
        self.assertEqual(
            set(function["parameters"]["properties"]),
            {"file_id"},
        )
        self.assertFalse(function["parameters"]["additionalProperties"])

    def test_occupied_floor_area_schema_requires_only_file_id(self) -> None:
        function = OCCUPIED_FLOOR_AREA_TOOL["function"]
        self.assertEqual(function["name"], "occupied_floor_area")
        self.assertEqual(function["parameters"]["required"], ["file_id"])
        self.assertEqual(
            set(function["parameters"]["properties"]),
            {"file_id"},
        )
        self.assertFalse(function["parameters"]["additionalProperties"])

    def test_calculator_schema_requires_operands_and_operator(self) -> None:
        function = CALCULATOR_TOOL["function"]
        self.assertEqual(function["name"], "calculator")
        self.assertEqual(
            function["parameters"]["required"],
            ["operand_1", "operand_2", "operator"],
        )
        self.assertEqual(
            function["parameters"]["properties"]["operator"]["enum"],
            ["add", "sub", "mul", "div"],
        )
        self.assertFalse(function["parameters"]["additionalProperties"])

    def test_shortest_path_schema_requires_both_ids_and_file_id(self) -> None:
        function = SHORTEST_PATH_TOOL["function"]
        self.assertEqual(function["name"], "shortest_path")
        self.assertEqual(
            function["parameters"]["required"],
            ["object_id_1", "object_id_2", "file_id"],
        )
        self.assertEqual(
            set(function["parameters"]["properties"]),
            {"object_id_1", "object_id_2", "file_id"},
        )
        self.assertFalse(function["parameters"]["additionalProperties"])

    def test_movement_schema_requires_object_direction_and_file_id(self) -> None:
        function = TEST_MOVEMENT_TOOL["function"]
        self.assertEqual(function["name"], "test_movement")
        self.assertEqual(
            function["parameters"]["required"],
            ["object_id", "direction", "file_id"],
        )
        self.assertEqual(
            function["parameters"]["properties"]["direction"]["enum"],
            ["up", "down", "left", "right"],
        )
        self.assertFalse(function["parameters"]["additionalProperties"])

    def test_replaced_bounding_box_name_is_not_executable(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown tool"):
            self.runtime.execute(
                "bounding_box",
                {"object_id": "table", "file_id": self.file_id},
            )

    def test_tool_descriptions_do_not_prescribe_other_tool_calls(self) -> None:
        tool_names = {
            tool["function"]["name"]
            for tool in TOOLS
        }
        ordering_phrases = (
            "call this first",
            "use this first",
            "before calling",
            "after calling",
            "before using",
            "after using",
        )
        for tool in TOOLS:
            function = tool["function"]
            description = function["description"].casefold()
            for other_name in tool_names - {function["name"]}:
                self.assertNotIn(other_name.casefold(), description)
            for phrase in ordering_phrases:
                self.assertNotIn(phrase, description)


if __name__ == "__main__":
    unittest.main()
