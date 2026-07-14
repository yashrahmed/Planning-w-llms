from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from floorplan_qa.evaluate_jsonl import Example
from floorplan_qa.evaluate_tools import compact_question
from floorplan_qa.floorplan_tools import FloorplanToolRuntime, tool_names


def rectangle(name: str, x1: float, y1: float, x2: float, y2: float) -> dict:
    return {
        "label": name,
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
        room_dir = self.layout_dir / "kitchen"
        room_dir.mkdir()
        layout = {
            "layout_id": 7,
            "room_type": "kitchen",
            "room_boundary": [
                {"x": 0.0, "y": 0.0},
                {"x": 5.0, "y": 0.0},
                {"x": 5.0, "y": 5.0},
                {"x": 0.0, "y": 5.0},
                {"x": 0.0, "y": 0.0},
            ],
            "openings": {"doors": [], "windows": []},
            "objects": [
                rectangle("first", 0.0, 0.0, 1.0, 1.0),
                rectangle("blocker", 2.0, 0.0, 3.0, 1.0),
                rectangle("second", 4.0, 0.0, 5.0, 1.0),
            ],
        }
        (room_dir / "room_7.json").write_text(json.dumps(layout), encoding="utf-8")
        self.example = Example(
            line_number=1,
            example_id="pair-distance-kitchen-7",
            messages=[
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": (
                        "Given the kitchen layout below in JSON, calculate distance.\n\n"
                        "Room layout:\n{\"large\":\"payload\"}\n\nBriefly answer."
                    ),
                },
            ],
            expected="4.000",
            task="pair_distance",
            reference_answer=4.0,
            source_layout="kitchen/room_7.json",
            parameters={"object_1": "first", "object_2": "second"},
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_toolsets_grow_cumulatively(self) -> None:
        counts = [len(tool_names(iteration)) for iteration in range(1, 6)]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts, [3, 5, 7, 8, 9])

    def test_pair_tool_returns_all_simple_relationships(self) -> None:
        runtime = FloorplanToolRuntime(self.example, self.layout_dir, seed=0)
        result = runtime.measure_pair("first", "second")
        self.assertEqual(result["distance_answer"], "4.000")
        self.assertEqual(result["angle_answer"], "90.000")
        self.assertEqual(result["intersections_in_order"], ["blocker"])

    def test_current_question_solver_recomputes_answer(self) -> None:
        runtime = FloorplanToolRuntime(self.example, self.layout_dir, seed=0)
        self.assertEqual(
            runtime.solve_current_question(),
            {"task": "pair_distance", "final_answer": "4.000"},
        )

    def test_compact_prompt_removes_raw_layout(self) -> None:
        compact = compact_question(self.example)
        self.assertNotIn("large", compact)
        self.assertIn("Final answer", compact)


if __name__ == "__main__":
    unittest.main()
