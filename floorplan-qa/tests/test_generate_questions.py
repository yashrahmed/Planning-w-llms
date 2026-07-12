from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from shapely.geometry import Polygon

from floorplan_qa.generate_questions import (
    TASKS,
    entity_centroid,
    generate_record,
    label,
    layout_paths,
    load_layout,
    maximum_slide_distance,
)
from floorplan_qa.quality_metrics import RATE_THRESHOLDS, assess


def rectangle(label_value: str, x1: float, y1: float, x2: float, y2: float) -> dict:
    return {
        "label": label_value,
        "points": [
            {"x": x1, "y": y1},
            {"x": x2, "y": y1},
            {"x": x2, "y": y2},
            {"x": x1, "y": y2},
        ],
    }


def sample_layout() -> dict:
    return {
        "layout_id": 0,
        "room_type": "kitchen",
        "room_boundary": [
            {"x": 0.0, "y": 0.0},
            {"x": 6.0, "y": 0.0},
            {"x": 6.0, "y": 5.0},
            {"x": 0.0, "y": 5.0},
            {"x": 0.0, "y": 0.0},
        ],
        "openings": {
            "windows": [rectangle("window", 0.0, 2.0, 0.05, 3.0)],
            "doors": [rectangle("door", 2.5, 4.85, 3.5, 5.0)],
        },
        "objects": [
            rectangle("fridge", 0.0, 0.0, 0.8, 0.8),
            rectangle("stove", 5.2, 0.0, 6.0, 0.8),
            rectangle("sink", 0.0, 4.2, 0.8, 5.0),
            rectangle("table", 2.5, 2.0, 3.5, 3.0),
            rectangle("rug", 1.5, 1.0, 4.5, 4.0),
        ],
    }


class GeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.layout_dir = Path(self.temporary_directory.name) / "layouts"
        room_dir = self.layout_dir / "kitchen"
        room_dir.mkdir(parents=True)
        self.layout_path = room_dir / "room_0.json"
        self.layout_path.write_text(json.dumps(sample_layout()), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_uniform_layout_shuffle_is_stable_and_complete(self) -> None:
        bedroom = self.layout_dir / "bedroom"
        bedroom.mkdir()
        for index in range(3):
            (bedroom / f"room_{index}.json").write_text("{}", encoding="utf-8")
        first = layout_paths(self.layout_dir, seed=12)
        second = layout_paths(self.layout_dir, seed=12)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(set(first), set(self.layout_dir.glob("*/*.json")))

    def test_continuous_repositioning_finds_first_contact(self) -> None:
        moving = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        obstacle = Polygon([(3, 0), (4, 0), (4, 1), (3, 1)])
        room = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
        distance = maximum_slide_distance(moving, room, [obstacle], (1.0, 0.0))
        self.assertAlmostEqual(distance, 2.0, delta=2e-5)

    def test_all_tasks_are_deterministic_and_pass_quality_gates(self) -> None:
        context = load_layout(self.layout_path)
        first = [
            generate_record(context, self.layout_dir, task, seed=19) for task in TASKS
        ]
        second = [
            generate_record(context, self.layout_dir, task, seed=19) for task in TASKS
        ]
        self.assertEqual(first, second)
        self.assertEqual({record["task"] for record in first}, set(TASKS))
        metrics, failures = assess(first, self.layout_dir, compare_equal=True)
        self.assertEqual(failures, [])
        for metric, threshold in RATE_THRESHOLDS.items():
            if metric in metrics:
                self.assertGreaterEqual(metrics[metric], threshold, metric)

    def test_released_numeric_regressions_when_dataset_is_available(self) -> None:
        released_root = (
            Path(__file__).resolve().parents[1]
            / "datasets"
            / "FloorplanQA-Layouts"
            / "layouts"
        )
        hssd_path = released_root / "hssd" / "room_14.json"
        kitchen_path = released_root / "kitchen" / "room_50.json"
        if not hssd_path.is_file() or not kitchen_path.is_file():
            self.skipTest("released layouts have not been downloaded")

        hssd = load_layout(hssd_path)
        hssd_entities = {label(entity): entity for entity in hssd.entities}
        distance = math.dist(
            entity_centroid(hssd_entities["sink"]),
            entity_centroid(hssd_entities["shower_2"]),
        )
        self.assertAlmostEqual(distance, 1.514, places=3)

        kitchen = load_layout(kitchen_path)
        kitchen_entities = {label(entity): entity for entity in kitchen.entities}
        start = entity_centroid(kitchen_entities["chair_2"])
        end = entity_centroid(kitchen_entities["window_1"])
        dx, dy = end[0] - start[0], end[1] - start[1]
        angle = math.degrees(math.acos(dy / math.hypot(dx, dy)))
        self.assertAlmostEqual(angle, 100.081, places=3)


if __name__ == "__main__":
    unittest.main()
