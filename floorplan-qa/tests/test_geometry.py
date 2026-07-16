from __future__ import annotations

import math
import unittest

from shapely.affinity import rotate
from shapely.geometry import Point, Polygon, box

from floorplan_qa.cgal_geometry import backend_name
from floorplan_qa.geometry import (
    GEOMETRY_TOLERANCE,
    configuration_space_region,
    maximum_slide_distance,
    optimize_maximum_rectangle,
    placement_witness,
    rectangle_from_parameters,
    type_a_contact_candidates,
)


class MaximumRectangleContactTests(unittest.TestCase):
    def test_type_a_enumerates_square_from_opposite_boundary_vertices(self) -> None:
        square = box(0.0, 0.0, 2.0, 2.0)

        candidates = type_a_contact_candidates(square)

        self.assertTrue(
            any(math.isclose(item[2] * item[3], 4.0, rel_tol=1e-7) for item in candidates)
        )

    def test_contact_optimizer_recovers_rotated_rectangle(self) -> None:
        polygon = rotate(
            box(-2.0, -1.0, 2.0, 1.0),
            30.0,
            origin=(0.0, 0.0),
            use_radians=False,
        )

        best, metadata = optimize_maximum_rectangle(polygon, polygon)

        self.assertAlmostEqual(best[2] * best[3], 8.0, delta=1e-4)
        self.assertTrue(
            polygon.buffer(GEOMETRY_TOLERANCE).covers(
                rectangle_from_parameters(best)
            )
        )
        self.assertEqual(metadata["algorithm"], "contact_event_shgo")
        self.assertGreater(metadata["shgo"]["function_evaluations"], 0)
        self.assertIn("contacts", metadata)

    def test_contact_optimizer_is_repeatable(self) -> None:
        polygon = box(0.0, 0.0, 5.0, 3.0)

        first, first_metadata = optimize_maximum_rectangle(polygon, polygon)
        second, second_metadata = optimize_maximum_rectangle(polygon, polygon)

        self.assertEqual(first, second)
        self.assertEqual(first_metadata, second_metadata)


class CgalConfigurationSpaceTests(unittest.TestCase):
    def test_native_backend_is_loaded(self) -> None:
        self.assertEqual(backend_name(), "CGAL exact configuration space")

    def test_rectangle_center_region_matches_room_erosion(self) -> None:
        region = configuration_space_region(
            box(0.0, 0.0, 4.0, 4.0), 2.0, 2.0, 0.0
        )

        self.assertAlmostEqual(region.area, 4.0, delta=1e-6)
        for actual, expected in zip(region.bounds, (1.0, 1.0, 3.0, 3.0)):
            self.assertAlmostEqual(actual, expected, delta=2e-7)

    def test_center_region_rejects_body_overlap_when_corners_are_clear(self) -> None:
        free_space = box(0.0, 0.0, 6.0, 6.0).difference(
            box(2.75, 2.75, 3.25, 3.25)
        )

        region = configuration_space_region(free_space, 2.0, 2.0, 0.0)

        self.assertFalse(region.covers(Point(3.0, 3.0)))

    def test_impossible_rectangle_has_no_placement_witness(self) -> None:
        self.assertIsNone(placement_witness(box(0.0, 0.0, 2.0, 2.0), 3.0, 1.0))

    def test_slide_distance_uses_first_configuration_space_contact(self) -> None:
        moving = box(0.0, 0.0, 1.0, 1.0)
        obstacle = box(3.0, 0.0, 4.0, 1.0)
        room = box(0.0, 0.0, 10.0, 4.0)

        distance = maximum_slide_distance(moving, room, [obstacle], (1.0, 0.0))

        self.assertAlmostEqual(distance, 2.0, delta=2e-5)

    def test_slide_can_move_away_from_an_initial_boundary_contact(self) -> None:
        moving = box(1.0, 1.0, 2.0, 2.0)
        obstacle = box(0.0, 1.0, 1.0, 2.0)
        room = Polygon(
            [(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)]
        )

        distance = maximum_slide_distance(moving, room, [obstacle], (1.0, 0.0))

        self.assertAlmostEqual(distance, 3.0, delta=2e-5)


if __name__ == "__main__":
    unittest.main()
