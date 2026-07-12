from __future__ import annotations

import unittest

from floorplan_qa.evaluate_jsonl import (
    answers_match,
    discrete_frechet,
    parse_sequence,
)


class EvaluatorTests(unittest.TestCase):
    def test_scalar_tasks_use_paper_relative_error(self) -> None:
        self.assertTrue(
            answers_match("10.190", "10.000", "pair_distance", 10.0)
        )
        self.assertFalse(
            answers_match("10.210", "10.000", "pair_distance", 10.0)
        )

    def test_free_space_uses_five_percent_relative_error(self) -> None:
        self.assertTrue(answers_match("20.900", "20.000", "free_space", 20.0))
        self.assertFalse(answers_match("21.100", "20.000", "free_space", 20.0))

    def test_visibility_uses_set_equality(self) -> None:
        reference = ["table", "chair"]
        self.assertTrue(
            answers_match("[chair, table]", "[]", "visibility", reference)
        )
        self.assertFalse(
            answers_match("[chair]", "[]", "visibility", reference)
        )
        self.assertEqual(parse_sequence("[chair, table]"), ["chair", "table"])

    def test_discrete_frechet_distance(self) -> None:
        reference = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        nearby = [(0.0, 0.0), (1.0, 1.2), (2.0, 0.0)]
        self.assertAlmostEqual(discrete_frechet(reference, nearby), 0.2)


if __name__ == "__main__":
    unittest.main()
