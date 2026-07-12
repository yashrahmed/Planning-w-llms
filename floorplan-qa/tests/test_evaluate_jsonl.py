from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from floorplan_qa.evaluate_jsonl import (
    Example,
    answers_match,
    discrete_frechet,
    parse_sequence,
    sample_examples,
    write_evaluation_report,
)


class EvaluatorTests(unittest.TestCase):
    @staticmethod
    def examples(count: int) -> list[Example]:
        return [
            Example(
                line_number=index + 1,
                example_id=f"example-{index}",
                messages=[{"role": "user", "content": "question"}],
                expected="0.000",
                task="pair_distance",
                reference_answer=0.0,
                source_layout="kitchen/room_0.json",
                parameters={},
            )
            for index in range(count)
        ]

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

    def test_seeded_sampling_is_uniform_without_replacement(self) -> None:
        examples = self.examples(200)
        first = sample_examples(examples, 150, seed=0)
        second = sample_examples(examples, 150, seed=0)
        different = sample_examples(examples, 150, seed=1)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(len({example.example_id for example in first}), 150)

    def test_sampling_rejects_an_oversized_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            sample_examples(self.examples(2), 3, seed=0)

    def test_json_report_write_is_complete_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "result.json"
            report = {"status": "running", "results": [{"correct": True}]}
            write_evaluation_report(output_path, report)
            self.assertEqual(json.loads(output_path.read_text()), report)
            self.assertFalse((output_path.parent / ".result.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
