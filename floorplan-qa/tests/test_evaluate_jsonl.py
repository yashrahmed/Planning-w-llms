from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from floorplan_qa.evaluate_jsonl import (
    Example,
    answers_match,
    build_ollama_payload,
    discrete_frechet,
    parse_sequence,
    sample_examples,
    write_evaluation_report,
)
from floorplan_qa.evaluate_tools import (
    agent_messages,
    build_ollama_payload as build_tool_ollama_payload,
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

    def test_v1_ollama_payload_keeps_raw_layout_and_has_no_tools(self) -> None:
        example = Example(
            line_number=1,
            example_id="raw-context",
            messages=[
                {"role": "system", "content": "Solve the floorplan question."},
                {
                    "role": "user",
                    "content": "Question\n\nRoom layout:\n{\"objects\":[{\"label\":\"chair\"}]}",
                },
            ],
            expected="0.000",
            task="pair_distance",
            reference_answer=0.0,
            source_layout="kitchen/room_0.json",
            parameters={},
        )

        payload = build_ollama_payload(
            example,
            model="qwen3.5:4b",
            thinking=False,
            seed=0,
            max_tokens=2500,
        )

        self.assertEqual(payload["messages"], example.messages)
        self.assertIn("Room layout:", payload["messages"][1]["content"])
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

    def test_json_report_write_is_complete_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "result.json"
            report = {"status": "running", "results": [{"correct": True}]}
            write_evaluation_report(output_path, report)
            self.assertEqual(json.loads(output_path.read_text()), report)
            self.assertFalse((output_path.parent / ".result.json.tmp").exists())

    def test_tool_payload_has_tools_without_scorer_metadata(self) -> None:
        example = Example(
            line_number=1,
            example_id="explicit-file",
            messages=[
                {"role": "system", "content": "Use exact geometry."},
                {
                    "role": "user",
                    "content": (
                        "Calculate the distance.\n\nRoom layout can be found in "
                        "file : kitchen-7-train.json"
                    ),
                },
            ],
            expected="4.000",
            task="pair_distance",
            reference_answer=4.0,
            source_layout="kitchen/room_7.json",
            parameters={"object_1": "first", "object_2": "second"},
            layout_file="kitchen-7-train.json",
        )

        messages = agent_messages(example)
        payload = build_tool_ollama_payload(
            messages,
            model="qwen3.5:4b",
            seed=0,
            max_tokens=2500,
        )
        serialized_messages = json.dumps(payload["messages"])
        self.assertIn("kitchen-7-train.json", serialized_messages)
        self.assertIn(
            "your next response must be the final answer", serialized_messages
        )
        self.assertIn(
            "Never repeat a tool call with the same arguments", serialized_messages
        )
        self.assertNotIn("kitchen/room_7.json", serialized_messages)
        self.assertNotIn("pair_distance", serialized_messages)
        self.assertNotIn("4.000", serialized_messages)
        self.assertNotIn("object_1", serialized_messages)
        self.assertEqual(len(payload["tools"]), 11)
        self.assertIn(
            "find_space_with_size",
            [tool["function"]["name"] for tool in payload["tools"]],
        )
        self.assertNotIn("tool_choice", payload)


if __name__ == "__main__":
    unittest.main()
