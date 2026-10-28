"""Scoring contracts using invented inputs and no model calls."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev


class ScoringGaps(unittest.TestCase):
    def test_json_tail_pointer_selects_last_complete_container(self):
        turns = [{"id": 7, "text": 'Keep the closing brace } and quote " here.'}]
        batch = {"raw/turns": turns, "nested": {"last": [9]}}
        cases = (
            ("Read these instructions. " + json.dumps(batch), "/raw~1turns", turns),
            ('Example: {"raw/turns": []}\nActual batch:\n' + json.dumps(batch)
             + '\nContinue after this. {"unfinished":', "/raw~1turns", turns),
            ("Use this batch: " + json.dumps(turns) + "\nEnd.", "/0/id", 7),
            (json.dumps(batch), "", batch),
        )
        for content, suffix, expected in cases:
            with self.subTest(content=content):
                try:
                    actual = ev._input_pointer({"content": content}, "/content/~json-tail" + suffix)
                except ValueError as error:
                    actual = str(error)
                self.assertEqual(actual, expected)
        for content in ("No batch here.", "Only 123 and true.", '{"unfinished":', "[NaN]"):
            with self.subTest(invalid=content):
                with self.assertRaisesRegex(ValueError, "input pointer.*did not locate source data"):
                    ev._input_pointer({"content": content}, "/content/~json-tail")
        self.assertEqual(ev._pointer({"~json-tail": 4}, "/~0json-tail"), 4)
        self.assertEqual(ev._pointer('{"turns":[7]}', "/turns/0"), 7)

    def test_cleanup_can_locate_batch_after_instructions(self):
        schema = {"type": "object", "x-eval": {"cleanup": {
            "input_turns": "/messages/0/content/~json-tail/turns",
            "output_turns": "/turns", "id": "/id", "text": "/text"}}}
        batch = {"turns": [{"id": 7, "text": "The gate is ready"}]}
        messages = [{"role": "user", "content": "Preserve the words.\n" + json.dumps(batch)}]
        try:
            scores = ev.check_output("transcript_cleanup", json.dumps(batch), schema=schema, messages=messages)
        except ValueError as error:
            scores = {"structurally_bad": str(error)}
        self.assertFalse(scores["structurally_bad"], scores)
        self.assertEqual(scores["missing_ids"], [])
        self.assertEqual(scores["overall_overlap"], 1)


if __name__ == "__main__":
    unittest.main()
