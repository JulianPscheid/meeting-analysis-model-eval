"""Scoring contracts using invented inputs and no model calls."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev


class ScoringGaps(unittest.TestCase):
    def test_xml_any_of_accepts_nonempty_fields_at_any_depth(self):
        from test_review_regressions import ReviewRegressions
        schema = {"mode": "any_of", "fields": ["meaning", "evidence"]}
        for content in (
            "<other><wrapper><evidence> The gate is ready. </evidence></wrapper><extra/></other>",
            "<meaning>The gate is ready.</meaning>",
            "<other><meaning/><meaning><bold>The gate</bold> is ready.</meaning></other>",
        ):
            with self.subTest(content=content):
                try:
                    scores = ev.check_output("highlight_analysis", content, schema=schema)
                except ValueError as error:
                    scores = {"structurally_bad": str(error)}
                self.assertFalse(scores["structurally_bad"], scores)
                self.assertTrue(scores["xml_valid"])
                self.assertIsNone(scores["root_matches"])
                self.assertTrue(scores["matched_fields"])
                result = ReviewRegressions().score([{"intent": "highlight_analysis", "content": content}], schema)
                self.assertTrue(result["runs"][0]["usable"], result)

    def test_xml_any_of_rejects_empty_fields_and_preserves_parse_guards(self):
        schema = {"mode": "any_of", "fields": ["meaning", "evidence"]}
        for content in (
            "<other><meaning> \t\n </meaning><evidence/></other>",
            '<other><meaning note="The gate is ready"/><extra>The gate is ready.</extra></other>',
            "<other><wrapper>The gate is ready.</wrapper></other>",
            "<other><meaning>The gate is ready.</other>",
            '<!DOCTYPE other [<!ENTITY x "The gate is ready">]><other><meaning>&x;</meaning></other>',
        ):
            with self.subTest(content=content):
                try:
                    scores = ev.check_output("highlight_analysis", content, schema=schema)
                except ValueError as error:
                    scores = {"structurally_bad": str(error)}
                self.assertIs(scores["structurally_bad"], True, scores)
        for bad in ({"mode": "unknown"}, {"mode": "any_of", "fields": []},
                    {"mode": "any_of", "fields": "meaning"},
                    {"mode": "any_of", "fields": ["bad/name"]}):
            with self.subTest(schema=bad):
                with self.assertRaises(ValueError):
                    ev.check_output("highlight_analysis", "<other/>", schema=bad)

    def test_xml_strict_remains_the_default(self):
        schema = {"root": "highlight", "children": ["meaning", "evidence"]}
        for content, expected_bad in (
            ("<highlight><meaning>The gate is ready.</meaning><evidence/></highlight>", False),
            ("<other><meaning>The gate is ready.</meaning><evidence/></other>", True),
            ("<highlight><meaning>The gate is ready.</meaning></highlight>", True),
            ("<highlight><meaning>The gate is ready.</meaning><evidence/><extra/></highlight>", True),
            ("<highlight><wrapper><meaning>The gate is ready.</meaning></wrapper><evidence/></highlight>", True),
        ):
            with self.subTest(content=content):
                default = ev.check_output("highlight_analysis", content, schema=schema)
                explicit = ev.check_output("highlight_analysis", content, schema={**schema, "mode": "strict"})
                self.assertEqual(default, explicit)
                self.assertEqual(default["structurally_bad"], expected_bad)

    def test_integer_speaker_labels_match_discovered_text_and_report_unknowns(self):
        schema = {"type": "array", "items": {"type": "object", "properties": {
            "label": {"type": ["string", "integer"]}, "name": {"type": "string"}}},
            "x-eval": {"speakers": {"mappings": "", "label": "/label", "name": "/name",
                                     "input_labels_pattern": r"(?<=speaker_id=)\d+"}}}
        messages = [{"role": "user", "content": "speaker_id=7: Ari Lane is ready."}]
        for label in (7, "7"):
            with self.subTest(known=label):
                scores = ev.check_output("speaker_identification", json.dumps([
                    {"label": label, "name": "Ari Lane"}]), schema=schema, messages=messages)
                self.assertEqual(scores["unknown_labels"], [])
                self.assertFalse(scores["structurally_bad"], scores)
        content = json.dumps([{"label": label, "name": "Ari Lane"} for label in (9, "10", 9, "10")])
        scores = ev.check_output("speaker_identification", content, schema=schema, messages=messages)
        self.assertEqual(scores["unknown_labels"], ["10", "9"])
        self.assertTrue(scores["structurally_bad"], scores)
        # Booleans are not integer ids, even though Python bool subclasses int.
        self.assertEqual(ev._speaker_scores('[{"label":true}]', messages, schema)["unknown_labels"], [])

    def test_schema_valid_json_without_words_is_clean_and_usable(self):
        from test_review_regressions import ReviewRegressions
        cases = (
            ({"count": 4, "ready": True}, {"type": "object", "properties": {
                "count": {"type": "integer"}, "ready": {"type": "boolean"}}}),
            ([], {"type": "array", "items": {"type": "integer"}}),
            ([0, False], {"type": "array", "items": {"type": ["integer", "boolean"]}}),
            (0, {"type": "integer"}), (False, {"type": "boolean"}),
            (None, {"type": "null"}),
        )
        for value, schema in cases:
            with self.subTest(value=value):
                row = {"intent": "feedback_analysis", "content": json.dumps(value)}
                scores = ev.check_output(row["intent"], row["content"], schema=schema)
                self.assertTrue(scores["empty_visible_text"])
                self.assertEqual(scores["schema_errors"], [])
                self.assertFalse(scores["structurally_bad"], scores)
                result = ReviewRegressions().score([row], schema)
                self.assertTrue(result["runs"][0]["usable"], result)
                self.assertEqual(result["runs"][0]["unusable_reasons"], [])
                self.assertEqual(result["summary"]["arm"]["usable"], 1)

    def test_schema_valid_wordless_text_stays_unusable(self):
        # Only payloads with no text fields at all are exempt; text that carries no words is still empty.
        cases = (("", {"type": "string"}),
                 ({"suggestion": "..."}, {"type": "object", "properties": {"suggestion": {"type": "string"}}}),
                 ({"suggestion": " \u200b"}, {"type": "object"}))
        for value, schema in cases:
            with self.subTest(value=value):
                scores = ev.check_output("feedback_analysis", json.dumps(value), schema=schema)
                self.assertEqual(scores["schema_errors"], [])
                self.assertTrue(scores["structurally_bad"], scores)
        xml_schema = {"mode": "any_of", "fields": ["title", "insight"]}
        scores = ev.check_output("highlight_analysis", "<analysis><title>...</title></analysis>", schema=xml_schema)
        self.assertTrue(scores["structurally_bad"], scores)

    def test_empty_json_keeps_other_failure_gates(self):
        from test_review_regressions import ReviewRegressions
        for content, schema in (("[]", {"type": "array", "minItems": 1}),
                                ("[broken", {"type": "array"}),
                                ("```json\n[]\n```", {"type": "array"})):
            with self.subTest(content=content):
                row = {"intent": "feedback_analysis", "content": content}
                scored = ReviewRegressions().score([row], schema)["runs"][0]
                self.assertTrue(scored["structurally_bad"], scored)
                self.assertFalse(scored["usable"])
        for extra, reason in (({"finish": "length"}, "truncation"),
                              ({"reasoning_chars": 8}, "reasoning_leakage")):
            row = {"intent": "feedback_analysis", "content": "[]", **extra}
            scored = ReviewRegressions().score([row], {"type": "array"})["runs"][0]
            self.assertTrue(scored["structurally_bad"], scored)
            self.assertIn(reason, scored["unusable_reasons"])
        self.assertTrue(ev.check_output("chat", "...")["structurally_bad"])
        self.assertTrue(ev.check_output("summary", "[]")["structurally_bad"])

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
