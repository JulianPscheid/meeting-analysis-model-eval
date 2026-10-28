"""Regressions for mechanical scoring; invented data, no model calls."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev

FIXTURES = Path(ev.__file__).parent / "fixtures"


class ReviewRegressions(unittest.TestCase):
    def score(self, rows, schema=None, **flags):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "transcript.txt"
            transcript.write_text("Invented meeting evidence.", encoding="utf-8")
            records = root / "runs.jsonl"
            base = {"label": "arm", "finish": "stop", "error": None,
                    "reasoning_chars": 0, "transcript_sha256": ev._file_sha256(transcript)}
            records.write_text("\n".join(json.dumps({**base, **row}) for row in rows), encoding="utf-8")
            argv = ["score", "--transcript", str(transcript), str(records)]
            if schema is not None:
                schema_file = root / "verdict-schema.json"
                schema_file.write_text(json.dumps(schema), encoding="utf-8")
                argv += ["--schema-file", str(schema_file)]
            args = ev.build_parser().parse_args(argv)
            for key, value in flags.items():
                setattr(args, key, value)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                self.assertEqual(ev.command_score(args), 0)
            return json.loads(printed.getvalue())

    def test_zero_word_visible_output_for_every_intent(self):
        for intent, spec in ev.INTENTS.items():
            schema, messages = None, []
            if spec["kind"] == "json":
                content, schema = '{"text":" \\t\\u200b\\u200d\\ufeff..."}', {"type": "object"}
            else:
                content = " \t\u200b\u200d\ufeff..."
            if intent == "summary":
                content = json.dumps({"title": "...", "recap_markdown": "...", "user_todos": []})
            elif intent == "highlight_analysis":
                schema = {"root": "highlight", "children": ["meaning", "evidence", "follow_up"]}
                content = '<highlight note="Words only in attributes"><meaning/><evidence/><follow_up/></highlight>'
            elif intent == "speaker_identification":
                schema["x-eval"] = {"speakers": {"mappings": "/speakers", "label": "/label", "name": "/name"}}
                content = '{"speakers":[]}'
            elif intent == "transcript_cleanup":
                schema["x-eval"] = {"cleanup": {"input_turns": "/messages/0/content/turns",
                    "output_turns": "/turns", "id": "/id", "text": "/text"}}
                messages = [{"role": "user", "content": '{"turns":[{"id":1,"text":"Original words"}]}'}]
                content = '{"turns":[{"id":1,"text":" \\u200b..."}]}'
            with self.subTest(intent=intent):
                scores = ev.check_output(intent, content, schema=schema, messages=messages)
                self.assertEqual(scores["words"], 0)
                self.assertTrue(scores["structurally_bad"], scores)
                self.assertIs(scores["empty_visible_text"], True)

    def test_zero_words_are_bad_for_whitespace_and_punctuation(self):
        for text in ("", " \t\n", "\u200b\u200d\ufeff", "..."):
            for intent, content, schema in (
                ("chat", text, None), ("transcript_summary", text, None),
                ("feedback_analysis", json.dumps({"words": text}), {"type": "object"}),
                ("highlight_analysis", f"<highlight><meaning>{text}</meaning></highlight>",
                 {"root": "highlight", "children": ["meaning"]}),
            ):
                with self.subTest(intent=intent, text=repr(text)):
                    scores = ev.check_output(intent, content, schema=schema)
                    self.assertTrue(scores["structurally_bad"], scores)
                    self.assertTrue(scores["empty_visible_text"])
        self.assertFalse(ev.check_output("chat", "Visible words.")["empty_visible_text"])

    def test_ordinary_analysis_prose_keeps_legacy_paths_clean(self):
        for line in ("Analysis: the budget moved.", "reasoning: the scope was reduced."):
            rows = [
                {"intent": "notes", "grammar": "off", "content": "# Review\n## Budget\n" + line},
                {"intent": "summary", "grammar": "on", "content": json.dumps({
                    "title": "Budget", "recap_markdown": "## Review\n" + line, "user_todos": []})},
            ]
            scored = self.score(rows)
            for row in scored["runs"]:
                with self.subTest(intent=row["intent"], line=line):
                    self.assertFalse(row["reasoning_leak"])
                    self.assertFalse(row["structurally_bad"], row)

    def test_explicit_reasoning_is_still_detected(self):
        for marker in ("<think>Plan</think>", "<thinking>Plan</thinking>",
                       "<|analysis|>Plan", "<|channel|>analysis"):
            scored = self.score([{"intent": "chat", "content": marker + " Answer."}])["runs"][0]
            self.assertTrue(scored["reasoning_leak"], marker)
            self.assertTrue(scored["structurally_bad"])
        row = {"intent": "chat", "content": "Answer.", "reasoning_chars": 10}
        self.assertTrue(self.score([row])["runs"][0]["structurally_bad"])
        self.assertFalse(self.score([row], allow_reasoning=True)["runs"][0]["structurally_bad"])

    def verdict_schema(self, explicit=False):
        schema = {"type": "object", "properties": {"rows": {"type": "array", "items": {
            "type": "object", "properties": {"verdict": {"enum": ["show", "hide"]}}}}}}
        if explicit:
            schema["properties"]["strength"] = {"enum": ["low", "high"]}
            schema["x-eval"] = {"verdict": "/rows/verdict"}
        return schema

    def test_array_verdict_discovery_pointer_and_distribution(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                rows = [{"intent": "feedback_evaluation", "grammar": "on", "content": json.dumps({
                    "rows": [{"verdict": verdict} for verdict in values], "strength": "low"})}
                    for values in (("hide", "show", "hide"), ("show", "invalid"))]
                result = self.score(rows, self.verdict_schema(explicit))
                self.assertEqual(result["summary"]["arm"]["verdict_distribution"],
                                 {"hide": 2, "show": 2, "invalid": 1})
                self.assertEqual(result["runs"][0]["verdicts"], ["hide", "show", "hide"])
                self.assertTrue(result["runs"][1]["schema_errors"])

    def test_misconfigured_verdict_pointer_is_reported_per_row(self):
        schema = self.verdict_schema()
        schema["x-eval"] = {"verdict": "/rows/missing"}
        rows = [{"intent": "feedback_evaluation", "content": content}
                for content in ('{"rows":[{"verdict":"show"}]}', '{"rows":[]}', '{broken')]
        scored = self.score(rows, schema)
        self.assertEqual(len(scored["runs"]), 3)
        for row in scored["runs"]:
            self.assertIn("x-eval.verdict", row["verdict_error"])
            self.assertIn("/rows/missing", row["verdict_error"])
            self.assertTrue(row["structurally_bad"])

    def test_nested_arrays_and_escaped_verdict_pointer(self):
        schema = {"type": "array", "items": {"type": "object", "properties": {
            "a/b": {"type": "array", "items": {"type": "array", "items": {
                "type": "object", "properties": {"v~d": {"enum": ["show", "hide"]}}}}}}},
            "x-eval": {"verdict": "/a~1b/v~0d"}}
        content = json.dumps([{"a/b": [[{"v~d": "show"}, {"v~d": "hide"}]]},
                              {"a/b": [[{"v~d": "show"}]]}])
        result = self.score([{"intent": "feedback_evaluation", "content": content}], schema)
        self.assertEqual(result["summary"]["arm"]["verdict_distribution"], {"show": 2, "hide": 1})

    def test_summary_accepts_trailing_suggested_topic(self):
        base = {"title": "T", "recap_markdown": "## Brief\n\nFacts", "user_todos": []}
        with_topic = json.dumps({**base, "suggested_topic": None})
        self.assertTrue(ev.score_content(with_topic, "transcript")["exact_fields_in_order"])
        reordered = json.dumps({"suggested_topic": None, **base})
        self.assertFalse(ev.score_content(reordered, "transcript")["exact_fields_in_order"])
        extra = json.dumps({**base, "suggested_topic": None, "other": 1})
        self.assertFalse(ev.score_content(extra, "transcript")["exact_fields_in_order"])

    def test_text_code_fences_are_structurally_bad(self):
        for intent, body in (("chat", "Answer."), ("transcript_summary", "Summary."),
                             ("notes", "# Review\n## Budget\nThe budget moved.")):
            for language in ("", "text", "markdown"):
                with self.subTest(intent=intent, language=language):
                    scores = ev.check_output(intent, f"```{language}\n{body}\n```")
                    self.assertTrue(scores["structurally_bad"], scores)
                    self.assertIs(scores["code_fence"], True)

    def test_text_schema_applies_to_visible_text(self):
        for intent, content in (("chat", "Facts."), ("transcript_summary", "Facts."),
                                ("notes", "# Review\nFacts.")):
            for schema, error in (({"type": "object"}, "expected object"),
                                  ({"type": "string", "minLength": 100}, "minLength"),
                                  ({"type": "string", "maxLength": 2}, "maxLength")):
                with self.subTest(intent=intent, schema=schema):
                    scores = ev.check_output(intent, content, schema=schema)
                    self.assertTrue(scores["structurally_bad"], scores)
                    self.assertTrue(any(error in item for item in scores["schema_errors"]))
            self.assertFalse(ev.check_output(intent, content, schema={"type": "string"})["structurally_bad"])
        scores = ev.check_output("chat", "```markdown\nFacts.\n```",
                                 schema={"type": "string", "minLength": 6, "maxLength": 6})
        self.assertEqual(scores["schema_errors"], [])
        with self.assertRaisesRegex(ValueError, "unsupported JSON schema"):
            ev.check_output("chat", "Facts.", schema={"pattern": "Facts"})

    def test_flat_cleanup_uses_text_checks_and_keeps_proxies_report_only(self):
        schema = json.loads((FIXTURES / "transcript_cleanup/schema.flat.json").read_text())
        request = json.loads((FIXTURES / "transcript_cleanup/request.flat.example.json").read_text())
        for content in ("...", "```text\nFacts.\n```"):
            scores = ev.check_output("transcript_cleanup", content, schema=schema, messages=request["messages"])
            self.assertTrue(scores["structurally_bad"])
        scores = ev.check_output("transcript_cleanup", "Completely different words.",
                                 schema=schema, messages=request["messages"])
        self.assertEqual(scores["overall_overlap"], 0)
        self.assertFalse(scores["structurally_bad"])
        schema["maxLength"] = 2
        scores = ev.check_output("transcript_cleanup", "Facts.", schema=schema, messages=request["messages"])
        self.assertTrue(any("maxLength" in error for error in scores["schema_errors"]))
        self.assertTrue(scores["structurally_bad"])

    def test_speaker_names_collapse_whitespace(self):
        schema = json.loads((FIXTURES / "speaker_identification/schema.json").read_text())
        for name, source in (("Mina  Vale", "Mina Vale"), ("Mina Vale", "Mina \t Vale")):
            scores = ev.check_output("speaker_identification", json.dumps({"speakers": [
                {"label": "Speaker 1", "name": name}]}), schema=schema,
                messages=[{"role": "user", "content": f"Speaker 1: {source} speaks."}])
            self.assertEqual(scores["unsupported_names"], [])
            self.assertFalse(scores["structurally_bad"], scores)

    def test_speaker_labels_are_case_insensitive(self):
        schema = json.loads((FIXTURES / "speaker_identification/schema.json").read_text())
        for label, source in (("speaker 1", "Speaker 1"), ("Speaker 1", "speaker 1")):
            scores = ev.check_output("speaker_identification", json.dumps({"speakers": [
                {"label": label, "name": "Mina Vale"}]}), schema=schema,
                messages=[{"role": "user", "content": f"{source}: Mina Vale speaks."}])
            self.assertEqual(scores["unknown_labels"], [])
            self.assertFalse(scores["structurally_bad"], scores)


if __name__ == "__main__":
    unittest.main()
