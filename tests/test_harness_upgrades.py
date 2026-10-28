"""Harness regressions using public, invented data and fake services only."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev


class SamplerParity(unittest.TestCase):
    def test_request_neutralizes_extra_stages_and_keys_them(self):
        from test_contracts import Requests
        fixture = Requests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        (fixture.root / "server").write_bytes(b"fake")
        (fixture.root / "model.gguf").write_bytes(b"fake")
        args = ev.build_parser().parse_args(fixture.argv())
        sent = []

        def reply(url, body, timeout):
            sent.append(body)
            settings = {key: value for key, value in body.items() if key not in
                        {"messages", "grammar", "stream", "verbose", "cache_prompt", "max_tokens"}}
            return {"choices": [{"finish_reason": "stop", "message": {"content": "okay"}}],
                    "__verbose": {"generation_settings": {**settings, "server_extra": 42}}}

        with mock.patch.object(ev, "_host", return_value={}), \
             mock.patch.object(ev, "_command_text", return_value="version: fake"), \
             mock.patch.object(ev.subprocess, "Popen"), \
             mock.patch.object(ev, "_wait_for_completion"), \
             mock.patch.object(ev, "_assert_metal"), \
             mock.patch.object(ev, "_post_json", side_effect=reply):
            self.assertEqual(ev.command_run(args), 0)
        neutrals = {"repeat_penalty": 1.0, "dry_multiplier": 0, "xtc_probability": 0,
                    "typical_p": 1.0, "min_p": 0, "frequency_penalty": 0,
                    "dynatemp_range": 0, "mirostat": 0, "top_n_sigma": -1,
                    "adaptive_target": -1}
        self.assertTrue(all(sent[0].get(key) == value for key, value in neutrals.items()), sent[0])
        manifest = json.loads((fixture.root / "out/arm.manifest.json").read_text())
        records = [json.loads(line) for line in (fixture.root / "out/arm.runs.jsonl").read_text().splitlines()]
        self.assertEqual(records[0].get("generation_settings", {}).get("server_extra"), 42)
        self.assertEqual(manifest.get("generation_settings"), [r["generation_settings"] for r in records])
        self.assertTrue(all(manifest["reuse_key"].get("sampler_settings", {}).get(k) == v
                            for k, v in neutrals.items()))

    def test_server_sampler_mismatch_is_refused(self):
        self.assertTrue(callable(getattr(ev, "_verify_sampler", None)), "runtime verification is required")
        with self.assertRaisesRegex(ValueError, "repeat_penalty"):
            ev._verify_sampler({"repeat_penalty": 1.1}, {"repeat_penalty": 1.0})
        with self.assertRaisesRegex(ValueError, "missing"):
            ev._verify_sampler({}, {"repeat_penalty": 1.0})
        ev._verify_sampler({"top_p": 0.80000001}, {"top_p": 0.8})


class MetalGate(unittest.TestCase):
    def test_device_listing_formats_and_cpu_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text("")
            for devices in ("MTL0: Apple M4", "  - MTL0 : Apple M4"):
                result = subprocess.CompletedProcess([], 0, devices, "")
                error = None
                with mock.patch.object(ev.subprocess, "run", return_value=result):
                    try:
                        ev._assert_metal(log, "fake-server")
                    except RuntimeError as caught:
                        error = str(caught)
                self.assertIsNone(error, devices)
            for devices in ("CPU: Apple M4", "BLAS: Apple Accelerate", "(none)"):
                result = subprocess.CompletedProcess([], 0, devices, "")
                with mock.patch.object(ev.subprocess, "run", return_value=result):
                    with self.assertRaisesRegex(RuntimeError, "refusing CPU/BLAS"):
                        ev._assert_metal(log, "fake-server")


class UsableTally(unittest.TestCase):
    def test_unusable_reasons_and_two_arm_table_are_additive(self):
        from test_review_regressions import ReviewRegressions
        scorer = ReviewRegressions()
        phrase = "one two three four five six seven eight nine ten eleven twelve "
        good = "The team has a plan and we will review the results with our colleagues."
        rows = [
            {"intent": "chat", "label": "candidate", "content": good},
            {"intent": "chat", "label": "candidate", "content": "..."},
            {"intent": "chat", "label": "candidate", "content": "Okay."},
            {"intent": "chat", "label": "candidate", "content": good, "finish": "length"},
            {"intent": "chat", "label": "incumbent", "content": phrase * 8},
            {"intent": "chat", "label": "incumbent", "content": "<think>plan</think>" + good},
            {"intent": "chat", "label": "incumbent", "content": good, "reasoning_chars": 30},
            {"intent": "summary", "label": "incumbent", "grammar": "on", "content": "{broken"},
            {"intent": "summary", "label": "incumbent", "grammar": "on", "content": json.dumps({
                "title": "Plan", "recap_markdown": good,
                "user_todos": [{"text": "Review the plan", "dueDate": None}]})},
        ]
        printed = io.StringIO()
        with contextlib.redirect_stderr(printed):
            output = scorer.score(rows, expect_no_todos=True)
        expected = [[], ["empty_visible_text"], ["near_empty_visible_text"], ["truncation"],
                    ["runaway_repetition"], ["reasoning_leakage"], ["reasoning_leakage"],
                    ["invalid_output"], ["unexpected_user_todos"]]
        for row, reasons in zip(output["runs"], expected):
            self.assertTrue(set(reasons) <= set(row.get("unusable_reasons", [])), row)
        self.assertEqual(sum(row.get("usable", False) for row in output["runs"]), 1)
        self.assertIn("candidate", printed.getvalue())
        self.assertIn("incumbent", printed.getvalue())
        self.assertIn("usable", printed.getvalue())
        group = output["grouped_summary"]["chat"][""]["candidate"]
        self.assertEqual((group.get("usable"), group.get("unusable")), (1, 3))

    def test_expected_language_and_headings_affect_usable_without_strict_flag(self):
        from test_review_regressions import ReviewRegressions
        scorer = ReviewRegressions()
        output = scorer.score([{"intent": "chat", "content":
            "The team has a plan and we will review the results with our colleagues."}], expect_language="de")
        row = output["runs"][0]
        self.assertFalse(row["structurally_bad"])
        self.assertIn("wrong_language", row.get("unusable_reasons", []))
        self.assertTrue(ev.check_output("chat", "## The plan and the next steps\nDas Team hat einen Plan und wir werden die Ergebnisse besprechen.",
                                       expect_language="de").get("untranslated_headings"))
