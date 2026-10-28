"""Harness regressions using public, invented data and fake services only."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
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

        def reply(url, body, *unused):
            sent.append(body)
            settings = {key: value for key, value in body.items() if key not in
                        {"messages", "grammar", "stream", "verbose", "cache_prompt", "max_tokens"}}
            return {"choices": [{"finish_reason": "stop", "message": {"content": "okay"}}],
                    "generation_settings": {**settings, "server_extra": 42}}

        with mock.patch.object(ev, "_host", return_value={}), \
             mock.patch.object(ev, "_command_text", return_value="version: fake"), \
             mock.patch.object(ev.subprocess, "Popen"), \
             mock.patch.object(ev, "_wait_for_completion"), \
             mock.patch.object(ev, "_assert_metal"), \
             mock.patch.object(ev, "_stream_completion", side_effect=reply):
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


class JudgeReliability(unittest.TestCase):
    def test_pinned_model(self):
        with mock.patch.object(ev.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
            ev._judge_once("public invented prompt")
        command = run.call_args.args[0]
        self.assertIn("-m", command)
        self.assertEqual(command[command.index("-m") + 1], "gpt-6.1-sol")

    def test_rate_limit_backoff(self):
        with mock.patch.object(ev, "_judge_once", side_effect=[RuntimeError("HTTP 429 rate limited"),
                               RuntimeError("HTTP 429 rate limited"), ({}, None, "{}")]) as judge, \
             mock.patch.object(ev.time, "sleep") as sleep:
            result = ev._judge_with_retry("invented prompt")
        self.assertIsNone(result[1])
        self.assertEqual(judge.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    def test_rate_limit_is_retained_when_cli_log_tail_has_no_status(self):
        result = subprocess.CompletedProcess([], 1, "HTTP 429: rate limited\n" + "cleanup log\n" * 50, "")
        with mock.patch.object(ev.subprocess, "run", return_value=result):
            with self.assertRaises(RuntimeError) as caught:
                ev._judge_once("invented prompt")
        self.assertIn("429", str(caught.exception))

    def test_zero_judgments_threshold_and_bounded_concurrency(self):
        from test_contracts import Requests
        fixture = Requests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        out = fixture.root / "judge.json"
        args = ev.build_parser().parse_args(["judge", "--request-file", str(fixture.request),
                                           "--out", str(out), str(fixture.results)])
        args.judge_model = "gpt-6.1-sol"
        args.judge_concurrency = 2
        args.max_judge_errors = 1
        fixture.results.write_text("\n".join(json.dumps(fixture.record(trial=i)) for i in range(4)))
        valid = {"unsupported_claims": [], "coverage": {"early": True, "middle": True, "late": True},
                 "length_verdict": "appropriate", "length_note": "Fits.", "structural_leaks": [], "todos": []}
        lock = threading.Lock()
        active = peak = calls = 0

        def judge(*unused):
            nonlocal active, peak, calls
            with lock:
                active += 1
                peak = max(peak, active)
                calls += 1
                number = calls
            time.sleep(0.02)
            with lock:
                active -= 1
            return (None, "bad JSON", "bad") if number == 1 else (valid, None, None)

        with mock.patch.object(ev, "_judge_with_retry", side_effect=judge), \
             mock.patch.object(ev, "_command_text", return_value="codex-cli test"):
            self.assertEqual(ev.command_judge(args), 0, "one tolerated error with successful judgments")
        result = json.loads(out.read_text())
        self.assertEqual(peak, 2)
        self.assertEqual([r["trial"] for r in result["runs"]], list(range(4)))
        self.assertEqual(result.get("judge_model"), args.judge_model)
        self.assertEqual(result.get("judge_cli_version"), "codex-cli test")
        args.max_judge_errors = 0
        with mock.patch.object(ev, "_judge_with_retry", return_value=(None, "bad", None)), \
             mock.patch.object(ev, "_command_text", return_value="codex-cli test"):
            self.assertEqual(ev.command_judge(args), 1)
        fixture.results.write_text("")
        with mock.patch.object(ev, "_command_text", return_value="codex-cli test"):
            self.assertEqual(ev.command_judge(args), 1, "zero judged trials must fail")


class RunSpeed(unittest.TestCase):
    def test_interrupt_stops_owned_server_before_waiting_for_threads(self):
        from test_contracts import Requests
        fixture = Requests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        args = ev.build_parser().parse_args(fixture.argv())
        observations = []
        with mock.patch.object(ev, "reuse_key", return_value={}), \
             mock.patch.object(ev, "_wait_for_completion"), mock.patch.object(ev, "_assert_metal"), \
             mock.patch.object(ev.subprocess, "Popen") as process, \
             mock.patch.object(ev.concurrent.futures, "ThreadPoolExecutor") as pool:
            pool.return_value.__enter__.return_value.map.side_effect = KeyboardInterrupt
            pool.return_value.__exit__.side_effect = lambda *unused: observations.append(process.return_value.terminate.called)
            with self.assertRaises(KeyboardInterrupt):
                ev.command_run(args)
        self.assertEqual(observations, [True], "interrupt must not wait for the request timeout")

    def test_stream_stops_runaway_and_preserves_partial_output_and_slot_settings(self):
        self.assertTrue(callable(getattr(ev, "_stream_completion", None)), "run must consume SSE incrementally")
        phrase = "one two three four five six seven eight nine ten eleven twelve "
        body = {"seed": 123, "id_slot": 0}
        events = [{"choices": [{"delta": {"content": phrase}, "finish_reason": None}]}] * 10
        events += [{"choices": [{"delta": {}, "finish_reason": "stop"}]}]
        raw = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"
        for threshold, finish in ((6, "runaway_stopped"), (0, "stop")):
            response = io.BytesIO(raw)
            def open_url(request, **kwargs):
                if isinstance(request, str):
                    return io.BytesIO(b'[{"id":0,"params":{"seed":123,"server_extra":42}}]')
                return response
            with mock.patch.object(ev.urllib.request, "urlopen", side_effect=open_url):
                result = ev._stream_completion("http://fake/v1/chat/completions", body, 10, threshold)
            self.assertTrue(response.closed)
            choice = result["choices"][0]
            self.assertEqual(choice["finish_reason"], finish)
            if threshold:
                self.assertLess(len(choice["message"]["content"]), len(phrase) * 10)
                self.assertEqual(result["generation_settings"]["server_extra"], 42)

    def test_stream_final_metadata_reasoning_usage_and_errors(self):
        self.assertTrue(callable(getattr(ev, "_stream_completion", None)), "run must preserve SSE metadata")
        events = [
            {"choices": [{"delta": {"reasoning_content": "thought"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "visible answer"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}],
             "__verbose": {"generation_settings": {"seed": 1}}},
            {"choices": [], "usage": {"completion_tokens": 9}, "timings": {"predicted_per_second": 4}},
        ]
        raw = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
        with mock.patch.object(ev.urllib.request, "urlopen", return_value=io.BytesIO(raw.encode())):
            result = ev._stream_completion("http://fake/v1/chat/completions", {}, 10, 6)
        self.assertEqual(result["choices"][0]["message"], {"content": "visible answer", "reasoning_content": "thought"})
        self.assertEqual(result["generation_settings"], {"seed": 1})
        self.assertEqual(result["usage"]["completion_tokens"], 9)
        self.assertEqual(result["timings"]["predicted_per_second"], 4)
        with mock.patch.object(ev.urllib.request, "urlopen", return_value=io.BytesIO(b'data: {"error":"bad request"}\n\n')):
            result = ev._stream_completion("http://fake/v1/chat/completions", {}, 10, 6)
        self.assertEqual(result["error"], "bad request")

    def test_parallel_runs_keep_order_seed_context_and_suppress_speed_claims(self):
        from test_contracts import Requests
        fixture = Requests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        (fixture.root / "server").write_bytes(b"fake")
        (fixture.root / "model.gguf").write_bytes(b"fake")
        args = ev.build_parser().parse_args(fixture.argv())
        self.assertEqual(getattr(args, "parallel", None), 1, "default remains serial")
        args.parallel = 2
        active = peak = 0
        lock = threading.Lock()

        def reply(url, body, *unused):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.04 if body["seed"] == args.seed else 0.01)
            with lock:
                active -= 1
            return {"choices": [{"finish_reason": "stop", "message": {"content": "A usable response"}}],
                    "generation_settings": ev._sampler_settings(args, body["seed"]),
                    "timings": {"predicted_per_second": 100}}

        with mock.patch.object(ev, "_host", return_value={}), \
             mock.patch.object(ev, "_command_text", return_value="version: fake"), \
             mock.patch.object(ev.subprocess, "Popen") as popen, \
             mock.patch.object(ev, "_wait_for_completion"), mock.patch.object(ev, "_assert_metal"), \
             mock.patch.object(ev, "_stream_completion", side_effect=reply):
            self.assertEqual(ev.command_run(args), 0)
        self.assertEqual(peak, 2)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-np") + 1], "2")
        self.assertEqual(command[command.index("-c") + 1], "8192")
        output = fixture.root / "out/arm.runs.jsonl"
        records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([r["seed"] for r in records], [args.seed, args.seed + 1] * 2)
        self.assertEqual([r["case_label"] for r in records], ["case one"] * 2 + ["case two"] * 2)
        self.assertTrue(all(r.get("parallel") == 2 for r in records))
        manifest = json.loads((fixture.root / "out/arm.manifest.json").read_text())
        self.assertEqual(manifest.get("parallel"), 2)
        self.assertEqual(manifest["reuse_key"].get("parallel"), 2)
        score = ev.build_parser().parse_args(["score", "--request-file", str(fixture.request), str(output)])
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            ev.command_score(score)
        result = json.loads(printed.getvalue())
        self.assertIsNone(result["summary"]["chat"]["case one"]["arm"]["median_predicted_per_second"])

    def test_stream_http_disconnect_reaches_fake_server(self):
        import http.server
        disconnected = threading.Event()
        phrase = "one two three four five six seven eight nine ten eleven twelve "

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                data = b'[{"id":0,"params":{"seed":123}}]'
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                frame = ("data: " + json.dumps({"choices": [{"delta": {"content": phrase}}]}) + "\n\n").encode()
                try:
                    for _ in range(100000):
                        self.wfile.write(frame)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    disconnected.set()

            def log_message(self, *args):
                pass

        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = ev._stream_completion(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                                               {"seed": 123, "id_slot": 0}, 5)
                self.assertEqual(result["choices"][0]["finish_reason"], "runaway_stopped")
                self.assertTrue(disconnected.wait(2), "server must observe the cancelled stream")
            finally:
                server.shutdown()
                thread.join()
