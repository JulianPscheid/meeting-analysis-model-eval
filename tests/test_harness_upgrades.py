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
