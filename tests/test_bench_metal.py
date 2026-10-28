"""bench.sh refuses CPU fallback but accepts every real Metal log shape; invented logs only."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

BENCH = Path(__file__).resolve().parents[1] / "bench.sh"


def require_metal(log: str) -> int:
    function = re.search(r"^require_metal\(\) \{.*?^\}", BENCH.read_text(), re.S | re.M).group(0)
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as handle:
        handle.write(log)
    script = f'{function}\nrequire_metal "{handle.name}"'
    try:
        return subprocess.run(["bash", "-c", script], capture_output=True).returncode
    finally:
        Path(handle.name).unlink()


class BenchMetal(unittest.TestCase):
    def test_accepts_metal_device_lines(self):
        for log in ("  MTL0: Apple M1\n", "- MTL0 : Apple M2 Pro\n",
                    "0.00.124.574 I cmn common_param:   - MTL0    : Apple M2 Pro (12124 MiB, 12123 MiB free)\n",
                    "ggml_metal_init: GPU name:   MTL0 (Apple M2 Pro)\n"):
            with self.subTest(log=log):
                self.assertEqual(require_metal(log), 0)

    def test_refuses_cpu_and_failure_logs(self):
        for log in ("  - CPU     : Apple M2 Pro\n  - BLAS    : Accelerate\n",
                    "0.00.124.574 I cmn common_param:   - CPU     : Apple M2 Pro\n",
                    "note: skipped MTL0: Apple M1 because CPU fallback\n",
                    "GPU name: (null)\n  MTL0: Apple M1\n",
                    "failed to create command queue\n  MTL0: Apple M1\n"):
            with self.subTest(log=log):
                self.assertNotEqual(require_metal(log), 0)


if __name__ == "__main__":
    unittest.main()
