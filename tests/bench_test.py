"""Check benchmark parsing and context boundaries without running inference."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SESSION = "Session initialization time: 0.100000 s\n"


def turn(tokens=3, reason=None, phases=True):
    text = (
        "  Tokenizer Encode: 0.010000 s\n"
        "  Prefill tokens: 10\n"
        "  Runstate initialization: 0.002000 s\n"
    )
    if phases:
        text += "    Prefill: 2.000000 s\n    Decode: 0.600000 s\n"
    text += (
        "  Generate: 2.601000 s\n"
        "  Tokenizer Decode: 0.000010 s\n"
        "Turn: 2.614000 s\n"
    )
    if reason is not None:
        text += f"  Stop reason: {reason}\n"
    return text + f"  Generated tokens: {tokens}\n"


class BenchTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="bench-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.bench = self.root / "benckmark"
        self.bench.mkdir()
        shutil.copyfile(PROJECT / "benckmark/bench.sh", self.bench / "bench.sh")

    def seed_reports(self):
        for name in ("profile.txt", "bench.txt"):
            (self.bench / name).write_text("previous report\n")

    def parse(self, raw):
        self.seed_reports()
        (self.bench / "input.txt").write_text(raw)
        return subprocess.run(
            ["bash", "bench.sh", "benckmark/input.txt"],
            cwd=self.bench, text=True, capture_output=True, timeout=10,
        )

    def reject(self, raw, message):
        result = self.parse(raw)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual((self.bench / "profile.txt").read_text(), "")
        self.assertEqual((self.bench / "bench.txt").read_text(), "")

    def test_normal_metrics_and_legacy_logs(self):
        for reason in (None, "eos", "max_new_tokens"):
            with self.subTest(reason=reason):
                result = self.parse(SESSION + turn(reason=reason))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((self.bench / "bench.txt").read_text(), (
                    "--- Turn 1 ---\nPrompt Processing:\n"
                    "  TTFT: 2012.00 ms\n"
                    "  Throughput: 5.00 tokens/s (10 tokens)\n"
                    "Token Generation:\n"
                    "  Average ITL: 200.00 ms/token\n"
                    "  Throughput: 5.00 tokens/s (3 tokens)\n"
                ))
                profile = (self.bench / "profile.txt").read_text()
                self.assertIn("    Prefill: 200.000 ms/token", profile)
                self.assertIn("    Decode: 200.000 ms/token", profile)
                self.assertNotIn("tokens:", profile)

    def test_zero_output_and_zero_timers(self):
        result = self.parse(SESSION + turn(tokens=0, reason="eos"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TTFT: N/A", result.stdout)
        self.assertIn("Average ITL: N/A", result.stdout)
        self.assertIn("Prefill: 200.000 ms/token", result.stdout)
        self.assertIn("Throughput: 0.00 tokens/s (0 tokens)", result.stdout)
        raw = (SESSION + turn()).replace("Prefill: 2.000000", "Prefill: 0.000000")
        raw = raw.replace("Decode: 0.600000", "Decode: 0.000000")
        result = self.parse(raw)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("Throughput: N/A"), 2)

    def test_full_context_and_zero_generation_budget(self):
        for reason in ("context_limit", "max_new_tokens", None):
            with self.subTest(reason=reason):
                raw = SESSION + turn(tokens=0, reason=reason, phases=False)
                raw = raw.replace("Prefill tokens: 10", "Prefill tokens: 2048")
                result = self.parse(raw)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Prefill: N/A", result.stdout)
                self.assertIn("Decode: N/A", result.stdout)
                self.assertIn("TTFT: N/A", result.stdout)
                self.assertIn("Average ITL: N/A", result.stdout)
                if reason == "context_limit":
                    self.assertIn("context limit reached; no generation", result.stdout)

    def test_context_limit_during_generation(self):
        result = self.parse(SESSION + turn(tokens=1, reason="context_limit"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("context limit reached; output truncated"), 2)
        self.assertIn("Average ITL: 600.00 ms/token", result.stdout)

    def test_context_overflow_and_generation_failure(self):
        for message in (
            "完整历史的 token 数超出上下文上限，请 /clear 后重试",
            "injected forward failure",
        ):
            with self.subTest(message=message):
                raw = (SESSION + turn() + "  Tokenizer Encode: 0.001000 s\n"
                       "  Prefill tokens: 2049\n生成失败: " + message + "\n")
                self.reject(raw, message)

    def test_incomplete_turn_cannot_reuse_previous_values(self):
        for prefix in (
            "  Prefill tokens:", "  Runstate initialization:", "    Prefill:",
            "    Decode:", "  Generate:", "  Tokenizer Decode:", "Turn:",
            "  Generated tokens:",
        ):
            with self.subTest(prefix=prefix):
                incomplete = "".join(line + "\n" for line in turn().splitlines()
                                     if not line.startswith(prefix))
                self.reject(SESSION + turn() + incomplete, "bench.sh:")

    def test_invalid_fields_and_duplicate_values(self):
        for original, replacement in (
            ("0.010000", "-0.010000"), ("0.010000", "nan"),
            ("0.010000", "1e999"), ("0.010000 s", "0.010000 ms"),
            ("tokens: 10", "tokens: 0"), ("tokens: 10", "tokens: 1.5"),
            ("tokens: 3", "tokens: -1"),
            ("  Generate: 2.601000 s\n", "  Generate: 2.601000 s\n" * 2),
        ):
            with self.subTest(replacement=replacement):
                self.reject(SESSION + turn().replace(original, replacement), "bench.sh:")

    def test_empty_unrelated_and_incomplete_logs(self):
        for raw in ("", "unrelated text\n", SESSION, SESSION + "  Tokenizer Encode: 0.1 s\n"):
            with self.subTest(raw=raw):
                self.reject(raw, "bench.sh:")
        self.reject(turn(), "missing session initialization")

    def test_missing_file_and_report_input_collision(self):
        self.seed_reports()
        result = subprocess.run(
            ["bash", "bench.sh", "benckmark/missing.txt"],
            cwd=self.bench, text=True, capture_output=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.bench / "bench.txt").read_text(), "")
        self.seed_reports()
        result = subprocess.run(
            ["bash", "bench.sh", "benckmark/bench.txt"],
            cwd=self.bench, text=True, capture_output=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("raw log must not", result.stderr)
        self.assertEqual((self.bench / "bench.txt").read_text(), "previous report\n")

    def test_failed_inference_does_not_publish_reports(self):
        # Substitute the process, not CUDA/model execution.
        tools = self.root / "tools"
        tools.mkdir()
        make = tools / "make"
        make.write_text("#!/bin/sh\nexit 0\n")
        make.chmod(0o755)
        build = self.root / "build"
        build.mkdir()
        repl = build / "qwen_repl"
        repl.write_text("#!/bin/sh\ncat >/dev/null\nexit 7\n")
        repl.chmod(0o755)
        self.seed_reports()
        env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"])
        result = subprocess.run(["bash", "bench.sh"], cwd=self.bench,
                                env=env, text=True, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.bench / "profile.txt").read_text(), "")
        self.assertEqual((self.bench / "bench.txt").read_text(), "")


if __name__ == "__main__":
    unittest.main()
