"""Verify report joins and unavailable metrics without running inference."""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "benckmark"))
from plot_benchmark import parse_reports, write_charts


PROFILE = """--- Turn 1 ---
    Prefill: 198.607 ms/token
    Decode: 199.596 ms/token
"""
BENCH = """--- Turn 1 ---
Prompt Processing:
  TTFT: 6360.71 ms
  Throughput: 5.04 tokens/s (32 tokens)
Token Generation:
  Average ITL: 199.60 ms/token
  Throughput: 5.01 tokens/s (9 tokens)
"""


class PlotBenchmarkTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="plot-benchmark-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def report(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_combined_and_split_reports_match(self):
        combined = self.report("combined.txt", PROFILE + BENCH)
        split = [self.report("profile.txt", PROFILE), self.report("bench.txt", BENCH)]
        turns = parse_reports([combined])
        self.assertEqual(turns, parse_reports(split))
        self.assertEqual(turns[0]["prefill"], 198.607)
        self.assertEqual(turns[0]["decode"], 199.596)
        self.assertEqual(turns[0]["ttft"], 6360.71)
        self.assertEqual(turns[0]["itl"], 199.60)
        self.assertEqual(turns[0]["prompt_throughput"], 5.04)
        self.assertEqual(turns[0]["generation_throughput"], 5.01)
        self.assertEqual(turns[0]["prompt_throughput_tokens"], 32)
        self.assertEqual(turns[0]["generation_throughput_tokens"], 9)

    def test_join_by_turn_id_and_sort_numerically(self):
        profile = PROFILE.replace("Turn 1", "Turn 12") + PROFILE
        bench = BENCH + BENCH.replace("Turn 1", "Turn 12").replace("6360.71", "252110.07")
        turns = parse_reports([self.report("turns.txt", profile + bench)])
        self.assertEqual([turn["turn"] for turn in turns], [1, 12])
        self.assertEqual(turns[1]["ttft"], 252110.07)

    def test_na_and_zero_remain_distinct(self):
        report = (PROFILE + BENCH).replace("199.596 ms/token", "N/A")
        report = report.replace("199.60 ms/token", "N/A").replace("5.01 tokens/s", "0.00 tokens/s")
        turns = parse_reports([self.report("missing.txt", report)])
        self.assertIsNone(turns[0]["decode"])
        self.assertIsNone(turns[0]["itl"])
        self.assertEqual(turns[0]["generation_throughput"], 0)

    def test_invalid_units_values_and_duplicates_fail(self):
        for bad in ("-1 ms/token", "nan ms/token", "1e999 ms/token", "1 s", "oops"):
            with self.subTest(bad=bad):
                path = self.report("invalid.txt", (PROFILE + BENCH).replace("198.607 ms/token", bad))
                with self.assertRaises(ValueError):
                    parse_reports([path])
        conflict = self.report("conflict.txt", PROFILE + BENCH + PROFILE.replace("198.607", "200"))
        with self.assertRaisesRegex(ValueError, "conflicting values"):
            parse_reports([conflict])

    def test_incomplete_report_and_missing_section_fail(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            parse_reports([self.report("partial.txt", PROFILE)])
        with self.assertRaisesRegex(ValueError, "section heading"):
            parse_reports([self.report("bad-section.txt", (PROFILE + BENCH).replace("Prompt Processing:", ""))])
        with self.assertRaisesRegex(ValueError, "no turns"):
            parse_reports([self.report("empty.txt", "")])

    def test_html_contains_exact_data_and_safe_paths(self):
        path = self.report("v1.txt", PROFILE + BENCH)
        turns = parse_reports([path])
        outputs = write_charts(turns, ["</script><script>alert(1)</script>"], self.root, "chart")
        self.assertEqual([path.name for path in outputs], ["chart_latency.html", "chart_metrics.html"])
        for output in outputs:
            document = output.read_text(encoding="utf-8")
            payload = re.search(r'<script id="chart-data" type="application/json">(.*?)</script>', document).group(1)
            self.assertEqual(json.loads(payload)["turns"], turns)
            self.assertNotIn("<script>", payload)
            self.assertNotIn("__CHART_DATA__", document)
            self.assertNotIn("<script src=", document)


if __name__ == "__main__":
    unittest.main()
