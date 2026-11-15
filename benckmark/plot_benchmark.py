#!/usr/bin/env python3
"""Parse bench.sh reports and write two standalone, offline interactive charts.

Examples (run from the project root):
  python3 benckmark/plot_benchmark.py benckmark/v1.txt
  python3 benckmark/plot_benchmark.py benckmark/profile.txt benckmark/bench.txt
"""

import argparse
import html
import json
import math
from pathlib import Path
import re
import sys


METRICS = {
    "prefill": {"label": "Prefill", "unit": "ms/token", "color": "#5096f5"},
    "decode": {"label": "Decode", "unit": "ms/token", "color": "#61bd7c"},
    "ttft": {"label": "TTFT", "unit": "ms", "color": "#5096f5"},
    "prompt_throughput": {
        "label": "Prompt processing throughput", "unit": "tokens/s", "color": "#61bd7c",
        "axis": "right",
    },
    "itl": {
        "label": "Average ITL", "unit": "ms/token", "color": "#db8b2f", "dash": True,
    },
    "generation_throughput": {
        "label": "Token Generation throughput", "unit": "tokens/s", "color": "#9971ce",
        "axis": "right", "dash": True,
    },
}
TURN = re.compile(r"---\s*Turn\s+(\d+)\s*---", re.IGNORECASE)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def parse_value(text, unit):
    """Preserve unavailable data as None; never turn it into zero."""
    if text.upper() == "N/A":
        return None
    match = re.fullmatch(rf"({NUMBER})\s+{re.escape(unit)}", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"expected a nonnegative number in {unit} or N/A, got {text!r}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError("metric must be finite")
    return value


def save_value(turn, key, value):
    if key in turn and turn[key] != value:
        raise ValueError(f"conflicting values for {key}; provide reports from one run only")
    turn[key] = value


def parse_reports(paths):
    """Join profile and benchmark sections by turn ID, including across files."""
    turns = {}
    for path in paths:
        current = None
        section = None
        text = ANSI.sub("", Path(path).read_text(encoding="utf-8-sig"))
        for line_number, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            match = TURN.fullmatch(line)
            if match:
                turn_id = int(match.group(1))
                if turn_id < 1:
                    raise ValueError(f"{path}:{line_number}: turn IDs must be positive")
                current = turns.setdefault(turn_id, {"turn": turn_id})
                section = None
                continue
            if line.lower() in ("prompt processing:", "token generation:"):
                section = "prompt" if line.lower().startswith("prompt") else "generation"
                continue
            key = None
            payload = None
            for label, metric in (("Prefill:", "prefill"), ("Decode:", "decode"),
                                  ("TTFT:", "ttft"), ("Average ITL:", "itl")):
                if line.lower().startswith(label.lower()):
                    key = metric
                    payload = line[len(label):].strip()
                    break
            if line.lower().startswith("throughput:"):
                if section is None:
                    raise ValueError(f"{path}:{line_number}: Throughput needs a section heading")
                key = "prompt_throughput" if section == "prompt" else "generation_throughput"
                payload = line.split(":", 1)[1].strip()
            if key is None:
                continue
            try:
                if current is None:
                    raise ValueError("metric appears before a --- Turn N --- heading")
                if key.endswith("throughput"):
                    count = re.search(r"\s*\((\d+)\s+tokens\)\s*$", payload, re.IGNORECASE)
                    if count:
                        save_value(current, key + "_tokens", int(count.group(1)))
                        payload = payload[:count.start()].strip()
                save_value(current, key, parse_value(payload, METRICS[key]["unit"]))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not turns:
        raise ValueError("no turns found; input must be a bench.sh report, not the raw inference log")
    for turn_id in sorted(turns):
        missing = [key for key in METRICS if key not in turns[turn_id]]
        if missing:
            raise ValueError(
                f"Turn {turn_id}: missing {', '.join(missing)}. "
                "Provide a combined report like v1.txt, or both profile.txt and bench.txt. "
                "Unavailable metrics must be explicitly recorded as N/A."
            )
    return [turns[turn_id] for turn_id in sorted(turns)]


def write_charts(turns, paths, output_dir, prefix):
    template = Path(__file__).with_name("benchmark_chart.html").read_text(encoding="utf-8")
    definitions = (
        ("latency", "Prefill / Decode latency", ["prefill", "decode"]),
        ("metrics", "Prompt processing / Token Generation", [
            "ttft", "prompt_throughput", "itl", "generation_throughput",
        ]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for kind, title, keys in definitions:
        data = {
            "kind": kind, "title": title, "sources": [str(path) for path in paths],
            "turns": turns, "series": [dict(METRICS[key], key=key) for key in keys],
        }
        # Prevent input paths from closing the JSON script element.
        payload = json.dumps(data, ensure_ascii=True, allow_nan=False)
        payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
        document = template.replace("__CHART_DATA__", payload).replace("__CHART_TITLE__", html.escape(title))
        destination = output_dir / f"{prefix}_{kind}.html"
        if destination.resolve() in [Path(path).resolve() for path in paths]:
            raise ValueError(f"output would overwrite input: {destination}")
        destination.write_text(document, encoding="utf-8")
        outputs.append(destination)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", type=Path, nargs="+", help="combined report, or profile.txt and bench.txt")
    parser.add_argument("--output-dir", type=Path, help="default: charts/ beside the first input")
    parser.add_argument("--prefix", help="output filename prefix; default: first input stem")
    args = parser.parse_args()
    prefix = args.prefix if args.prefix is not None else args.inputs[0].stem
    if not prefix or prefix in (".", "..") or "/" in prefix or "\\" in prefix:
        parser.error("--prefix must be a simple filename prefix")
    try:
        turns = parse_reports(args.inputs)
        output_dir = args.output_dir if args.output_dir is not None else args.inputs[0].parent / "charts"
        outputs = write_charts(turns, args.inputs, output_dir, prefix)
    except (OSError, ValueError) as error:
        parser.exit(1, f"plot_benchmark.py: {error}\n")
    print(f"Parsed {len(turns)} turns. Open these offline charts in your browser:")
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()
