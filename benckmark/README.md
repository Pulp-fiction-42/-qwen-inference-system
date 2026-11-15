# Interactive benchmark charts

From the project root, run:

```bash
python3 benckmark/plot_benchmark.py benckmark/v1.txt
```

Open the two generated files in any modern browser:

- `benckmark/charts/v1_latency.html`: Prefill and Decode in ms/token.
- `benckmark/charts/v1_metrics.html`: TTFT, prompt processing throughput,
  Average ITL, and Token Generation throughput.

The script uses Python's standard library. The generated HTML files are
self-contained and work offline, with no Python server or chart package needed.
Keep `plot_benchmark.py` and `benchmark_chart.html` together when copying the tool.

For the separate reports produced by `bench.sh`:

```bash
python3 benckmark/plot_benchmark.py benckmark/profile.txt benckmark/bench.txt
```

Choose another destination and filename prefix with:

```bash
python3 benckmark/plot_benchmark.py benckmark/v1.txt --output-dir benckmark/charts --prefix experiment1
```

Existing chart files with the same prefix are replaced. Input reports are never
modified. Reports must come from the same run: turns are joined by turn number,
and conflicting values are rejected. Raw inference logs must first be converted
with `bash benckmark/bench.sh path/to/raw-log.txt`.

Hover or touch the plot for a grouped tooltip. Use **Inspect** to select a turn
with the keyboard. Uncheck legend entries to hide series. Exact values remain
available in the scrollable table above each plot.

The second chart uses latency on the left axis and throughput on the right.
Latency defaults to a logarithmic scale because TTFT and ITL have very different
magnitudes; **Latency scale** switches it to linear. Zero latency is shown only
in linear mode. TTFT uses ms, ITL uses ms/token, and throughput uses tokens/s.
Each tooltip includes its metric's unit and available token counts.

Values come directly from the reports without recomputation or normalization.
In `bench.sh`, TTFT approximates encode + runstate initialization + prefill,
excluding first-token sampling. N/A remains missing and creates a line gap;
missing fields, invalid numbers, and wrong units cause a readable error.

Run the parser and output checks with:

```bash
python3 -m unittest discover -s tests -p 'plot_benchmark_test.py'
```
