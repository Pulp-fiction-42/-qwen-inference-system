#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 1 ]; then
  echo "Usage: bash benckmark/bench.sh [raw-log-path-relative-to-project-root]" >&2
  exit 1
fi

# An explicit input log skips inference and computes metrics from that run.
raw_log=${1:-benckmark/bench_raw.txt}
if [[ "$raw_log" -ef benckmark/profile.txt || "$raw_log" -ef benckmark/bench.txt ]]; then
  echo "bench.sh: raw log must not be one of the output reports" >&2
  exit 1
fi

# Clear previous reports before running or validating a new log.
: > benckmark/profile.txt
: > benckmark/bench.txt

if [ "$#" -eq 0 ]; then
  make -j4 repl

  weights=/home/bgi-chao/wdr/main/model.bin
  vocab=/home/bgi-chao/llama-deploy/models/qwen2.5-1.5b-instruct-q8_0.gguf

  printf '%s\n' \
    'hello, world' \
    'what is CUDA?' \
    'summarize your last answer in one sentence' \
    'what is a GPU kernel?' \
    'explain threads, blocks, and grids briefly' \
    'what is shared memory?' \
    'how does memory coalescing work?' \
    'what is a warp?' \
    'what is the difference between CPU and GPU?' \
    'name three common CUDA libraries' \
    'what is quantization in LLM inference?' \
    'summarize our conversation in one sentence' \
    '/exit' |
    ./build/qwen_repl "$weights" "$vocab" \
      --max-new-tokens 320 --context-limit 2048 \
      >/dev/null 2>"$raw_log"
fi

if [[ ! -f "$raw_log" || ! -r "$raw_log" || ! -s "$raw_log" ]]; then
  echo "bench.sh: raw log is missing, unreadable, or empty: $raw_log" >&2
  exit 1
fi

# One run, one raw log, two reports computed from it:
#   profile.txt: project call hierarchy; Prefill/Decode are per-token latencies.
#   bench.txt:   TTFT, prompt throughput, average ITL, generation throughput.
# TTFT = tokenizer encode + runstate init + prefill (first-token sampling excluded, approximation).
# ITL  = decode time / generated tokens (decode time includes sampling, not tokenizer decode).
awk -v profile_out=benckmark/profile.txt -v bench_out=benckmark/bench.txt '
function fail(message) {
    print "bench.sh: line " NR ", turn " n ": " message > "/dev/stderr"
    failed = 1
    exit 1
}

function number(value) {
    return value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/ &&
           sprintf("%g", value + 0) !~ /[iI][nN][fF]|[nN][aA][nN]/
}

function save(key, value, unit, is_count) {
    if (!active) fail("field outside a turn: " key)
    if (key in v) fail("duplicate field: " key)
    if (!number(value)) fail("invalid or negative value for " key ": " value)
    if (is_count && value !~ /^[0-9]+$/) fail("non-integer token count: " key)
    if (!is_count && unit != "s") fail("expected seconds for " key)
    v[key] = value + 0
}

# Failed turns do not print Generated tokens, and the REPL still exits with 0.
/^生成失败:|^启动或运行失败:|^读取输入失败/ { fail($0) }
/^Session initialization time:/ {
    if (session_seen || n) fail("unexpected session initialization")
    if (!number($4) || $5 != "s") fail("invalid session initialization time")
    session_seen = 1
    profile = sprintf("Session initialization: %.6f s\n", $4)
}
/^  Tokenizer Encode:/ {
    if (active) fail("previous turn is incomplete")
    if (!session_seen) fail("missing session initialization")
    delete v
    stop_reason = ""
    n++
    active = 1
    save("enc", $3, $4, 0)
}
/^  Prefill tokens:/          { save("ptok", $3, "", 1) }
/^  Runstate initialization:/ { save("init", $3, $4, 0) }
/^    Prefill:/               { save("pre", $2, $3, 0) }
/^    Decode:/                { save("dec", $2, $3, 0) }
/^  Generate:/               { save("gen", $2, $3, 0) }
/^  Tokenizer Decode:/        { save("tdec", $3, $4, 0) }
/^Turn:/                     { save("turn", $2, $3, 0) }
/^  Stop reason:/ {
    if (!active || stop_reason != "") fail("unexpected stop reason")
    if ($3 !~ /^(eos|max_new_tokens|context_limit)$/) fail("invalid stop reason")
    stop_reason = $3
}
/^  Generated tokens:/ { save("gtok", $3, "", 1); report(); active = 0 }

function ratio(numerator, denominator, precision, unit) {
    if (denominator <= 0) return "N/A"
    return sprintf("%.*f %s", precision, numerator / denominator, unit)
}

function report(   keys, count, i, pre_present, dec_present, status, ttft, pre_ms, dec_ms, pp_tps, tg_tps) {
    count = split("enc ptok init gen tdec turn gtok", keys, " ")
    for (i = 1; i <= count; i++) {
        if (!(keys[i] in v)) fail("missing field: " keys[i])
    }
    if (v["ptok"] <= 0) fail("prefill token count must be positive")
    pre_present = ("pre" in v)
    dec_present = ("dec" in v)
    if (pre_present != dec_present) fail("missing Prefill or Decode time")
    if (!pre_present && v["gtok"] > 0) fail("output tokens without Prefill/Decode times")
    if (!pre_present && stop_reason == "eos") fail("EOS without Prefill/Decode times")

    # Older logs have no stop reason. Do not guess which limit stopped them.
    status = ""
    if (stop_reason == "context_limit") {
        if (v["gtok"] == 0) status = "context limit reached; no generation"
        else status = "context limit reached; output truncated"
    } else if (!pre_present) {
        status = "generation not executed (zero budget or full context)"
    } else if (v["gtok"] == 0) {
        status = "no output tokens"
    }

    ttft = "N/A"
    pre_ms = dec_ms = pp_tps = tg_tps = "N/A"
    if (pre_present) {
        pre_ms = ratio(v["pre"] * 1000, v["ptok"], 3, "ms/token")
        dec_ms = ratio(v["dec"] * 1000, v["gtok"], 3, "ms/token")
        pp_tps = ratio(v["ptok"], v["pre"], 2, "tokens/s")
        tg_tps = ratio(v["gtok"], v["dec"], 2, "tokens/s")
        if (v["gtok"] > 0) ttft = sprintf("%.2f ms", (v["enc"] + v["init"] + v["pre"]) * 1000)
    }

    profile = profile sprintf("--- Turn %d ---\n", n)
    if (status != "") profile = profile "  Status: " status "\n"
    profile = profile sprintf("Turn: %.6f s\n", v["turn"])
    profile = profile sprintf("  Tokenizer Encode: %.6f s\n", v["enc"])
    profile = profile sprintf("  Runstate initialization: %.6f s\n", v["init"])
    profile = profile sprintf("  Generate: %.6f s\n", v["gen"])
    profile = profile "    Prefill: " pre_ms "\n"
    profile = profile "    Decode: " dec_ms "\n"
    profile = profile sprintf("  Tokenizer Decode: %.6f s\n", v["tdec"])

    bench = bench sprintf("--- Turn %d ---\n", n)
    if (status != "") bench = bench "  Status: " status "\n"
    bench = bench "Prompt Processing:\n  TTFT: " ttft "\n"
    bench = bench sprintf("  Throughput: %s (%d tokens)\n", pp_tps, v["ptok"])
    bench = bench "Token Generation:\n  Average ITL: " (dec_present ? ratio(v["dec"] * 1000, v["gtok"], 2, "ms/token") : "N/A") "\n"
    bench = bench sprintf("  Throughput: %s (%d tokens)\n", tg_tps, v["gtok"])
}

END {
    if (failed) exit 1
    if (active) fail("last turn is incomplete")
    if (n == 0) fail("no completed turns")
    # Publish only after the entire log has passed validation.
    printf "%s", profile > profile_out
    printf "%s", bench > bench_out
}
' "$raw_log"

cat benckmark/profile.txt
echo
cat benckmark/bench.txt
