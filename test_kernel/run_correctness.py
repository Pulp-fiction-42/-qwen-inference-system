"""顺序运行原仓库的三项测试，保留日志、退出码和生成 token diff。"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "test_kernel/results")
    args = parser.parse_args()
    model_path = str(args.model_path.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "8"
    environment["MKL_NUM_THREADS"] = "8"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    results = {}

    def hashes():
        files = list((ROOT / "ops").glob("*"))
        files += list((ROOT / "src/model").rglob("*"))
        return {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files) if path.is_file()
        }

    before = hashes()

    def save():
        (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

    def run(name, command, stdout_name, stderr_name, timeout, stdin_name=None):
        print(f"START {name}", flush=True)
        started = time.monotonic()
        stdin = (output / stdin_name).open("r") if stdin_name else subprocess.DEVNULL
        try:
            with (output / stdout_name).open("w") as stdout, (output / stderr_name).open("w") as stderr:
                try:
                    result = subprocess.run(
                        command, cwd=ROOT, env=environment, stdin=stdin,
                        stdout=stdout, stderr=stderr, timeout=timeout,
                    )
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            results[name] = {
                "exit_code": code, "seconds": round(time.monotonic() - started, 3),
                "command": command, "stdout": stdout_name, "stderr": stderr_name,
            }
            save()
            print(f"END {name}: exit={code}, seconds={results[name]['seconds']}", flush=True)
            return code
        finally:
            if stdin_name:
                stdin.close()

    run("environment", [
        sys.executable, "-c",
        "import torch, transformers; print('torch', torch.__version__); "
        "print('transformers', transformers.__version__); "
        "print('CUDA available', torch.cuda.is_available()); "
        "print('GPU', torch.cuda.get_device_name(0)); "
        "print('matmul precision', torch.get_float32_matmul_precision())",
    ], "environment.txt", "environment.stderr.txt", 60)

    weight_status = run("weights", [
        sys.executable, str(ROOT / "test_kernel/verify_weights.py"), model_path,
    ], "weights.json", "weights.stderr.txt", 300)
    if weight_status != 0:
        print("权重未通过一致性检查，停止数值对照。", flush=True)
        return 1

    device_status = run("single_forward_dump", [
        str(ROOT / "test_kernel/test_single_forward/test_single_forward"),
    ], "device_state.txt", "single_forward.stderr.txt", 180)
    if device_status == 0:
        run("single_forward_compare", [
            sys.executable, str(ROOT / "test_kernel/test_single_forward/inspect_forward.py"),
            model_path, str(output / "device_state.txt"),
            "--token-id", "123", "--max-seq-len", "2048",
        ], "single_forward_compare.txt", "single_forward_compare.stderr.txt", 300)

    reference_status = run("generation_reference", [
        sys.executable, str(ROOT / "test_kernel/test_generation/test_generation.py"),
        model_path, "-prompt", "你好，请介绍一下自己。",
        "-out", str(output / "reference.txt"),
    ], "prompt.txt", "generation_reference.stderr.txt", 600)
    if reference_status == 0:
        engine_status = run("generation_engine", [str(ROOT / "engine")],
                            "engine.txt", "generation_engine.stderr.txt", 1200, "prompt.txt")
        reference = (output / "reference.txt").read_text().splitlines(keepends=True)
        actual = (output / "engine.txt").read_text().splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(reference, actual, fromfile="reference.txt", tofile="engine.txt"))
        (output / "generation.diff").write_text(diff)
        results["generation_compare"] = {
            "passed": engine_status == 0 and reference == actual,
            "reference_tokens": sum(line.strip() != "DONE" for line in reference),
            "engine_tokens": sum(line.strip() != "DONE" for line in actual),
            "first_different_line": next(
                (i + 1 for i in range(max(len(reference), len(actual)))
                 if reference[i:i+1] != actual[i:i+1]), None),
        }
        save()

    run("multi_turn", [
        sys.executable, str(ROOT / "test_kernel/test_multi_turn/test_multi_turn.py"), model_path,
    ], "multi_turn.txt", "multi_turn.stderr.txt", 1200)
    results["production_sources_unchanged"] = before == hashes()
    results["multi_turn_note"] = "原测试没有自动数值判据；使用 README 描述的姓名/颜色三轮场景。"
    save()
    passed = (
        results.get("single_forward_compare", {}).get("exit_code") == 0
        and results.get("generation_compare", {}).get("passed", False)
        and results["multi_turn"]["exit_code"] == 0
        and results["production_sources_unchanged"]
    )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
