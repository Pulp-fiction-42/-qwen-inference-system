"""运行真实 REPL：多轮记忆、/clear、/exit、EOF、长度和错误后的继续输入。"""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("weights")
parser.add_argument("vocab")
parser.add_argument("--output-dir", type=Path, default=Path("test_kernel/results/session_integration"))
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
base = ["./build/qwen_repl", args.weights, args.vocab]
results = {}


def run(name, command, text):
    print("START", name, flush=True)
    process = subprocess.run(command, input=text, text=True, capture_output=True, timeout=300)
    (args.output_dir / (name + ".txt")).write_text(process.stdout)
    (args.output_dir / (name + ".stderr.txt")).write_text(process.stderr)
    results[name] = {"exit_code": process.returncode}
    return process


multi = run("repl_multi_turn", base,
            "我叫李三，我最喜欢的颜色是蓝色，请记住。\n"
            "我们刚才聊到了什么？请简短回答。\n"
            "我叫什么名字？我最喜欢什么颜色？\n/clear\n/exit\n")
answers = multi.stdout.split(">>> ")
results["repl_multi_turn"]["passed"] = (
    multi.returncode == 0 and len(answers) >= 4
    and "李三" in answers[3] and "蓝色" in answers[3]
    and "历史已清空" in multi.stdout and not multi.stderr
)
limited = run("repl_length_eof", base + ["--max-new-tokens", "1"], "你好，请介绍一下自己。\n")
results["repl_length_eof"]["passed"] = (
    limited.returncode == 0 and "本轮生成长度上限" in limited.stdout and not limited.stderr
)
zero = run("repl_zero", base + ["--max-new-tokens", "0"], "你好\n/exit\n")
results["repl_zero"]["passed"] = zero.returncode == 0 and "本轮生成长度上限" in zero.stdout
# 34-token prompt 剩余一格；溢出请求失败后，/clear 后仍可正常生成。
context = run("repl_context", base + ["--context-limit", "35"],
              "你好，请介绍一下自己。\n" + "1" * 3000 + "\n/clear\n你好，请介绍一下自己。\n/exit\n")
results["repl_context"]["passed"] = (
    context.returncode == 0 and context.stdout.count("已达到上下文上限") == 2
    and "超出上下文上限" in context.stderr
)
invalid = run("repl_invalid_args", base + ["--max-new-tokens", "-1"], "")
results["repl_invalid_args"]["passed"] = invalid.returncode != 0 and "非负整数" in invalid.stderr
missing = run("repl_missing_weights", [base[0], "/tmp/nonexistent-qwen-weights.bin", args.vocab], "")
results["repl_missing_weights"]["passed"] = missing.returncode != 0 and "无法打开权重文件" in missing.stderr
empty = args.output_dir / "truncated_weights.bin"
empty.write_bytes(b"1234")
truncated = run("repl_truncated_weights", [base[0], str(empty), args.vocab], "")
results["repl_truncated_weights"]["passed"] = truncated.returncode != 0 and "header 不完整" in truncated.stderr
(args.output_dir / "repl_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
raise SystemExit(0 if all(item["passed"] for item in results.values()) else 1)
