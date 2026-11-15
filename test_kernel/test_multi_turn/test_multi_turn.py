"""向同一个 engine 依次发送三轮中文输入，在终端显示问答。"""

import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="本地 Hugging Face 模型目录")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    # 使用原测试 README 描述的姓名/颜色记忆场景。
    prompts = [
        "我叫李三，我最喜欢的颜色是蓝色，请记住。",
        "我们刚才聊到了什么？请简短回答。",
        "我叫什么名字？我最喜欢什么颜色？",
    ]
    messages = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    project_dir = Path(__file__).resolve().parents[2]
    engine = subprocess.Popen(
        [str(project_dir / "engine")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    try:
        for prompt in prompts:
            print(f"\nYou: {prompt}", flush=True)
            messages.append({"role": "user", "content": prompt})
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            token_ids = encoded["input_ids"]
            if not token_ids or len(token_ids) > 2048:
                parser.error("套用模板后的每轮输入长度必须为 1 到 2048")

            # 每轮发送完整历史，与 repl.py 的请求格式一致。
            ids_text = " ".join(str(token_id) for token_id in token_ids)
            engine.stdin.write(f"{len(token_ids)} {ids_text}\n")
            engine.stdin.flush()

            print("Assistant: ", end="", flush=True)
            generated_ids = []
            while True:
                line = engine.stdout.readline()
                if not line:
                    raise RuntimeError("engine 在本轮结束前退出")
                if line.strip() == "DONE":
                    break
                generated_ids.append(int(line))

            # 收齐本轮 ID 后解码，避免中文字符被拆开。
            reply = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            print(reply, flush=True)
            messages.append({"role": "assistant", "content": reply})
    finally:
        if engine.poll() is None:
            engine.terminate()
        engine.wait()
        engine.stdin.close()
        engine.stdout.close()


if __name__ == "__main__":
    main()
