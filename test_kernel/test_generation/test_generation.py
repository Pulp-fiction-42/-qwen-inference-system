"""按 repl.py/run.cu 的顺序执行对话模板、逐 token PREFILL 和 Argmax GENERATION。

stdout 输出 engine 请求；-out 文件输出参考 token IDs 和 DONE。
用法：python test_kernel/test_generation/test_generation.py MODEL_PATH -prompt "你好" -out reference.txt
"""

import argparse
import sys

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM


MAX_SEQ_LEN = 2048  # 与 engine 的总长度上限一致。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="本地 Hugging Face 模型目录")
    parser.add_argument("-prompt", "--prompt", default="人工智能的未来是", help="初始提示词")
    parser.add_argument("-out", "--out", required=True, help="参考 token IDs 输出文件")
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("prompt 不能为空")

    # 与 C/CUDA 参考实现保持一致，权重和计算使用 float32。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = Qwen2ForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float32,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()

    # 与 repl.py 一致，在用户消息后添加 assistant 的起始标记。
    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    prompt_ids = inputs["input_ids"]  # [1, prompt_len]
    attention_mask = inputs["attention_mask"]  # [1, prompt_len]
    prompt_length = prompt_ids.shape[1]
    if prompt_length <= 0 or prompt_length > MAX_SEQ_LEN:
        parser.error("套用对话模板后的 prompt 长度必须为 1 到 2048")
    print(f"运行设备: {device}", file=sys.stderr, flush=True)

    # 与 run.cu 使用相同的停止条件，EOS 不写入输出。
    eos_token_ids = [151643, 151645]

    generated_tokens = []
    past_key_values = None
    position = 0
    with torch.no_grad():
        # PREFILL：每次输入一个 prompt token，逐步填充所有层的 KV Cache。
        while position < prompt_length:
            input_ids = prompt_ids[:, position:position + 1]  # [1, 1]
            prefix_mask = attention_mask[:, :position + 1]  # [1, position + 1]
            outputs = model(
                input_ids=input_ids,
                attention_mask=prefix_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            position += 1

        print(f"PREFILL 完成: {position} tokens", file=sys.stderr, flush=True)

        # GENERATION：从上次前向的 logits 选 token，再输入该 token。
        while position < MAX_SEQ_LEN:
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            token_id = next_token_id.item()
            if token_id in eos_token_ids:
                break

            generated_tokens.append(token_id)
            if position == MAX_SEQ_LEN - 1:
                break

            input_ids = next_token_id
            new_mask = torch.ones((1, 1), device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            position += 1

    with open(args.out, "w", encoding="utf-8") as output_file:
        for token_id in generated_tokens:
            output_file.write(f"{token_id}\n")
        output_file.write("DONE\n")

    # 参考结果保存成功后，stdout 只发送一条完整的 engine 请求。
    ids_text = " ".join(str(token_id) for token_id in prompt_ids.tolist()[0])
    print(f"{prompt_length} {ids_text}", flush=True)


if __name__ == "__main__":
    main()
