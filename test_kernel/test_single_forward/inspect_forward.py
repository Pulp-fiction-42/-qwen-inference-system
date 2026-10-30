"""比较 layers[0]、position=0 的 C/PyTorch 状态，Mean AE 阈值为 1e-5。

用法：python inspect_forward.py MODEL_PATH device_state.txt
输入每行：x[0]: -0.0305175781（也接受 state->x[0]: ...）。
索引使用 C 缓冲区原始下标；必须提供每个 state 的全部有效元素。
缺失元素时不计算 Mean AE，直接判定未通过；建议 C 使用 %.9g 保留 float32 精度。
position=0 的 score 索引为 h * max_seq_len；cache 只需 layer0/token0 的 kv_dim 个值。
"""

import argparse
import math
import re
from pathlib import Path


def read_states(path):
    """读取 name[index]: value，保留原始索引，忽略空行。"""
    states = {}
    pattern = re.compile(r"(?:state->)?(\w+)\[(\d+)\]:\s*(\S+)")
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        match = pattern.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"第 {line_number} 行格式错误：需要 name[index]: value")
        name, index_text, value_text = match.groups()
        index = int(index_text)
        value = float(value_text)
        values = states.setdefault(name, {})
        if index in values:
            raise ValueError(f"第 {line_number} 行重复：{name}[{index}]")
        values[index] = value
    if not states:
        raise ValueError("输入文件没有 state 数据")
    return states


def compare_states(reference, actual, max_seq_len, n_layers):
    """reference 为有效元素列表；actual 的索引对应 C 的实际内存布局。"""
    threshold = 1e-5
    all_passed = True
    for name in actual:
        if name not in reference:
            print(f"state->{name}: 未通过❌（未知 state）")
            all_passed = False

    for name, expected in reference.items():
        label = "state->" + name
        values = actual.get(name, {})
        errors = []
        present = 0
        skipped = 0
        invalid = False
        capacity = len(expected)
        is_score = name in ("attn_score", "softmax_attn_score")
        is_cache = name in ("key_cache", "value_cache")
        if is_score:
            capacity *= max_seq_len
        if is_cache:
            capacity *= max_seq_len * n_layers

        for index, value in values.items():
            if index < 0 or index >= capacity:
                invalid = True
                continue
            # position=0：每个头仅第 0 个 score 有效；cache 仅 layer0/token0 有效。
            if is_score and index % max_seq_len != 0:
                skipped += 1
                continue
            if is_cache and index >= len(expected):
                skipped += 1
                continue
            reference_index = index // max_seq_len if is_score else index
            present += 1
            expected_value = expected[reference_index]
            if not math.isfinite(value) or not math.isfinite(expected_value):
                invalid = True
                continue
            errors.append(abs(value - expected_value))

        count = len(errors)
        missing = len(expected) - present
        complete = count == len(expected) and count > 0 and not invalid
        mean_ae_text = "N/A"
        passed = False
        if complete:
            mean_ae = math.fsum(errors) / len(expected)
            mean_ae_text = f"{mean_ae:.8e}"
            passed = mean_ae < threshold
        status = "通过✅" if passed else "未通过❌"
        detail = f"有效索引覆盖={present}/{len(expected)}，忽略未写入位置={skipped}"
        if missing:
            detail += f"，缺少 {missing} 个有效元素，必须完整输出"
        if invalid:
            detail += "，含非有限值或越界索引"
        if not values:
            detail += "，缺少该 state"
        print(f"{label:28s} Mean AE={mean_ae_text} {status}（{detail}）")
        all_passed = all_passed and passed
    return all_passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="本地 Hugging Face 模型目录，包含 config.json 和权重")
    parser.add_argument("state_file", type=Path, help="test_forward.cu 输出的 name[index]: value 文本")
    parser.add_argument("--token-id", type=int, default=123, help="与 C 测试一致的 token，默认 123；position 固定为 0")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="C 缓冲区的 max_seq_len，默认 2048")
    args = parser.parse_args()
    model_path = Path(args.model_path)
    if not (model_path / "config.json").is_file():
        parser.error(f"config.json not found in {model_path}")
    if args.max_seq_len < 1:
        parser.error("max_seq_len 必须大于 0")
    try:
        actual_states = read_states(args.state_file)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    import torch
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    # 权重和前向计算均使用 float32；模型在 CPU 上执行。
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    model.eval()

    captured_outputs = {}
    captured_cos_sin = {}
    handles = []

    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // n_heads

    def save(name, tensor):
        captured_outputs[name] = tensor.detach().float().cpu().flatten()

    def out_hook(name):
        def hook(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            save(name, out)
        return hook

    def in_hook(name):
        def hook(_module, inputs):
            save(name, inputs[0])
        return hook

    # block 的输入可能走 positional 或 kwarg
    def block_in_hook(_module, args, kwargs):
        hidden = args[0] if len(args) > 0 else kwargs["hidden_states"]
        save("x", hidden)

    # RoPE 是函数不是 module：在 self_attn 入口取 cos/sin，稍后自己复现
    def attn_in_hook(_module, args, kwargs):
        cos, sin = kwargs["position_embeddings"]
        captured_cos_sin["cos"] = cos
        captured_cos_sin["sin"] = sin

    # Qwen2ForCausalLM -> .model (Qwen2Model) -> .layers (ModuleList)
    first_block = model.model.layers[0]
    attn = first_block.self_attn
    mlp = first_block.mlp

    # 名字对应 RunState 成员
    # 所有 block 中间结果均取 layers[0]，x 为 embedding 输出。
    handles.append(first_block.register_forward_pre_hook(block_in_hook, with_kwargs=True))
    handles.append(first_block.input_layernorm.register_forward_hook(out_hook("before_attn_norm_x")))
    handles.append(attn.q_proj.register_forward_hook(out_hook("q")))
    handles.append(attn.k_proj.register_forward_hook(out_hook("k")))
    handles.append(attn.v_proj.register_forward_hook(out_hook("v")))
    handles.append(attn.register_forward_pre_hook(attn_in_hook, with_kwargs=True))
    handles.append(attn.o_proj.register_forward_pre_hook(in_hook("weighted_value")))  # o_proj 的输入
    handles.append(attn.o_proj.register_forward_hook(out_hook("attn_o")))
    handles.append(first_block.post_attention_layernorm.register_forward_pre_hook(in_hook("residual_after_attn")))
    handles.append(first_block.post_attention_layernorm.register_forward_hook(out_hook("after_attn_norm_x")))
    handles.append(mlp.gate_proj.register_forward_hook(out_hook("g")))
    handles.append(mlp.up_proj.register_forward_hook(out_hook("b")))
    handles.append(mlp.act_fn.register_forward_hook(out_hook("s")))
    handles.append(mlp.down_proj.register_forward_pre_hook(in_hook("f")))  # down_proj 的输入
    handles.append(mlp.down_proj.register_forward_hook(out_hook("m")))
    handles.append(first_block.register_forward_hook(out_hook("residual_after_mlp")))

    inputs = torch.tensor([[args.token_id]])  # [batch=1, seq_len=1]，position=0
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    # q_rot / k_rot：用抓到的 q、k 和 cos/sin 复现 RoPE
    cos = captured_cos_sin["cos"]
    sin = captured_cos_sin["sin"]
    q = captured_outputs["q"].view(1, n_heads, 1, head_dim).to(cos.dtype)
    k = captured_outputs["k"].view(1, n_kv_heads, 1, head_dim).to(cos.dtype)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    save("q_rot", q_rot)
    save("k_rot", k_rot)

    # GQA：每组 query 头共享一个 key 头，分数形状为 [1, n_heads, 1, 1]。
    heads_per_group = n_heads // n_kv_heads
    repeated_k = k_rot.repeat_interleave(heads_per_group, dim=1)
    scores = torch.matmul(q_rot, repeated_k.transpose(-2, -1))
    scores = scores / math.sqrt(head_dim)
    probabilities = torch.softmax(scores, dim=-1)
    save("attn_score", scores)
    save("softmax_attn_score", probabilities)
    save("key_cache", k_rot)  # 仅 layer0、position0 的有效区域
    save("value_cache", captured_outputs["v"])

    # C 的 forward_block0 在第 0 层后直接做 norm/head，与完整模型输出不同。
    with torch.no_grad():
        block_output = captured_outputs["residual_after_mlp"].view(1, 1, -1)
        final_norm = model.model.norm(block_output)
        logits = model.lm_head(final_norm)
    save("final_norm", final_norm)
    save("logits", logits)

    order = ["x", "before_attn_norm_x", "q", "k", "v", "q_rot", "k_rot",
             "key_cache", "value_cache", "attn_score", "softmax_attn_score", "weighted_value", "attn_o",
             "residual_after_attn", "after_attn_norm_x", "g", "b", "s", "f", "m",
             "residual_after_mlp", "final_norm", "logits"]
    reference = {}
    for name in order:
        reference[name] = captured_outputs[name].tolist()
    print(f"layers[0], token={args.token_id}, position=0, float32, threshold=1e-5（严格小于）")
    all_passed = compare_states(reference, actual_states, args.max_seq_len, model.config.num_hidden_layers)

    # model->... 为模型最终权重，不属于 layers[0] 的中间状态。
    model_weights = [
        ("model->final_norm", model.model.norm.weight),  # [dim]
        ("model->lm_head", model.lm_head.weight),       # [vocab_size, dim]
    ]
    for name, weight in model_weights:
        # 只转换前 10 个值，避免复制整个 LM head 权重矩阵。
        first10 = weight.detach().reshape(-1)[:10].float().cpu().tolist()
        first10 = [round(value, 5) for value in first10]
        print(f"{name:28s} shape={tuple(weight.shape)} numel={weight.numel()} first10={first10}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
