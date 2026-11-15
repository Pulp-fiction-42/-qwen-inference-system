"""核对原 model.bin 的布局和全部 FP32 权重，防止使用不同权重做数值对照。"""
import json
from pathlib import Path
import struct
import sys

import numpy as np
from safetensors import safe_open
import torch

model_dir = Path(sys.argv[1])
binary_path = Path("/home/bgi-chao/wdr/main/model.bin")
config = json.loads((model_dir / "config.json").read_text())
expected_header = (
    config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"],
    config["num_attention_heads"], config["num_key_value_heads"], config["vocab_size"],
)
with binary_path.open("rb") as stream:
    header = struct.unpack("<6i", stream.read(24))
if header != expected_header:
    raise ValueError(f"header mismatch: {header} != {expected_header}")

names = ["model.embed_tokens.weight"]
for index in range(config["num_hidden_layers"]):
    prefix = f"model.layers.{index}."
    names.extend(prefix + suffix for suffix in [
        "input_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.q_proj.bias",
        "self_attn.k_proj.weight", "self_attn.k_proj.bias",
        "self_attn.v_proj.weight", "self_attn.v_proj.bias",
        "self_attn.o_proj.weight", "post_attention_layernorm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    ])
names.append("model.norm.weight")
values = np.memmap(binary_path, mode="r", dtype="<f4", offset=24)
offset = 0
chunk_size = 1024 * 1024
with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source:
    for name in names:
        tensor = source.get_tensor(name).reshape(-1)
        for start in range(0, tensor.numel(), chunk_size):
            end = min(start + chunk_size, tensor.numel())
            actual = torch.from_numpy(np.array(values[offset + start:offset + end], copy=True))
            reference = tensor[start:end].float()
            if not torch.equal(actual, reference):
                raise ValueError(f"weight mismatch: {name}, elements [{start}, {end})")
        offset += tensor.numel()
if offset != len(values):
    raise ValueError(f"binary length mismatch: {offset} != {len(values)}")
print(json.dumps({
    "header": header, "tensors": len(names), "float32_elements": offset,
    "exact_match": True, "lm_head_tied_to_embedding": config["tie_word_embeddings"],
}, ensure_ascii=False, indent=2))
