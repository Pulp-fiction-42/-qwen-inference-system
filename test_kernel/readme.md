# 推理测试

| 目录 | 测什么 | 怎么测 |
|---|---|---|
| `test_single_forward/` | 单 token、单 Transformer block 的前向计算。最初为排查 logits 不一致而创建，用于定位哪个计算步骤的 CUDA kernel 出错。 | 固定 token=123、position=0，只计算模型的第 0 个 block，再接 final norm / LM head；导出中间值，由 `inspect_forward.py` 与 PyTorch 对照，要求有效元素齐全且各项平均绝对误差小于 `1e-5`。 |
| `test_generation/` | 完整模型的单轮生成过程是否与参考实现一致。 | Python 逐 token 前向、argmax 生成；stdout 保存套用模板后的输入请求，`-out` 保存参考输出。把同一请求送入实际 `engine`，比较两份输出 txt。 |
| `test_multi_turn/` | 人工观察 engine 的多轮对话表现。 | 预设三轮中文输入，维护实际问答历史，向同一个 engine 逐轮发送完整历史，在终端显示中文问答。 |

## 单步辅助测试

- `test_single_forward/test_single_forward.cu`：导出第 0 个 Transformer block 的中间状态；这里的 block 指模型层。
- `test_single_forward/inspect_forward.py`：读取导出值，逐项比较 PyTorch 参考值。
- `test_single_forward/test_sampler.cu`：构造随机 logits，在已知位置放入唯一最大值，打印 argmax 的期望位置和实际位置。
- `test_single_forward/test_rms_norm.cu`：单独调用 token=123 的 embedding 和首层 RMSNorm，目前没有数值比较或通过判定。

## 单轮生成对照的输出约定

- 输入文件：每行一轮请求，格式为 `N id1 … idN`，包含对话模板 token。
- `reference.txt` 和 `engine.txt`：每行一个生成 token ID，每轮以 `DONE` 结束；不输出 EOS。
- 两边都使用 FP32、argmax、EOS `151643` / `151645`，输入加生成的长度上限为 2048。
- 用 `diff -u reference.txt engine.txt` 比较；优先检查首次不一致的轮次和 token，之后的生成可能因前缀不同而继续分叉。

## 多轮测试流程

1. 脚本启动一个 engine，依次发送 `prompts` 中的三条中文输入。
2. 每轮将完整 `messages` 套用与 `repl.py` 相同的模板；收齐回复后解码，在终端打印 `You` / `Assistant`。
3. 将 engine 的实际回复加入历史，再进行下一轮。直接阅读回答，观察是否记住前文；不导出文件，不自动判定通过或失败。

## 使用方法

以下命令均在项目根目录执行，Python 环境需安装 PyTorch 和 Transformers。先将模型目录替换为实际路径：

```bash
MODEL_PATH="你的本地HuggingFace模型目录"
```

该目录的权重应与 C/CUDA 加载的 `model.bin` 对应。目前 `model.cu` 的权重路径仍是硬编码，运行前确认其指向正确文件。

### 1. 单 token、单 block forward

编译测试，保存中间状态，再与 PyTorch 对照：

```bash
make test_kernel/test_single_forward/test_single_forward &&
./test_kernel/test_single_forward/test_single_forward > test_kernel/test_single_forward/device_state.txt &&
python test_kernel/test_single_forward/inspect_forward.py "$MODEL_PATH" \
    test_kernel/test_single_forward/device_state.txt --token-id 123 --max-seq-len 2048
```

查看各项“通过 / 未通过”和误差；全部通过时 Python 返回 0，否则返回非零。若修改 C 测试里的 token，需同步修改 `--token-id`。

### 2. 单轮 generation

先运行参考模型，保存请求和参考生成 ID；再启动 engine 处理同一请求：

```bash
make engine &&
python test_kernel/test_generation/test_generation.py "$MODEL_PATH" \
    -prompt "你好，请介绍一下自己。" \
    -out test_kernel/test_generation/reference.txt \
    > test_kernel/test_generation/prompt.txt &&
./engine < test_kernel/test_generation/prompt.txt \
    > test_kernel/test_generation/engine.txt &&
diff -u test_kernel/test_generation/reference.txt test_kernel/test_generation/engine.txt
```

`prompt.txt` 保存输入 ID；两份结果文件保存生成 ID 和 `DONE`。`diff` 没有输出表示完全一致，有输出则显示差异。

### 3. 多轮对话

三轮提示词在脚本的 `prompts` 列表中修改。运行后自动完成三轮问答，每轮回复收齐后显示为人类可读的文字：

```bash
make engine &&
python test_kernel/test_multi_turn/test_multi_turn.py "$MODEL_PATH"
```

默认场景先告诉模型姓名“李三”和喜欢的颜色“蓝色”，第三轮询问这两项信息。此脚本只加载 tokenizer，推理由 engine 完成。
