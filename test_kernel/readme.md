# 重构后的三项 GPU 测试

测试定义来自[原仓库 test_kernel](https://github.com/Pulp-fiction-42/-qwen-inference-system/tree/main/test_kernel)。
本目录测试真实的 Qwen2Layer、Qwen2Model、Generator、Sampler 和 ops，不使用 host 调度测试里的替代算子。

## 执行

本机已有可用环境，无需安装依赖：

```sh
make -j4 test-kernel-build

/home/bgi-chao/miniconda3/envs/inference/bin/python \
    test_kernel/run_correctness.py \
    /home/bgi-chao/wdr/main/Qwen2.5-1.5B-Instruct

# 补充：原融合 kernel 与拆分算子对照
make test_kernel/test_single_forward/test_score_split
./test_kernel/test_single_forward/test_score_split
```

GPU 命令需要能够访问 NVIDIA 设备；本次在沙箱外执行。
编译沿用原项目的 `-g -G -arch=sm_120`。
运行器按顺序执行，保留各项退出码，某项数值失败后仍执行其余测试。
总退出码为 0 才表示所有自动条件通过；多轮脚本完成不等同于数值验证。

## 原测试与接口适配

| 测试 | 保留的内容 | 适配内容 |
| --- | --- | --- |
| 单 block | token=123、position=0、block 0 后接 final norm/head；23 项 Mean AE 严格小于 1e-5 | C++ 入口调用 Qwen2Layer 和 op_*；只输出有效 cache/score 元素并保留原下标，以 %.9g 打印；inspect_forward.py 原样保留 |
| 单轮生成 | 原 Python 参考脚本、FP32、逐 token prefill、argmax、EOS、2048 上限、输出协议 | 测试 engine 调用生产 Generator/Sampler；保持原 token 协议及停止条件，每轮独立 RunState |
| 多轮对话 | 同一 engine 接收三轮完整对话历史，打印回答，无自动数值判据 | 原脚本实际 prompts 与其 README 不同；采用 README 中姓名“李三”、颜色“蓝色”的三轮记忆场景 |

测试 engine 位于 `test_kernel/engine.cu`，现在调用生产 Generator/Sampler。
原采样 kernel 已原样迁入 `ops/argmax.cu`；`legacy_sampler.cu` 仅留作对照，不再链接。
三项旧测试中的模板/历史仍由 Python 处理；真实 C++ Session 和 REPL 由下述补充测试覆盖。

## 权重核对

`verify_weights.py` 核对二进制头、原加载顺序及所有 338 个张量：
将 Hugging Face 权重转换为 float32 后，与 model.bin 逐元素精确比较。
LM head 继续与 embedding 绑定。权重不一致时停止对照。

## 结果

[最初报告](results/REPORT.md) 保留不变。
Session 接入后的结果见 [最新报告](results/session_integration/REPORT.md) 和
[机器可读结果](results/session_integration/results.json)。
`results/device_state.txt` 是实际 GPU 中间值，`generation.diff` 是生成 ID 差异。
`test_score_split.cu` 为独立诊断，对 8 个连续位置的有效历史分别比较
原融合 kernel 和当前拆分算子的 score/probability 位模式，不更改生产实现。

## Session / REPL 补充验证

```sh
make -j4 repl test-model test-runtime build/runtime/session_gpu_test

# 先复跑三项原测试，生成新的参考文件（已有 score 偏差使总退出码为 1）。
/home/bgi-chao/miniconda3/envs/inference/bin/python test_kernel/run_correctness.py \
    /home/bgi-chao/wdr/main/Qwen2.5-1.5B-Instruct \
    --output-dir test_kernel/results/session_integration

./build/runtime/session_gpu_test /home/bgi-chao/wdr/main/model.bin \
    /home/bgi-chao/llama-deploy/models/qwen2.5-1.5b-instruct-q8_0.gguf \
    test_kernel/results/session_integration/reference.txt

python3 tests/repl_test.py /home/bgi-chao/wdr/main/model.bin \
    /home/bgi-chao/llama-deploy/models/qwen2.5-1.5b-instruct-q8_0.gguf
```

`test-runtime` 使用真实 tokenizer、替代 GPU 入口验证边界、历史和失败清理，
可通过 `VOCAB_PATH=/path/to/vocab.gguf` 改路径。
GPU 测试验证 Session 的 token 对照、长度限制和显存回收；
REPL 测试通过标准输入验证多轮、命令、EOF、启动错误和请求失败后的继续输入。
