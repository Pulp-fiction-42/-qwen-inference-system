# Model / Layer 与算子迁移说明

## 阅读顺序

1. `src/model/model_types.h`：原来的四个数据结构。
2. `src/model/model.cpp`：模型级计算顺序。
3. `src/model/layer/layer.cpp`：一个 decoder block 的完整计算顺序。
4. `ops/ops.h`：不包含模型类型的 host 算子接口。
5. `ops/*.cu`：原 kernel 及普通函数包装。

## 数据和资源

- `Qwen2Model::model_` 引用已加载的 `Model`。
- `layers_[i].weights_` 引用 `Model::layers[i]`，不复制权重。
- 使用这些对象期间，不能释放、替换或重新加载被引用的权重数组。
- `RunState` 保留原来的临时结果和各层 KV cache。第一层输入是
  `state.x`，后续层输入是 `state.residual_after_mlp`。
- `position` 从 0 开始，调用前需要保证所有层的历史
  `[0, position)` 已写入，且 `position < max_seq_len`。
- 一次 generate 的所有 token 共享同一个 RunState，由 Session 在生成前后
  调用 `load_runstate()` / `free_runstate()`。Session/Generator 现已接通。
- 原 forward 临时申请的 RMSNorm 标量改为 `RunState::rms_scratch`，
  放在原连续缓冲区末尾，不改变其他缓冲区的相对位置。
- 权重保持整块加载和释放方式；REPL 通过命令行传入 model.bin 路径。
  权重不是每个 token 的临时资源，可以跨 generate 使用。

## 算子划分

| 文件 | 算子 |
| --- | --- |
| embedding.cu | embedding lookup |
| rms_norm.cu | RMSNorm；分别保留层内和 final 的启动方式 |
| linear.cu | 原 proj_bias、proj、mlp_proj、lm_head_proj |
| rope.cu | Q/K 的 RoPE |
| attention_score.cu | GQA 的缩放 QKᵀ |
| softmax.cu | 每行有效历史位置的 softmax |
| weighted_sum.cu | GQA 的 V 加权求和 |
| elementwise.cu | 原两种残差相加、SiLU、逐元素乘法 |
| copy.cu | 默认 stream 的设备复制及 forward 结束时的同步 |
| argmax.cu | 原两阶段 argmax，Sampler 调用；临时缓冲来自 RunState |

模型类没有 CUDA launch、内存申请或释放。所有 kernel 使用默认 stream；
KV 复制使用默认 stream 的 cudaMemcpyAsync，消费这些数据的 kernel 随后执行。
不再创建 attention 专用 stream。

## 本次允许的 kernel 调整

源代码来自
[原 layer.cu](https://github.com/Pulp-fiction-42/-qwen-inference-system/blob/main/layer.cu)、
[原 forward.cu](https://github.com/Pulp-fiction-42/-qwen-inference-system/blob/main/forward.cu) 和
[原 model.cu](https://github.com/Pulp-fiction-42/-qwen-inference-system/blob/main/model.cu)。

- 原 attn_score_softmax 在 score / softmax 分界处拆成两个 kernel，
  保留各自的计算、索引和归约语句。softmax 原先使用的 seq_len 字段
  改为独立 valid_columns 参数；max_seq_len 改为 row_stride 参数。
- 只读输入参数增加 const，与 host 接口一致，不改变函数体计算。
- 原 GQAConfig 移到 ops 内部头文件，保留其字段，算子不包含模型头文件。
- 原权重地址分配中的 lambda 改为显式指针赋值和 cursor 累加，
  保持原排列次序，遵守当前 C++ 风格约定。

未修复原 kernel 问题，未合并投影或残差 kernel，也未引入新的优化。
特别保留：

- proj_bias 的固定 6 × 256 启动尺寸，以及 K/V 投影的越界问题。
- final RMSNorm 的 6 个 block 启动方式及其重复写入问题。
- RoPE 内部的 128 常量。

原采样 kernel 不属于 Model/Layer forward。现已原样迁入 ops/argmax.cu，
由 Sampler 调用，保留原固定 640 的实现及启动尺寸。
test_kernel/legacy_sampler.cu 留作源码对照，不参与生产或测试 engine 链接。
为支持 Session 异常清理，资源和部分算子 host 包装中的 exit 已改为抛出异常；
kernel 函数体未改动。

## 验证边界

`make all` 编译 model、ops、resources 三份静态库。
`make test-model` 使用替代算子检查一层/两层模型、连续位置、cache 偏移、
中间结果的连接、权重指针、scratch 复用和 final norm / LM head 顺序。
这是 host 调度测试，不运行 CUDA kernel，也不证明数值正确性。

本次还检查了完整静态库的 CUDA 链接，以及迁移前后 12 个未拆分 kernel
的函数体一致性。拆出的 score 和 softmax 计算段也分别与原代码比对。
首次构建时未能从沙箱访问 NVIDIA 驱动。后续已在沙箱外完成三项真实 GPU 测试：
单 block 22/23 项通过，attention score 超出原阈值；单轮生成 63 个 token 完全一致；
三轮记忆对话完成。原融合 kernel 与拆分算子的补充对照也逐位一致。
详见 [GPU 测试报告](../test_kernel/results/REPORT.md)，不将该结果视为全部数值通过。

Session 接入后的回归结果见 [最新报告](../test_kernel/results/session_integration/REPORT.md)。
