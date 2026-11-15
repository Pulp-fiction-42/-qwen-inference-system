# 重构后计算正确性测试结果

日期：2026-11-05。测试已在真实 GPU 上运行，整体结论是：**没有全部通过原测试标准**。

## 环境和权重

- GPU：NVIDIA GeForce RTX 5090 Laptop GPU。
- PyTorch：2.14.0+cu130；Transformers：5.16.1。
- 编译沿用原配置：`-g -G -arch=sm_120`。
- 参考模型：`/home/bgi-chao/wdr/main/Qwen2.5-1.5B-Instruct`。
- CUDA 权重：`/home/bgi-chao/wdr/main/model.bin`。
- 338 个权重张量、1,543,714,304 个 float32 元素全部精确一致。
- 本次测试未改动 `ops/` 或 `src/model/` 中的任何文件。

权重记录：[weights.json](weights.json)。运行信息：[environment.txt](environment.txt)。

## 三项测试

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| 单 token / 单 block 数值对照 | **未通过**：23 项中 22 项通过，attention score 超出原误差标准 | [逐项误差](single_forward_compare.txt) |
| 完整模型单轮生成 | **通过本测试用例**：63 个生成 token 与参考逐个一致，DONE 一致，diff 为空 | [参考输出](reference.txt)、[实际输出](engine.txt)、[diff](generation.diff) |
| 多轮对话 | 三轮均完成；第三轮正确回答姓名李三、颜色蓝色。原脚本无自动数值通过标准 | [完整对话](multi_turn.txt) |

单轮输入为“你好，请介绍一下自己。”，套用模板后 34 个 prompt token。
EOS 和最大长度条件与原脚本一致，没有减少生成长度或放宽比对标准。
多轮采用原 README 的姓名/颜色场景；这是对实际脚本 prompts 的明确调整。

## 单 block 未通过项

固定 token=123、position=0，只经过第 0 个 block 后做 final RMSNorm / LM head。

| 状态 | Mean AE | 标准 |
| --- | --- | --- |
| attention score | **8.34121623e-05** | 严格小于 1e-5，未通过 |
| attention softmax | 0 | 通过 |
| final norm | 4.49821147e-06 | 通过 |
| logits | 4.89830671e-06 | 通过 |

23 项有效元素全部输出，没有缺失、非有限值或越界索引报告。
position=0 时每个 head 仅有一个有效历史位置，所以 softmax 恒为 1；
softmax 通过不能证明 score 的偏差无关紧要。
原误差阈值保持不变，比较脚本退出码为 1。

## 偏差是否来自 score / softmax 拆分

额外使用同一组实际 Q、K 和 cache，对比原 `attn_score_softmax` kernel
和当前拆分算子。测试位置为 0～7，token 为 123～130，使用真实首层权重。

- 共比较 432 个有效 score 元素及 432 个 probability 元素。
- 所有值均有限，位模式完全相同。
- 每个位置的最大绝对差均为 0。
- 对照程序退出码为 0。

记录：[score_split_check.txt](score_split_check.txt)。

这说明在本次输入下，拆分没有改变原融合 kernel 的输出。
它不解释 CUDA 相对 PyTorch 的偏差来源，也不代表所有输入或整个模型都已数值对齐。
本次没有修改 kernel、归约顺序、采样逻辑或阈值。

## 可复现性及范围

执行方式见 [测试说明](../readme.md)。各阶段命令、退出码和耗时保存在
[results.json](results.json)。测试 engine 使用当前 Qwen2Model 和原采样代码，
不经过尚未接入的生产 Session/Generator。
