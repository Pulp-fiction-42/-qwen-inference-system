# Session / Generator 接入验证

日期：2026-11-06（Asia/Shanghai）。GPU：NVIDIA GeForce RTX 5090 Laptop GPU。
CUDA 构建沿用 `-g -G -arch=sm_120`；参考环境为 torch 2.14.0+cu130、transformers 5.16.1。

## 结论

Session、Generator、Sampler、Tokenizer 多轮模板和 REPL 已接通。
新生成路径、历史、长度边界、异常清理及退出释放均通过本次针对性验证。
原三项回归仍有一项数值检查未通过：单 block attention score 的已有误差保持不变。
因此不能将结果表述为全部数值正确。

## 原三项回归

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| 单 block，23 项 Mean AE < 1e-5 | 22/23 通过；score = 8.34121623e-05；logits = 4.89830671e-06 | single_forward_compare.txt |
| 单轮 FP32 参考 vs 新 Generator/Sampler | 63 个新增 token 和 DONE 完全一致；diff 为空 | reference.txt、engine.txt、generation.diff |
| 同一 engine 三轮完整历史 | 正常完成，正确回忆“李三”“蓝色”；原脚本无自动数值判据 | multi_turn.txt |

三项测试通过 `test_kernel/run_correctness.py` 顺序运行，退出码为 1，
原因是保留的 score 超阈值，其他步骤仍完整执行。详细退出码和耗时见 results.json。
所有 338 个权重张量与参考精确一致，见 weights.json。
测试 engine 现已调用生产 Generator/Sampler，不再链接 legacy_sampler.cu。
原 Python 多轮测试继续采用其 README 的姓名/颜色场景。

## 新增功能验证

- `make test-model`：原模型编排测试通过。
- `make test-runtime`：真实 tokenizer + 替代 GPU 入口；两种 EOS、prefill/生成位置、
  新增长度、上下文边界、非法输入、历史提交、/clear、分配失败和 forward 失败清理通过。
- `session_gpu_test`：从 C++ Session 的模板和编码入口开始，生成的 63 个 token 与参考一致；
  超长请求失败后清空历史再生成，结果仍一致；连续轮次未发现整轮 GPU 缓冲增长。
- Session 析构后可用 GPU 显存增加 6,176,112,640 字节；RunState 释放检查允许
  16 MiB 驱动波动，因此不是逐字节内存检查器，但能够发现整轮 KV cache 未释放。
- GPU 验证新增预算 0/1、prompt 恰好填满上下文、只剩一个 token 空间，返回原因与数量正确。
- `tests/repl_test.py` 的 7 项全部通过：真实终端多轮记忆、/clear、/exit、EOF、
  新增长度限制、上下文限制与请求失败后继续输入，以及错误参数/缺失权重/截断文件的启动失败。
  退出码为 1 的启动错误用例属于预期结果。

日志：session_gpu.txt、session_gpu.stderr.txt、repl_results.json 及 repl_*.txt。
构建只出现既有 llama.cpp unicode.cpp 的未使用函数警告；本次新增 C++ 代码无编译警告。

## 迁移边界

source_audit.json 记录对照：14 个既有计算 kernel 的函数体保持不变，
两个 argmax kernel 与 legacy_sampler.cu 完全相同；Tokenizer 编码/解码实现保持不变。
默认 stream、原数学逻辑、原 kernel 尺寸限制均保留。
新增 argmax 临时区由 Session 随 RunState 管理；没有每 token GPU 分配。
部分 host 包装的 exit 改为异常，以便 Session 清理资源；不涉及 kernel 内部修复。

本次没有修复原 score 偏差、K/V 固定投影尺寸、final RMSNorm 多 block 写入，
或原 argmax 固定 640 的内部实现。已通过的生成样例不代表所有输入均保证数值正确。

复现命令见 [测试说明](../../readme.md)，接口和生命周期见
[Session / Generator 说明](../../../docs/session_generation.md)。
