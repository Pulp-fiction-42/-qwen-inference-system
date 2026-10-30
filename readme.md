# Qwen Inference System

## 当前状态

- 基本的前向传播已经完成，`engine` 已支持完整 generation 和多轮对话。
- 此前的非法内存访问问题已修复。
- `test_kernel/` 下三项检查均已通过，结果由项目维护者确认：

| 检查 | Python 文件 | 验证方式 | 状态 |
|---|---|---|---|
| 单 token、单 block forward | `test_kernel/test_single_forward/inspect_forward.py` | 将 C/CUDA 导出的中间状态与 PyTorch 参考值逐项对照 | 已通过 |
| 完整 generation | `test_kernel/test_generation/test_generation.py` | 生成参考 token 序列，与 `engine` 对同一输入的输出逐 token 比较 | 已通过 |
| 多轮对话 | `test_kernel/test_multi_turn/test_multi_turn.py` | 向同一个 `engine` 发送完整对话历史，人工检查回复及上下文记忆 | 已通过 |

测试方法与运行命令详见 [test_kernel/readme.md](test_kernel/readme.md)。

## 构建

- 使用 `make` 或 `make engine` 编译 `engine`，根据文件修改时间增量编译并链接。
- 使用 `make <测试目标路径>` 编译 `test_kernel/` 下的 C/CUDA 测试文件，例如 `make test_kernel/test_single_forward/test_single_forward`。
- Python 测试脚本直接运行，无需通过 `make` 编译。

## 历史待确认记录

此前使用 `compute-sanitizer` 运行 `engine` 时，曾出现进程异常退出、但 `ERROR SUMMARY: 0 errors` 的情况。本次进度更新未重新验证该问题。

## 文件职责

- `forward.h` / `forward.cu`：负责单 token 前向流程。
- `layer.h` / `layer.cu`：负责各层计算与 attention / MLP 编排，计算直接写在各层 kernel 中。
