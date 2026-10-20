1. 未完成版本
2. pure operator 和“具体某一层的计算”没有分开（下一版需要实现）。
3. 下一版需要明确抽象层级和分工职责：明确 pure operator 由 `__device__` 在 kernel 函数中发起，`forward.cu` 中存放执行某一具体层算子的函数，由 `__global__` 发起（并在其中通过 `__device__` 调用 operator）。
