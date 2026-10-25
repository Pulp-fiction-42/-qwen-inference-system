# Qwen Inference System

## 当前状态

### 已修复问题 / 改进
1. **修复问题**：
```shell
(base) bgi-chao@BGI123:~/repl$ python repl.py
PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.

You: hi
Assistant: 
copy v to v-cache failed: an illegal memory access was encountered
copy k to K-cache failed: an illegal memory access was encountered
gate projection launch failed: an illegal memory access was encountered
'sTraceback (most recent call last):
  File "/home/bgi-chao/repl/repl.py", line 95, in <module>
    main()
    ~~~~^^
  File "/home/bgi-chao/repl/repl.py", line 63, in main
    raise RuntimeError("Inference process exited unexpectedly.")
RuntimeError: Inference process exited unexpectedly.
```

2. **新增 Makefile 简化构建过程**：
   - 支持根据文件修改时间增量编译 `.cu` 并链接生成 `engine`。
   - 支持 `-lineinfo` 用于调试与 sanitizer 分析。
   - 支持 `test_kernel/` 目录下各独立 kernel 测试用例的编译规则。

### 待解决 bug
执行 `echo "1 99" | compute-sanitizer ./engine` 遇到问题：
```shell
========= COMPUTE-SANITIZER
========= Error: process didn't terminate successfully
========= Target application returned an error
========= ERROR SUMMARY: 0 errors
```

## 文件职责

- `forward.h` / `forward.cu`：负责单 token 前向流程。
- `layer.h` / `layer.cu`：负责各层计算与 attention / MLP 编排，计算直接写在各层 kernel 中。
