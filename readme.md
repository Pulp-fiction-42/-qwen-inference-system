# Qwen Inference System

## 当前状态

1. 已成功完成编译和链接；可执行产物位于 `make/` 目录。
2. 运行 `python repl.py` 时，CUDA kernel 会触发非法内存访问：

```shell
PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.

You: fuck you
Assistant: copy v to v-cache failed: an illegal memory access was encountered
copy k to K-cache failed: an illegal memory access was encountered
gate projection launch failed: an illegal memory access was encountered
Traceback (most recent call last):
  File "/home/bgi-chao/repl/repl.py", line 95, in <module>
    main()
    ~~~~^^
  File "/home/bgi-chao/repl/repl.py", line 63, in main
    raise RuntimeError("Inference process exited unexpectedly.")
RuntimeError: Inference process exited unexpectedly
```

这表明 CUDA kernel 中存在非法访问问题。下一版本将重点定位并修复该问题。

## 文件职责

- `forward.h` / `forward.cu`：负责单 token 前向流程。
- `layer.h` / `layer.cu`：负责各层计算与 attention / MLP 编排，计算直接写在各层 kernel 中。
