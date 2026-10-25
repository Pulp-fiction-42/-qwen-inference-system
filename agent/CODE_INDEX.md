# Code Index

Qwen2.5-1.5B-Instruct 本地推理 REPL。Python 负责 tokenizer 与终端交互，常驻的 C/CUDA 进程（`engine`）负责推理。协议与请求流程详见 [readme.md](../readme.md)；协作与编码规范见 [AGENTS.md](../AGENTS.md)。

> 维护规则：新增/删除/重命名文件或公开符号时，同步更新本文件。本文件只记录"是什么、在哪、依赖谁"，不记录行号。

## 1. 心智模型

```
repl.py ──stdin: "N id1 … idN\n"──▶ engine (run.cu)
        ◀─stdout: "token_id\n"… "DONE\n"─┘
                                       │
                 load_model_weights ◀──┤ model.cu   （权重一次性搬到 GPU，生成每层地址表）
                 load_runstate      ◀──┤ model.cu   （激活 / KV-cache 一次性分配）
                 forward()          ◀──┘ forward.cu （单 token 前向，调用 layer.cu 中的层函数与 kernel）
```

核心思想：**权重只读、RunState 常驻、`position` 单调递增**；每个请求从 `position = 0` 重新 PREFILL，覆盖旧 KV-cache。

## 2. 文件地图

| 文件 | 职责 | 状态 |
|---|---|---|
| [repl.py](../repl.py) | 启动 `engine` 子进程；chat template → token ID；流式解码输出 | 完成 |
| [run.cu](../run.cu) | `main`：读请求 → PREFILL → GENERATION（argmax）→ 打印 token / `DONE` | 完成（依赖未实现的 `forward`、`sample_argmax`） |
| [model.h](../model.h) | `Config` / `blocks` / `Model` / `RunState` 定义，加载与释放函数声明 | 完成 |
| [model.cu](../model.cu) | `load_model_weights` / `free_model_weights` / `load_runstate` / `free_runstate` | 完成 |
| [forward.h](../forward.h) | `forward` 声明 | 进行中 |
| [forward.cu](../forward.cu) | 单 token 前向流程，调用层函数与 kernel | 进行中（TODO 较多） |
| [layer.h](../layer.h) | 模型层 kernel 声明及 `attention` / `mlp` host 编排函数声明 | 进行中 |
| [layer.cu](../layer.cu) | 模型层 kernel 与 `attention` / `mlp` 编排；残差、MLP、LM head 已实现 | 进行中（仍有 TODO） |
| [test/test_model.cu](../test/test_model.cu) | 校验 `Config` 与 embedding 前 8 个 float 是否与 `model.bin` 一致 | 完成 |
| [test/test_print.cu](../test/test_print.cu) | 打印 `RunState` 各成员的字节数、GPU 地址、偏移与到下一成员的间距；并简要打印 `Model` 权重地址 | 完成 |
| `test/test_model` | 已构建的二进制（未被 git 跟踪，不要手改） | 产物 |
| [Makefile](../Makefile) | 构建 `engine`（`-arch=sm_120`） | 完成 |
| [readme.md](../readme.md) | 版本备注及前向编排 / 层计算的文件分工 | 备注 |
| [AGENTS.md](../AGENTS.md) | 协作方式、编码规范（C++ 当 C 用）、文件分层规则 | 规范 |
| [gpu.png](./gpu.png) | GPU 微架构心智模型 | 参考图 |
| [qwen25_forward.png](./qwen25_forward.png) | Qwen2.5 前向结构图 | 参考图 |
| `.learn/` | 个人学习草稿（`pointer.cu`、`model.cu`、`hf_to_bin_minimal.ipynb` 及编译产物） | 非项目代码 |
| `.clangd` / `.vscode/` | 编辑器配置（clangd CUDA 模式，`sm_80`） | 配置 |

## 3. 数据结构（[model.h](../model.h)）

| 符号 | 含义 |
|---|---|
| `MAX_SEQ_LEN` | 最大序列长度 2048 |
| `Config` | `dim=1536, hidden_dim=8960, n_layers=28, n_heads=12, n_kv_heads=2, vocab_size=151936`；`head_dim=128`，`kv_dim=256`，GQA 每个 KV 头对应 6 个 Q 头 |
| `blocks` | 单层权重指针：`before_attn_norm, wq/bq, wk/bk, wv/bv, wo, post_attn_norm, w_gate, w_up, w_down`（均为 GPU `const float*`） |
| `Model` | `config`、`embedding`、`layers`（host 上的 `blocks[]` 地址表）、`final_norm`、`lm_head`（与 `embedding` 共享）、`device_data`、`mapped_size` |
| `RunState` | 单块 GPU 内存 `device_data` 上切出的全部中间缓冲：`x`、`q/k/v`、`q_rot/k_rot`、`key_cache/value_cache`（`[n_layers, max_seq_len, kv_dim]`）、`attn_score`、`softmax_attn_score`、`weighted_value`、`attn_o`、MLP 缓冲 `g/b/s/f/m`、`residual_*`、`final_norm`、`logits` |

## 4. `model.bin` 布局

路径当前硬编码为 `/home/bgi-chao/wdr/main/model.bin`（由同目录 `convert.py` 生成，约 6.17 GB，fp32）。

```
header: 6 × int32  = dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size
embedding            [vocab_size, dim]          （同时作为 lm_head，权重共享）
for each layer (×n_layers):
  before_attn_norm   [dim]
  wq [dim, dim]      bq [dim]
  wk [kv_dim, dim]   bk [kv_dim]
  wv [kv_dim, dim]   bv [kv_dim]
  wo [dim, dim]
  post_attn_norm     [dim]
  w_gate [hidden, dim]   w_up [hidden, dim]   w_down [dim, hidden]
final_norm           [dim]
```

`load_model_weights` 用"整块 `cudaMemcpy` + `cursor` 顺序切分"得到各指针；顺序必须与此布局严格一致。

## 5. 函数索引

| 函数 | 位置 | 说明 |
|---|---|---|
| `main` | run.cu | 请求循环；PREFILL / GENERATION；EOS = `151643`、`151645` |
| `load_model_weights(Model*)` | model.cu | mmap → 解析 header → `cudaMalloc/Memcpy` → 填 `layers[]` |
| `free_model_weights(Model*)` | model.cu | `cudaFree` + `delete[] layers` |
| `load_runstate(RunState*, Config*)` | model.cu | 一次 `cudaMalloc`，`cursor` 切出各缓冲 |
| `free_runstate(RunState*)` | model.cu | `cudaFree` |
| `forward(...)` | forward.h / forward.cu | 单 token 前向，返回 `logits` 指针；`cudaDeviceSynchronize` 后返回。声明 4 参、定义 5 参（多 `Config*`） |
| `embedding_lookup` | layer.h / layer.cu | `__global__`，token → `x` |
| `rms_norm` | layer.h / layer.cu | `__global__`，RMS 计算待实现，归一化表达式已写 |
| `proj / lm_head_proj` | layer.h / layer.cu | `__global__`，一线程一行累加；`lm_head_proj` 支持矩形权重和可选 bias（NULL 表示无 bias） |
| `rope` | layer.h / layer.cu | `__global__`，旋转位置编码（TODO） |
| `attn_residual / mlp_residual` | layer.h / layer.cu | `__global__`，逐元素残差相加，带 dim 边界判断 |
| `compute_attn_score_softmax` | layer.h / layer.cu | `__global__`，grid=`(seq_len, n_heads)`、block=`head_dim`；点积与 softmax 待实现 |
| `attention` | layer.h / layer.cu | host 编排：QKV 投影 → RoPE（host 算 `freqs[64]`，θ=1e6）→ V 写 cache（`v_to_v_cache_stream`）、K 写 cache + score（`computation_stream`）；已写到 score kernel 启动，之后的 softmax / 加权和 / `wo` / stream 同步与销毁未写 |
| `mlp` | layer.h / layer.cu | host 编排：gate/up → SiLU → 逐元素乘法 → down；复用 RunState 缓冲，在默认 stream 中依次启动 |
| `mlp_proj / silu / elementwise_mul` | layer.cu | MLP 内部 kernel：矩形矩阵向量乘、SiLU、逐元素乘法；一线程一个输出元素 |
| `sample_argmax` | 仅在 run.cu 使用 | 尚未声明与实现 |

## 6. 前向流程（对应 `qwen25_forward.png`）

```
embedding_lookup → x
for layer in 0..n_layers-1:
  rms_norm(before_attn_norm) → q/k/v 投影(+bias) → RoPE(q,k) → 写 K/V cache
  → scores = q·Kᵀ/√128 → softmax → weighted sum V → wo → + residual
  rms_norm(post_attn_norm) → gate/up → SiLU(g)⊙up → down → + residual
final_norm → lm_head(=embedding) → logits
```

## 7. 当前已知不一致（仅记录，未修改）

1. `forward` 签名：[forward.h](../forward.h) 为 4 参数，[forward.cu](../forward.cu) 定义为 5 参数（多 `Config*`）；[run.cu](../run.cu) 按 4 参数调用。
2. `MAX_SEQ_LEN` 在 model.h、model.cu、run.cu、repl.py 各定义一次；model.cu、run.cu 都 include model.h，C++ 中会重定义。`model.h` 中 `lm_head` 注释写 `[dim, vocab_size]`，实际与 embedding 同为 `[vocab_size, dim]`。
3. `RunState::attn_score` 未在 `load_runstate` 中计入 `total_size`，也未赋指针。
4. `sample_argmax` 在 run.cu 使用但无声明/实现；`model.h`、`forward.h`、`layer.h` 均无 include guard，重复包含可能导致重定义。
5. [forward.cu](../forward.cu) 的 LM head 已改为传 NULL（无 bias），不再分配零 bias；`forward` 仍每 token 分配 rms，[layer.cu](../layer.cu) 的 `attention` 每次调用创建两个 stream 而未销毁。违反 AGENTS.md 6.5（per-token 不得分配或创建 stream）。
6. repl.py 将权重路径作为 `argv[1]` 传给 `engine`，但 `model.cu` 忽略参数并硬编码路径；`engine` 由 Makefile 生成，Makefile 目前为空。
7. 层 kernel 大多未实现，forward.cu 中部分 grid/block 仍为未定义的占位变量；`attention` 调用的 `q_proj / k_proj / v_proj` 与 layer.h / layer.cu 中的 `proj` 不一致。Makefile 为空。
8. [model.cu](../model.cu) 中有不符合 AGENTS.md 6.0（C++ 当 C 用）的写法：lambda `take`、`auto*`、`static_cast`/`reinterpret_cast`、`new[]`/`delete[]`、`nullptr`；[run.cu](../run.cu) 也用了 `constexpr`、`nullptr`。
9. [layer.cu](../layer.cu) 的 `compute_attn_score_softmax` 硬编码 `h / 6`（应为 `n_heads / n_kv_heads`）。
