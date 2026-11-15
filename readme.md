# Qwen 推理 REPL

## 本次重构

- `Qwen2Model` 编排完整前向计算，`Qwen2Layer` 组织单层 decoder；CUDA 算子和启动配置迁入 `ops/`，score 与 softmax 独立执行。
- Session 管理权重、历史和每轮运行资源；Generator 负责 prefill 与逐 token 生成，Sampler 负责 argmax，Tokenizer 支持多轮模板。
- 接通 REPL，支持 `/clear`、`/exit` 和长度限制。算子使用默认 stream，原 kernel 内部逻辑保持不变。
- 验证中单轮生成的 63 个 token 与参考一致，多轮测试完成；单 block 的 attention score 仍超出误差阈值。详见 [测试报告](test_kernel/results/session_integration/REPORT.md)。

```sh
make -j4 repl
./build/qwen_repl /path/to/model.bin /path/to/vocab.gguf \
    --max-new-tokens 256 --context-limit 2048
```

权重使用原项目 FP32 model.bin 布局；GGUF 仅提供词表和合并规则。
输入 `/exit` 或 EOF 退出，`/clear` 清空历史。普通输入生成完整回答后一次显示。
默认本轮最多新增 256 个 token，prompt 与回答合计不超过 2048 个 token。
达到长度限制时显示停止原因，超长 prompt 不自动截断。

Session 持有权重和文本历史，每轮从 position 0 重算完整历史。
RunState（包括 KV cache、RMSNorm 和 argmax 临时缓冲）每轮申请、结束后释放；
Generator 和 Sampler 只借用资源。正常生成或长度截断后提交历史，发生异常则保留原历史。
具体接口和生命周期见 [Session / Generator 说明](docs/session_generation.md)。

## Qwen tokenizer

`src/session/tokenizer.cpp` 迁入了 llama.cpp 的 Qwen2 BPE 分词逻辑：特殊 token
切分、按 merge rank 合并、token ID 查找及字节解码。当前支持
`tokenizer.ggml.model=gpt2`、`tokenizer.ggml.pre=qwen2` 的 GGUF（包含当前部署的
Qwen2.5-1.5B-Instruct Q8_0），聊天格式采用 Qwen ChatML 多轮模板，也保留单轮调用接口。

Tokenizer 拥有从 GGUF 加载的词表和合并规则，借用外部的 `Template`，不保存会话历史。
构造函数现在需要额外传入 GGUF 路径；不加载模型权重，不创建推理上下文。
GGUF 读取依赖 `libggml-base`，Unicode 预切分和字节映射复用 llama.cpp 的
`unicode.cpp` / `unicode-data.cpp`。本实现不调用 `llama_tokenize()`，也不需要链接
`libllama` 或 CUDA。

```cpp
#include "src/session/tokenizer.h"

Template chat_template;  // 默认 Qwen 系统提示词；可修改 system_prompt，清空则省略系统消息
Tokenizer tokenizer(chat_template,
    "/home/bgi-chao/llama-deploy/models/qwen2.5-1.5b-instruct-q8_0.gguf");

const auto prompt = tokenizer.apply_chat_template("你好");
const auto input_ids = tokenizer.encode(prompt);
// Session 将 input_ids 交给 Generator；这里只展示编解码接口。
const auto text = tokenizer.decode(input_ids);
```

`encode()` 识别文本中的特殊 token，并按 GGUF 配置添加 BOS/EOS，不重复应用模板。
`decode()` 合并整组 token 的字节，过滤控制、未知和未使用 token；正常文本和
user-defined token 保留。上例解码的是完整输入，因此仍包含角色名和系统提示词；
实际显示回答时应传入 Generator 返回的新增 tokens。
`Template` 的生命周期必须覆盖 Tokenizer；无法读取的模型、错误元数据和越界
token ID 会抛出标准异常。

构建 tokenizer 静态库：

```sh
make tokenizer
# llama.cpp 在其他位置时：
make tokenizer LLAMA_DIR=/path/to/llama.cpp LLAMA_LIB_DIR=/path/to/llama.cpp/build/bin
```

输出为 `build/libinference_tokenizer.a`。最终可执行文件需在该静态库之后链接
`-lggml-base`，并指定其库目录和运行时搜索目录，例如：

```sh
g++ -std=c++17 -I. example.cpp build/libinference_tokenizer.a \
  -L../llama-deploy/llama.cpp/build/bin \
  -Wl,-rpath,/home/bgi-chao/llama-deploy/llama.cpp/build/bin \
  -lggml-base -o example
```

`make tokenizer` 和 `make clean-tokenizer` 保留独立入口；
`make repl` 将其与 Session、Generator、Sampler 和模型算子链接。

## Model 与 Layer

`src/model/model.h/.cpp` 定义 `Qwen2Model`，
`src/model/layer/layer.h/.cpp` 定义 `Qwen2Layer`。
原 `Config`、`blocks`、`Model`、`RunState` 放在
`src/model/model_types.h` 中，保留原来的权重和缓冲区布局。

`Qwen2Model` 引用已经加载的 `Model`，用
`std::vector<Qwen2Layer>` 按配置中的层数组织 decoder。
每个 `Qwen2Layer` 只引用一个 `blocks`，不复制 GPU 权重。
计算顺序为 embedding → decoder blocks → final RMSNorm → LM head。
一次 forward 处理一个 token，输出保存在 GPU 上的 `RunState::logits`。

`ops/ops.h` 提供普通 host 函数，CUDA kernel 和启动配置放在
`ops/*.cu` 中。Attention score 和 softmax 已拆分；
所有操作均在默认 stream 上顺序执行。算子不依赖模型结构体。

运行资源由 Session 决定申请和释放时机。资源函数迁入
`src/model/model_resources.cu`：
一次 generate 前调用 `load_runstate()`，各 token 复用同一个
`RunState`，结束后调用 `free_runstate()`。
新增 `rms_scratch` 位于原缓冲区末尾，随整块显存一起申请和释放。
Model/Layer 不申请或释放 GPU 资源；模型权重仍通过原来的
`load_model_weights()` / `free_model_weights()` 管理，REPL 从命令行传入权重路径。

Session/Generator 已接入这些接口；异常通过 host 包装函数传回 Session，
由 Session 清理 RunState。原测试的无路径调用仍保留默认 model.bin 路径。

本次保留原 kernel 内部计算及已知问题，包括 K/V 投影的固定启动尺寸、
final RMSNorm 的 6 个 block 和 RoPE 内部固定的 128。
这些问题留待后续修复；构建和 host 测试通过不代表 GPU 推理正确。
具体迁移边界见 [重构说明](docs/model_refactor.md)。

构建与验证：

```sh
make -j4 all      # model、ops、resources 三份静态库，需要 nvcc
make test-model  # 无需 GPU，检查 host forward 的调用顺序和参数
```

也可以分别运行 `make model`、`make ops`、`make resources`。
输出为 `build/libinference_model.a`、`build/libinference_ops.a` 和
`build/libinference_resources.a`；最终程序需要将所用静态库与 CUDA runtime 链接。
`make tokenizer` 保留原来的独立构建方式。

原仓库三项 GPU 测试已适配，见 [运行说明](test_kernel/readme.md) 和
[最新测试报告](test_kernel/results/session_integration/REPORT.md)。单 block 尚有一项 score 误差超阈值；
单轮生成 token 完全一致，多轮记忆场景完成。
