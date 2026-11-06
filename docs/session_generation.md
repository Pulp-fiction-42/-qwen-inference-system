# Session / Generator 接口与生命周期

## 阅读顺序

1. `src/main.cpp`：解析路径和长度参数，创建 Session，调用 Terminal。
2. `src/terminal/terminal.cpp`：逐行读取、处理 `/exit` 和 `/clear`、显示回答。
3. `src/session/session.cpp`：历史、模板、编码、资源、生成、解码。
4. `src/generator/generator.cpp`：prefill、逐 token forward 和停止条件。
5. `src/generator/sampler/sampler.cpp`：借用 RunState 调用 argmax。
6. `ops/argmax.cu`：原两个 kernel 及其启动函数。

新增代码使用普通函数、简单 class/struct、string/vector、引用和显式循环。
Tokenizer 现有 BPE 编解码内部实现保持不变，仅扩展消息列表模板。

## 接口

```cpp
struct ChatMessage {
    std::string role;     // user 或 assistant
    std::string content;
};

struct GenerationOptions {
    int max_new_tokens = 256;
    int max_context_tokens = MAX_SEQ_LEN;  // 2048
};

// tokens 只包含新增内容，排除 prompt 和 EOS。
struct GenerationResult {
    std::vector<int> tokens;
    StopReason stop_reason = STOP_EOS;
};

struct TurnResult {
    std::string text;
    GenerationResult generation;
};
```

`Tokenizer::apply_chat_template(messages)` 先加入 Template 的系统消息，
再依次加入 messages，最后追加 assistant 起始标记。
旧的 string 重载等价于一条 user 消息。Tokenizer 不保存历史。

`Generator(Qwen2Model&, RunState&)` 借用资源。
`generate(input_ids, options)` 每次从 position 0 开始，不复用上轮 KV。
`Sampler::sample()` 使用 RunState 中的 logits 和采样临时区返回一个 token ID。

`Session(weights_path, vocab_path, options)` 加载一次模型和 tokenizer。
`run_turn(input)` 返回 TurnResult，`clear()` 清空历史、保留权重与 tokenizer。
复制 Session 被禁止，避免重复释放 GPU 权重。

## 一轮资源和数据流

```text
Session 已持有：Model 权重、Tokenizer、成功提交的历史
    |
    +-- 复制历史，追加本轮 user（暂不提交）
    +-- apply_chat_template -> encode -> 检查 prompt 长度
    +-- load_runstate：一次申请 KV cache + 中间值 + RMS/argmax 临时区
    +-- 栈上创建 Qwen2Model 和 Generator，借用权重和 RunState
    +-- generate -> decode
    +-- 追加 assistant 并提交历史
    +-- free_runstate

异常：free_runstate，保留原历史，错误交回 Terminal
退出：Session 析构 -> free_model_weights；Tokenizer 自动析构
```

Qwen2Model 在每轮栈上创建，其 vector 只有 28 个层权重引用，不复制 GPU 权重。
RunState 保留原连续布局，argmax 的三个临时区追加在原 RMS scratch 后面。
不创建新 stream，kernel 和 KV 复制均按默认 stream 顺序执行。
没有每 token 的 GPU 分配，也不在层对象中保存跨轮运行状态。

## 停止和失败语义

- token 151643 或 151645：返回 `STOP_EOS`，结束 token 不加入结果。
- 新增内容达到 `max_new_tokens`：返回 `STOP_MAX_NEW_TOKENS`。
- prompt 与新增内容合计达到 `max_context_tokens`：返回 `STOP_CONTEXT_LIMIT`。
- 生成非空内容时，若两种长度上限同时达到，返回上下文上限原因。
- 最后一个新增 token 不再送入 forward；无需为没有后续预测的 token 写 KV。
- prompt 恰好填满上下文：返回空 tokens 和上下文上限，不启动 forward。
- `max_new_tokens == 0`：返回空 tokens 和生成长度上限，不启动 forward。
- 空 prompt、无效 token/参数或 prompt 超长：抛出异常，不截断历史。
- 正常结束或产生内容后被长度截断：将本轮 user/assistant 提交到历史。
- 预算为 0 或 prompt 已满、没有生成内容：不提交本轮；立即 EOS 仍算成功回复。
- 生成、解码或内存分配失败：释放本轮资源、不提交历史，Terminal 显示错误后继续读取。

文件映射、GPU 权重复制、CPU 地址表分配失败的清理在资源函数内部完成，
因为这些错误发生时 Session 可能尚未构造成功。
既有 host `exit()` 出口改为异常，便于外层执行清理；未更改 kernel 内部计算。

## 构建、运行及验证

```sh
make -j4 repl
./build/qwen_repl /home/bgi-chao/wdr/main/model.bin \
    /home/bgi-chao/llama-deploy/models/qwen2.5-1.5b-instruct-q8_0.gguf \
    --max-new-tokens 256 --context-limit 2048
```

权重仍为原 FP32 二进制布局，LM head 与 embedding 共享。
GGUF 仅用作词表，不替代 model.bin 权重。
原 kernel 固定的维度限制、argmax 的固定 640 实现及已有数值偏差均保留。
测试命令与结果见 [测试说明](../test_kernel/readme.md) 和
[最新报告](../test_kernel/results/session_integration/REPORT.md)。
