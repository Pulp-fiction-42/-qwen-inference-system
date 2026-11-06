#pragma once

#include <memory>
#include <string>
#include <vector>

// Qwen2/2.5 的 ChatML 模板；空 system_prompt 表示不添加系统消息。
struct Template {
    std::string system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
};

struct ChatMessage {
    std::string role;     // user 或 assistant；系统消息统一来自 Template。
    std::string content;
};

// 负责聊天模板格式化与编解码，不保存会话历史。
class Tokenizer {
public:
    // 模板由外部持有，其生命周期必须覆盖 Tokenizer。
    // 从 GGUF 加载 Qwen2 BPE 词表和合并规则，不加载权重或创建推理上下文。
    Tokenizer(const Template& chat_template, const std::string& model_path);
    ~Tokenizer();

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) = delete;
    Tokenizer& operator=(Tokenizer&&) = delete;

    // 多轮版本按消息顺序格式化，最后添加 assistant 生成提示。
    std::string apply_chat_template(const std::vector<ChatMessage>& messages) const;
    // 保留原单轮调用方式。
    std::string apply_chat_template(const std::string& input) const;

    // 编码已经完成模板格式化的完整文本，不再重复应用模板。
    // 识别特殊 token，并按 GGUF 配置添加 BOS/EOS。
    std::vector<int> encode(const std::string& text) const;

    // 整体解码本轮生成的 tokens，过滤不需要展示的特殊 tokens。
    std::string decode(const std::vector<int>& token_ids) const;

private:
    struct Impl;
    const Template& chat_template_;
    std::unique_ptr<Impl> impl_;
};
