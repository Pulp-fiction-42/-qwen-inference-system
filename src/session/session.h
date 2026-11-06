#pragma once

#include <string>
#include <vector>
#include "tokenizer.h"
#include "../generator/generator.h"

struct TurnResult {
    std::string text;
    GenerationResult generation;
};

// 权重跨轮复用，历史保存在 CPU；每轮申请并释放独立 RunState。
class Session {
public:
    Session(const std::string& weights_path, const std::string& vocab_path,
            const GenerationOptions& options);
    ~Session();

    TurnResult run_turn(const std::string& input);
    void clear();

private:
    // Session 拥有裸 GPU 资源，禁止复制，避免重复释放。
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    Template chat_template_;  // 必须先于借用它的 Tokenizer 构造。
    Tokenizer tokenizer_;
    Model model_;
    GenerationOptions options_;
    std::vector<ChatMessage> history_;
};
