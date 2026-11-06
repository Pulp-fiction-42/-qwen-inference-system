#include "session.h"

#include <stdexcept>

Session::Session(const std::string& weights_path, const std::string& vocab_path,
                 const GenerationOptions& options)
    : tokenizer_(chat_template_, vocab_path), model_{}, options_(options)
{
    if (options_.max_new_tokens < 0 || options_.max_context_tokens <= 0 ||
        options_.max_context_tokens > MAX_SEQ_LEN) {
        throw std::invalid_argument("生成长度或上下文上限无效");
    }
    load_model_weights(&model_, weights_path.c_str());
}

Session::~Session()
{
    free_model_weights(&model_);
}

TurnResult Session::run_turn(const std::string& input)
{
    // 只在生成和解码均成功后提交历史；失败不留下半轮对话。
    std::vector<ChatMessage> messages = history_;
    ChatMessage user = {"user", input};
    messages.push_back(user);
    std::string prompt = tokenizer_.apply_chat_template(messages);
    std::vector<int> input_ids = tokenizer_.encode(prompt);
    if (input_ids.size() > (size_t)options_.max_context_tokens) {
        throw std::length_error("完整历史的 token 数超出上下文上限，请 /clear 后重试");
    }

    RunState state = {};
    TurnResult result;
    try {
        load_runstate(&state, &model_.config);
        Qwen2Model qwen(model_);
        Generator generator(qwen, state);
        result.generation = generator.generate(input_ids, options_);
        result.text = tokenizer_.decode(result.generation.tokens);

        // 预算为 0 或 prompt 已填满上下文时，不提交一轮未生成的对话。
        if (!result.generation.tokens.empty() || result.generation.stop_reason == STOP_EOS) {
            ChatMessage assistant = {"assistant", result.text};
            messages.push_back(assistant);
            history_.swap(messages);
        }
    } catch (...) {
        free_runstate(&state);
        throw;
    }
    free_runstate(&state);
    return result;
}

void Session::clear()
{
    history_.clear();
}
