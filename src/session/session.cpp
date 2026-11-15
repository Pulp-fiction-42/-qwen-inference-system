#include "session.h"
#include <chrono>
#include <stdexcept>
#include <iostream>

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
    // profile encode time: <拼接对话+应用模版+encode>
    auto encode_start_time = std::chrono::steady_clock::now();
    // 只在生成和解码均成功后提交历史；失败不留下半轮对话。
    // asseemble multi turn chat history 
    std::vector<ChatMessage> messages = history_;
    ChatMessage user = {"user", input};
    messages.push_back(user);
    std::string prompt = tokenizer_.apply_chat_template(messages);
    std::vector<int> input_ids = tokenizer_.encode(prompt);
    auto encode_end_time = std::chrono::steady_clock::now();
    auto encode_duration = std::chrono::duration_cast<std::chrono::microseconds>(encode_end_time - encode_start_time).count();
    std::cerr << "  Tokenizer Encode: " << encode_duration / 1e6 << " s\n";
    std::cerr << "  Prefill tokens: " << input_ids.size() << "\n";

    if (input_ids.size() > (size_t)options_.max_context_tokens) {
        throw std::length_error("完整历史的 token 数超出上下文上限，请 /clear 后重试");
    }

    // profile forward state initialization time for each turn 
    RunState state = {};
    TurnResult result;
    try {  
        auto init_start_time = std::chrono::steady_clock::now();
        load_runstate(&state, &model_.config);
        Qwen2Model qwen(model_);
        auto init_end_time = std::chrono::steady_clock::now();
        auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(init_end_time - init_start_time).count();
        std::cerr << "  Runstate initialization: " << init_duration / 1e6 << " s\n";

        // profile generation time 
        auto start_time = std::chrono::steady_clock::now();
        Generator generator(qwen, state);
        result.generation = generator.generate(input_ids, options_);
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cerr << "  Generate: " << duration / 1e6 << " s\n";


        // profile decode time
        auto decode_start_time = std::chrono::steady_clock::now();
        result.text = tokenizer_.decode(result.generation.tokens);
        auto decode_end_time = std::chrono::steady_clock::now();
        auto decode_duration = std::chrono::duration_cast<std::chrono::microseconds>(decode_end_time - decode_start_time).count();
        std::cerr << "  Tokenizer Decode: " << decode_duration / 1e6 << " s\n";


        // 预算为 0 或 prompt 已达到上下文上限时，不提交一轮未生成的对话。
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
