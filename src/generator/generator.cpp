#include "generator.h"
#include <iostream>
#include <stdexcept>
#include <chrono>

Generator::Generator(Qwen2Model& model, RunState& state)
    : model_(model), state_(state), sampler_(state, model.config().vocab_size)
{
}

GenerationResult Generator::generate(const std::vector<int>& input_ids,
                                     const GenerationOptions& options)
{
    if (options.max_new_tokens < 0 || options.max_context_tokens <= 0 ||
        options.max_context_tokens > state_.max_seq_len) {
        throw std::invalid_argument("生成长度或上下文上限无效");
    }
    if (input_ids.empty() || input_ids.size() > (size_t)options.max_context_tokens) {
        throw std::invalid_argument("完整历史的 token 数超出上下文上限，请 /clear 后重试");
    }
    for (size_t i = 0; i < input_ids.size(); ++i) {
        if (input_ids[i] < 0 || input_ids[i] >= model_.config().vocab_size) {
            throw std::out_of_range("输入 token ID 超出词表范围");
        }
    }

    GenerationResult result;
    if (options.max_new_tokens == 0) {
        result.stop_reason = STOP_MAX_NEW_TOKENS;
        return result;
    }
    if (input_ids.size() == (size_t)options.max_context_tokens) {
        result.stop_reason = STOP_CONTEXT_LIMIT;
        return result;
    }

    // profile prefill phase 
    auto prefill_start_time = std::chrono::steady_clock::now();
    int position = 0;
    for (size_t i = 0; i < input_ids.size(); ++i) {
        model_.forward(state_, input_ids[i], position);
        ++position;
    }
    auto prefill_end_time = std::chrono::steady_clock::now();
    auto prefill_duration = std::chrono::duration_cast<std::chrono::microseconds>(prefill_end_time - prefill_start_time).count();
    std::cerr << "    Prefill: " << prefill_duration / 1e6 << " s\n";

    // profile decode phase
    auto decode_start_time = std::chrono::steady_clock::now();
    while (true) {
        // `model_.forward(...)`结束时，gpu上state.logits存放着下一个词在词表上的分布
        // 直接基于state.logits进行采样，得到下一个 token ID。
        int token = sampler_.sample();
        if (token < 0 || token >= model_.config().vocab_size) {
            throw std::runtime_error("argmax 返回了无效 token ID");
        }
        if (token == 151643 || token == 151645) {
            result.stop_reason = STOP_EOS;
            break;
        }
        result.tokens.push_back(token);// type of `result` is `GenerationResult`

        // prompt + 新增 tokens 不超过上下文容量；最后一个 token 无需再 forward。
        if (position + 1 >= options.max_context_tokens) {
            result.stop_reason = STOP_CONTEXT_LIMIT;
            break;
        }
        if (result.tokens.size() >= (size_t)options.max_new_tokens) {
            result.stop_reason = STOP_MAX_NEW_TOKENS;
            break;
        }
        model_.forward(state_, token, position);
        ++position;
    }
    auto decode_end_time = std::chrono::steady_clock::now();
    auto decode_duration = std::chrono::duration_cast<std::chrono::microseconds>(decode_end_time - decode_start_time).count();
    std::cerr << "    Decode: " << decode_duration / 1e6 << " s\n";
    return result;

}
