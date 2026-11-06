#include "generator.h"

#include <stdexcept>

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

    // 每轮从 position 0 重新计算完整 prompt，逐个写入所有层的 KV cache。
    int position = 0;
    for (size_t i = 0; i < input_ids.size(); ++i) {
        model_.forward(state_, input_ids[i], position);
        ++position;
    }

    while (true) {
        // 最后一个 prompt token 的 logits 用来预测第一个新增 token。
        int token = sampler_.sample();
        if (token < 0 || token >= model_.config().vocab_size) {
            throw std::runtime_error("argmax 返回了无效 token ID");
        }
        if (token == 151643 || token == 151645) {
            result.stop_reason = STOP_EOS;
            return result;
        }
        result.tokens.push_back(token);

        // prompt + 新增 tokens 不超过上下文容量；最后一个 token 无需再 forward。
        if (position + 1 >= options.max_context_tokens) {
            result.stop_reason = STOP_CONTEXT_LIMIT;
            return result;
        }
        if (result.tokens.size() >= (size_t)options.max_new_tokens) {
            result.stop_reason = STOP_MAX_NEW_TOKENS;
            return result;
        }
        model_.forward(state_, token, position);
        ++position;
    }
}
