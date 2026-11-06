#pragma once

#include <vector>
#include "../model/model.h"
#include "sampler/sampler.h"

struct GenerationOptions {
    int max_new_tokens = 256;
    int max_context_tokens = MAX_SEQ_LEN;
};

enum StopReason {
    STOP_EOS,
    STOP_MAX_NEW_TOKENS,
    STOP_CONTEXT_LIMIT
};

struct GenerationResult {
    std::vector<int> tokens;  // 仅新增 tokens，不含 prompt 和结束 token。
    StopReason stop_reason = STOP_EOS;
};

// 借用模型与本轮 RunState，不申请或释放 GPU 内存。
class Generator {
public:
    Generator(Qwen2Model& model, RunState& state);
    GenerationResult generate(const std::vector<int>& input_ids,
                              const GenerationOptions& options);

private:
    Qwen2Model& model_;
    RunState& state_;
    Sampler sampler_;
};
