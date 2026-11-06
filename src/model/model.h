#pragma once

#include <vector>

#include "model_types.h"
#include "layer/layer.h"

// 借用已加载的 Model；使用期间不得释放或重新加载权重。
class Qwen2Model {
public:
    explicit Qwen2Model(Model& model);
    const Config& config() const;

    // 单 token forward，结果保存在 GPU 上的 state.logits。
    // Session 必须先准备 RunState，以及 [0, position) 的 KV 历史。
    void forward(RunState& state, int token_id, int position);

private:
    Model& model_;
    std::vector<Qwen2Layer> layers_;
};
