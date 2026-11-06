#pragma once

#include "../model_types.h"

// 一个对象对应一个 decoder block，只引用这一层的权重。
class Qwen2Layer {
public:
    explicit Qwen2Layer(const blocks& weights);

    // input 是 GPU 输入，输出保存在 state.residual_after_mlp。
    void forward(
        RunState& state,
        const Config& config,
        const float* input,
        int layer_index,
        int position
    );

private:
    const blocks& weights_;
};
