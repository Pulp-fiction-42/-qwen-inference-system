#include "model.h"
#include "ops/ops.h"

const Config& Qwen2Model::config() const
{
    return model_.config;
}

Qwen2Model::Qwen2Model(Model& model)
    : model_(model)
{
    int layer_count = model_.config.n_layers;
    layers_.reserve(layer_count);

    for (int i = 0; i < layer_count; ++i) {
        Qwen2Layer layer(model_.layers[i]);
        layers_.push_back(layer);
    }
}

void Qwen2Model::forward(RunState& state, int token_id, int position)
{
    const Config& config = model_.config;

    op_embedding(token_id, model_.embedding, config.dim, state.x);

    const float* input = state.x;
    for (int i = 0; i < config.n_layers; ++i) {
        layers_[i].forward(state, config, input, i, position);
        input = state.residual_after_mlp;
    }

    // 单独保留原 final RMSNorm 的启动方式，不在此次重构中修复。
    op_final_rms_norm(model_.final_norm, input, state.final_norm,
                      config.dim, state.rms_scratch, 1e-6f);

    op_lm_head_proj(model_.lm_head, nullptr, state.final_norm,
                    state.logits, config.vocab_size, config.dim, config.dim);

    // 保留原 forward 返回前等待 GPU 完成的行为。
    op_synchronize();
}
