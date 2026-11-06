#include "layer.h"

#include <cstddef>

#include "ops/ops.h"

Qwen2Layer::Qwen2Layer(const blocks& weights)
    : weights_(weights)
{
}

void Qwen2Layer::forward(
    RunState& state,
    const Config& config,
    const float* input,
    int layer_index,
    int position
) {
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int head_dim = dim / config.n_heads;
    int kv_dim = config.n_kv_heads * head_dim;
    int seq_len = position + 1;

    // 本层 cache 的布局仍为 [max_seq_len, kv_dim]。
    std::size_t layer_offset =
        (std::size_t)layer_index * state.max_seq_len * kv_dim;

    float* layer_keys = state.key_cache + layer_offset;
    float* layer_values = state.value_cache + layer_offset;
    float* current_key = layer_keys + (std::size_t)position * kv_dim;
    float* current_value = layer_values + (std::size_t)position * kv_dim;

    // 1. Attention 前的 RMSNorm。
    op_rms_norm(weights_.before_attn_norm, input, state.before_attn_norm_x,
                dim, state.rms_scratch, 1e-6f);

    // 2. Q、K、V 投影，分别沿用原 proj_bias kernel。
    op_proj_bias(weights_.wq, weights_.bq, state.before_attn_norm_x, state.q, dim);
    op_proj_bias(weights_.wk, weights_.bk, state.before_attn_norm_x, state.k, dim);
    op_proj_bias(weights_.wv, weights_.bv, state.before_attn_norm_x, state.v, dim);

    // 3. RoPE。原 kernel 内部仍假定 head_dim 为 128。
    op_rope(state.q, state.q_rot, position, config.n_heads, head_dim, 1000000.0f);
    op_rope(state.k, state.k_rot, position, config.n_kv_heads, head_dim, 1000000.0f);

    // 4. 将当前 token 的 K、V 写入对应层的 cache。
    op_copy(current_key, state.k_rot, kv_dim);
    op_copy(current_value, state.v, kv_dim);

    // 5. GQA score 和 softmax 分开调用，在默认 stream 上顺序执行。
    op_gqa_score(state.q_rot, layer_keys, state.attn_score,
                 config.n_heads, config.n_kv_heads, head_dim,
                 seq_len, state.max_seq_len);

    op_softmax(state.attn_score, state.softmax_attn_score,
               config.n_heads, seq_len, state.max_seq_len);

    op_gqa_weighted_sum(state.softmax_attn_score, layer_values,
                        state.weighted_value, config.n_heads,
                        config.n_kv_heads, head_dim, seq_len, state.max_seq_len);

    // 6. 输出投影和第一次残差相加。
    op_proj(weights_.wo, state.weighted_value, state.attn_o, dim);
    op_attn_residual(input, state.attn_o, state.residual_after_attn, dim);

    // 7. MLP 前的 RMSNorm。
    op_rms_norm(weights_.post_attn_norm, state.residual_after_attn,
                state.after_attn_norm_x, dim, state.rms_scratch, 1e-6f);

    // 8. Gate / up 投影，然后计算 SiLU(gate) * up。
    op_mlp_proj(weights_.w_gate, state.after_attn_norm_x, state.g, hidden_dim, dim);
    op_mlp_proj(weights_.w_up, state.after_attn_norm_x, state.b, hidden_dim, dim);
    op_silu(state.g, state.s, hidden_dim);
    op_mul(state.s, state.b, state.f, hidden_dim);

    // 9. Down 投影和第二次残差相加。
    op_mlp_proj(weights_.w_down, state.f, state.m, dim, hidden_dim);
    op_mlp_residual(state.residual_after_attn, state.m, state.residual_after_mlp, dim);
}
