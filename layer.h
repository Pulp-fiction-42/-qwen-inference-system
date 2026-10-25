#pragma once
// GQA config 
// for `attn_score_softmax` and `weighted_sum` kernels casue they need these config to do GQA 
struct GQAConfig {
    int seq_len;
    int stride_q_head;
    int stride_kv_token; // 256
    int stride_kv_group; // 128
    int stride_score_head;
    int max_seq_len;
    float* layer_kv;
};


// forward函数所用算子
__global__ void embedding_lookup(int token_id, const float* embedding, int dim, float* des);
__global__ void rms_norm(const float* rmn_norm_weight, float* input, float* output, int dim, float* rms, float eps);
__global__ void proj_bias(const float* weights, const float* bias, const float* input, float* output, int dim);
__global__ void proj(const float* weights, const float* input, float* output, int dim);
__global__ void rope(float* input, float* rot_input, int position, float theta = 1000000.0); //(state->q, state->k) => (state->q_rot, state->k_rot)
__global__ void attn_score_softmax(
    const float* q_rot,
    const float* k_cache,
    float* attn_score,
    GQAConfig gqa_config
);
__global__ void weighted_sum(float* attn_score, const float* value_cache, float* output, GQAConfig gqa_config);
__global__ void attn_residual(const float* x, const float* attn_o, float* residual_after_attn, int dim);
__global__ void mlp_residual(const float* residual_after_attn, const float* m, float* residual_after_mlp, int dim);
__global__ void lm_head_proj(const float* weights, const float* bias, const float* input, float* output, int w_row, int w_col, int in_dim);

void attention(blocks* layer, RunState* state, int layer_index, Config* config, int position);
void mlp(
    const blocks* layer,
    RunState* state,
    const float* input,   // GPU 输入: after_attn_norm_x [dim]
    float* output,        // GPU 输出: m [dim]
    int dim,              // 1536
    int hidden_dim        // 8960
);


// sample argamax所用算子
__global__ void kernel_1(float* logits, float* kernel_1_output, int* kernel_1_output_idx, int vocab_size);
__global__ void kernel_2(float* kernel_1_output_logits, int* kernel_1_output_idx, int* d_max_value_index);