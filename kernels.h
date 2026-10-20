#pragma once

typedef enum {
    REDUCE_SUM,
    REDUCE_MAX
} ReduceOp;

typedef enum {
    ELEMENTWISE_ADD,
    ELEMENTWISE_MUL
} ELEMENTWISEOp;


// these are some universal use operator
__global__ void mat_vec(const float* weights, const float* bias, const float* input, float* output, int w_row, int w_col, int in_dim);
void reduce(float* input, float* output, int in_dim, ReduceOp op);
__device__ void product_reduce(float* input1, float* input2, float* output, int dim);
__global__ void rope(float* freqs, float* q, float* k, int p, float* rot_q, float* rot_k); //(state->q, state->k) => (state->q_rot, state->k_rot)
void elementwise(float* input_1, float* input_2, float* output, int dim, ELEMENTWISEOp op);


// these are some specific layer's operator 
void embedding_lookup(int token, const float* embedding, int dim, float* des);
void rms_norm(const float* rmn_norm_weight, float* input, float* output, int dim, float eps);
void attention(blocks* layer, RunState* state, int layer_index, Config* config, int position);
__global__ void compute_attn_score_softmax(
    const float* q_rot,
    const float* k_cache,
    float* attn_score,
    int seq_len,// p + 1
    int head_dim,
    int stride_q_head,// 128
    int stride_k_token,// 256
    int stride_k_group,// 128
    int stride_score_head
);
void mlp(
    const blocks* layer,
    RunState* state,
    const float* input,   // GPU 输入: after_attn_norm_x [dim]
    float* output,        // GPU 输出: m [dim]
    int dim,              // 1536
    int hidden_dim        // 8960
);


