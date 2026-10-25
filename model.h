#pragma once
#include <cstdint>
constexpr int MAX_SEQ_LEN = 2048;
struct Config {
    int dim;          // 隐藏维度：1536
    int hidden_dim;   // MLP 中间维度：8960
    int n_layers;     // blocks 层数：28
    int n_heads;      // Q 注意力头数：12
    int n_kv_heads;   // K/V 注意力头数：2
    int vocab_size;   // 词表大小：151936
};

struct blocks {
    const float* before_attn_norm;

    const float* wq;
    const float* bq;
    const float* wk;
    const float* bk;
    const float* wv;
    const float* bv;
    const float* wo; // final step of atten layer 

    // MLP
    const float* post_attn_norm;

    const float* w_gate;
    const float* w_up;
    const float* w_down;
};

struct Model {
    Config config;

    const float* embedding;
    blocks* layers;
    const float* final_norm;
    const float* lm_head;// [dim, vocab_size]，LM head 权重矩阵

    void* device_data;
    std::size_t mapped_size;
};

struct RunState {
    float* device_data; // GPU memory base pointer
    int max_seq_len; // allocate the max memory of KV-cache

    // 1. Before attention
    float* x;                    // [dim]，当前 block 的输入
    float* before_attn_norm_x;    // [dim]，RMSNorm 1 输出

    // 2. Attention
    float* q;                    // [dim]，Q projection 输出
    float* k;                    // [kv_dim]，K projection 输出
    float* v;                    // [kv_dim]，V projection 输出

    //这里跳过了给reshape分配空间，直接用..._rot表示

    float* q_rot;                // [dim]，RoPE(Q) 输出
    float* k_rot;                // [kv_dim]，RoPE(K) 输出

    float* key_cache;            // [n_layers, max_seq_len, kv_dim]
    float* value_cache;          // [n_layers, max_seq_len, kv_dim]

    float* attn_score;           // [n_heads, max_seq_len]，缩放后的 QKᵀ 
    float* softmax_attn_score;   // [n_heads, max_seq_len]，缩放后的 QKᵀ
    float* weighted_value;       // [n_heads, 128]，每个 Q head 的维度
    float* attn_o;               // [dim]，wo * (1536)

    // 3. After attention
    float* residual_after_attn;  // [dim]，x + attn_o
    float* after_attn_norm_x;    // [dim]，RMSNorm 2 输出

    // 4. MLP
    float* g;                    // [hidden_dim]，gate projection 输出
    float* b;                    // [hidden_dim]，up projection 输出
    float* s;                    // [hidden_dim]，SiLU(g)
    float* f;                    // [hidden_dim]，s ⊙ b
    float* m;                    // [dim]，down projection 输出

    // 5. After MLP
    float* residual_after_mlp;   // [dim]，residual_after_attn + m

    // 6. After all blocks
    float* final_norm;           // [dim]，Final RMSNorm 输出
    float* logits;               // [vocab_size]，LM head 输出
};
// 根据当前项目设计，RunState 已涵盖推理过程中所需的全部中间状态，无需额外的状态结构体。

void load_model_weights(Model* model);
void load_runstate(RunState* runstate, Config* config);
void free_model_weights(Model* model);
void free_runstate(RunState* runstate);