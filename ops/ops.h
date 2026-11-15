#pragma once

// 所有指针都指向 GPU 内存。所有操作使用默认 stream。
// 此接口只使用指针和尺寸，不依赖 Model、blocks、RunState 或 Config。
// 未修复原 kernel 的尺寸限制；具体限制见各 .cu 文件。

void op_embedding(int token_id, const float* embedding, int dim, float* output);

void op_rms_norm(const float* weights, const float* input, float* output,
                 int dim, float* scratch, float eps);

// 保留原 final RMSNorm 的 6 个 block 启动方式。
void op_final_rms_norm(const float* weights, const float* input, float* output,
                       int dim, float* scratch, float eps);

// 保留原来不同的投影 kernel，此次仅迁移，不合并实现。
void op_proj_bias(const float* weights, const float* bias,
                  const float* input, float* output, int dim);
void op_proj(const float* weights, const float* input, float* output, int dim);
void op_mlp_proj(const float* weights, const float* input, float* output,
                 int out_dim, int in_dim);
void op_lm_head_proj(const float* weights, const float* bias,
                     const float* input, float* output,
                     int w_row, int w_col, int in_dim);

void op_rope(const float* input, float* output, int position,
             int heads, int head_dim, float theta);

// scores/probabilities 布局为 [n_heads, max_seq_len]。
// seq_len 是有效历史长度；max_seq_len 是每行实际存储跨度。
void op_gqa_score(const float* query, const float* key_cache, float* scores,
                  int n_heads, int n_kv_heads, int head_dim,
                  int seq_len, int max_seq_len);
void op_softmax(const float* input, float* output,
                int rows, int valid_columns, int row_stride);
void op_gqa_weighted_sum(const float* probabilities, const float* value_cache,
                         float* output, int n_heads, int n_kv_heads,
                         int head_dim, int seq_len, int max_seq_len);

void op_attn_residual(const float* input, const float* attention,
                      float* output, int dim);
void op_mlp_residual(const float* residual, const float* mlp,
                     float* output, int dim);
void op_silu(const float* input, float* output, int dim);
void op_mul(const float* left, const float* right, float* output, int dim);

// count 的单位为 float 元素个数，不申请或释放内存。
void op_copy(float* destination, const float* source, int count);
void op_synchronize();

// workspace 由调用方持有；长度分别为 ceil(vocab_size / 256)、同上、1。
// 返回 CPU 上的 token ID；保留原 argmax kernel 和默认 stream 启动方式。
int op_argmax(float* logits, int vocab_size, float* block_values,
              int* block_indices, int* token_index);
