// 迁自原 layer.cu；除接口及 score/softmax 拆分外，保留 kernel 计算逻辑。
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "ops.h"

#include "attention_config.h"

__global__ void weighted_sum(const float* softmax_attn_score, const float* value_cache, float* output, GQAConfig gqa_config){

    int head_id = blockIdx.x;                 // 0~11
    int d = threadIdx.x;                      // 0~127
    int n_kv_heads = gqa_config.kv_head_dim / gqa_config.head_dim;
    int group_id = head_id / (gqa_config.heads_num / n_kv_heads); // same GQA mapping as k

    float accum_sum = 0.0f;
    for (int token = 0; token < gqa_config.seq_len; ++token){
        float score = softmax_attn_score[head_id * gqa_config.max_seq_len + token];
        float value = value_cache[token * gqa_config.kv_head_dim + group_id * gqa_config.head_dim + d];
        accum_sum += score * value;
    }

    // [n_heads, head_dim] flattened -> [dim]
    output[head_id * gqa_config.head_dim + d] = accum_sum;
}

void op_gqa_weighted_sum(const float* probabilities, const float* value_cache,
                         float* output, int n_heads, int n_kv_heads,
                         int head_dim, int seq_len, int max_seq_len)
{
    GQAConfig config;
    config.seq_len = seq_len;
    config.head_dim = head_dim;
    config.max_seq_len = max_seq_len;
    config.layer_kv = nullptr;
    config.heads_num = n_heads;
    config.kv_head_dim = n_kv_heads * head_dim;

    weighted_sum<<<n_heads, head_dim>>>(probabilities, value_cache, output, config);
}
