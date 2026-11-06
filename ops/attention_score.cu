// 迁自原 layer.cu；除接口及 score/softmax 拆分外，保留 kernel 计算逻辑。
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "ops.h"

#include "attention_config.h"

const int WARP_SIZE = 32;

__global__ void attn_score(
    const float* q_rot,
    const float* k_cache,
    float* attn_score,
    GQAConfig gqa_config
){

    // threads local variable
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_number = blockDim.x / WARP_SIZE;
    int n_kv_heads = gqa_config.kv_head_dim / gqa_config.head_dim;
    int group_id = blockIdx.x / (gqa_config.heads_num / n_kv_heads); // which kv head this q head uses
    int score_offset = blockIdx.x * gqa_config.max_seq_len; // this head's row in attn_score / softmax_attn_score

    // 12 heads parrallel
    //      each head:
                // handle warp_number tokens (one per warp) parallely
    for (int token = warp_id; token < gqa_config.seq_len; token += warp_number){
        float accum_sum = 0.0f; 

        // inside the warp: head_dim / 32 turns of accumulation
        for(int d = lane_id; d < gqa_config.head_dim; d += WARP_SIZE){
            accum_sum += q_rot[blockIdx.x * gqa_config.head_dim + d] * k_cache[token * gqa_config.kv_head_dim + group_id * gqa_config.head_dim + d];   

        }
        

        // Warp 内规约：32 → 16 → 8 → 4 → 2 → 1
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            accum_sum += __shfl_down_sync(0xffffffff, accum_sum, offset);
        }

        // 每个 Warp 只有 Lane 0 写入最终结果
        if (lane_id == 0) {
            attn_score[score_offset + token] = accum_sum * (1 / sqrt((float)gqa_config.head_dim));
        }


    }

    // 保留原同步点；后续 softmax 在默认 stream 的下一个 kernel 中执行。
    __syncthreads();

}

void op_gqa_score(const float* query, const float* key_cache, float* scores,
                  int n_heads, int n_kv_heads, int head_dim,
                  int seq_len, int max_seq_len)
{
    GQAConfig config;
    config.seq_len = seq_len;
    config.head_dim = head_dim;
    config.max_seq_len = max_seq_len;
    config.layer_kv = nullptr;
    config.heads_num = n_heads;
    config.kv_head_dim = n_kv_heads * head_dim;

    attn_score<<<n_heads, 256>>>(query, key_cache, scores, config);
}
