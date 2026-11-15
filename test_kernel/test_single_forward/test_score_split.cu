// 对同一组 Q/K/cache 比较原融合 kernel 与当前拆分的 score/softmax。
// 原融合 kernel 原样保留在这个测试文件中，不接入生产代码。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#include "src/model/layer/layer.h"
#include "ops/ops.h"
#include "ops/attention_config.h"

const int WARP_SIZE = 32;
const int MAX_WARPS_PER_BLOCK = 32;

__global__ void attn_score_softmax(
    const float* q_rot,
    const float* k_cache,
    float* attn_score,         // [n_heads, max_seq_len] raw scaled Q*K^T
    float* softmax_attn_score, // [n_heads, max_seq_len] softmax result
    GQAConfig gqa_config
){

    // threads local variable
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_number = blockDim.x / WARP_SIZE;
    int n_kv_heads = gqa_config.kv_head_dim / gqa_config.head_dim;
    int group_id = blockIdx.x / (gqa_config.heads_num / n_kv_heads); // which kv head this q head uses
    int tokens_per_warp = (gqa_config.max_seq_len + warp_number - 1) / warp_number; // scores one warp covers in softmax
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

    // all warps must finish writing attn_score before softmax reads other warps' scores
    __syncthreads();

    // softmax
    // shared memory for softmax intermediate results
    __shared__ float s_max[MAX_WARPS_PER_BLOCK]; // one per warp
    __shared__ float s_sum[MAX_WARPS_PER_BLOCK]; // one per warp
    int activae_warp_num = (gqa_config.seq_len + tokens_per_warp - 1) / tokens_per_warp; // number of warps actively handling tokens in this head

    // step1 : max-reduce 
    // launch the acivated warp
    if (warp_id < activae_warp_num) {
        // each thread handles tokens_per_warp / 32 elements (stride = 32); elements beyond seq_len are padded with -inf
        float thread_max = -INFINITY;
        for (int d = lane_id; d < tokens_per_warp; d += WARP_SIZE){
            int token_idx = warp_id * tokens_per_warp + d;
            float score = -INFINITY;
            if (token_idx < gqa_config.seq_len) {
                score = attn_score[score_offset + token_idx];
            }
            thread_max = fmaxf(thread_max, score);
        }

        // warp shuffle reduce: 32 -> 16 -> 8 -> 4 -> 2 -> 1, lane 0 gets the warp max
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
            thread_max = fmaxf(thread_max, other_max);
        }

        // lane 0 writes this warp's max to shared memory
        if (lane_id == 0) {
            s_max[warp_id] = thread_max;
        }
    } else {
        // idle warp: write -inf so it never wins the final reduction
        if (lane_id == 0) {
            s_max[warp_id] = -INFINITY;
        }
    }

    // s_max[0..warp_number-1] must be fully written before warp 0 reads it
    __syncthreads();

    // warp 0, first warp_number / 2 threads: tree reduction over s_max[warp_number] -> s_max[0]
    if (warp_id == 0) {
        for (int stride = warp_number / 2; stride > 0; stride /= 2) {
            if (lane_id < stride) {
                s_max[lane_id] = fmaxf(s_max[lane_id], s_max[lane_id + stride]);
            }
            // only warp 0 is involved, so a warp-level barrier between levels is enough
            __syncwarp();
        }
    }
    __syncthreads(); // ensure s_max[0] is fully reduced before using it


    // step2: exp(score - max_score), read raw attn_score and write softmax_attn_score
    float max_score = s_max[0]; // the final max value for this head
    for (int i = threadIdx.x; i < gqa_config.seq_len; i += blockDim.x){
        softmax_attn_score[score_offset + i] = expf(attn_score[score_offset + i] - max_score);
    }
    __syncthreads(); // ensure all threads have written softmax_attn_score before any further use


    // step3: sum-reduce
    if (warp_id < activae_warp_num) {
        // each thread handles tokens_per_warp / 32 elements (stride = 32); elements beyond seq_len are padded with 0
        float accum_sum = 0.0f;
        for (int d = lane_id; d < tokens_per_warp; d += WARP_SIZE){
            int token_idx = warp_id * tokens_per_warp + d;
            float score = 0.0f;
            if (token_idx < gqa_config.seq_len) {
                score = softmax_attn_score[score_offset + token_idx];
            }
            accum_sum += score;
        }

        // warp shuffle reduce: 32 -> 16 -> 8 -> 4 -> 2 -> 1, lane 0 gets the warp sum
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_sum = __shfl_down_sync(0xffffffff, accum_sum, offset);
            accum_sum += other_sum;
        }

        // lane 0 writes this warp's sum to shared memory
        if (lane_id == 0) {
            s_sum[warp_id] = accum_sum;
        }
    } else {
        // idle warp: write 0 so it does not change the final sum
        if (lane_id == 0) {
            s_sum[warp_id] = 0.0f;
        }
    }

    // s_sum[0..warp_number-1] must be fully written before warp 0 reads it
    __syncthreads();

    // warp 0, first warp_number / 2 threads: tree reduction over s_sum[warp_number] -> s_sum[0]
    if (warp_id == 0) {
        for (int stride = warp_number / 2; stride > 0; stride /= 2) {
            if (lane_id < stride) {
                s_sum[lane_id] = s_sum[lane_id] + s_sum[lane_id + stride];
            }
            __syncwarp();
        }
    }
    __syncthreads(); // ensure s_sum[0] is fully reduced before using it


    // step4: normalize softmax_attn_score by the sum
    float sum = s_sum[0]; // the final sum for this head
    for (int i = threadIdx.x; i < gqa_config.seq_len; i += blockDim.x){
        softmax_attn_score[score_offset + i] /= sum;
    }
    __syncthreads(); // ensure all threads have normalized softmax_attn_score before any further use

}

static void check_cuda(cudaError_t error)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "score diagnostic: %s\n", cudaGetErrorString(error));
        exit(1);
    }
}

int main()
{
    Model model = {};
    RunState state = {};
    load_model_weights(&model);
    load_runstate(&state, &model.config);
    Qwen2Layer layer(model.layers[0]);

    int heads = model.config.n_heads;
    int capacity = heads * state.max_seq_len;
    size_t bytes = (size_t)capacity * sizeof(float);
    float* fused_scores;
    float* fused_probabilities;
    check_cuda(cudaMalloc((void**)&fused_scores, bytes));
    check_cuda(cudaMalloc((void**)&fused_probabilities, bytes));
    float* a = (float*)malloc(bytes);
    float* b = (float*)malloc(bytes);
    float* pa = (float*)malloc(bytes);
    float* pb = (float*)malloc(bytes);
    if (!a || !b || !pa || !pb) {
        fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    GQAConfig config;
    config.head_dim = model.config.dim / heads;
    config.max_seq_len = state.max_seq_len;
    config.layer_kv = state.key_cache;
    config.heads_num = heads;
    config.kv_head_dim = config.head_dim * model.config.n_kv_heads;

    bool all_equal = true;
    for (int position = 0; position < 8; ++position) {
        op_embedding(123 + position, model.embedding, model.config.dim, state.x);
        layer.forward(state, model.config, state.x, 0, position);
        op_synchronize();

        config.seq_len = position + 1;
        attn_score_softmax<<<heads, 256>>>(
            state.q_rot, state.key_cache, fused_scores, fused_probabilities, config);
        check_cuda(cudaDeviceSynchronize());
        check_cuda(cudaMemcpy(a, state.attn_score, bytes, cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(b, fused_scores, bytes, cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(pa, state.softmax_attn_score, bytes, cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(pb, fused_probabilities, bytes, cudaMemcpyDeviceToHost));

        int score_mismatches = 0;
        int probability_mismatches = 0;
        float max_score_error = 0.0f;
        float max_probability_error = 0.0f;
        for (int h = 0; h < heads; ++h) {
            for (int t = 0; t <= position; ++t) {
                int i = h * state.max_seq_len + t;
                if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
                    std::memcmp(a + i, b + i, sizeof(float)) != 0) {
                    ++score_mismatches;
                }
                if (!std::isfinite(pa[i]) || !std::isfinite(pb[i]) ||
                    std::memcmp(pa + i, pb + i, sizeof(float)) != 0) {
                    ++probability_mismatches;
                }
                max_score_error = fmaxf(max_score_error, fabsf(a[i] - b[i]));
                max_probability_error = fmaxf(max_probability_error, fabsf(pa[i] - pb[i]));
            }
        }
        printf("position=%d score_bit_mismatches=%d probability_bit_mismatches=%d "
               "max_score_error=%.9g max_probability_error=%.9g\n",
               position, score_mismatches, probability_mismatches,
               max_score_error, max_probability_error);
        if (score_mismatches || probability_mismatches) {
            all_equal = false;
        }
    }

    free(a);
    free(b);
    free(pa);
    free(pb);
    check_cuda(cudaFree(fused_scores));
    check_cuda(cudaFree(fused_probabilities));
    free_runstate(&state);
    free_model_weights(&model);
    return all_equal ? 0 : 1;
}
