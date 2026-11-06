// 迁自原 layer.cu；除接口及 score/softmax 拆分外，保留 kernel 计算逻辑。
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "ops.h"

const int WARP_SIZE = 32;
const int MAX_WARPS_PER_BLOCK = 32;

// 从 attn_score_softmax 拆出的原 softmax 计算段。
// valid_columns 是有效列数；row_stride 是原 max_seq_len 存储跨度。
__global__ void softmax(
    const float* attn_score,
    float* softmax_attn_score,
    int valid_columns,
    int row_stride
){
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_number = blockDim.x / WARP_SIZE;
    int tokens_per_warp = (row_stride + warp_number - 1) / warp_number;
    int score_offset = blockIdx.x * row_stride;

    // softmax
    // shared memory for softmax intermediate results
    __shared__ float s_max[MAX_WARPS_PER_BLOCK]; // one per warp
    __shared__ float s_sum[MAX_WARPS_PER_BLOCK]; // one per warp
    int activae_warp_num = (valid_columns + tokens_per_warp - 1) / tokens_per_warp; // number of warps actively handling tokens in this head

    // step1 : max-reduce 
    // launch the acivated warp
    if (warp_id < activae_warp_num) {
        // each thread handles tokens_per_warp / 32 elements (stride = 32); elements beyond seq_len are padded with -inf
        float thread_max = -INFINITY;
        for (int d = lane_id; d < tokens_per_warp; d += WARP_SIZE){
            int token_idx = warp_id * tokens_per_warp + d;
            float score = -INFINITY;
            if (token_idx < valid_columns) {
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
    for (int i = threadIdx.x; i < valid_columns; i += blockDim.x){
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
            if (token_idx < valid_columns) {
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
    for (int i = threadIdx.x; i < valid_columns; i += blockDim.x){
        softmax_attn_score[score_offset + i] /= sum;
    }
    __syncthreads(); // ensure all threads have normalized softmax_attn_score before any further use

}

void op_softmax(const float* input, float* output,
                int rows, int valid_columns, int row_stride)
{
    softmax<<<rows, 256>>>(input, output, valid_columns, row_stride);
}
