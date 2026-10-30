// copy the dim into state->x 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "forward.h"
#include "model.h"
#include "layer.h"


__global__ void embedding_lookup(
    int token_id,
    const float* embedding,  // GPU 权重表：[vocab_size, dim]
    int dim,                 // 一行的长度
    float* des           // [dim]，写入 state->x
){
    const float* base_ptr = embedding + (size_t)token_id * dim;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dim)
        des[tid] = base_ptr[tid];

}

// 1 block, 256 threads, each threads handle 6 elements
// step1: reduce to get `r`
// step2; element wise: yi = xi * r * Wi
__global__ void rms_norm(const float* rmn_norm_weight, float* input, float* output, int dim, float* r, float eps){
    
    // reduce 1536 dim inputs -> rms, r = 1 / (sqrt(rms + eps))
    __shared__ float s_data[256];
    int tid = threadIdx.x;
    float six_number_sum = 0.0f;
    

    for(int i = tid; i < dim; i += blockDim.x){
        six_number_sum += input[i] * input[i];
        // printf("thread: %d six_number_sum: %f\n", tid, six_number_sum);
    }
    s_data[tid] = six_number_sum;
    __syncthreads();


    // reduce, totally 8 times
    for(int stride = 1; stride < (blockDim.x); stride *= 2){
        if (tid % (2 * stride) == 0){
            s_data[tid] += s_data[tid + stride];
            // printf("s_data[%d] after stride %d: %f\n", tid, stride, s_data[tid]);
        }
        __syncthreads();
    }
    // compute the `r` based on `eps` and `rms` and write to *rms by thread0
    if (tid == 0){
        // printf("rms is %f\n", s_data[0]);
        *r = 1.0f / sqrtf(s_data[0] / dim + eps);
        // printf("r is %f\n", *r);
    }
    __syncthreads();

    // element wise 
    for (int i = tid; i < dim; i += blockDim.x){
        output[i] = input[i] * (*r) * rmn_norm_weight[i];
        /*
        if (tid == 0){
            printf("the first input is this turn is: %f\n", input[i]);
        }        
        */
    }
}






// for QKV projection
// <<<6, 256>>>, each thread handles one row of the output
__global__ void proj_bias(const float* weights, const float* bias, const float* input, float* output, int dim){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid; // each thread handle one row's (product + bias)
    float sum = 0.0f;

    for (int i = 0; i < dim; ++i){
        sum += weights[row * dim + i] * input[i];
    }
    sum += bias[row];
    output[row] = sum;
 
}   


// rope: <<<haed-num, head-elements / 2>>> <=> q: <<<12, 64>>>, k: <<<2, 64>>>
/*
each block represents one head 
each thread handles one group(eg: (0, 64), (1, 65), (2, 66), ...(63, 127))
*/
__global__ void rope(
    float*  q_or_k,
    float*  rot_q_or_k,
    int position,
    float theta
    )

{
    // group variable for each thread 
    int group_stride = blockDim.x; // 64
    int group_id = threadIdx.x % group_stride;
    
    // head variable for each thread
    int head_stride = blockDim.x * 2; // 128
    int global_idx = blockIdx.x * head_stride + threadIdx.x;

    // each thread handles one group's rotation
    float frequency = 1.0f / powf(theta, 2.0f * group_id / 128.0f);
    float angle = position * frequency;

    rot_q_or_k[global_idx] =
        q_or_k[global_idx] * cosf(angle)
        - q_or_k[global_idx + group_stride] * sinf(angle);

    rot_q_or_k[global_idx + group_stride] =
        q_or_k[global_idx] * sinf(angle)
        + q_or_k[global_idx + group_stride] * cosf(angle);
}



//  strategy: <<<heads number, 256>>>
//  attn_score:
//      each warp handles one token. so for each head(block), needs '(p + 1) / warp-number(8)' turn to handles all tokens in sequence
//  softmax: each block handle 1 head, for each head(p + 1): max-reduce -> element-wise(`score - max`) -> sum-reduce -> element-wise normalization
//      inside the block, each warp handle `tokens_per_warp` scores (= max_seq_len / warp_number, e.g. 2048 / 8 = 256)
//          inside the warp, each thread handle `tokens_per_warp / 32` elements (e.g. 256 / 32 = 8)
//  requirement: blockDim.x is a multiple of 32, blockDim.x / 32 is a power of 2 and <= MAX_WARPS_PER_BLOCK
const int WARP_SIZE = 32;
const int MAX_WARPS_PER_BLOCK = 32; // 1024 threads / 32
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




// weighted sum 
// strategy: <<<heads number, head_dim(128)>>>
// each block handles one head, each thread handles one output dim `d`
// so a thread walks over all tokens: output[d] = sum_t softmax_attn_score[t] * v[t][d]
__global__ void weighted_sum(float* softmax_attn_score, const float* value_cache, float* output, GQAConfig gqa_config){

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


//  projection without bias for attention layer's final output
//  each thread handles one row of the output
__global__ void proj(const float* weights, const float* input, float* output, int dim){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid; // each thread handle one row's product
    float sum = 0.0f;

    for (int i = 0; i < dim; ++i){
        sum += weights[row * dim + i] * input[i];
    }
    output[row] = sum;
}


void attention(blocks* layer, RunState* state, int layer_index, Config* config, int position){

    // parameters for GQA
    size_t layer_offset = layer_index * state->max_seq_len * (config->n_kv_heads * (config->dim / config->n_heads));// for lacate the v and k
    float* layer_values = state->value_cache + layer_offset;
    float* current_value = layer_values + position * (config->n_kv_heads * (config->dim / config->n_heads));
    float* layer_keys = state->key_cache + layer_offset;
    float* current_key = layer_keys + position * (config->n_kv_heads * (config->dim / config->n_heads));


  // initialize GQA config
    GQAConfig gqa_k_config = {
        .seq_len = position + 1,
        .head_dim = config->dim / config->n_heads,
        .max_seq_len = state->max_seq_len,
        .layer_kv = layer_keys,
        .heads_num = config->n_heads,
        .kv_head_dim = config->dim /  config->n_heads * config->n_kv_heads
    };

    GQAConfig gqa_v_config = {
        .seq_len = position + 1,
        .head_dim = config->dim / config->n_heads,
        .max_seq_len = state->max_seq_len,
        .layer_kv = layer_values,
        .heads_num = config->n_heads,
        .kv_head_dim = config->dim /  config->n_heads * config->n_kv_heads
    };


    cudaStream_t computation_stream , v_to_v_cache_stream;
    cudaError_t err = cudaStreamCreate(&computation_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "create computation stream failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaStreamCreate(&v_to_v_cache_stream); 
    if (err != cudaSuccess) {
        fprintf(stderr, "create v_to_v_cache stream failed: %s\n", cudaGetErrorString(err));
    }

    // Q, K, V projection
    proj_bias<<<dim3(6), dim3(256), 0, computation_stream>>>(layer->wq, layer->bq, state->before_attn_norm_x, state->q, config->dim);
    proj_bias<<<dim3(6), dim3(256), 0, computation_stream>>>(layer->wk, layer->bk, state->before_attn_norm_x, state->k, config->dim);
    proj_bias<<<dim3(6), dim3(256), 0, computation_stream>>>(layer->wv, layer->bv, state->before_attn_norm_x, state->v, config->dim);
    cudaStreamSynchronize(computation_stream);




    // reshape for `q`, `k`, `v`
    // but, it's just logical shape...hhh, the reshape is in my mental 


    // rope for q, k 
    rope<<<12, 64, 0, computation_stream>>>(state->q, state->q_rot, position);
    rope<<<2, 64, 0, computation_stream>>>(state->k, state->k_rot, position);
    cudaStreamSynchronize(computation_stream);


    /*
    rope之后的计算并行2条线。
        computation：k_rot write into k_cache -> softmax(Q*KT) and get attn_score (raw) and softmax_attn_score
        v_to_v_cache：v write into v-cache
    synchronize streams before weighted sum
        weighted_sum =  softmax_attn_score([12, p+1]) * v([p+1], 2, 128])
            for each head: [p+1, 1] * [1, 128] -> [p+1, 128]
            combine results from all heads to get [12, 128] -> [1536]
    */

    /*
    1. current v to v cache stream
    */
    err = cudaMemcpyAsync(current_value, state->v, (config->n_kv_heads) * (config->dim / config->n_heads) * sizeof(float), cudaMemcpyDeviceToDevice, v_to_v_cache_stream);
    if (err != cudaSuccess) {
    fprintf(stderr, "copy v to v-cache failed: %s\n", cudaGetErrorString(err));
    }


    /*
    2. GQA computation stream
    */

    // firstly, write k into k-cache
    err = cudaMemcpyAsync(current_key, state->k_rot, (config->n_kv_heads * (config->dim / config->n_heads)) * sizeof(float), cudaMemcpyDeviceToDevice, computation_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "copy k to K-cache failed: %s\n", cudaGetErrorString(err));
    }


    // compute attn_score and softmax_attn_score
    attn_score_softmax<<<dim3(config->n_heads), dim3(256), 0, computation_stream>>>(
        state->q_rot,
        state->key_cache + layer_offset,
        state->attn_score,
        state->softmax_attn_score,
        gqa_k_config
    );


    /*
    3. synchronize streams before weighted sum
    */
    cudaStreamSynchronize(computation_stream);
    cudaStreamSynchronize(v_to_v_cache_stream);

    weighted_sum<<<dim3(config->n_heads), dim3(config->dim / config->n_heads), 0, computation_stream>>>(
        state->softmax_attn_score,
        state->value_cache + layer_offset,
        state->weighted_value,
        gqa_v_config
    );


    /*
    4. projection 
    */
    proj<<<dim3(6), dim3(256), 0, computation_stream>>>(
        layer->wo,
        state->weighted_value,
        state->attn_o,
        config->dim
    );

    // later kernels run on the default stream, so wait before destroying
    cudaStreamSynchronize(computation_stream);
    cudaStreamDestroy(computation_stream);
    cudaStreamDestroy(v_to_v_cache_stream);

}


// after attention, before MLP
__global__ void attn_residual(const float* x, const float* attn_o, float* residual_after_attn, int dim){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dim){
        residual_after_attn[tid] = x[tid] + attn_o[tid];
    }
}


/*
grid: (ceil(out_dim / 256), 1, 1), block: (256, 1, 1), 256 threads per block.
gate/up: <<<35, 256>>> for 8960 rows; down: <<<6, 256>>> for 1536 rows.
Each block handles up to 256 consecutive output rows.
Each thread handles one row: loop over in_dim, accumulate weights[row, i] * input[i],
then write one output[row]. No bias or reduction between threads; extra threads skip.
*/
__global__ void mlp_proj(
    const float* weights, // [out_dim, in_dim]
    const float* input,   // [in_dim]
    float* output,        // [out_dim]
    int out_dim,
    int in_dim
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid;

    if (row < out_dim){
        float sum = 0.0f;

        for (int i = 0; i < in_dim; ++i){
            sum += weights[row * in_dim + i] * input[i];
        }
        output[row] = sum;
    }
}


__global__ void silu(const float* g, float* s, int hidden_dim){
    // g, s: [hidden_dim]
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < hidden_dim){
        float gate = g[tid];
        float sigmoid = 1.0f / (1.0f + expf(-gate));
        s[tid] = gate * sigmoid;
    }
}


__global__ void elementwise_mul(const float* s, const float* b, float* f, int hidden_dim){
    // s, b, f: [hidden_dim]
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < hidden_dim){
        f[tid] = s[tid] * b[tid];
    }
}


// MLP: host launches kernels in order on the default stream
void mlp(
    const blocks* layer,
    RunState* state,
    const float* input,   // GPU 输入: after_attn_norm_x [dim]
    float* output,        // GPU 输出: m [dim]
    int dim,              // 1536
    int hidden_dim        // 8960
){
    int threads_per_block = 256;
    int hidden_blocks_per_grid = (hidden_dim + threads_per_block - 1) / threads_per_block;
    int dim_blocks_per_grid = (dim + threads_per_block - 1) / threads_per_block;

    // gate/up: [hidden_dim, dim] * [dim] -> [hidden_dim]
    mlp_proj<<<hidden_blocks_per_grid, threads_per_block>>>(layer->w_gate, input, state->g, hidden_dim, dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "gate projection launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    mlp_proj<<<hidden_blocks_per_grid, threads_per_block>>>(layer->w_up, input, state->b, hidden_dim, dim);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "up projection launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    silu<<<hidden_blocks_per_grid, threads_per_block>>>(state->g, state->s, hidden_dim);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "silu launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    elementwise_mul<<<hidden_blocks_per_grid, threads_per_block>>>(state->s, state->b, state->f, hidden_dim);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "elementwise multiply launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // down: [dim, hidden_dim] * [hidden_dim] -> [dim]
    mlp_proj<<<dim_blocks_per_grid, threads_per_block>>>(layer->w_down, state->f, output, dim, hidden_dim);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "down projection launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}


__global__ void mlp_residual(const float* residual_after_attn, const float* m, float* residual_after_mlp, int dim){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dim){
        residual_after_mlp[tid] = residual_after_attn[tid] + m[tid];
    }
}


/*
Launch to use: grid (ceil(w_row / 256), 1, 1), block (256, 1, 1), 256 threads per block.
For 151936 vocabulary rows: <<<594, 256>>>; forward.cu still needs this launch config.
Each block handles up to 256 consecutive vocabulary rows (128 in the last block).
Each thread handles one logit: loop over in_dim (1536), accumulate weights[row, i] * input[i],
add bias[row] if present, then write output[row]. Row stride is w_col (= in_dim).
No reduction between threads; threads with row >= w_row skip.
*/
__global__ void lm_head_proj(const float* weights, const float* bias, const float* input, float* output, int w_row, int w_col, int in_dim){
    // weights: [w_row, w_col], input: [in_dim], output/bias: [w_row]
    // w_col == in_dim; Qwen LM head has no bias, so pass NULL
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid;

    if (row < w_row){
        float sum = 0.0f;

        for (int i = 0; i < in_dim; ++i){
            sum += weights[row * w_col + i] * input[i];
        }
        if (bias != NULL){
            sum += bias[row];
        }
        output[row] = sum;
    }
}








// 2 kernels for sample_argmax


/*
`kernel_1` config as <<<151936 / 256 + 1, 256>>>
Compare the values ​​of 2 threads, reduce from 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
finally, each block write the maximum value to global memory for `kernel 2` to find the overall maximum of 256 produced by kernel 1
*/
__global__ void kernel_1(float* logits, float* kernel_1_output_logits, int* kernel_1_output_idx, int vocab_size){
    
    // move 256 logits from VRAM -> l1 cache (shared memory)
    // 用到shared memory是因为每个threads的寄存器都是私有的    
    __shared__ float s_logits[256];
    __shared__ int s_idx[256]; 

    // move data from global memory to shared memory
    int block_offset = blockIdx.x * blockDim.x;
    int global_idx = block_offset + threadIdx.x;
    if (global_idx < vocab_size){
        s_logits[threadIdx.x] = logits[global_idx];
        s_idx[threadIdx.x] = global_idx; 
    }else{
        s_logits[threadIdx.x] = -INFINITY; // final block
    }
    __syncthreads();



    // 每个block内负责: 256->128->64->32->16->8->4->2->1
    // 共8轮规约
    // 每轮规约: 每个线程与相邻线程比较取最大值，并将最大值放在比较组的前面位置
    // 每轮实例：turn1: (0, 1); turn2: (0, 2); turn3: (0, 4); ...
    // 每轮的t_stride(表示当前轮次：2个做比较线程相邻的跨度): 1, 2, 4, 8, 16, 32, 64, 128
    int tid = threadIdx.x;
    for (int t_stride = 1; t_stride < blockDim.x; t_stride *= 2){
        if (tid % (2 * t_stride) == 0){ // only selected threads 
        
            if (s_logits[tid] < s_logits[tid + t_stride]){
                s_logits[tid] = s_logits[tid + t_stride];
                s_idx[tid] = s_idx[tid + t_stride];
            }
        }
        __syncthreads(); // synchronize after each reduction step
    }
    // write the maximum value and it's index of each block to global memory by `thread 0`
    if (threadIdx.x == 0){
        kernel_1_output_logits[blockIdx.x] = s_logits[0];
        kernel_1_output_idx[blockIdx.x] = s_idx[0];
    }
}


/*
kernel 2: find the overall maximum value's index from the outputs of kernel 1
config: <<<1, 640>>>, 640 threads in each block to cover the number of blocks of kernel 1: 594
*/

__global__ void kernel_2(float* kernel_1_output_logits, int* kernel_1_output_idx, int* d_max_value_index){

    __shared__ float s_max[640];
    __shared__ int s_idx[640];

    // move data form glolbal memory to shared memory
    int tid = threadIdx.x;
    if (tid < 640){
        s_max[tid] = kernel_1_output_logits[tid];
        s_idx[tid] = kernel_1_output_idx[tid];
    }else{
        s_max[tid] = -INFINITY;
    }

    __syncthreads(); 

    // 每个block内规约找最大值
    for (int t_stride = 1; t_stride < 640; t_stride *= 2) {
            // 只有当前跨度内的起始线程执行，且必须严格限制目标地址小于 640
            if (tid % (2 * t_stride) == 0) {
                int target_idx = tid + t_stride;
                if (target_idx < 640) { // 严格防止尾部越界
                    if (s_max[tid] < s_max[target_idx]) {
                        s_max[tid] = s_max[target_idx];
                        s_idx[tid] = s_idx[target_idx];
                    }
                }
            }
            __syncthreads();
        }
    
    if (tid == 0){
        *d_max_value_index = s_idx[0];
    }
}