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


__global__ void rms_norm(const float* rmn_norm_weight, float* input, float* output, int dim, float* rms, float eps){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // normalization
    // x' = x / RMS
    // out = x' * weight
    output[tid] = input[tid] / *(rms) * rmn_norm_weight[tid];

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


// <<< 12, 64 >>>, each block handle one head, for each head, there are 64 groups, so use 64 threads in each block
/*
each thread:
1. local variable: i(group), freq, angle, position
2. rotate for this group, eg: group 0: (q[0], q[64])
    2.1. compute: qi' = qi * cos(angle) - q(i+ 64) * sin(angle)
    2.2. compute: q(i+64)' = qi * sin(angle) + q(i+64) * cos(angle)
    "for 2.1 and 2.2, each warp just do one 内存事务 (memory transaction), cause the cache line is 128 bytes,
    and after warp0 execute the `qi' = qi * cos(angle) - q(i+ 64) * sin(angle)`, the total 128 bytes of q-head
    will store in cache" 
*/
__global__ void rope(float* input, float* rot_input, int position, float theta){
    // assign local variable to each thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float freq = powf(theta, -2.0f * i / 128.0f);
    float angle = position * freq;

    // rotate and write to rot_input
    rot_input[i] = input[i] * cosf(angle) - input[i + 64] * sinf(angle);
    rot_input[i + 64] = input[i] * sinf(angle) + input[i + 64] * cosf(angle);

}


//  strategy: 
//      each `block` handle one head
//      in each `block`,there are "p + 1" threads, each threads handle one token's score which is: (1, 128) * (128, 1) -> 1
//      for the 128-product above: q-head * k-head^T (should consider use which head in one group based on GQA)
__global__ void attn_score_softmax(
const float* q_rot,
const float* k_cache,
float* attn_score,
GQAConfig gqa_config
)
    {
    // the must local variable for each thread
    int token_pos = threadIdx.x; 
    int current_h = blockIdx.x;
    int group_for_each_h = current_h / 6;
    float product_sum = 0.0f;


    // (1, 128) * (128, 1)
    for (int i = 0; i < 128; ++i){
        product_sum += q_rot[(current_h * gqa_config.stride_q_head) + i] * k_cache[(token_pos * gqa_config.stride_kv_token) + (group_for_each_h * gqa_config.stride_kv_group) + i];
    }
    attn_score[(current_h * gqa_config.stride_score_head) + token_pos] = product_sum;

    }


// weighted sum 
// strategy same like attn_score_softmax
__global__ void weighted_sum(float* attn_score, const float* value_cache, float* output, GQAConfig gqa_config){
    // each block handles one head, each thread handles one of the 128 output dims
    int dim_idx = threadIdx.x;
    int current_h = blockIdx.x;
    int group_for_each_h = current_h / 6;
    float sum = 0.0f;

    if (dim_idx < 128){
        // (1, seq_len) * (seq_len, 128) -> (1, 128)
        for (int t = 0; t < gqa_config.seq_len; ++t){
            sum += attn_score[(current_h * gqa_config.stride_score_head) + t] * value_cache[(t * gqa_config.stride_kv_token) + (group_for_each_h * gqa_config.stride_kv_group) + dim_idx];
        }
        output[(current_h * gqa_config.stride_q_head) + dim_idx] = sum;
    }


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
        .stride_q_head = config->dim, // 128
        .stride_kv_token = config->n_kv_heads * (config->dim / config->n_heads), // 256
        .stride_kv_group = config->dim, // 128
        .stride_score_head = state->max_seq_len,
        .max_seq_len = state->max_seq_len,
        .layer_kv = layer_keys
    };

    GQAConfig gqa_v_config = {
        .seq_len = position + 1,
        .stride_q_head = config->dim,
        .stride_kv_token = config->n_kv_heads * (config->dim / config->n_heads), // 256
        .stride_kv_group = config->dim, // 128
        .stride_score_head = state->max_seq_len,
        .max_seq_len = state->max_seq_len,
        .layer_kv = layer_values
    };


    // Q, K, V projection
    proj_bias<<<dim3(6), dim3(256)>>>(layer->wq, layer->bq, state->before_attn_norm_x, state->q, config->dim);
    proj_bias<<<dim3(6), dim3(256)>>>(layer->wk, layer->bk, state->before_attn_norm_x, state->k, config->dim);
    proj_bias<<<dim3(6), dim3(256)>>>(layer->wv, layer->bv, state->before_attn_norm_x, state->v, config->dim);
    cudaDeviceSynchronize();




    // reshape for `q`, `k`, `v`
    // but, it's just logical shape...hhh, the reshape is in my mental 


    // rope for q, k 
    rope<<<12, 64>>>(state->q, state->q_rot, position);
    rope<<<12, 64>>>(state->k, state->k_rot, position);
    cudaDeviceSynchronize();


    /*
    rope之后的计算并行2条线。
        computation：k_rot write into k_cache -> softmax(Q*KT) and get attn_score
        v_to_v_cache：v write into v-cache
    synchronize streams before weighted sum
        weighted_sum =  attn_score([12, p+1]) * v([p+1], 2, 128])
            for each head: [p+1, 1] * [1, 128] -> [p+1, 128]
            combine results from all heads to get [12, 128] -> [1536]
    */

    cudaStream_t computation_stream , v_to_v_cache_stream;
    cudaStreamCreate(&computation_stream);
    cudaStreamCreate(&v_to_v_cache_stream); 



    /*
    1. current v to v cache stream
    */
    cudaError_t err = cudaMemcpyAsync(current_value, state->v, (config->n_kv_heads) * (config->dim / config->n_heads) * sizeof(float), cudaMemcpyDeviceToDevice, v_to_v_cache_stream);
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


    // compute attn_score 
    attn_score_softmax<<<dim3(config->n_heads), dim3(position + 1)>>>(
        state->q_rot,
        state->key_cache + layer_offset,
        state->attn_score,
        gqa_k_config
    );


    /*
    3. synchronize streams before weighted sum
    */
    cudaStreamSynchronize(computation_stream);
    cudaStreamSynchronize(v_to_v_cache_stream);
    
    weighted_sum<<<dim3(config->n_heads), dim3(config->dim / config->n_heads)>>>(
        state->attn_score,
        state->value_cache + layer_offset,
        state->weighted_value,
        gqa_v_config
    );


    /*
    4. projection 
    */
    proj<<<dim3(6), dim3(256)>>>(
        layer->wo,
        state->weighted_value,
        state->attn_o,
        config->dim
    );




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