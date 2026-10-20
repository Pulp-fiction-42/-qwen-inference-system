#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "model.h"
#include "kernels.h"

void embedding_lookup(
    int token_id,
    const float* embedding,  // GPU 权重表：[vocab_size, dim]
    int dim,                 // 一行的长度
    float* des           // 写入state->x指向的 gpu buffer 区域
){
    //todo

}




void rms_norm(const float* rmn_norm_weight, float* input, float* output, int dim, float eps){
    //todo
}





void attention(blocks* layer, RunState* state, int layer_index, Config* config, int position){
    int head_dim = config->dim / config->n_heads; // 128
    int kv_dim = config->n_kv_heads * head_dim;
    int heads_per_kv = config->n_heads / config->n_kv_heads; // 6

    // Q, K, V projection
    int blocks_per_grid = 1;      // todo: 取决于 mat_vec 的并行方式
    int threads_per_block = 128;  // todo
    mat_vec<<<blocks_per_grid, threads_per_block>>>(layer->wq, layer->bq, state->before_attn_norm_x, state->q, config->dim, config->dim, config->dim);
    mat_vec<<<blocks_per_grid, threads_per_block>>>(layer->wk, layer->bk, state->before_attn_norm_x, state->k, kv_dim, config->dim, config->dim);
    mat_vec<<<blocks_per_grid, threads_per_block>>>(layer->wv, layer->bv, state->before_attn_norm_x, state->v, kv_dim, config->dim, config->dim);
    cudaDeviceSynchronize();


    // reshape for `q`, `k`, `v`
   

    //rope for `q`, `k`
    // frequency for 64-pair in each head
    float freqs[64];
    for (int d = 0; d < 64; ++d){
        freqs[d] = powf(10.0f, -6.0f * d / 64.0f); 
    }
    //allocate memory and copy `freqs[64]` from host to device
    //这里是之前runstate分配时存在的纰漏，没有考虑分配`freqs`
    
    float* freqs_dev_ptr = NULL; // store the address of `freqs` in device
    cudaError_t err = cudaMalloc(&freqs_dev_ptr, 64 * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for `rope frequency` failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    err = cudaMemcpy(freqs_dev_ptr, freqs, 64 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "frequency copy from host to device failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    //apply rope 
    rope<<<1, 64>>>(freqs_dev_ptr, state->q, state->k, position, state->q_rot, state->k_rot);


    /*
    rope之后的计算并行2条线。
    首先是从：
        computation：k_rot写入k_cache->...->softmax计算Q*KT这一过程
        v_to_v_cache：v写入v-cache这一过程
    最后在计算weighted sum时同步
    */
    cudaStream_t computation_stream , v_to_v_cache_stream;
    cudaStreamCreate(&computation_stream);
    cudaStreamCreate(&v_to_v_cache_stream); 

    size_t layer_offset = layer_index * state->max_seq_len * kv_dim;// for lacate the v and k


    /*
    current v to v cache stream
    */
    float* layer_values = state->value_cache + layer_offset;
    float* current_value = layer_values + position * kv_dim;
    err = cudaMemcpyAsync(
    current_value,
    state->v,
    (config->n_kv_heads) * (config->dim / config->n_heads) * sizeof(float),
    cudaMemcpyDeviceToDevice,
    v_to_v_cache_stream
    );
    if (err != cudaSuccess) {
    fprintf(stderr, "copy v to v-cache failed: %s\n", cudaGetErrorString(err));
    }


    /*
    GQA computation stream:
    1. write k into k-cache
    2. attn score: q_rot [12, 128] * KT in K-cache [12, p+1] -> [12, p+1], 这一步计算思路是：12个head并行，每个head做product([1, 128] * [128, p+1] -> [1, p+1])   
    3. apply softamx for each head: 同样是12个head并行，每个head做softamx计算，在kernel内部：
        找到p+1序列中的最大值Max并减去Max，然后应用e^(xi - Max) (elemetwise operation)
        计算分母sum (reduce)
        除以分母sum (elementwise)
    4. weighted sum: attn_score[12, p+1] * v[12, 128] -> [12, 128]
        12个head并行计算，每个head做mat_vec
    5. Output projection: [1536, 1536] * [1536, 1] -> [1536, 1]
    */

    //write k into k-cache
    float * current_key = state->key_cache + layer_offset + position * kv_dim;
    err = cudaMemcpyAsync(
        current_key,//destination
        state->k_rot,//source
        kv_dim * sizeof(float),
        cudaMemcpyDeviceToDevice,
        computation_stream
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "copy k to K-cache failed: %s\n", cudaGetErrorString(err));
    }


    //attn score: q_rot [12, 128] * KT in K-cache [12, p+1] -> [12, p+1]
    // compute attention scores: q_rot [12, 128] * KT in K-cache [12, p+1] -> [12, p+1]
    int seq_len = position + 1;
    int stride_q_head = head_dim;               // 128
    int stride_k_token = kv_dim;                // 256
    int stride_k_group = head_dim;              // 相邻 kv head 在 同一个 K-head 内的跨度
    int stride_score_head = state->max_seq_len; // state->attn_score 每个 head 的跨度    

    compute_attn_score_softmax<<<dim3(seq_len, config->n_heads), dim3(head_dim), 0, computation_stream>>>(
        state->q_rot,
        state->key_cache + layer_offset,
        state->attn_score,
        seq_len,
        head_dim,
        stride_q_head,
        stride_k_token,
        stride_k_group,
        stride_score_head
    );





    

}




// this is inside the attention 
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
){
    // use `__device__ product_reduce` to compute the 128-q 128-kT product and write partial sum to shared memory
    // calculate properly the base-pointer of q and kT for each `__device__ product_reduce`

    // assign must local variable to each thread
    int h = blockIdx.y;
    int k_h_group = h / 6;// k_h_group can be 0 or 1
    int p = blockIdx.x;
    float* q_ptr = (float*)(q_rot) + h * stride_q_head;
    float* k_ptr = (float*)(k_cache) + (p * stride_k_token) + (k_h_group * stride_k_group);// base pointer, token, which head in a 256 
    
    // call product_reduce for each thread 
    product_reduce(q_ptr, k_ptr, attn_score + h * stride_score_head + p, head_dim);

    // softmax
    
    



    
    

}



