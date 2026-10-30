#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "model.h"
#include "forward.h"
#include "layer.h"


// return state->logits
float* forward(Model* model, RunState* state, Config* config, int token_id, int position){

    // allocate rms for later 3 times rmsnorm
    float* r;
    cudaMalloc((void**)&r, sizeof(float));

    embedding_lookup<<<dim3(6), dim3(256)>>>(token_id, model->embedding, model->config.dim, state->x);
    
    
    float* layer_input = state->x;
    for (int layer_index = 0; layer_index < model->config.n_layers; layer_index++){
        blocks* layer = &(model->layers[layer_index]);

        // before attn rmsnorm
        // each thread handle 6 elements of input
        // step: reduce to get `r` -> elementwise: yi = xi * r * wi
        rms_norm<<<dim3(1), dim3(256)>>>(layer->before_attn_norm, layer_input, state->before_attn_norm_x, model->config.dim, r, 1e-6);                    
 

        attention(layer, state, layer_index, config, position);

        // Residual connection
        attn_residual<<<dim3(6), dim3(256)>>>(layer_input, state->attn_o, state->residual_after_attn, model->config.dim);

        // after attn rmsnormb
        rms_norm<<<dim3(1), dim3(256)>>>(layer->post_attn_norm, state->residual_after_attn, state->after_attn_norm_x, model->config.dim, r, 1e-6);
        
        // mlp
        mlp((const blocks*)layer, state, state->after_attn_norm_x, state->m, model->config.dim, model->config.hidden_dim);

        // residual connection after mlp
        mlp_residual<<<dim3(6), dim3(256)>>>(state->residual_after_attn, state->m, state->residual_after_mlp, model->config.dim);

        // update the float* layer_input for the next layer
        layer_input = state->residual_after_mlp;
    }
    // final layer normalization
    rms_norm<<<dim3(6), dim3(256)>>>(model->final_norm, state->residual_after_mlp, state->final_norm, model->config.dim, r, 1e-6);
    cudaFree(r);// after all rms_norm calls are done, then can be freed

    // lm head linear without bias
    // here, we don't apply softmax for logists, cause use max strategy for sampling
    int threads_per_block = 256;
    int blocks_per_grid = (config->vocab_size + threads_per_block - 1) / threads_per_block; // (151936 + 256 - 1) / 256 = 594
    
    lm_head_proj<<<blocks_per_grid, threads_per_block>>>(model->lm_head, NULL, state->final_norm, state->logits, config->vocab_size, config->dim, config->dim);
    cudaDeviceSynchronize();

    return state->logits;

}

// logits on device -> 
int sample_argmax(float* logits, int vocab_size){
    // allocate global memory for kernel 1's output
    float* kernel_1_output_logits;
    cudaMalloc((void**)&kernel_1_output_logits,  ((vocab_size + 256 - 1) / 256) * sizeof(float));

    int* kernel_1_output_idx;
    cudaMalloc((void**)&kernel_1_output_idx,  ((vocab_size + 256 - 1) / 256) * sizeof(int));

    // allocate an int memory, after kernel2 finished, it will store the desire index 
    int* kernel_2_output_idx;
    cudaMalloc((void**)&kernel_2_output_idx, sizeof(int));


    int num_blocks = (vocab_size + 256 - 1) / 256; // 640
    // launch kernel1 
    kernel_1<<<num_blocks, 256>>>(logits, kernel_1_output_logits, kernel_1_output_idx, vocab_size);
    cudaDeviceSynchronize();

    // launch kernel2 
    kernel_2<<<1, num_blocks>>>(kernel_1_output_logits, kernel_1_output_idx, kernel_2_output_idx);
    cudaDeviceSynchronize();
  
    
    // declare an int(as return value), and copy the result from device to host
    int max_value_index;
    cudaMemcpy(&max_value_index, kernel_2_output_idx, sizeof(int), cudaMemcpyDeviceToHost);
  
    cudaFree(kernel_1_output_logits);
    cudaFree(kernel_1_output_idx);
    cudaFree(kernel_2_output_idx);
    cudaDeviceSynchronize();

    return max_value_index;

}











