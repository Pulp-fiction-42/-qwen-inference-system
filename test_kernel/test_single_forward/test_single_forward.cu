// test the issue part of forward 
#include "layer.h"
#include <stdio.h>
#include <cuda_runtime.h>


// just forward the first block 
float* forward_block0(Model* model, RunState* state, Config* config, int token_id, int position){

    // allocate rms for later 3 times rmsnorm
    float* r;
    cudaMalloc((void**)&r, sizeof(float));

    embedding_lookup<<<dim3(6), dim3(256)>>>(token_id, model->embedding, model->config.dim, state->x);
    

    float* layer_input = state->x;
    
        blocks* layer = &(model->layers[0]);

        // before attn rmsnorm
        // each thread handle 6 elements of input
        // step: reduce to get `r` -> elementwise: yi = xi * r * wi
        rms_norm<<<dim3(1), dim3(256)>>>(layer->before_attn_norm, layer_input, state->before_attn_norm_x, model->config.dim, r, 1e-6);                    
 

        attention(layer, state, 0, config, position);

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




void print_runstate(RunState* state, Config* config){
  // malloc space for each members in runstate
    size_t size_x = config->dim * sizeof(float);
    size_t size_before_attn_norm_x = config->dim * sizeof(float);   
    size_t size_q = config->dim * sizeof(float); // 1536
    size_t size_k = config->dim / config->n_heads * config->n_kv_heads * sizeof(float); // 256
    size_t size_v = size_k; // 256
    size_t size_q_rot = size_q; //包含reshape->rope，shape transform from [dim] to [n_heads, dim/n_heads]
    size_t size_k_rot = size_k; // size_K: [256], size_k_rot: [2, 128] - **但都表示256个float的大小**    
    // kv-cache
    size_t size_key_cache = config->n_layers * state->max_seq_len * size_k_rot;
    size_t size_value_cache = config->n_layers * state->max_seq_len * size_v;
    //...
    size_t size_attn_score = config->n_heads * state->max_seq_len * sizeof(float);
    size_t size_softmax_attn_score = config->n_heads * state->max_seq_len * sizeof(float);    
    size_t size_weighted_value = config->n_heads * (config->dim / config->n_heads) * sizeof(float); // [6, 128], [6, 128]
    size_t size_attn_o = config->dim * sizeof(float); // 1536
    // after attention 
    size_t size_residual_after_attn = config->dim * sizeof(float);
    size_t size_after_attn_norm_x = config->dim * sizeof(float);
    //MLP
    size_t size_g = config->hidden_dim * sizeof(float);
    size_t size_b = config->hidden_dim * sizeof(float);
    size_t size_s = config->hidden_dim * sizeof(float);
    size_t size_f = config->hidden_dim * sizeof(float);
    size_t size_m = config->dim * sizeof(float);
    //after MLP 
    size_t size_residual_after_mlp = config->dim * sizeof(float);
    //after all blocks
    size_t size_final_norm = config->dim * sizeof(float);
    size_t size_logits = config->vocab_size * sizeof(float);

    float* h_x = (float*)malloc(size_x);
    float* h_before_attn_norm_x = (float*)malloc(size_before_attn_norm_x);
    float* h_q = (float*)malloc(size_q);
    float* h_k = (float*)malloc(size_k);
    float* h_v = (float*)malloc(size_v);
    float* h_q_rot = (float*)malloc(size_q_rot);
    float* h_k_rot = (float*)malloc(size_k_rot);
    float* h_key_cache = (float*)malloc(size_key_cache);
    float* h_value_cache = (float*)malloc(size_value_cache);
    float* h_attn_score = (float*)malloc(size_attn_score);
    float* h_softmax_attn_score = (float*)malloc(size_softmax_attn_score);
    float* h_weighted_value = (float*)malloc(size_weighted_value);
    float* h_attn_o = (float*)malloc(size_attn_o);
    float* h_residual_after_attn = (float*)malloc(size_residual_after_attn);
    float* h_after_attn_norm_x = (float*)malloc(size_after_attn_norm_x);
    float* h_g = (float*)malloc(size_g);
    float* h_b = (float*)malloc(size_b);
    float* h_s = (float*)malloc(size_s);
    float* h_f = (float*)malloc(size_f);
    float* h_m = (float*)malloc(size_m);
    float* h_residual_after_mlp = (float*)malloc(size_residual_after_mlp);
    float* h_final_norm = (float*)malloc(size_final_norm);
    float* h_logits = (float*)malloc(size_logits);
 
    // copy from device to host 
    cudaMemcpy(h_x, state->x, size_x, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_before_attn_norm_x, state->before_attn_norm_x, size_before_attn_norm_x, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_q, state->q, size_q, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k, state->k, size_k, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, state->v, size_v, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_q_rot, state->q_rot, size_q_rot, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k_rot, state->k_rot, size_k_rot, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_key_cache, state->key_cache, size_key_cache, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_value_cache, state->value_cache, size_value_cache, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_attn_score, state->attn_score, size_attn_score, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_softmax_attn_score, state->softmax_attn_score, size_softmax_attn_score, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_weighted_value, state->weighted_value, size_weighted_value, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_attn_o, state->attn_o, size_attn_o, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_residual_after_attn, state->residual_after_attn, size_residual_after_attn, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_after_attn_norm_x, state->after_attn_norm_x, size_after_attn_norm_x, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g, state->g, size_g, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, state->b, size_b, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_s, state->s, size_s, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f, state->f, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m, state->m, size_m, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_residual_after_mlp, state->residual_after_mlp, size_residual_after_mlp, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_norm, state->final_norm, size_final_norm, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_logits, state->logits, size_logits, cudaMemcpyDeviceToHost);

    // print all members and redirect into a new file, each member speerate with 4 '\n'
    // just print each member's previus 10 values
    // for K V cache, just print the 10 values on position 0
    #define PRINTALL(name, arr, bytes) \
        for(size_t i = 0; i < (bytes) / sizeof(float); i++){ \
            printf(name "[%zu]: %f\n", i, arr[i]); \
        } \
        printf("\n\n\n\n");

    PRINTALL("x", h_x, size_x)
    PRINTALL("before_attn_norm_x", h_before_attn_norm_x, size_before_attn_norm_x)
    PRINTALL("q", h_q, size_q)
    PRINTALL("k", h_k, size_k)
    PRINTALL("v", h_v, size_v)
    PRINTALL("q_rot", h_q_rot, size_q_rot)
    PRINTALL("k_rot", h_k_rot, size_k_rot)
    PRINTALL("key_cache", h_key_cache, size_key_cache)
    PRINTALL("value_cache", h_value_cache, size_value_cache)
    PRINTALL("attn_score", h_attn_score, size_attn_score)
    PRINTALL("softmax_attn_score", h_softmax_attn_score, size_softmax_attn_score)
    PRINTALL("weighted_value", h_weighted_value, size_weighted_value)
    PRINTALL("attn_o", h_attn_o, size_attn_o)
    PRINTALL("residual_after_attn", h_residual_after_attn, size_residual_after_attn)
    PRINTALL("after_attn_norm_x", h_after_attn_norm_x, size_after_attn_norm_x)
    PRINTALL("g", h_g, size_g)
    PRINTALL("b", h_b, size_b)
    PRINTALL("s", h_s, size_s)
    PRINTALL("f", h_f, size_f)
    PRINTALL("m", h_m, size_m)
    PRINTALL("residual_after_mlp", h_residual_after_mlp, size_residual_after_mlp)
    PRINTALL("final_norm", h_final_norm, size_final_norm)
    PRINTALL("logits", h_logits, size_logits)
    #undef PRINTALL

    free(h_x); free(h_before_attn_norm_x); free(h_q); free(h_k); free(h_v);
    free(h_q_rot); free(h_k_rot); free(h_key_cache); free(h_value_cache);
    free(h_attn_score); free(h_softmax_attn_score); free(h_weighted_value);
    free(h_attn_o); free(h_residual_after_attn); free(h_after_attn_norm_x);
    free(h_g); free(h_b); free(h_s); free(h_f); free(h_m);
    free(h_residual_after_mlp); free(h_final_norm); free(h_logits);

}


int main(){
    Model model;
    RunState d_state;
    Config config;
    load_model_weights(&model);
    load_runstate(&d_state, &model.config);
    config = model.config;
    int token = 123; // random token for testing
    int position = 0;

  forward_block0(&model, &d_state, &config, token, position);
  print_runstate(&d_state, &config);
}











