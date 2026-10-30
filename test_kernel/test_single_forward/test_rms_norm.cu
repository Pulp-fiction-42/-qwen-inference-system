#include <stdio.h>
#include <cuda_runtime.h>
#include "model.h"
#include "layer.h"
/* 
__global__ void rms_norm(const float* rmn_norm_weight, float* input, float* output, int dim, float* rms, float eps){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // normalization
    // x' = x / RMS
    // out = x' * weight
    output[tid] = input[tid] / *(rms) * rmn_norm_weight[tid];

}
*/


int main(){
    Model model;
    RunState state;
    Config config;
    
    load_model_weights(&model);
    load_runstate(&state, &model.config);
    config = model.config;    

    // use first layer's before attention normalization
    blocks* layer = &model.layers[0];
    float* r;
    cudaMalloc((void**)&r, sizeof(float));
    
    // construct input 
    int token_id = 123;
    
    
    embedding_lookup<<<dim3(6), dim3(256)>>>(token_id, model.embedding, config.dim, state.x);
    rms_norm<<<dim3(1), dim3(256)>>>(
        layer->before_attn_norm, 
        state.x, 
        state.before_attn_norm_x, 
        config.dim, 
        r, 1e-6f);        
    
    cudaFree(r);

    return 0;
}