// 迁自原 layer.cu；除接口及 score/softmax 拆分外，保留 kernel 计算逻辑。
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "ops.h"
#include <stdexcept>
#include <string>

__global__ void attn_residual(const float* x, const float* attn_o, float* residual_after_attn, int dim){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dim){
        residual_after_attn[tid] = x[tid] + attn_o[tid];
    }
}

__global__ void mlp_residual(const float* residual_after_attn, const float* m, float* residual_after_mlp, int dim){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dim){
        residual_after_mlp[tid] = residual_after_attn[tid] + m[tid];
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

void op_attn_residual(const float* input, const float* attention,
                      float* output, int dim)
{
    attn_residual<<<6, 256>>>(input, attention, output, dim);
}

void op_mlp_residual(const float* residual, const float* mlp,
                     float* output, int dim)
{
    mlp_residual<<<6, 256>>>(residual, mlp, output, dim);
}

void op_silu(const float* input, float* output, int dim)
{
    int threads_per_block = 256;
    int blocks_per_grid = (dim + threads_per_block - 1) / threads_per_block;

    silu<<<blocks_per_grid, threads_per_block>>>(input, output, dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("silu launch failed: ") + cudaGetErrorString(err));
    }
}

void op_mul(const float* left, const float* right, float* output, int dim)
{
    int threads_per_block = 256;
    int blocks_per_grid = (dim + threads_per_block - 1) / threads_per_block;

    elementwise_mul<<<blocks_per_grid, threads_per_block>>>(left, right, output, dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("elementwise multiply launch failed: ") + cudaGetErrorString(err));
    }
}
