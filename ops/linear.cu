// 迁自原 layer.cu；除接口及 score/softmax 拆分外，保留 kernel 计算逻辑。
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "ops.h"
#include <stdexcept>
#include <string>

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

__global__ void proj(const float* weights, const float* input, float* output, int dim){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid; // each thread handle one row's product
    float sum = 0.0f;

    for (int i = 0; i < dim; ++i){
        sum += weights[row * dim + i] * input[i];
    }
    output[row] = sum;
}

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

void op_proj_bias(const float* weights, const float* bias,
                  const float* input, float* output, int dim)
{
    // 沿用原 Q/K/V 的固定启动尺寸；K/V 的越界问题未在此次重构中修复。
    proj_bias<<<6, 256>>>(weights, bias, input, output, dim);
}

void op_proj(const float* weights, const float* input, float* output, int dim)
{
    proj<<<6, 256>>>(weights, input, output, dim);
}

void op_mlp_proj(const float* weights, const float* input, float* output,
                 int out_dim, int in_dim)
{
    int threads_per_block = 256;
    int blocks_per_grid = (out_dim + threads_per_block - 1) / threads_per_block;

    mlp_proj<<<blocks_per_grid, threads_per_block>>>(
        weights, input, output, out_dim, in_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("mlp projection launch failed: ") + cudaGetErrorString(err));
    }
}

void op_lm_head_proj(const float* weights, const float* bias,
                     const float* input, float* output,
                     int w_row, int w_col, int in_dim)
{
    int threads_per_block = 256;
    int blocks_per_grid = (w_row + threads_per_block - 1) / threads_per_block;

    lm_head_proj<<<blocks_per_grid, threads_per_block>>>(
        weights, bias, input, output, w_row, w_col, in_dim);
}
