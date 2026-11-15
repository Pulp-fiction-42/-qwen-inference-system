// 迁自原 layer.cu；除接口及 score/softmax 拆分外，保留 kernel 计算逻辑。
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "ops.h"

__global__ void rms_norm(const float* rmn_norm_weight, const float* input, float* output, int dim, float* r, float eps){
    
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


    // reduce, totally 8 times®
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

void op_rms_norm(const float* weights, const float* input, float* output,
                 int dim, float* scratch, float eps)
{
    rms_norm<<<1, 256>>>(weights, input, output, dim, scratch, eps);
}

void op_final_rms_norm(const float* weights, const float* input, float* output,
                       int dim, float* scratch, float eps)
{
    // 原 forward 使用 6 个 block；重复写入的问题按要求留待后续处理。
    rms_norm<<<6, 256>>>(weights, input, output, dim, scratch, eps);
}
