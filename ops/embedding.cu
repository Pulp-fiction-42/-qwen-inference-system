// 迁自原 layer.cu；除接口及 score/softmax 拆分外，保留 kernel 计算逻辑。
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "ops.h"

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

void op_embedding(int token_id, const float* embedding, int dim, float* output)
{
    embedding_lookup<<<6, 256>>>(token_id, embedding, dim, output);
}
