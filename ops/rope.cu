// 迁自原 layer.cu；除接口及 score/softmax 拆分外，保留 kernel 计算逻辑。
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "ops.h"

__global__ void rope(
    const float* q_or_k,
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

void op_rope(const float* input, float* output, int position,
             int heads, int head_dim, float theta)
{
    // 这里只封装启动尺寸，kernel 内部的 128 常量保持不变。
    rope<<<heads, head_dim / 2>>>(input, output, position, theta);
}
