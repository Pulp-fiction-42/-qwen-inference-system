// 迁自原 layer.cu；除接口及 score/softmax 拆分外，保留 kernel 计算逻辑。
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "ops.h"
#include <stdexcept>
#include <string>

void op_copy(float* destination, const float* source, int count)
{
    cudaError_t err = cudaMemcpyAsync(
        destination, source, (size_t)count * sizeof(float), cudaMemcpyDeviceToDevice);

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("copy device buffer failed: ") + cudaGetErrorString(err));
    }
}

void op_synchronize()
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("forward synchronization failed: ") + cudaGetErrorString(err));
    }
}
