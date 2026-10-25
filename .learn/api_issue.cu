#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel_A(int* p) {
    p[0] = 123;  // 故意访问非法地址
}

int main() {
    int* d;
    int h = 0;

    cudaMalloc(&d, sizeof(int));   
    cudaMemset(d, 0, sizeof(int));

    // ① 启动错误 kernel
    kernel_A<<<1, 1>>>(nullptr);
    cudaError_t kernel_err = cudaGetLastError();

    printf("kernel_A error: %s\n",
           cudaGetErrorString(kernel_err));

    // ② 完全合法的内存拷贝
    cudaError_t cpy_err = cudaMemcpy(
        (int*) &h, d, sizeof(int), cudaMemcpyDeviceToHost
    );

    printf("cudaMemcpy: %s\n", cudaGetErrorString(cpy_err));

    // ③ 再调用一个合法的 API
    cudaError_t memset_err = cudaMemset(d, 0, sizeof(int));

    printf("cudaMemset: %s\n", cudaGetErrorString(memset_err));

    cudaFree(d);

    return 0;
}
