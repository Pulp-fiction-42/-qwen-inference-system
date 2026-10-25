#include <stdio.h>
#include <cuda_runtime.h>

__global__ void task_a(int* out) {
    // 工作流 A 的计算
    *out = 10;
}

__global__ void task_b(int* out) {
    // 工作流 B 的计算
    *out = 20;
}

int main() {
    int h_a = 0;
    int h_b = 0;
    int* d_a = NULL;
    int* d_b = NULL;

    cudaMalloc((void**)&d_a, sizeof(int));
    cudaMalloc((void**)&d_b, sizeof(int));

    // 1. 创建两条独立的 stream
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 2. 分别在两条 stream 中异步启动不同工作流
    // 第四个参数传入对应的 stream
    task_a<<<1, 1, 0, stream1>>>(d_a);
    task_b<<<1, 1, 0, stream2>>>(d_b);

    // 3. block the cpu thread until both streams finish their tasks
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 4. 依赖数据：两路完成后安全读取结果
    cudaMemcpy(&h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost); // test if the data copied to host is 10?
    cudaMemcpy(&h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost);

    printf("result: a = %d, b = %d, sum = %d\n", h_a, h_b, h_a + h_b);

    // 5. 销毁 stream 与释放显存
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}