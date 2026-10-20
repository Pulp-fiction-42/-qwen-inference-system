#include <cstddef>
#include <cuda_runtime.h>
#include "../model.h"
#include <stdio.h>

int main() {
    Model model{};
    RunState state{};


    // 1. 加载模型并分配运行缓冲区。
    load_model_weights(&model);
    load_runstate(&state, &model.config);


    // 2. 检查mdoel.bin的前6个int_32是否被正确赋給config
    
    // 核对host端的config与devuce端的config是否一致：将device_data的前几个字节复制回host端并打印，检查是否与model.config一致。
    int* h_data = ((int*)malloc(6 * sizeof(int)));
    int* d_data = (int*)model.device_data;
    cudaError_t err = cudaMemcpy(h_data, d_data, 6 * sizeof(int), cudaMemcpyDeviceToHost);    
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    //依次打印传递到host的6个int值
    printf("dim: %d\n", h_data[0]);
    printf("hidden_dim: %d\n", h_data[1]);
    printf("n_layers: %d\n", h_data[2]);
    printf("n_heads: %d\n", h_data[3]);
    printf("n_kv_heads: %d\n", h_data[4]);
    printf("vocab_size: %d\n", h_data[5]);
    
    // 3. 检查`model.bin`是否正确，这里只检查在device上的embedding(model跳过config的初始成员指针)
    // 跳过前6个int32，指向权重数据(embedding)的起始位置
    float* model_weights_base_pointer_device = reinterpret_cast<float*>(d_data + 6); 
    //打开model.bin
    FILE* file = fopen("/home/bgi-chao/wdr/main/model.bin", "rb");
    if (file == NULL) {
        perror("无法打开 model.bin");
        return -1;
    }
    //将model.bin的embedding的前8个float存入数组A-需要跳过前6个int32
    float A[8];
    fseek(file, 6 * sizeof(int), SEEK_SET); // 跳过前6个int32
    fread(A, sizeof(float), 8, file);
    fclose(file);
    
    //将device_data的embedding的前8个float存入数组B
    float B[8];
    err = cudaMemcpy(B, model_weights_base_pointer_device, 8 * sizeof(float),  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    //compare A and B
    for (int i = 0; i < 8; ++i) {
        if (A[i] != B[i]){
            printf("模型权重不匹配: 第 %d 个元素，GPU端: %f, 文件端: %f\n", i, B[i], A[i]);
        }
    }
    printf("模型权重检查完成。\n");




    // 5. 释放资源。
    free_runstate(&state);
    free_model_weights(&model);

    return 0;
}