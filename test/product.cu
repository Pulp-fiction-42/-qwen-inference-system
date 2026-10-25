#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

__global__ void proj(const float* weights, const float* bias, const float* input, float* output){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid; // each thread handle one row's (product + bias)
    float sum = 0.0f;

    for (int i = 0; i < 1536; ++i){
        sum += weights[row * 1536 + i] * input[i];
    }
    sum += bias[row];
    output[row] = sum;
    // print each block's first theread's result 
    if (threadIdx.x == 0) {
        printf("Thread %d computed output: %f\n", tid, sum);
}   
}




int main() {
    // Let's assume the matrix's shape is (1536, 1536), vector shape: (1536)
    int matrix_size = 1536 * 1536;
    int vector_size = 1536;
    float* d_weights;
    float* d_bias;
    float* d_input;
    float* d_output;

    // test data initialization
    std::vector<float> h_weights(matrix_size, 1.0f);
    std::vector<float> h_bias(vector_size, 0.0f);
    std::vector<float> h_input(vector_size);
    for (int i = 0; i < vector_size; ++i) {
        h_input[i] = i + 1.0f;
    }

    cudaMalloc(&d_weights, matrix_size * sizeof(float));
    cudaMalloc(&d_bias, vector_size * sizeof(float));
    cudaMalloc(&d_input, vector_size * sizeof(float));
    cudaMalloc(&d_output, vector_size * sizeof(float));

    cudaMemcpy(d_weights, h_weights.data(), matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);

    
    dim3 blockDim(256);
    dim3 gridDim(6);
    proj<<<gridDim, blockDim>>>(d_weights, d_bias, d_input, d_output);


    // cudaFree 本身会隐式同步，但显式同步可以尽早捕获 kernel 执行错误
    cudaDeviceSynchronize();


    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
