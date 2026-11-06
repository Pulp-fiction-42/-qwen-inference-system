// 原 argmax 的两个 kernel 原样迁入，包含原固定 640 的实现。
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "ops.h"

__global__ void kernel_1(float* logits, float* kernel_1_output_logits, int* kernel_1_output_idx, int vocab_size){
    
    // move 256 logits from VRAM -> l1 cache (shared memory)
    // 用到shared memory是因为每个threads的寄存器都是私有的    
    __shared__ float s_logits[256];
    __shared__ int s_idx[256]; 

    // move data from global memory to shared memory
    int block_offset = blockIdx.x * blockDim.x;
    int global_idx = block_offset + threadIdx.x;
    if (global_idx < vocab_size){
        s_logits[threadIdx.x] = logits[global_idx];
        s_idx[threadIdx.x] = global_idx; 
    }else{
        s_logits[threadIdx.x] = -INFINITY; // final block
    }
    __syncthreads();



    // 每个block内负责: 256->128->64->32->16->8->4->2->1
    // 共8轮规约
    // 每轮规约: 每个线程与相邻线程比较取最大值，并将最大值放在比较组的前面位置
    // 每轮实例：turn1: (0, 1); turn2: (0, 2); turn3: (0, 4); ...
    // 每轮的t_stride(表示当前轮次：2个做比较线程相邻的跨度): 1, 2, 4, 8, 16, 32, 64, 128
    int tid = threadIdx.x;
    for (int t_stride = 1; t_stride < blockDim.x; t_stride *= 2){
        if (tid % (2 * t_stride) == 0){ // only selected threads 
        
            if (s_logits[tid] < s_logits[tid + t_stride]){
                s_logits[tid] = s_logits[tid + t_stride];
                s_idx[tid] = s_idx[tid + t_stride];
            }
        }
        __syncthreads(); // synchronize after each reduction step
    }
    // write the maximum value and it's index of each block to global memory by `thread 0`
    if (threadIdx.x == 0){
        kernel_1_output_logits[blockIdx.x] = s_logits[0];
        kernel_1_output_idx[blockIdx.x] = s_idx[0];
    }
}

__global__ void kernel_2(float* kernel_1_output_logits, int* kernel_1_output_idx, int* d_max_value_index){

    __shared__ float s_max[640];
    __shared__ int s_idx[640];

    // move data form glolbal memory to shared memory
    int tid = threadIdx.x;
    if (tid < 640){
        s_max[tid] = kernel_1_output_logits[tid];
        s_idx[tid] = kernel_1_output_idx[tid];
    }else{
        s_max[tid] = -INFINITY;
    }

    __syncthreads(); 

    // 每个block内规约找最大值
    for (int t_stride = 1; t_stride < 640; t_stride *= 2) {
            // 只有当前跨度内的起始线程执行，且必须严格限制目标地址小于 640
            if (tid % (2 * t_stride) == 0) {
                int target_idx = tid + t_stride;
                if (target_idx < 640) { // 严格防止尾部越界
                    if (s_max[tid] < s_max[target_idx]) {
                        s_max[tid] = s_max[target_idx];
                        s_idx[tid] = s_idx[target_idx];
                    }
                }
            }
            __syncthreads();
        }
    
    if (tid == 0){
        *d_max_value_index = s_idx[0];
    }
}

static void check_argmax(cudaError_t error)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("argmax: ") + cudaGetErrorString(error));
    }
}

int op_argmax(float* logits, int vocab_size, float* block_values,
              int* block_indices, int* token_index)
{
    int num_blocks = (vocab_size + 256 - 1) / 256;
    kernel_1<<<num_blocks, 256>>>(logits, block_values, block_indices, vocab_size);
    check_argmax(cudaGetLastError());
    check_argmax(cudaDeviceSynchronize());

    kernel_2<<<1, num_blocks>>>(block_values, block_indices, token_index);
    check_argmax(cudaGetLastError());
    check_argmax(cudaDeviceSynchronize());

    int token;
    check_argmax(cudaMemcpy(&token, token_index, sizeof(int), cudaMemcpyDeviceToHost));
    return token;
}
