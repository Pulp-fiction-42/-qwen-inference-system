#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
    // 每轮的t_stride: 1, 2, 4, 8, 16, 32, 64, 128
    int tid = threadIdx.x;
    for (int t_stride = 1; t_stride < blockDim.x; t_stride *= 2){
        if (tid % (2 * t_stride) == 0){ // only selected threads 
            int idx = tid;
        
            if (s_logits[idx] < s_logits[idx + t_stride]){
                s_logits[idx] = s_logits[idx + t_stride];
                s_idx[idx] = s_idx[idx + t_stride];
            }
        }
        __syncthreads(); // synchronize after each reduction step
    }


    // write the maximum value and it's index of each block to global memory by `thread 0`
    if (threadIdx.x == 0){
        kernel_1_output_logits[blockIdx.x] = s_logits[0];
        kernel_1_output_idx[blockIdx.x] = s_idx[0];
        // identifier "printf" is undefined
        printf("block %d, max=%f idx=%d\n", blockIdx.x, s_logits[0], s_idx[0]);

    }
}




// ---- simple test ----

int main(){
    const int vocab_size = 1000;
    const int threads = 256;
    const int blocks = (vocab_size + threads - 1) / threads; // 4

    // construct the host logits array
    float *h_logits = (float*)malloc(vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) h_logits[i] = (float)(i);
    h_logits[777] = -INFINITY; // the minimum value to ensure it's not the max

    // copy from host -> device
    float *d_logits, *d_out_logits;
    int *d_out_idx;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMalloc(&d_out_logits, blocks * sizeof(float));
    cudaMalloc(&d_out_idx, blocks * sizeof(int));
    cudaMemcpy(d_logits, h_logits, vocab_size * sizeof(float), cudaMemcpyHostToDevice);




    kernel_1<<<blocks, threads>>>(d_logits, d_out_logits, d_out_idx, vocab_size);
    
    cudaDeviceSynchronize();


    cudaFree(d_logits);
    cudaFree(d_out_logits);
    cudaFree(d_out_idx);
    free(h_logits);
    return 0;
}


