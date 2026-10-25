#include <stdio.h>
#include <cuda_runtime.h>

//for a 8 * 8 metrix, use each thread print it's location
__global__ void hello() { 
    int row = blockIdx.y + blockDim.y + threadIdx.y; 
    int col = blockIdx.x + blockDim.x + threadIdx.x;
    printf("Block(%d %d), Thread(%d %d) -> Matrix(%d %d)", blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, row, col);
}

int main() {
        dim3 blocks(4, 4);
        dim3 grid(2, 2);
        hello<<<grid, blocks>>>();
    cudaDeviceReset();
    return 0;
}
