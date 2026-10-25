# include <cuda_runtime.h>
# include <stdio.h>

// this file is for testing the  
__global__ void read(float* a, float* out) {
    int tid = threadIdx.x;
    out[tid] = a[tid];
}

// Host




