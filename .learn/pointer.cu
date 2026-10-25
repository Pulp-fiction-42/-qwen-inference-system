#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    float *d_a = NULL;

    printf("Before:\n");
    printf("  d_a  = %p\n", (void *)d_a);
    printf("  &d_a = %p\n", (void *)&d_a);

    void **devPtr = (void **)&d_a;

    printf("\nInside pointer relationship:\n");
    printf("  devPtr  = %p\n", (void *)devPtr);
    printf("  *devPtr = %p\n", *devPtr);

    cudaError_t err = cudaMalloc(devPtr, sizeof(float));

    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("\nAfter:\n");
    printf("  d_a     = %p\n", (void *)d_a);
    printf("  &d_a    = %p\n", (void *)&d_a);
    printf("  devPtr  = %p\n", (void *)devPtr);
    printf("  *devPtr = %p\n", *devPtr);

    cudaFree(d_a);
    return 0;
}