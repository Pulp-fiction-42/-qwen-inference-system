#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../forward.h"
#include "../model.h"
#include <time.h>





// ---- simple test ----

int main(){
    // config 
    Model model;
    Config config = model.config;
    size_t member_count = sizeof(Config) / sizeof(int);
    FILE *fp = fopen("/home/bgi-chao/wdr/main/model.bin", "rb");
    if (!fp) {
        perror("Failed to open model file");
        return 1;
    }
    fread(&config, sizeof(int), member_count, fp);
    fclose(fp);

    
    // construct logists, use random numbers to fill the logists
    // run 10 tests, each test, put the max float number: 2.0f in different positions to check if the sampling works correctly
    int vocab = config.vocab_size;
    float* h_logits = (float *)malloc(vocab * sizeof(float));
    // 初始化随机数种子
    srand((unsigned int)time(NULL));
    for (int test = 0; test < 10; test++) {
        for (int i = 0; i < vocab; i++) {
            h_logits[i] = (float)rand() / (float)RAND_MAX;
        }
        h_logits[test % vocab] = 2.0f; // put the max value in different positions


        // copy logists on host to device
        float* d_logits;
        cudaMalloc((void**)&d_logits, vocab * sizeof(float));
        cudaMemcpy(d_logits, h_logits, vocab * sizeof(float), cudaMemcpyHostToDevice);

        // sampling 
        int result = sample_argmax(d_logits, vocab);
        printf("test turn: %d, expect: %d\n", test, test % vocab); // the max value index should be test % vocab
        printf("test turn: %d, result: %d\n", test, result);
        cudaFree(d_logits);
        free(h_logits);
        h_logits = (float *)malloc(vocab * sizeof(float)); // reallocate for the next test
    }
}


