#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "model.h"
#include <cstdio>
#include <cstdlib>




/*
allocate GPU memory, load model weights from file to GPU
initialize the Struct: Model, Config, blocks
*/
void load_model_weights(Model* model)
{

    // 1. 将文件映射到 CPU 地址空间
    int fd = open("/home/bgi-chao/wdr/main/model.bin", O_RDONLY);

    struct stat info;
    fstat(fd, &info);
    std::size_t bytes = info.st_size;

    void* host_model_bin = mmap(
        nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    // 2. CPU 解析 header
    const auto* header = static_cast<const std::int32_t*>(host_model_bin);
    model->config = {
        header[0], header[1], header[2],
        header[3], header[4], header[5]
    };

    const Config& c = model->config;// alias to model->config 
    std::size_t dim = c.dim;
    std::size_t hidden = c.hidden_dim;
    std::size_t kv_dim = dim / c.n_heads * c.n_kv_heads;

    // 3. 整块复制到 GPU
    cudaError_t err = cudaMalloc(&model->device_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        munmap(host_model_bin, bytes);
        exit(1);
    }
    err = cudaMemcpy(model->device_data, host_model_bin,
                     bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(model->device_data);
        munmap(host_model_bin, bytes);
        exit(1);
    }

    // 同步复制完成后，CPU 文件映射即可释放
    munmap(host_model_bin, bytes);


    // 4. CPU 创建每层的“地址表”
    model->layers = new blocks[c.n_layers]{};

    // 跳过 header，cursor 保存 GPU 地址, initial cursor is first element of Weights's address
    auto* cursor = reinterpret_cast<const float*>(
        static_cast<char*>(model->device_data)
        + 6 * sizeof(std::int32_t));

    // 记录当前 GPU 地址，再移动到下一个张量
    auto take = [&](const float*& field, std::size_t count) {
        field = cursor;
        cursor += count;
    };

    take(model->embedding, c.vocab_size * dim);

    for (int i = 0; i < c.n_layers; ++i) {
        blocks& layer = model->layers[i];

        take(layer.before_attn_norm, dim);
        take(layer.wq, dim * dim);
        take(layer.bq, dim);
        take(layer.wk, kv_dim * dim);
        take(layer.bk, kv_dim);
        take(layer.wv, kv_dim * dim);
        take(layer.bv, kv_dim);
        take(layer.wo, dim * dim);
        take(layer.post_attn_norm, dim);
        take(layer.w_gate, hidden * dim);
        take(layer.w_up, hidden * dim);
        take(layer.w_down, dim * hidden);
    }

    take(model->final_norm, dim);
    model->lm_head = model->embedding;
}



void free_model_weights(Model* model){
    cudaError_t err = cudaFree(model->device_data);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
    }
    delete[] model->layers;
}




void load_runstate(RunState* state, Config* config) {
    // 1. 根据 config，计算每个缓冲区需要的字节数。
    state->max_seq_len = MAX_SEQ_LEN;
    size_t size_x = config->dim * sizeof(float);
    size_t size_before_attn_norm_x = config->dim * sizeof(float);
    size_t size_q = config->dim * sizeof(float); // 1536
    size_t size_k = config->dim / config->n_heads * config->n_kv_heads * sizeof(float); // 256
    size_t size_v = size_k; // 256
    size_t size_q_rot = size_q; //包含reshape->rope，shape transform from [dim] to [n_heads, dim/n_heads]
    size_t size_k_rot = size_k; // size_K: [256], size_k_rot: [2, 128] - **但都表示256个float的大小**    
    // kv-cache
    size_t size_key_cache = config->n_layers * state->max_seq_len * size_k_rot;
    size_t size_value_cache = config->n_layers * state->max_seq_len * size_v;
    //...
    size_t size_attn_score = config->n_heads * state->max_seq_len * sizeof(float);
    size_t size_softmax_attn_score = config->n_heads * state->max_seq_len * sizeof(float);    
    size_t size_weighted_value = config->n_heads * (config->dim / config->n_heads) * sizeof(float); // [6, 128], [6, 128]
    size_t size_attn_o = config->dim * sizeof(float); // 1536
    // after attention 
    size_t size_residual_after_attn = config->dim * sizeof(float);
    size_t size_after_attn_norm_x = config->dim * sizeof(float);
    //MLP
    size_t size_g = config->hidden_dim * sizeof(float);
    size_t size_b = config->hidden_dim * sizeof(float);
    size_t size_s = config->hidden_dim * sizeof(float);
    size_t size_f = config->hidden_dim * sizeof(float);
    size_t size_m = config->dim * sizeof(float);
    //after MLP 
    size_t size_residual_after_mlp = config->dim * sizeof(float);
    //after all blocks
    size_t size_final_norm = config->dim * sizeof(float);
    size_t size_logits = config->vocab_size * sizeof(float);


    // 2. 将各缓冲区大小相加，得到总字节数。
    size_t total_size = size_x + size_before_attn_norm_x + size_q + size_k + size_v +
                        size_q_rot + size_k_rot + size_key_cache + size_value_cache +
                        size_attn_score + 
                        size_softmax_attn_score + size_weighted_value + size_attn_o +
                        size_residual_after_attn + size_after_attn_norm_x +
                        size_g + size_b + size_s + size_f + size_m +
                        size_residual_after_mlp +
                        size_final_norm + size_logits;


    // 3. 一次性分配 GPU 内存，将基地址保存到 state。
    float* base_ptr_in_device;
    cudaError_t err = cudaMalloc((void**)&base_ptr_in_device, total_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    state->device_data = base_ptr_in_device;


    // 4. 给RunState中的每个成员指针赋值
    /*
    初始化：指向 base pointer。
    赋值：让当前成员指向 cursor。
    移动：cursor 加上当前缓冲区的元素数。
    对下一个成员重复第 2、3 步。
    */    
    float* cursor = base_ptr_in_device;
    state->x = cursor ; cursor += config->dim;
    state->before_attn_norm_x = cursor ; cursor += config->dim;
    state->q = cursor ; cursor += config->dim;
    state->k = cursor ; cursor += config->dim / config->n_heads * config->n_kv_heads;
    state->v = cursor ; cursor += config->dim / config->n_heads * config->n_kv_heads;
    state->q_rot = cursor ; cursor += config->dim;
    state->k_rot = cursor ; cursor += config->dim / config->n_heads * config->n_kv_heads;
    state->key_cache = cursor ; cursor += config->n_layers * state->max_seq_len * (config->dim / config->n_heads * config->n_kv_heads);
    state->value_cache = cursor ; cursor += config->n_layers * state->max_seq_len * (config->dim / config->n_heads * config->n_kv_heads);
    state->attn_score = cursor ; cursor += config->n_heads * state->max_seq_len;
    state->softmax_attn_score = cursor ; cursor += config->n_heads * state->max_seq_len;
    state->weighted_value = cursor ; cursor += config->n_heads * (config->dim / config->n_heads);
    state->attn_o = cursor ; cursor += config->dim;
    state->residual_after_attn = cursor ; cursor += config->dim;
    state->after_attn_norm_x = cursor ; cursor += config->dim;
    state->g = cursor ; cursor += config->hidden_dim;
    state->b = cursor ; cursor += config->hidden_dim;
    state->s = cursor ; cursor += config->hidden_dim;
    state->f = cursor ; cursor += config->hidden_dim;
    state->m = cursor ; cursor += config->dim;
    state->residual_after_mlp = cursor ; cursor += config->dim;
    state->final_norm = cursor ; cursor += config->dim;
    state->logits = cursor ; cursor += config->vocab_size;  
}



void free_runstate(RunState* state) {
    cudaError_t err = cudaFree(state->device_data);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
    }
}
