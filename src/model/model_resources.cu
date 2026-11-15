// 迁自原 model.cu。Session 决定 RunState 的申请和释放时机。
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "model_types.h"
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>




/*
allocate GPU memory, load model weights from file to GPU
initialize the Struct: Model, Config, blocks
*/
void load_model_weights(Model* model, const char* path)
{

    // 文件与 GPU 分配失败时逐项释放，不能 exit，否则 Session 无法清理。
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(std::string("无法打开权重文件: ") + path);
    }
    struct stat info;
    if (fstat(fd, &info) != 0 || info.st_size < (off_t)(6 * sizeof(std::int32_t))) {
        close(fd);
        throw std::runtime_error("权重文件 header 不完整");
    }
    std::size_t bytes = info.st_size;
    void* host_model_bin = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (host_model_bin == MAP_FAILED) {
        throw std::runtime_error("映射权重文件失败");
    }

    const std::int32_t* header = (const std::int32_t*)host_model_bin;
    Config c = {header[0], header[1], header[2], header[3], header[4], header[5]};
    // 原 kernel 使用固定维度；当前只接受这个布局，不声称支持任意 Qwen 尺寸。
    if (c.dim != 1536 || c.hidden_dim != 8960 || c.n_layers <= 0 ||
        c.n_heads != 12 || c.n_kv_heads != 2 || c.vocab_size != 151936) {
        munmap(host_model_bin, bytes);
        throw std::runtime_error("权重配置不符合当前 CUDA kernel 的固定尺寸");
    }
    std::size_t dim = c.dim;
    std::size_t hidden = c.hidden_dim;
    std::size_t kv_dim = dim / c.n_heads * c.n_kv_heads;
    std::size_t layer_floats = 2 * dim * dim + 2 * kv_dim * dim +
                               3 * hidden * dim + 3 * dim + 2 * kv_dim;
    std::size_t expected_bytes = 6 * sizeof(std::int32_t) + sizeof(float) *
        (c.vocab_size * dim + c.n_layers * layer_floats + dim);
    if (bytes != expected_bytes) {
        munmap(host_model_bin, bytes);
        throw std::runtime_error("权重文件大小与 header 不符，要求原 FP32、共享 LM head 布局");
    }

    void* device_data = nullptr;
    cudaError_t err = cudaMalloc(&device_data, bytes);
    if (err != cudaSuccess) {
        munmap(host_model_bin, bytes);
        throw std::runtime_error(std::string("权重 cudaMalloc: ") + cudaGetErrorString(err));
    }
    err = cudaMemcpy(device_data, host_model_bin, bytes, cudaMemcpyHostToDevice);
    munmap(host_model_bin, bytes);
    if (err != cudaSuccess) {
        cudaFree(device_data);
        throw std::runtime_error(std::string("权重 cudaMemcpy: ") + cudaGetErrorString(err));
    }
    blocks* layers = new (std::nothrow) blocks[c.n_layers]{};
    if (layers == nullptr) {
        cudaFree(device_data);
        throw std::runtime_error("分配权重地址表失败");
    }
    model->config = c;
    model->device_data = device_data;
    model->mapped_size = bytes;
    model->layers = layers;

    // 跳过 header，cursor 保存 GPU 地址, initial cursor is first element of Weights's address
    const float* cursor = (const float*)((char*)model->device_data
        + 6 * sizeof(std::int32_t));

    // 显式记录地址并移动 cursor，保持原权重布局。

    model->embedding = cursor;

    cursor += c.vocab_size * dim;

    for (int i = 0; i < c.n_layers; ++i) {
        blocks& layer = model->layers[i];

        layer.before_attn_norm = cursor;

        cursor += dim;
        layer.wq = cursor;
        cursor += dim * dim;
        layer.bq = cursor;
        cursor += dim;
        layer.wk = cursor;
        cursor += kv_dim * dim;
        layer.bk = cursor;
        cursor += kv_dim;
        layer.wv = cursor;
        cursor += kv_dim * dim;
        layer.bv = cursor;
        cursor += kv_dim;
        layer.wo = cursor;
        cursor += dim * dim;
        layer.post_attn_norm = cursor;
        cursor += dim;
        layer.w_gate = cursor;
        cursor += hidden * dim;
        layer.w_up = cursor;
        cursor += hidden * dim;
        layer.w_down = cursor;
        cursor += dim * hidden;
    }

    model->final_norm = cursor;

    cursor += dim;
    model->lm_head = model->embedding;
}



void free_model_weights(Model* model){
    cudaError_t err = cudaFree(model->device_data);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
    }
    delete[] model->layers;
    *model = {};
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


    // argmax 临时区追加在原缓冲区末尾，不改变已有数据的偏移。
    size_t sampler_blocks = (config->vocab_size + 255) / 256;
    size_t sampler_bytes = sampler_blocks * (sizeof(float) + sizeof(int)) + sizeof(int);

    // 2. 将各缓冲区大小相加，得到总字节数。
    size_t total_size = size_x + size_before_attn_norm_x + size_q + size_k + size_v +
                        size_q_rot + size_k_rot + size_key_cache + size_value_cache +
                        size_attn_score + 
                        size_softmax_attn_score + size_weighted_value + size_attn_o +
                        size_residual_after_attn + size_after_attn_norm_x +
                        size_g + size_b + size_s + size_f + size_m +
                        size_residual_after_mlp +
                        size_final_norm + size_logits + sizeof(float) + sampler_bytes;


    // 3. 一次性分配 GPU 内存，将基地址保存到 state。
    float* base_ptr_in_device;
    cudaError_t err = cudaMalloc((void**)&base_ptr_in_device, total_size);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("RunState cudaMalloc: ") + cudaGetErrorString(err));
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
    // 原 forward 中的临时 r，随本次 generate 的 RunState 一起释放。
    state->rms_scratch = cursor;
    ++cursor;
    state->sampler_values = cursor;
    cursor += sampler_blocks;
    state->sampler_indices = (int*)cursor;
    state->sampler_token = state->sampler_indices + sampler_blocks;
}



void free_runstate(RunState* state) {
    if (state->device_data == nullptr) {
        return;
    }
    cudaError_t err = cudaFree(state->device_data);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
    }
    *state = {};
}
