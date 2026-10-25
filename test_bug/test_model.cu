#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "../model.h"

const int N_MEMBERS = 23;

void print_line() {
    printf("------------------------------------------------------------------------------------------------\n");
}

// 打印 Model 在 GPU 上的整体布局（只看几个关键张量）
void print_model(Model* model) {
    Config* c = &model->config;
    printf("\n[Model weights on GPU]\n");
    print_line();
    printf("config: dim=%d hidden_dim=%d n_layers=%d n_heads=%d n_kv_heads=%d vocab_size=%d\n",
           c->dim, c->hidden_dim, c->n_layers, c->n_heads, c->n_kv_heads, c->vocab_size);
    printf("%-24s %18p\n", "device_data (base)", model->device_data);
    printf("%-24s %18p\n", "embedding", (void*)model->embedding);
    printf("%-24s %18p\n", "layers[0].wq", (void*)model->layers[0].wq);
    printf("%-24s %18p\n", "layers[last].w_down", (void*)model->layers[c->n_layers - 1].w_down);
    printf("%-24s %18p\n", "final_norm", (void*)model->final_norm);
    printf("%-24s %18p (== embedding: %s)\n", "lm_head", (void*)model->lm_head,
           model->lm_head == model->embedding ? "yes" : "no");
}

// 打印 RunState 每个成员：期望字节数、GPU 地址、相对基地址偏移、到下一个成员的实际间距
void print_runstate(RunState* state, Config* c) {
    int dim = c->dim;
    int kv_dim = c->dim / c->n_heads * c->n_kv_heads;
    int head_dim = c->dim / c->n_heads;
    size_t f = sizeof(float);

    const char* names[N_MEMBERS] = {
        "x", "before_attn_norm_x", "q", "k", "v", "q_rot", "k_rot",
        "key_cache", "value_cache", "attn_score", "softmax_attn_score",
        "weighted_value", "attn_o", "residual_after_attn", "after_attn_norm_x",
        "g", "b", "s", "f", "m", "residual_after_mlp", "final_norm", "logits"
    };

    float* ptrs[N_MEMBERS] = {
        state->x, state->before_attn_norm_x, state->q, state->k, state->v,
        state->q_rot, state->k_rot, state->key_cache, state->value_cache,
        state->attn_score, state->softmax_attn_score, state->weighted_value,
        state->attn_o, state->residual_after_attn, state->after_attn_norm_x,
        state->g, state->b, state->s, state->f, state->m,
        state->residual_after_mlp, state->final_norm, state->logits
    };

    // 期望字节数：在这里独立按 config 重新算一遍，用来对照 load_runstate 的实际布局
    size_t sizes[N_MEMBERS] = {
        dim * f, dim * f, dim * f, kv_dim * f, kv_dim * f, dim * f, kv_dim * f,
        (size_t)c->n_layers * state->max_seq_len * kv_dim * f,
        (size_t)c->n_layers * state->max_seq_len * kv_dim * f,
        (size_t)c->n_heads * state->max_seq_len * f,
        (size_t)c->n_heads * state->max_seq_len * f,
        (size_t)c->n_heads * head_dim * f,
        dim * f, dim * f, dim * f,
        (size_t)c->hidden_dim * f, (size_t)c->hidden_dim * f,
        (size_t)c->hidden_dim * f, (size_t)c->hidden_dim * f,
        dim * f, dim * f, dim * f,
        (size_t)c->vocab_size * f
    };

    size_t total_expected = 0;
    for (int i = 0; i < N_MEMBERS; i++) {
        total_expected += sizes[i];
    }

    char* base = (char*)state->device_data;
    char* end_of_block = base + total_expected;

    printf("\n[RunState on GPU]  max_seq_len=%d\n", state->max_seq_len);
    printf("base (device_data) = %p, expected total = %zu bytes (%.2f MiB)\n",
           (void*)base, total_expected, total_expected / 1048576.0);
    print_line();
    printf("%-20s %12s %10s  %-16s %12s %12s  %s\n",
           "member", "bytes", "MiB", "gpu address", "offset", "gap_to_next", "check");
    print_line();

    for (int i = 0; i < N_MEMBERS; i++) {
        if (ptrs[i] == NULL) {
            printf("%-20s %12zu %10.4f  %-16s %12s %12s  %s\n",
                   names[i], sizes[i], sizes[i] / 1048576.0,
                   "(null)", "-", "-", "NOT ASSIGNED in load_runstate");
            continue;
        }

        // 下一个已赋值成员的地址；最后一个成员则对照整块内存的末尾
        char* next_addr = end_of_block;
        for (int j = i + 1; j < N_MEMBERS; j++) {
            if (ptrs[j] != NULL) {
                next_addr = (char*)ptrs[j];
                break;
            }
        }

        long long offset = (char*)ptrs[i] - base;
        long long gap = next_addr - (char*)ptrs[i];
        const char* check = ((size_t)gap == sizes[i]) ? "ok" : "MISMATCH (gap != bytes)";

        printf("%-20s %12zu %10.4f  %-16p %12lld %12lld  %s\n",
               names[i], sizes[i], sizes[i] / 1048576.0,
               (void*)ptrs[i], offset, gap, check);
    }
    print_line();
}

int main() {
    Model model;
    RunState state;
    memset(&model, 0, sizeof(Model));
    memset(&state, 0, sizeof(RunState)); // 未赋值的指针保持 NULL，便于发现遗漏

    load_model_weights(&model);
    load_runstate(&state, &model.config);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // 确认基地址确实是 device 内存
    cudaPointerAttributes attr;
    err = cudaPointerGetAttributes(&attr, state.device_data);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaPointerGetAttributes failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    printf("RunState base pointer type: %s\n",
           attr.type == cudaMemoryTypeDevice ? "device (GPU)" : "NOT device");

    print_model(&model);
    print_runstate(&state, &model.config);

    free_runstate(&state);
    free_model_weights(&model);
    return 0;
}
