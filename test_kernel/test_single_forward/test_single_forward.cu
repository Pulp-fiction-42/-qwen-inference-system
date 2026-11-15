// 按原单 block 测试的输入和输出约定，调用重构后的 Qwen2Layer。
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "src/model/layer/layer.h"
#include "ops/ops.h"

static void check_cuda(cudaError_t error)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "single forward test: %s\n", cudaGetErrorString(error));
        exit(1);
    }
}

// 仅输出有效元素，score 保留原来的 head * max_seq_len 下标。
// %.9g 保留 float32 有效精度，符合原 inspect_forward.py 的输入约定。
static void print_values(const char* name, const float* device, int count, int stride = 1)
{
    int capacity = (count - 1) * stride + 1;
    float* host = (float*)malloc((size_t)capacity * sizeof(float));
    if (host == nullptr) {
        fprintf(stderr, "host allocation failed\n");
        exit(1);
    }
    check_cuda(cudaMemcpy(host, device, (size_t)capacity * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < count; ++i) {
        int index = i * stride;
        printf("%s[%d]: %.9g\n", name, index, host[index]);
    }
    free(host);
}

int main()
{
    Model model = {};
    RunState state = {};
    load_model_weights(&model);
    load_runstate(&state, &model.config);
    const Config& config = model.config;

    int token = 123;
    int position = 0;
    int dim = config.dim;
    int kv_dim = dim / config.n_heads * config.n_kv_heads;

    op_embedding(token, model.embedding, dim, state.x);
    Qwen2Layer layer(model.layers[0]);
    layer.forward(state, config, state.x, 0, position);
    op_final_rms_norm(model.final_norm, state.residual_after_mlp,
                      state.final_norm, dim, state.rms_scratch, 1e-6f);
    op_lm_head_proj(model.lm_head, nullptr, state.final_norm, state.logits,
                    config.vocab_size, dim, dim);
    op_synchronize();

    print_values("x", state.x, dim);
    print_values("before_attn_norm_x", state.before_attn_norm_x, dim);
    print_values("q", state.q, dim);
    print_values("k", state.k, kv_dim);
    print_values("v", state.v, kv_dim);
    print_values("q_rot", state.q_rot, dim);
    print_values("k_rot", state.k_rot, kv_dim);
    print_values("key_cache", state.key_cache, kv_dim);
    print_values("value_cache", state.value_cache, kv_dim);
    print_values("attn_score", state.attn_score, config.n_heads, state.max_seq_len);
    print_values("softmax_attn_score", state.softmax_attn_score,
                  config.n_heads, state.max_seq_len);
    print_values("weighted_value", state.weighted_value, dim);
    print_values("attn_o", state.attn_o, dim);
    print_values("residual_after_attn", state.residual_after_attn, dim);
    print_values("after_attn_norm_x", state.after_attn_norm_x, dim);
    print_values("g", state.g, config.hidden_dim);
    print_values("b", state.b, config.hidden_dim);
    print_values("s", state.s, config.hidden_dim);
    print_values("f", state.f, config.hidden_dim);
    print_values("m", state.m, dim);
    print_values("residual_after_mlp", state.residual_after_mlp, dim);
    print_values("final_norm", state.final_norm, dim);
    print_values("logits", state.logits, config.vocab_size);

    free_runstate(&state);
    free_model_weights(&model);
    return 0;
}
