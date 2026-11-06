// Host 调度测试：用普通函数替代 CUDA 算子，检查顺序、指针和尺寸。
// 不执行 kernel，不验证原 kernel 的数值正确性。
#include <cassert>
#include <cstdio>
#include <cstring>

#include "src/model/model.h"
#include "ops/ops.h"

static Model model;
static RunState state;
static int calls;
static int layer_index;
static int position;
static int token_id;

static const char* layer_calls[] = {
    "norm", "proj_bias", "proj_bias", "proj_bias", "rope", "rope",
    "copy", "copy", "score", "softmax", "weighted_sum", "proj",
    "attn_residual", "norm", "mlp_proj", "mlp_proj", "silu", "mul",
    "mlp_proj", "mlp_residual"
};

static void record(const char* name)
{
    const char* expected;
    int layers_end = 1 + model.config.n_layers * 20;
    if (calls == 0) {
        expected = "embedding";
    } else if (calls < layers_end) {
        expected = layer_calls[(calls - 1) % 20];
    } else {
        const char* final_calls[] = {"final_norm", "lm_head", "synchronize"};
        assert(calls < layers_end + 3);
        expected = final_calls[calls - layers_end];
    }
    assert(std::strcmp(name, expected) == 0);
    ++calls;
}

static const float* layer_input()
{
    if (layer_index == 0) {
        return state.x;
    }
    return state.residual_after_mlp;
}

static int kv_dim()
{
    return model.config.dim / model.config.n_heads * model.config.n_kv_heads;
}

static int cache_offset()
{
    return layer_index * state.max_seq_len * kv_dim();
}

void op_embedding(int token, const float* weights, int dim, float* output)
{
    record("embedding");
    assert(token == token_id && weights == model.embedding);
    assert(dim == model.config.dim && output == state.x);
}

void op_rms_norm(const float* weights, const float* input, float* output,
                 int dim, float* scratch, float eps)
{
    record("norm");
    assert(dim == model.config.dim && scratch == state.rms_scratch);
    assert(eps == 1e-6f);
    if (output == state.before_attn_norm_x) {
        assert(weights == model.layers[layer_index].before_attn_norm);
        assert(input == layer_input());
    } else {
        assert(output == state.after_attn_norm_x);
        assert(weights == model.layers[layer_index].post_attn_norm);
        assert(input == state.residual_after_attn);
    }
}

void op_final_rms_norm(const float* weights, const float* input, float* output,
                       int dim, float* scratch, float eps)
{
    record("final_norm");
    assert(weights == model.final_norm && input == state.residual_after_mlp);
    assert(output == state.final_norm && dim == model.config.dim);
    assert(scratch == state.rms_scratch && eps == 1e-6f);
}

void op_proj_bias(const float* weights, const float* bias,
                  const float* input, float* output, int dim)
{
    record("proj_bias");
    const blocks& layer = model.layers[layer_index];
    assert(input == state.before_attn_norm_x && dim == model.config.dim);
    if (output == state.q) {
        assert(weights == layer.wq && bias == layer.bq);
    } else if (output == state.k) {
        assert(weights == layer.wk && bias == layer.bk);
    } else {
        assert(output == state.v && weights == layer.wv && bias == layer.bv);
    }
}

void op_rope(const float* input, float* output, int p,
             int heads, int head_dim, float theta)
{
    record("rope");
    assert(p == position && theta == 1000000.0f);
    assert(head_dim == model.config.dim / model.config.n_heads);
    if (input == state.q) {
        assert(output == state.q_rot && heads == model.config.n_heads);
    } else {
        assert(input == state.k && output == state.k_rot);
        assert(heads == model.config.n_kv_heads);
    }
}

void op_copy(float* destination, const float* source, int count)
{
    record("copy");
    assert(count == kv_dim());
    int offset = cache_offset() + position * kv_dim();
    if (source == state.k_rot) {
        assert(destination == state.key_cache + offset);
    } else {
        assert(source == state.v && destination == state.value_cache + offset);
    }
}

void op_gqa_score(const float* query, const float* keys, float* scores,
                  int heads, int kv_heads, int head_dim, int length, int stride)
{
    record("score");
    assert(query == state.q_rot && keys == state.key_cache + cache_offset());
    assert(scores == state.attn_score);
    assert(heads == model.config.n_heads && kv_heads == model.config.n_kv_heads);
    assert(head_dim == model.config.dim / heads);
    assert(length == position + 1 && stride == state.max_seq_len);
}

void op_softmax(const float* input, float* output, int rows, int length, int stride)
{
    record("softmax");
    assert(input == state.attn_score && output == state.softmax_attn_score);
    assert(rows == model.config.n_heads);
    assert(length == position + 1 && stride == state.max_seq_len);
}

void op_gqa_weighted_sum(const float* probabilities, const float* values,
                         float* output, int heads, int kv_heads,
                         int head_dim, int length, int stride)
{
    record("weighted_sum");
    assert(probabilities == state.softmax_attn_score);
    assert(values == state.value_cache + cache_offset());
    assert(output == state.weighted_value);
    assert(heads == model.config.n_heads && kv_heads == model.config.n_kv_heads);
    assert(head_dim == model.config.dim / heads);
    assert(length == position + 1 && stride == state.max_seq_len);
}

void op_proj(const float* weights, const float* input, float* output, int dim)
{
    record("proj");
    assert(weights == model.layers[layer_index].wo);
    assert(input == state.weighted_value && output == state.attn_o);
    assert(dim == model.config.dim);
}

void op_attn_residual(const float* input, const float* attention, float* output, int dim)
{
    record("attn_residual");
    assert(input == layer_input() && attention == state.attn_o);
    assert(output == state.residual_after_attn && dim == model.config.dim);
}

void op_mlp_proj(const float* weights, const float* input, float* output,
                 int out_dim, int in_dim)
{
    record("mlp_proj");
    const blocks& layer = model.layers[layer_index];
    if (output == state.m) {
        assert(weights == layer.w_down && input == state.f);
        assert(out_dim == model.config.dim && in_dim == model.config.hidden_dim);
    } else {
        assert(input == state.after_attn_norm_x);
        assert(out_dim == model.config.hidden_dim && in_dim == model.config.dim);
        if (output == state.g) {
            assert(weights == layer.w_gate);
        } else {
            assert(output == state.b && weights == layer.w_up);
        }
    }
}

void op_silu(const float* input, float* output, int dim)
{
    record("silu");
    assert(input == state.g && output == state.s && dim == model.config.hidden_dim);
}

void op_mul(const float* left, const float* right, float* output, int dim)
{
    record("mul");
    assert(left == state.s && right == state.b && output == state.f);
    assert(dim == model.config.hidden_dim);
}

void op_mlp_residual(const float* residual, const float* mlp, float* output, int dim)
{
    record("mlp_residual");
    assert(residual == state.residual_after_attn && mlp == state.m);
    assert(output == state.residual_after_mlp && dim == model.config.dim);
    ++layer_index;
}

void op_lm_head_proj(const float* weights, const float* bias,
                     const float* input, float* output, int rows, int cols, int dim)
{
    record("lm_head");
    assert(weights == model.lm_head && bias == nullptr);
    assert(input == state.final_norm && output == state.logits);
    assert(rows == model.config.vocab_size);
    assert(cols == model.config.dim && dim == model.config.dim);
}

void op_synchronize()
{
    record("synchronize");
}

int main()
{
    // 小尺寸仅用于 host 指针/参数检查，不表示原 CUDA kernel 支持这些尺寸。
    float buffers[24][64] = {};
    float weights[2][12] = {};
    float embedding[64] = {};
    float final_norm[8] = {};
    blocks layers[2] = {};

    model.config = {8, 16, 2, 4, 2, 8};
    model.embedding = embedding;
    model.lm_head = embedding;
    model.final_norm = final_norm;
    model.layers = layers;

    for (int i = 0; i < 2; ++i) {
        layers[i].before_attn_norm = &weights[i][0];
        layers[i].wq = &weights[i][1];
        layers[i].bq = &weights[i][2];
        layers[i].wk = &weights[i][3];
        layers[i].bk = &weights[i][4];
        layers[i].wv = &weights[i][5];
        layers[i].bv = &weights[i][6];
        layers[i].wo = &weights[i][7];
        layers[i].post_attn_norm = &weights[i][8];
        layers[i].w_gate = &weights[i][9];
        layers[i].w_up = &weights[i][10];
        layers[i].w_down = &weights[i][11];
    }

    state.max_seq_len = 4;
    state.x = buffers[0];
    state.before_attn_norm_x = buffers[1];
    state.q = buffers[2];
    state.k = buffers[3];
    state.v = buffers[4];
    state.q_rot = buffers[5];
    state.k_rot = buffers[6];
    state.key_cache = buffers[7];
    state.value_cache = buffers[8];
    state.attn_score = buffers[9];
    state.softmax_attn_score = buffers[10];
    state.weighted_value = buffers[11];
    state.attn_o = buffers[12];
    state.residual_after_attn = buffers[13];
    state.after_attn_norm_x = buffers[14];
    state.g = buffers[15];
    state.b = buffers[16];
    state.s = buffers[17];
    state.f = buffers[18];
    state.m = buffers[19];
    state.residual_after_mlp = buffers[20];
    state.final_norm = buffers[21];
    state.logits = buffers[22];
    state.rms_scratch = buffers[23];

    for (int layer_count = 1; layer_count <= 2; ++layer_count) {
        model.config.n_layers = layer_count;
        Qwen2Model qwen(model);
        for (position = 0; position < state.max_seq_len; ++position) {
            calls = 0;
            layer_index = 0;
            token_id = position + 1;
            qwen.forward(state, token_id, position);
            assert(layer_index == layer_count);
            assert(calls == 4 + layer_count * 20);
            // forward 不应改变这些由外部所有者持有的缓冲区指针。
            assert(state.x == buffers[0]);
            assert(state.residual_after_mlp == buffers[20]);
            assert(state.rms_scratch == buffers[23]);
        }
    }
    std::puts("model forward host checks passed");
}
