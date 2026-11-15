// 不启动 GPU：替代 Model/资源/argmax，验证生成边界、历史提交和异常清理。
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "src/session/session.h"
#include "ops/ops.h"

static std::vector<int> forwarded;
static std::vector<int> sampled;
static size_t sample_index;
static int live_models;
static int live_states;
static bool fail_forward;
static bool fail_allocate;
static bool fail_model;
static float buffer;
static int index_buffer;

Qwen2Model::Qwen2Model(Model& model) : model_(model) {}
const Config& Qwen2Model::config() const { return model_.config; }
void Qwen2Model::forward(RunState&, int token, int position)
{
    if (fail_forward) {
        throw std::runtime_error("injected forward failure");
    }
    assert(position == (int)forwarded.size());
    forwarded.push_back(token);
}
void load_model_weights(Model* model, const char*)
{
    if (fail_model) {
        throw std::runtime_error("injected load failure");
    }
    *model = {};
    model->config.vocab_size = 151936;
    model->device_data = &buffer;
    ++live_models;
}
void free_model_weights(Model* model)
{
    if (model->device_data != nullptr) {
        --live_models;
        *model = {};
    }
}
void load_runstate(RunState* state, Config*)
{
    if (fail_allocate) {
        throw std::runtime_error("injected allocation failure");
    }
    *state = {};
    state->device_data = &buffer;
    state->max_seq_len = MAX_SEQ_LEN;
    state->logits = &buffer;
    state->sampler_values = &buffer;
    state->sampler_indices = &index_buffer;
    state->sampler_token = &index_buffer;
    ++live_states;
}
void free_runstate(RunState* state)
{
    if (state->device_data != nullptr) {
        --live_states;
        *state = {};
    }
}
int op_argmax(float* logits, int vocab, float* values, int* indices, int* token)
{
    assert(logits == &buffer && values == &buffer);
    assert(indices == &index_buffer && token == &index_buffer && vocab == 151936);
    assert(sample_index < sampled.size());
    return sampled[sample_index++];
}
static void reset(const std::vector<int>& tokens)
{
    forwarded.clear();
    sampled = tokens;
    sample_index = 0;
}

static void test_generator()
{
    Model model = {};
    RunState state = {};
    load_model_weights(&model, "unused");
    load_runstate(&state, &model.config);
    Qwen2Model qwen(model);
    Generator generator(qwen, state);
    GenerationOptions options;
    std::vector<int> prompt = {1, 2, 3};

    reset({4, 5, 151645});
    GenerationResult result = generator.generate(prompt, options);
    assert(result.tokens == std::vector<int>({4, 5}));
    assert(result.stop_reason == STOP_EOS);
    assert(forwarded == std::vector<int>({1, 2, 3, 4, 5}));

    reset({151643});
    result = generator.generate(prompt, options);
    assert(result.tokens.empty() && result.stop_reason == STOP_EOS);
    assert(forwarded == prompt);

    options.max_new_tokens = 2;
    reset({4, 5});
    result = generator.generate(prompt, options);
    assert(result.tokens.size() == 2 && result.stop_reason == STOP_MAX_NEW_TOKENS);
    assert(forwarded == std::vector<int>({1, 2, 3, 4}));

    options.max_context_tokens = 4;
    reset({4});
    result = generator.generate(prompt, options);
    assert(result.tokens.size() == 1 && result.stop_reason == STOP_CONTEXT_LIMIT);
    assert(forwarded == prompt);

    options.max_context_tokens = 3;
    reset({});
    result = generator.generate(prompt, options);
    assert(result.tokens.empty() && result.stop_reason == STOP_CONTEXT_LIMIT);
    assert(forwarded.empty());

    options.max_context_tokens = MAX_SEQ_LEN;
    options.max_new_tokens = 0;
    result = generator.generate(prompt, options);
    assert(result.tokens.empty() && result.stop_reason == STOP_MAX_NEW_TOKENS);
    assert(forwarded.empty());

    // 所有非法参数都必须在第一次 forward 前被拒绝。
    for (int case_id = 0; case_id < 6; ++case_id) {
        options = {};
        std::vector<int> invalid = prompt;
        if (case_id == 0) invalid.clear();
        if (case_id == 1) options.max_new_tokens = -1;
        if (case_id == 2) options.max_context_tokens = MAX_SEQ_LEN + 1;
        if (case_id == 3) options.max_context_tokens = 2;
        if (case_id == 4) invalid[0] = -1;
        if (case_id == 5) invalid[2] = 151936;
        bool caught = false;
        try {
            generator.generate(invalid, options);
        } catch (const std::exception&) {
            caught = true;
        }
        assert(caught && forwarded.empty());
    }
    options = {};
    reset({151936});
    bool caught = false;
    try {
        generator.generate(prompt, options);
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught);
    free_runstate(&state);
    free_model_weights(&model);
    assert(live_states == 0 && live_models == 0);
}

static void test_session(const char* vocab_path)
{
    Template chat_template;
    Tokenizer tokenizer(chat_template, vocab_path);
    GenerationOptions options;
    {
        Session session("unused", vocab_path, options);
        assert(live_models == 1);
        std::vector<ChatMessage> expected;
        expected.push_back({"user", "你好"});
        reset({32, 151645}); // token 32 = A
        TurnResult result = session.run_turn("你好");
        assert(result.text == "A" && live_states == 0);
        std::vector<int> expected_forward = tokenizer.encode(tokenizer.apply_chat_template(expected));
        expected_forward.push_back(32);
        assert(forwarded == expected_forward);
        expected.push_back({"assistant", "A"});

        // 生成中失败和分配失败均不提交历史；RunState 被释放。
        for (int case_id = 0; case_id < 2; ++case_id) {
            fail_forward = case_id == 0;
            fail_allocate = case_id == 1;
            bool caught = false;
            try {
                session.run_turn("不应保存这轮");
            } catch (const std::runtime_error&) {
                caught = true;
            }
            assert(caught && live_states == 0 && live_models == 1);
        }
        fail_forward = false;
        fail_allocate = false;
        expected.push_back({"user", "第二轮"});
        reset({151643});
        session.run_turn("第二轮");
        assert(forwarded == tokenizer.encode(tokenizer.apply_chat_template(expected)));
        assert(live_states == 0);

        session.clear();
        reset({151645});
        session.run_turn("重来");
        assert(forwarded == tokenizer.encode(tokenizer.apply_chat_template("重来")));
        assert(live_states == 0);
    }
    assert(live_models == 0);
    fail_model = true;
    bool caught = false;
    try {
        Session session("missing", vocab_path, options);
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught && live_models == 0);
    fail_model = false;
    std::puts("runtime checks passed: EOS, lengths, positions, history, clear, failure cleanup");
}

int main(int argc, char** argv)
{
    assert(argc == 2);
    test_generator();
    test_session(argv[1]);
}
