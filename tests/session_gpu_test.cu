// 真实 GPU：Session 端到端 token 对照、长度边界、跨轮及析构后的显存释放。
#include <cassert>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "src/session/session.h"

static size_t free_memory()
{
    size_t available, total;
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemGetInfo(&available, &total) != cudaSuccess) {
        throw std::runtime_error("cannot query GPU memory");
    }
    return available;
}

int main(int argc, char** argv)
{
    assert(argc == 4);
    std::ifstream reference_file(argv[3]);
    assert(reference_file.good());
    std::vector<int> reference;
    std::string line;
    while (std::getline(reference_file, line) && line != "DONE") {
        reference.push_back(std::stoi(line));
    }
    assert(!reference.empty() && line == "DONE");
    const std::string prompt = "你好，请介绍一下自己。";
    Template chat_template;
    Tokenizer tokenizer(chat_template, argv[2]);
    int prompt_length = tokenizer.encode(tokenizer.apply_chat_template(prompt)).size();
    GenerationOptions options;
    size_t weights_live;
    const size_t tolerance = 16 * 1024 * 1024; // 允许驱动小量懒加载，不允许一轮 KV cache 泄漏。
    {
        Session session(argv[1], argv[2], options);
        TurnResult first = session.run_turn(prompt);
        assert(first.generation.tokens == reference);
        assert(first.generation.stop_reason == STOP_EOS);
        printf("Session single turn: %zu token IDs match reference\n", reference.size());
        weights_live = free_memory();

        // 多次失败不应保存历史，也不能留下 RunState。
        for (int i = 0; i < 2; ++i) {
            bool caught = false;
            try {
                session.run_turn(std::string(5000, '1'));
            } catch (const std::length_error&) {
                caught = true;
            }
            assert(caught);
            assert(free_memory() + tolerance >= weights_live);
        }
        session.clear();
        TurnResult second = session.run_turn(prompt);
        assert(second.generation.tokens == reference);
        assert(free_memory() + tolerance >= weights_live);
        printf("Session clear + retry: identical tokens; no per-turn GPU memory growth\n");
    }
    size_t released = free_memory();
    assert(released > weights_live + 6000000000ULL);
    printf("Session destruction: GPU free bytes increased by %zu\n", released - weights_live);

    // 将有效上下文上限调小，在 GPU 上测试剩余 1 格及恰好填满的边界。
    options.max_context_tokens = prompt_length + 1;
    {
        Session session(argv[1], argv[2], options);
        TurnResult result = session.run_turn(prompt);
        assert(result.generation.tokens.size() == 1);
        assert(result.generation.tokens[0] == reference[0]);
        assert(result.generation.stop_reason == STOP_CONTEXT_LIMIT);
    }
    options.max_context_tokens = prompt_length;
    {
        Session session(argv[1], argv[2], options);
        TurnResult result = session.run_turn(prompt);
        assert(result.generation.tokens.empty());
        assert(result.generation.stop_reason == STOP_CONTEXT_LIMIT);
    }
    options.max_context_tokens = MAX_SEQ_LEN;
    options.max_new_tokens = 1;
    {
        Session session(argv[1], argv[2], options);
        TurnResult result = session.run_turn(prompt);
        assert(result.generation.tokens.size() == 1);
        assert(result.generation.tokens[0] == reference[0]);
        assert(result.generation.stop_reason == STOP_MAX_NEW_TOKENS);
    }
    options.max_new_tokens = 0;
    {
        Session session(argv[1], argv[2], options);
        TurnResult result = session.run_turn(prompt);
        assert(result.generation.tokens.empty());
        assert(result.generation.stop_reason == STOP_MAX_NEW_TOKENS);
    }
    printf("GPU length checks passed: max-new 0/1, context full/one free slot\n");
}
