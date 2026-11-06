// 保留原三项测试的 token 协议，调用生产 Generator/Sampler。
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>
#include "src/generator/generator.h"

int main()
{
    Model model = {};
    RunState state = {};
    int status = 0;
    try {
        load_model_weights(&model);
        Qwen2Model qwen(model);
        GenerationOptions options;
        options.max_new_tokens = MAX_SEQ_LEN; // 原测试仅受总上下文上限约束。
        int length;
        while (scanf("%d", &length) == 1) {
            if (length <= 0 || length > MAX_SEQ_LEN) {
                throw std::invalid_argument("invalid prompt length");
            }
            std::vector<int> prompt(length);
            for (int i = 0; i < length; ++i) {
                if (scanf("%d", &prompt[i]) != 1) {
                    throw std::runtime_error("incomplete prompt");
                }
            }
            load_runstate(&state, &model.config);
            Generator generator(qwen, state);
            GenerationResult result = generator.generate(prompt, options);
            free_runstate(&state);
            for (size_t i = 0; i < result.tokens.size(); ++i) {
                printf("%d\n", result.tokens[i]);
            }
            printf("DONE\n");
            fflush(stdout);
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "engine: %s\n", error.what());
        status = 1;
    }
    free_runstate(&state);
    free_model_weights(&model);
    return status;
}
