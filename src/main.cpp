#include "session/session.h"
#include "terminal/terminal.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

static int parse_count(const char* text)
{
    char* end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 || value > INT_MAX) {
        throw std::invalid_argument("长度参数必须是非负整数");
    }
    return (int)value;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "用法: " << argv[0]
                  << " WEIGHTS.bin VOCAB.gguf [--max-new-tokens N] [--context-limit N]\n";
        return 1;
    }
    try {
        GenerationOptions options;
        for (int i = 3; i < argc; i += 2) {
            if (i + 1 >= argc) {
                throw std::invalid_argument("选项缺少长度参数");
            }
            std::string option = argv[i];
            int count = parse_count(argv[i + 1]);
            if (option == "--max-new-tokens") {
                options.max_new_tokens = count;
            } else if (option == "--context-limit") {
                options.max_context_tokens = count;
            } else {
                throw std::invalid_argument("未知选项: " + option);
            }
        }
        if (options.max_context_tokens <= 0 || options.max_context_tokens > MAX_SEQ_LEN) {
            throw std::invalid_argument("上下文上限必须为 1 到 2048");
        }
        Session session(argv[1], argv[2], options);
        std::cout << "输入 /exit 退出，/clear 清空历史。\n";
        run_repl(session);
        // EOF 和 /exit 都返回这里，随后析构 Session 并释放模型权重。
    } catch (const std::exception& error) {
        std::cerr << "启动或运行失败: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
