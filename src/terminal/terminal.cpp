#include "terminal.h"

#include "../session/session.h"

#include <exception>
#include <iostream>
#include <string>

void run_repl(Session& session) {
    std::string input;

    while (true) {
        if (!(std::cout << ">>> " << std::flush)) {
            return;
        }

        if (!std::getline(std::cin, input)) {
            if (!std::cin.eof()) {
                std::cerr << "读取输入失败。\n";
            }
            std::cout << '\n';
            return;
        }

        if (input == "/exit") {
            return;
        }

        if (input == "/clear") {
            session.clear();
            std::cout << "历史已清空。\n";
            continue;
        }

        if (input.empty()) {
            continue;
        }

        TurnResult result;
        try {
            // Session 负责模板格式化、编码、整轮生成和解码。
            result = session.run_turn(input);
        } catch (const std::exception& error) {
            std::cerr << "生成失败: " << error.what() << '\n';
            continue;
        }

        std::cout << result.text;
        if (result.text.empty() || result.text.back() != '\n') {
            std::cout << '\n';
        }
        if (result.generation.stop_reason == STOP_MAX_NEW_TOKENS) {
            std::cout << "[已达到本轮生成长度上限]\n";
        } else if (result.generation.stop_reason == STOP_CONTEXT_LIMIT) {
            std::cout << "[已达到上下文上限，可 /clear 后继续]\n";
        }
        std::cout << std::flush;
    }
}
