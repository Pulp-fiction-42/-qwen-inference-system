// Qwen2 BPE adapted from llama.cpp/src/llama-vocab.cpp.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tokenizer.h"

#include "gguf.h"
#include "unicode.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

// GGUF tokenizer.ggml.token_type 的值（不是 llama_token_attr 位掩码）。
enum TokenType { normal = 1, unknown = 2, control = 3, user_defined = 4,
                 unused = 5, byte = 6 };

// 与 llama.cpp 的 LLAMA_VOCAB_PRE_TYPE_QWEN2 保持一致。
const std::vector<std::string> qwen_regex = {
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
};

int64_t require_key(const gguf_context* ctx, const char* name, gguf_type type) {
    const auto key = gguf_find_key(ctx, name);
    if (key < 0 || gguf_get_kv_type(ctx, key) != type) {
        throw std::runtime_error(std::string("Missing or invalid GGUF key: ") + name);
    }
    return key;
}

int64_t require_array(const gguf_context* ctx, const char* name, gguf_type type) {
    const auto key = require_key(ctx, name, GGUF_TYPE_ARRAY);
    if (gguf_get_arr_type(ctx, key) != type) {
        throw std::runtime_error(std::string("Invalid GGUF array type: ") + name);
    }
    return key;
}

bool optional_bool(const gguf_context* ctx, const char* name) {
    if (gguf_find_key(ctx, name) < 0) {
        return false;
    }
    return gguf_get_val_bool(ctx, require_key(ctx, name, GGUF_TYPE_BOOL));
}

} // namespace

struct Tokenizer::Impl {
    struct Token {
        std::string text;
        int type;
        bool special() const {
            return type == control || type == unknown || type == user_defined;
        }
    };

    std::vector<Token> vocab;
    std::unordered_map<std::string, int> token_to_id;
    std::map<std::pair<std::string, std::string>, int> ranks;
    std::vector<int> special_ids;
    bool add_bos = false;
    bool add_eos = false;
    int bos = -1;
    int eos = -1;

    explicit Impl(const std::string& model_path) {
        // no_alloc + ctx=nullptr：只读取元数据，不分配模型张量。
        std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(
            gguf_init_from_file(model_path.c_str(), {true, nullptr}), gguf_free);
        if (!ctx) {
            throw std::runtime_error("Cannot read GGUF: " + model_path);
        }
        const auto model = require_key(ctx.get(), "tokenizer.ggml.model", GGUF_TYPE_STRING);
        const auto pre = require_key(ctx.get(), "tokenizer.ggml.pre", GGUF_TYPE_STRING);
        if (std::string(gguf_get_val_str(ctx.get(), model)) != "gpt2" ||
            std::string(gguf_get_val_str(ctx.get(), pre)) != "qwen2") {
            throw std::invalid_argument("Tokenizer supports only Qwen2/2.5 BPE GGUF files");
        }

        const auto tokens = require_array(ctx.get(), "tokenizer.ggml.tokens", GGUF_TYPE_STRING);
        const auto types = require_array(ctx.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32);
        const auto count = gguf_get_arr_n(ctx.get(), tokens);
        if (count == 0 || count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            gguf_get_arr_n(ctx.get(), types) != count) {
            throw std::runtime_error("Invalid GGUF vocabulary size");
        }
        const auto* type_data = static_cast<const int32_t*>(gguf_get_arr_data(ctx.get(), types));
        vocab.reserve(count);
        token_to_id.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            Token token{gguf_get_arr_str(ctx.get(), tokens, i), type_data[i]};
            if (token.text.empty() || token.type < normal || token.type > byte) {
                throw std::runtime_error("Invalid GGUF vocabulary entry");
            }
            // llama.cpp 会把这些结束标记修正为 control，包括当前模型中的 </s>。
            if (token.text == "</s>" || token.text == "<|im_end|>" ||
                token.text == "<|endoftext|>") {
                token.type = control;
            }
            if (!token_to_id.emplace(token.text, static_cast<int>(i)).second) {
                throw std::runtime_error("Duplicate GGUF token: " + token.text);
            }
            vocab.push_back(std::move(token));
        }

        const auto merges = require_array(ctx.get(), "tokenizer.ggml.merges", GGUF_TYPE_STRING);
        const auto merge_count = gguf_get_arr_n(ctx.get(), merges);
        if (merge_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Too many BPE merges");
        }
        for (std::size_t i = 0; i < merge_count; ++i) {
            const std::string merge = gguf_get_arr_str(ctx.get(), merges, i);
            const auto sep = merge.find(' ');
            if (sep == std::string::npos || sep == 0 || sep + 1 == merge.size()) {
                throw std::runtime_error("Invalid BPE merge rule");
            }
            ranks.emplace(std::make_pair(merge.substr(0, sep), merge.substr(sep + 1)),
                          static_cast<int>(i));
        }
        add_bos = optional_bool(ctx.get(), "tokenizer.ggml.add_bos_token");
        add_eos = optional_bool(ctx.get(), "tokenizer.ggml.add_eos_token");
        auto special_id = [&](const char* name, bool required) {
            if (gguf_find_key(ctx.get(), name) < 0 && !required) {
                return -1;
            }
            const auto id = gguf_get_val_u32(ctx.get(), require_key(ctx.get(), name, GGUF_TYPE_UINT32));
            if (id >= vocab.size()) {
                throw std::runtime_error(std::string("Invalid special token ID: ") + name);
            }
            return static_cast<int>(id);
        };
        bos = special_id("tokenizer.ggml.bos_token_id", add_bos);
        eos = special_id("tokenizer.ggml.eos_token_id", add_eos);
        for (std::size_t i = 0; i < vocab.size(); ++i) {
            if (vocab[i].special()) {
                special_ids.push_back(static_cast<int>(i));
            }
        }
        // 与 llama.cpp 一样，先切出较长的特殊 token，再处理普通文本。
        std::sort(special_ids.begin(), special_ids.end(), [&](int a, int b) {
            return vocab[a].text.size() > vocab[b].text.size();
        });
        for (const auto& marker : {"<|im_start|>", "<|im_end|>"}) {
            const auto it = token_to_id.find(marker);
            if (it == token_to_id.end() || !vocab[it->second].special()) {
                throw std::runtime_error("Qwen ChatML special token is missing");
            }
        }
    }

    // 每次调用独立保存工作状态；合并按 rank 优先，rank 相同时从左到右。
    void encode_text(const std::string& text, std::vector<int>& output) const {
        struct Symbol {
            std::size_t offset;
            std::size_t size;
            int prev;
            int next;
        };
        struct Bigram {
            int left;
            int right;
            int rank;
            std::size_t size;
            bool operator<(const Bigram& other) const {
                return rank > other.rank || (rank == other.rank && left > other.left);
            }
        };

        for (const auto& word : unicode_regex_split(text, qwen_regex)) {
            if (word.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::length_error("BPE word is too long");
            }
            std::vector<Symbol> symbols;
            for (std::size_t offset = 0; offset < word.size();) {
                const auto length = std::min(unicode_len_utf8(word[offset]), word.size() - offset);
                const int index = static_cast<int>(symbols.size());
                symbols.push_back({offset, length, index - 1,
                                   offset + length < word.size() ? index + 1 : -1});
                offset += length;
            }
            std::priority_queue<Bigram> queue;
            auto add_pair = [&](int left, int right) {
                if (left < 0 || right < 0) {
                    return;
                }
                const auto& a = symbols[left];
                const auto& b = symbols[right];
                const auto it = ranks.find({word.substr(a.offset, a.size), word.substr(b.offset, b.size)});
                if (it != ranks.end()) {
                    queue.push({left, right, it->second, a.size + b.size});
                }
            };
            for (std::size_t i = 1; i < symbols.size(); ++i) {
                add_pair(static_cast<int>(i) - 1, static_cast<int>(i));
            }
            while (!queue.empty()) {
                const auto pair = queue.top();
                queue.pop();
                auto& left = symbols[pair.left];
                auto& right = symbols[pair.right];
                // 邻居被合并后队列里的旧项失效；符号只增大或被删除。
                if (left.size == 0 || right.size == 0 || left.next != pair.right ||
                    left.size + right.size != pair.size) {
                    continue;
                }
                left.size += right.size;
                right.size = 0;
                left.next = right.next;
                if (right.next >= 0) {
                    symbols[right.next].prev = pair.left;
                }
                add_pair(left.prev, pair.left);
                add_pair(pair.left, left.next);
            }
            for (const auto& symbol : symbols) {
                if (symbol.size == 0) {
                    continue;
                }
                const auto piece = word.substr(symbol.offset, symbol.size);
                const auto it = token_to_id.find(piece);
                if (it != token_to_id.end()) {
                    output.push_back(it->second);
                } else {
                    // 保留 llama.cpp byte-encoded BPE 的逐字节回退行为。
                    for (char c : piece) {
                        const auto fallback = token_to_id.find(std::string(1, c));
                        if (fallback != token_to_id.end()) {
                            output.push_back(fallback->second);
                        }
                    }
                }
            }
        }
    }

    std::vector<int> encode(const std::string& text) const {
        struct Fragment { std::size_t offset; std::size_t size; int token; };
        std::list<Fragment> fragments;
        if (!text.empty()) {
            fragments.push_back({0, text.size(), -1});
        }
        for (int id : special_ids) {
            const auto& marker = vocab[id].text;
            for (auto it = fragments.begin(); it != fragments.end(); ++it) {
                if (it->token >= 0) {
                    continue;
                }
                const auto end = it->offset + it->size;
                auto match = text.find(marker, it->offset);
                while (match != std::string::npos && match < end && marker.size() <= end - match) {
                    if (match > it->offset) {
                        fragments.insert(it, {it->offset, match - it->offset, -1});
                    }
                    fragments.insert(it, {match, marker.size(), id});
                    it->offset = match + marker.size();
                    it->size = end - it->offset;
                    match = text.find(marker, it->offset);
                }
            }
        }
        std::vector<int> result;
        if (add_bos) {
            result.push_back(bos);
        }
        for (const auto& fragment : fragments) {
            if (fragment.token >= 0) {
                result.push_back(fragment.token);
            } else if (fragment.size > 0) {
                encode_text(text.substr(fragment.offset, fragment.size), result);
            }
        }
        if (add_eos) {
            result.push_back(eos);
        }
        return result;
    }
};

Tokenizer::Tokenizer(const Template& chat_template, const std::string& model_path)
    : chat_template_(chat_template), impl_(new Impl(model_path)) {}

Tokenizer::~Tokenizer() = default;

std::string Tokenizer::apply_chat_template(const std::string& input) const {
    std::vector<ChatMessage> messages;
    ChatMessage user = {"user", input};
    messages.push_back(user);
    return apply_chat_template(messages);
}

std::string Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages) const {
    std::string prompt;
    if (!chat_template_.system_prompt.empty()) {
        prompt = "<|im_start|>system\n" + chat_template_.system_prompt + "<|im_end|>\n";
    }
    for (size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        if (message.role != "user" && message.role != "assistant") {
            throw std::invalid_argument("消息角色必须是 user 或 assistant");
        }
        prompt += "<|im_start|>" + message.role + "\n";
        prompt += message.content + "<|im_end|>\n";
    }
    return prompt + "<|im_start|>assistant\n";
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    return impl_->encode(text);
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    std::string text;
    for (int id : token_ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= impl_->vocab.size()) {
            throw std::out_of_range("Invalid token ID: " + std::to_string(id));
        }
        const auto& token = impl_->vocab[id];
        if (token.type == control || token.type == unknown || token.type == unused) {
            continue;
        }
        if (token.type == user_defined) {
            text += token.text;
        } else if (token.type == byte) {
            // GGUF byte token 使用 <0xXX> 文本表示。
            if (token.text.size() != 6 || token.text.compare(0, 3, "<0x") != 0 || token.text.back() != '>') {
                throw std::runtime_error("Invalid byte token");
            }
            text += static_cast<char>(std::stoul(token.text.substr(3, 2), nullptr, 16));
        } else {
            for (auto cpt : unicode_cpts_from_utf8(token.text)) {
                text += static_cast<char>(unicode_cpt_to_byte(cpt));
            }
        }
    }
    return text;
}
