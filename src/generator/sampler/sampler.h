#pragma once

#include "../../model/model_types.h"

// 只负责贪心 argmax；临时缓冲由 Session 随 RunState 管理。
class Sampler {
public:
    Sampler(RunState& state, int vocab_size);
    int sample();

private:
    RunState& state_;
    int vocab_size_;
};
