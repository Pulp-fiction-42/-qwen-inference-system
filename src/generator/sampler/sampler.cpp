#include "sampler.h"
#include "ops/ops.h"

Sampler::Sampler(RunState& state, int vocab_size)
    : state_(state), vocab_size_(vocab_size)
{
}

int Sampler::sample()
{
    return op_argmax(state_.logits, vocab_size_, state_.sampler_values,
                     state_.sampler_indices, state_.sampler_token);
}
