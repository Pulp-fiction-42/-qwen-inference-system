#pragma once
#include "model.h"

float* forward(Model* model, RunState* run_state, Config* config, int token_id, int position);

int sample_argmax(float* logits, int vocab_size);


