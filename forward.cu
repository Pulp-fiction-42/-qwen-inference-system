#include <cstdio>
#include "model.h"
#include "forward.h"
#include "kernels.h"


float* forward(Model* model, RunState* state, Config* config, int token_id, int position){
    embedding_lookup(token_id, model->embedding, model->config.dim, state->x);
    float* layer_input = state->x;
    for (int layer_index = 0; layer_index < model->config.n_layers; layer_index++){
        blocks* layer = &(model->layers[layer_index]);

        // before attn rmsnorm
        rms_norm(layer->before_attn_norm, layer_input, state->before_attn_norm_x, model->config.dim, 1e-6);                    

        attention(layer, state, layer_index, config, position);

        // Residual connection
        elementwise(layer_input, state->attn_o, state->residual_after_attn, model->config.dim, ELEMENTWISE_ADD);

        // after attn rmsnorm
        rms_norm(state->residual_after_attn, layer_input, state->after_attn_norm_x, model->config.dim, 1e-6);
        
        // mlp
        mlp((const blocks*)layer, state, state->after_attn_norm_x, state->m, model->config.dim, model->config.hidden_dim);

        // residual connection after mlp
        elementwise(state->residual_after_attn, state->m, state->residual_after_mlp, model->config.dim, ELEMENTWISE_ADD);

        // update the float* layer_input for the next layer
        layer_input = state->residual_after_mlp;
    }
    // final layer normalization
    rms_norm(layer_input, layer_input, state->final_norm, model->config.dim, 1e-6);

    // lm head linear without bias
    // here, we don't apply softmax for logists, cause use max strategy for sampling
    float zero_bias[config->dim] = {0.0f}; 
    mat_vec((float*)(model->lm_head), zero_bias, state->final_norm, state->logits, config->vocab_size, config->dim, config->dim);


    return state->logits;

}




