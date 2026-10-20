#include <cstdio>
#include "model.h"    // ModelWeights, RunState, load/free 函数
#include "forward.h"  // forward()
constexpr int MAX_SEQ_LEN = 2048;

int main() {

    Model model;
    RunState state;
    load_model_weights(&model);
    load_runstate(&state, &model.config);
    int prompt[MAX_SEQ_LEN];
    int prompt_length;

    // Wait for requests until Python closes the pipe.
    while (scanf("%d", &prompt_length) == 1) {
        if (prompt_length <= 0 || prompt_length > MAX_SEQ_LEN)
            return 1;

        for (int i = 0; i < prompt_length; ++i) {
            if (scanf("%d", &prompt[i]) != 1)
                return 1;
        }

        /*
        `position` track the prompt's position, `position` increment when PREFILL and GENERATION phase 
        `float* logits` always keep the previous token's address  
        */
        int position = 0;
        const float *logits = nullptr;
        
        // PREFILL
        /*
        forward token by token to populate the KV-cache 
        the `float* logits` point to the last token's logits 
        */
        while (position < prompt_length) {
            logits = forward(&model, &state, prompt[position], position);
            ++position;
        }

        // GENERATION
        /*
        select next token use `sample_argmax` with last token's logits and print it out(`print + fflush`) to pipe to Python process
        break when `position >= MAX_SEQ_LEN` and hit the `151645 or 151643`
        */
        while (position < MAX_SEQ_LEN) {
            int next_token = sample_argmax(logits, model.config.vocab_size);
            
            // if the token id signal means END
            if (next_token == 151645 || next_token == 151643)
                break;

            printf("%d\n", next_token);
            fflush(stdout);
            
            // if length exced the longest
            if (position == MAX_SEQ_LEN - 1)
                break;

            logits = forward(&model, &state, next_token, position);
            ++position;
        }
        

        printf("DONE\n");
        fflush(stdout);
    }


    free_runstate(&state);
    free_model_weights(&model);


    return 0;
}
