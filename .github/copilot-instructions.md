# Copilot Instructions for Qwen Inference Engine

## User Interaction Principles

1. **Mental Model First**: For each problem, prioritize user assertions, mental models, or proposed solutions. First determine whether they are **correct or incorrect**, then make necessary corrections or adopt them directly. Do not arbitrarily replace the user's approach.
2. **Scope Control**: Focus strictly on the current problem. Do not expand the scope, introduce unrelated topics, or take on additional tasks without being asked.
3. **Minimal Diff**: Make only the code changes necessary to solve the current problem. Do not refactor, optimize, or alter the existing architecture unless explicitly requested.
4. **User Decision Authority**: The user makes final decisions on technical direction and implementation strategies. Point out clear mistakes directly rather than blindly agreeing.
5. **Communication Style**: Be direct and concise. State conclusions first, followed by only the necessary reasoning. Avoid Socratic questioning and unsolicited exercises.
6. **On-Demand Verification**: Do not automatically compile, run tests, or perform additional verification. Only do so when explicitly requested.
7. **Assertive Interaction**: When the user provides an assertion, treat it as the starting point. Evaluate it directly (true, false, or incomplete) before proposing adjustments.

## Build and Test Commands

### Core Build Targets
- `make -j4 repl`: Builds the interactive CLI application `build/qwen_repl`.
- `make all`: Compiles `build/libinference_ops.a`, `build/libinference_model.a`, and `build/libinference_resources.a`.
- `make tokenizer`: Compiles `build/libinference_tokenizer.a` using llama.cpp headers and `libggml-base` (`LLAMA_DIR` and `LLAMA_LIB_DIR` can be overridden).
- `make model`: Compiles `build/libinference_model.a` (host forward orchestrator).
- `make ops`: Compiles `build/libinference_ops.a` (CUDA operators via `nvcc`).
- `make resources`: Compiles `build/libinference_resources.a` (weight & `RunState` allocators).
- `make clean`: Cleans operator objects and library. Module-specific clean targets: `clean-tokenizer`, `clean-model`, `clean-resources`.

### Running Tests
- **Host Model Forward Test** (No GPU required; validates layer call order, tensor sizes, and scratch buffer reuse using mock operators):
  ```sh
  make test-model
  ```
- **Host Runtime Test** (No GPU required; validates session flow, ChatML templating, stopping conditions, and exception safety):
  ```sh
  make test-runtime
  # Or override vocabulary path:
  make test-runtime VOCAB_PATH=/path/to/vocab.gguf
  ```
- **Single GPU Forward Test** (Requires GPU; tests single block 0 forward and final norm/LM head):
  ```sh
  make test_kernel/test_single_forward/test_single_forward
  ./test_kernel/test_single_forward/test_single_forward
  ```
- **Attention Score Split Diagnostic** (Verifies numerical bit-pattern parity between fused kernel and split score/softmax):
  ```sh
  make test_kernel/test_single_forward/test_score_split
  ./test_kernel/test_single_forward/test_score_split
  ```
- **GPU Session Test**:
  ```sh
  make build/runtime/session_gpu_test
  ./build/runtime/session_gpu_test /path/to/model.bin /path/to/vocab.gguf test_kernel/results/session_integration/reference.txt
  ```
- **Full Python Correctness Suite** (Runs single forward, single-turn generation, and multi-turn memory):
  ```sh
  make -j4 test-kernel-build
  /home/bgi-chao/miniconda3/envs/inference/bin/python test_kernel/run_correctness.py /path/to/Qwen2.5-1.5B-Instruct
  ```
- **REPL Integration Test** (Tests stdin commands `/clear`, `/exit`, EOF, length truncation, and recovery from input errors):
  ```sh
  python3 tests/repl_test.py /path/to/model.bin /path/to/vocab.gguf
  ```
- **Interactive REPL Run**:
  ```sh
  ./build/qwen_repl /path/to/model.bin /path/to/vocab.gguf --max-new-tokens 256 --context-limit 2048
  ```

## High-Level Architecture

The system is a C-style C++/CUDA inference engine for Qwen2.5-1.5B-Instruct in FP32 precision.

```text
src/main.cpp -> Terminal REPL Loop
      |
src/session/Session (Owns Model weights, CPU text history, Tokenizer)
      |  (Per turn: ChatML template -> encode -> allocate RunState)
      v
src/generator/Generator (Prefill from position 0 + autoregressive decode)
      |
      +---> src/model/Qwen2Model & Qwen2Layer (Token forward flow; NO CUDA launches)
      |            |
      |            v
      |     ops/ops.h (Host wrapper functions; NO model struct dependencies)
      |            |
      |            v
      |     ops/*.cu (CUDA device kernels, launch configurations, default stream)
      |
      +---> src/generator/sampler/Sampler -> ops/argmax.cu (Next-token argmax)
```

### Component Roles & Data Flow
1. **REPL & Terminal** (`src/main.cpp`, `src/terminal/`):
   Parses command-line arguments and runs the REPL loop. Dispatches `/clear` and `/exit` commands, displays generation outputs, and handles EOF.
2. **Session** (`src/session/`):
   - Owns `Model` weights (loaded once from `model.bin`) and `Tokenizer` (loaded from GGUF metadata via `libggml-base`).
   - Maintains multi-turn conversation history in CPU memory.
   - Per-turn lifecycle: formats history with ChatML template, encodes tokens, allocates temporary GPU `RunState` (`load_runstate`), runs generation, decodes output, appends response to history, and frees `RunState` (`free_runstate`).
   - Exception safety: any error during generation triggers `free_runstate()`, catches the error back to the terminal, and preserves existing conversation history.
3. **Generator & Sampler** (`src/generator/`):
   - `Generator` borrows `Qwen2Model` and `RunState` (performs zero GPU allocations).
   - Runs prompt prefill token-by-token from position 0 (KV cache is recomputed every turn; no cross-turn KV reuse).
   - Generates tokens autoregressively until hitting `STOP_EOS` (tokens 151643/151645), `STOP_MAX_NEW_TOKENS`, or `STOP_CONTEXT_LIMIT`.
   - `Sampler` calls `op_argmax` using temporary buffers in `RunState` to select next token.
4. **Model & Layer** (`src/model/`):
   - `Qwen2Model` contains 28 `Qwen2Layer` instances, referencing layer weights in `Model::layers`.
   - Forward pass per token: `embedding` -> 28 decoder layers (RMSNorm -> QKV projection -> RoPE -> KV cache copy -> GQA Score -> Softmax -> GQA Weighted Sum -> Out projection -> Residual -> RMSNorm -> Gate/Up MLP -> SiLU/Mul -> Down MLP -> Residual) -> `final_norm` -> `lm_head` -> `logits`.
   - **Strict architectural boundary**: Model and Layer classes define execution order only. They never launch CUDA kernels directly and never manage GPU memory.
5. **Operators** (`ops/`):
   - Host functions (`op_*`) encapsulate CUDA kernel launches on the default stream.
   - Kernels and operators work strictly with raw device pointers, dimensions, and strides.
   - Operators have zero dependency on model structs (`Model`, `RunState`, `Config`, `blocks`).

## Key Conventions & Constraints

### 1. C++ Complexity Constraints (Strict)
This codebase prioritizes **readability, simplicity, and explicit control flow** over modern C++ abstractions.
- **Allowed**:
  - `class` / `struct` for organizing data and clear single responsibilities.
  - `public` / `private` for basic encapsulation.
  - Simple constructors without complex initialization logic.
  - Raw pointers and references (`*`, `&`).
  - Standard member functions.
  - Basic containers (`std::vector`, `std::string`) when clearly beneficial over raw arrays.
- **Prohibited**:
  - Templates and template metaprogramming.
  - Inheritance, virtual functions, polymorphism.
  - Smart pointers (`std::unique_ptr`, `std::shared_ptr`).
  - Move semantics (`std::move`, `&&`).
  - Operator overloading.
  - Lambdas, `std::function`, callbacks, or functional paradigms.
  - Heavy STL algorithms or complex generic frameworks.
- Write code that a C programmer with basic C++ knowledge can read, modify, and debug effortlessly.

### 2. CUDA Architecture Boundaries
- Model classes define forward execution order.
- Host `op_*()` functions encapsulate CUDA kernel launches.
- Device kernels implement mathematical operations.
- Model classes must never launch CUDA kernels directly.
- Kernel launch configurations (`<<<grid, block>>>`) belong strictly inside `ops/*.cu`.
- Operator headers (`ops/ops.h`) must never include model headers or structs.

### 3. Memory & Resource Ownership
- **Model weights** (`Model`): Loaded once at startup (`load_model_weights`), freed on shutdown (`free_model_weights`). Weights persist across turns.
- **RunState**: Holds KV cache (`[n_layers, max_seq_len, kv_dim]`), intermediate activations, RMS scratchpad, and argmax workspaces. Allocated per turn by `Session` via `load_runstate()` and released via `free_runstate()`.
- Model, Layer, Generator, and Sampler borrow resources; they do not allocate or free device memory.

### 4. Preserved Kernel Behaviors
Retain existing kernel constraints and implementations unless explicitly instructed to change them:
- Fixed launch configuration in `op_proj_bias` (6 x 256 threads).
- Fixed block count in `op_final_rms_norm` (6 blocks).
- Constant 128 hardcoded inside `rope.cu`.
- Fixed 640 reduction dimension in `argmax.cu`.

