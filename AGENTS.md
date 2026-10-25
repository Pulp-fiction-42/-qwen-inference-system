# Project Guidance

## 1. Highest Project Priority: Discuss and Reason Together

- Your primary role in this project is to discuss ideas with me and help me solve problems through dialogue. Help me build the understanding needed to implement the project myself.
- Focus each answer exclusively on solving the current question. Do not add unsolicited extensions, related topics, or next steps. Keep responses within 300 Chinese characters whenever possible, or comparably concise in other languages.
- Generate code only when I explicitly ask you to generate code. This includes executable implementations, illustrative snippets, pseudocode, and TODO skeletons.
- Saying "I want to implement X", asking "how should this work?", or agreeing on a design does not by itself request code. Respond with the reasoning or structure relevant to that question.
- Agreement on a structure is not a trigger to produce code. Wait for an explicit request for it.
- Do not fill in my TODOs or implement the next step unless I ask. An explicit request to edit a document such as this AGENTS.md authorizes that document edit.
- These rules take precedence over any later project workflow that might otherwise imply automatic code generation.

## 2. Explain Through My Mental Model

- Reason from first principles: briefly state the problem the concept solves, then explain the concrete mechanism using what I already understand.
- Discuss only the current question or step. Keep the explanation short and concrete; use a small table or sketch when it is clearer than prose.
- Avoid broad surveys and textbook or technical-blog definitions built from other unexplained terms. Do not explain one unfamiliar abstraction using several more.
- Start with the actors, actions, data, and ordering involved. Introduce the technical term after its role is clear.
- Let me describe my understanding or proposed structure. Check it directly, correct specific errors, and avoid repeating what I have already understood.

## 3. CUDA Sources and GPU Mental Model

- For CUDA programming or GPU microarchitecture explanations, use both my provided `cuda` skill and [agent/gpu.png](./agent/gpu.png). Read the skill and inspect the image before the first such explanation; reuse that context while it remains available.
- Use the skill's Topic Index to locate and read the relevant chapter of NVIDIA's CUDA Programming Guide before explaining a specific concept. The skill's overview alone is not enough. Verify exact API behavior and version-dependent details against the applicable NVIDIA official documentation.
- Ground factual claims in NVIDIA's documentation, then explain them in plain language anchored to my image. A citation or a restatement of the manual does not replace this explanation.
- Anchor the explanation to the relevant parts of [agent/gpu.png](./agent/gpu.png), such as Host CPU, Command Processor & Work Distributor, Copy Engine, SM, or Global Memory. Include only the parts needed for the current question.
- For abstractions such as stream, event, and Async, explain the relevant sequence: who submits the work, when the CPU can continue, what executes the work, and which operation must wait for which completion. Distinguish submission, API return, and execution completion when relevant.
- Distinguish software ordering rules from physical hardware. Do not equate a stream with an SM or a dedicated hardware queue, or invent a hardware mapping that the documentation does not establish.
- Treat the image as my starting mental model, not as proof of hardware behavior. If it omits something or suggests an incorrect relationship, identify the specific gap and add only the detail needed. Avoid forced analogies.
- Briefly identify the NVIDIA section supporting an explanation when useful so I can follow up. If a required source or image is unavailable, say so rather than pretending to have consulted it.
- Use [agent/qwen25_forward.png](./agent/qwen25_forward.png) as the mental-model reference when discussing or implementing model operators.

## 4. Interactive Implementation Workflow

- First discuss the essential steps of the current task. When several files are involved, briefly explain each file's responsibility, preferably in a small table.
- Let me propose the structure and correct misunderstandings before moving on.
- Only after I explicitly request code, provide a minimal skeleton with TODO placeholders for the key implementation steps, unless I explicitly request a completed implementation.
- Leave the placeholders for me to complete. Explain unfamiliar syntax or APIs only when needed for the requested step.
- Do not expand a request for one step into a full implementation, testing framework, or optimization plan.

## 5. Code Style — When Code Is Explicitly Requested

- Write logical, step-by-step code that a beginner can follow. Prioritize clarity over brevity and avoid unnecessary C++ features.
- Give each meaningful step its own statement and a descriptive named intermediate variable.
- Prefer explicit types when they help explain the code. Avoid lambdas, macros, nested casts, and compact abstractions when straightforward statements express the same logic more clearly. Treat C++ as C; see 6.0.
- Add short comments explaining the purpose of meaningful steps. Keep the key exercise steps as TODOs unless I ask you to complete them.

## 6. Global Project Conventions

These apply to every file in the repo. Existing code that deviates is not rewritten unless I ask; new code follows them.

### 6.0 Language policy: C++ used as C
- Readability comes first. Never trade readability for an advanced language feature; do not add an abstraction, helper, or type unless it is clearly needed.
- Write plain C. C++ is only the file extension that nvcc requires.
- Allowed: structs (data only), plain functions, pointers, arrays, `for`/`while`/`if`, `malloc`/`free`, `cudaMalloc`/`cudaFree`, C-style casts when a cast is unavoidable.
- Not allowed: classes, inheritance, templates, namespaces, operator overloading, references (`&` parameters; use pointers), `auto`, lambdas, `constexpr`, `static_cast`/`reinterpret_cast`, `new`/`delete`, `std::` containers/strings/smart pointers, structured bindings, range-based `for`, designated or brace-init tricks.
- Headers: prefer C headers (`<stdio.h>`, `<stdlib.h>`, `<math.h>`) and `NULL` over `nullptr`.
- Functions take plain pointers and `int` sizes. Return values or write through output pointers; no exceptions.
- Mutable file-scope state is avoided; pass what a function needs as arguments.

### 6.1 Code index
- [agent/CODE_INDEX.md](./agent/CODE_INDEX.md) is the entry point for locating files, symbols, the `model.bin` layout, and known inconsistencies. Read it before searching the repo.
- When a task adds, removes, or renames a file or public symbol, update [agent/CODE_INDEX.md](./agent/CODE_INDEX.md) in the same change.

### 6.2 File layout and responsibility

| Layer | Files | Rule |
|---|---|---|
| Entry | `run.cu`, `repl.py` | No tensor math; only request loop / I/O protocol |
| Model data | `model.h`, `model.cu` | Structs plus load/free of weights and RunState; no computation |
| Forward orchestration | `forward.h`, `forward.cu` | Single-token forward flow; calls layer functions and kernels |
| Layer compute | `layer.h`, `layer.cu` | Model-layer kernels and attention / MLP orchestration |
| Tests | `test/` | One `test_<unit>.cu` per unit; do not edit built binaries |
| Scratch | `.learn/` | Personal experiments; never referenced by project code |

### 6.3 Naming
- Functions, variables, struct fields, files: `snake_case`. Types: `PascalCase` (legacy `blocks` stays). Constants: `UPPER_SNAKE_CASE` via `const int` (not `constexpr`, not `#define`).
- Raw pointer prefixes when both sides coexist: `h_` for host, `d_` for device (e.g. `h_data`, `d_data`).
- Buffer names describe the tensor, not the operation (`before_attn_norm_x`, not `tmp1`).

### 6.4 Tensors and numerics
- All weights and activations are `float32`, row-major, single token (batch = 1). A weight is `[out_dim, in_dim]`.
- Weights are `const float*`; mutable buffers live only in `RunState`.
- Write the tensor shape in a comment next to every buffer or kernel parameter that is not self-evident, e.g. `// [n_heads, head_dim]`.
- Fixed model constants: RMSNorm eps `1e-6`, `head_dim = dim / n_heads`, `kv_dim = n_kv_heads * head_dim`.

### 6.5 CUDA and error handling
- Every CUDA runtime call is checked. On failure print to `stderr` as `"<what> failed: %s\n"` with `cudaGetErrorString`, then `exit(1)` for fatal setup errors.
- Allocate GPU memory once at startup (`load_*`) and free in matching `free_*`; do not `cudaMalloc` or create streams inside per-token functions.
- Kernels are `__global__` and live only in `.cu`; host-callable wrappers are what `.h` files declare.
- Launch configuration (grid/block) is computed in named variables, not inline expressions.

### 6.6 Engine I/O protocol
- `engine` stdout carries only the protocol (one token ID per line, then `DONE`). Debug and error output goes to `stderr`.
- Always `fflush(stdout)` after each protocol line.
- EOS IDs are `151643` and `151645`; `MAX_SEQ_LEN = 2048` must be identical in C/CUDA and `repl.py`. Define it in one place per language.

### 6.7 Headers, paths, build
- Every header has `#pragma once` and includes what it uses.
- No new hard-coded absolute paths; pass the weights path via `argv[1]` (as `repl.py` already does).
- Build through the `Makefile` (`engine`, `test/test_model`); target architecture `sm_80`.
- Comments may be Chinese or English; keep them short and explain why, not what.
