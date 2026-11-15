## C++ Coding Style and Complexity Constraints

This project prioritizes **readability, simplicity, and explicit control flow** over advanced C++ abstractions.

The codebase is primarily C-style C/CUDA, with minimal C++ features introduced only when necessary for modular design.

### 1. Allowed C++ Features

- `class` / `struct` for organizing data and behavior.
- `public` / `private` for basic encapsulation.
- Constructors with simple initialization.
- Pointers and references (`*`, `&`) for passing and accessing existing objects.
- Ordinary member functions, such as `Qwen2Model::forward()`.
- Basic C++ containers only when they provide a clear benefit over simple arrays.

### 2. Avoid Advanced C++ Features

Do not introduce the following unless explicitly requested:

- Templates or template metaprogramming.
- Inheritance, virtual functions, or polymorphism.
- Smart pointers (`unique_ptr`, `shared_ptr`).
- Move semantics (`std::move`, `&&`).
- Operator overloading.
- Lambdas, callbacks, or complex functional programming.
- Design patterns, generic frameworks, or unnecessary abstraction layers.
- Complex STL containers or algorithms.

Prefer ordinary functions, simple classes, structs, arrays, pointers, and explicit loops.

### 3. Class Design

- Classes should have one clear responsibility.
- Keep class hierarchies flat.
- Prefer composition over inheritance.
- Reuse existing `Model`, `blocks`, `RunState`, and `Config` structs.
- Do not introduce additional classes unless they solve an immediate requirement.
- Keep constructors simple. Avoid complex initialization logic.
- Prefer straightforward code over compact or clever syntax.

### 4. CUDA Architecture

- Model classes define the forward computation order.
- Host-side `op_*()` functions encapsulate CUDA Kernel launches.
- Device kernels implement mathematical operations.
- Model classes must not launch CUDA kernels directly.
- Operator implementations must not depend on model-specific classes or structs.
- Keep CUDA launch configurations inside the operator implementation files.

### 5. Refactoring Rules

- Make the smallest changes necessary to achieve the requested goal.
- Preserve existing mathematical operations and numerical behavior.
- Avoid unrelated improvements, optimizations, or architectural redesigns.
- Do not introduce computation graphs, generic execution engines, or backend registries.
- Preserve the existing memory management design unless explicitly instructed otherwise.
- Prefer explicit, readable code even if it requires a few extra lines.

**Guiding principle: Write code that a C programmer with basic C++ knowledge can easily read, understand, and modify.**
