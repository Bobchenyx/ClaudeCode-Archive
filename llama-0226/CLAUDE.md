# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **IMPORTANT**: Read [AGENTS.md](AGENTS.md) before any work. This project does **not** accept PRs that are fully or predominantly AI-generated. AI may only assist in an advisory/review capacity. All AI usage requires explicit disclosure.

## Build Commands

```bash
# Standard build
cmake -B build
cmake --build build --config Release -j $(nproc)

# Build with specific backend
cmake -B build -DGGML_CUDA=ON        # NVIDIA GPU
cmake -B build -DGGML_METAL=ON       # Apple Metal (default on macOS)
cmake -B build -DGGML_VULKAN=ON      # Vulkan

# Build specific target
cmake --build build --target llama-cli
cmake --build build --target llama-server
cmake --build build --target llama-quantize
```

Binaries go to `build/bin/`. The old Makefile is deprecated; use CMake only.

## Testing

```bash
# Run all tests
cd build && ctest --output-on-failure

# Run a single test by name
ctest -R test-tokenizer-0-llama-bpe

# Build and run a specific test target
cmake --build build --target test-backend-ops && ./build/bin/test-backend-ops
```

Test sources are in `tests/`. The test CMakeLists.txt provides helper functions: `llama_build()`, `llama_test()`, `llama_build_and_test()`.

## Code Style

- **Formatter**: clang-format (v15+), config in `.clang-format`
- **Linter**: clang-tidy, config in `.clang-tidy`
- C++17, 120 char line limit, 4-space indentation, no tabs
- Pointer alignment: middle style (`void * ptr`)
- LF line endings

## Architecture

### Three-layer design

1. **ggml/** — Low-level tensor computation library (C). Handles memory allocation, tensor ops, quantization/dequantization, and hardware backend dispatch.
   - `ggml/src/ggml.c` — Core tensor operations
   - `ggml/src/ggml-backend.cpp` — Backend abstraction layer
   - `ggml/src/ggml-cpu/` — CPU backend (SIMD-optimized: NEON, AVX, etc.)
   - `ggml/src/ggml-cuda/` — CUDA backend (custom kernels)
   - `ggml/src/ggml-metal/` — Metal backend (Apple GPU)
   - `ggml/src/ggml-vulkan/` — Vulkan backend

2. **src/** — Core llama library (`libllama`). Model loading, inference orchestration, tokenization, sampling.
   - `src/llama.cpp` — Main entry point
   - `src/llama-model.cpp` — Model loading/management
   - `src/llama-context.cpp` — Inference context
   - `src/llama-kv-cache.cpp` — KV cache management
   - `src/llama-sampler.cpp` — Token sampling strategies
   - `src/llama-vocab.cpp` — Tokenizer/vocabulary
   - `src/llama-grammar.cpp` — Constrained generation via grammars
   - `src/llama-arch.cpp` — Architecture definitions
   - `src/models/` — Per-architecture model implementations (112+ models)

3. **common/** — Shared utilities for tools/examples. Argument parsing, chat templates, logging, HTTP.

### Key public API

- `include/llama.h` — C API header (the public interface)
- `include/llama-cpp.h` — C++ wrapper

### Tools (in `tools/`)

| Tool | Description |
|------|-------------|
| `tools/cli/` | `llama-cli` — Interactive inference CLI |
| `tools/server/` | `llama-server` — OpenAI-compatible HTTP API server |
| `tools/quantize/` | `llama-quantize` — Model quantization |
| `tools/llama-bench/` | `llama-bench` — Performance benchmarking |
| `tools/perplexity/` | `llama-perplexity` — Quality measurement |
| `tools/imatrix/` | Importance matrix for quantization calibration |

### Model format

Uses GGUF (`.gguf`) binary format. Python tooling in `gguf-py/` for model conversion and inspection.

## Key CMake Options

| Option | Description |
|--------|-------------|
| `LLAMA_BUILD_TESTS` | Build tests (default: ON) |
| `LLAMA_BUILD_TOOLS` | Build tools (default: ON) |
| `LLAMA_BUILD_EXAMPLES` | Build examples (default: ON) |
| `LLAMA_BUILD_SERVER` | Build server (default: ON) |
| `LLAMA_SANITIZE_THREAD` | Enable thread sanitizer |
| `LLAMA_SANITIZE_ADDRESS` | Enable address sanitizer |
| `BUILD_SHARED_LIBS` | Build shared libraries |

## Related Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) — Contribution guidelines
- [docs/build.md](docs/build.md) — Comprehensive build documentation
- [tools/server/README-dev.md](tools/server/README-dev.md) — Server development docs
