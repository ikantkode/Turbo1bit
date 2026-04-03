# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Turbo1Bit enables KV cache compression for PrismML's Bonsai 1-bit LLMs by combining **Flash Attention** with **quantized KV cache storage**. This reduces inference memory by up to 2.91x, making Bonsai-8B (8.2B parameters) with full 65K context fit on 8GB hardware.

### Key Discovery

The project's main contribution is discovering that llama.cpp's KV cache quantization (`--ctk`, `--ctv`) requires Flash Attention (`--fa`) to work with Bonsai models. Without Flash Attention, llama.cpp stores the V cache transposed (incompatible with block quantization like Q4_0). Flash Attention stores V non-transposed, making quantization possible.

## Build System

The project uses a **dual CMake build system**:

1. **Standalone Turbo1Bit library** (`CMakeLists.txt` in root) - builds the core TurboQuant compression code
2. **llama.cpp integration** (separate clone in `bonsai-llama.cpp/`) - builds the inference tools

### Initial Setup

```bash
# Clone the Bonsai fork of llama.cpp
git clone --branch prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git bonsai-llama.cpp

# Build with Metal GPU support (Apple Silicon)
cd bonsai-llama.cpp && mkdir build && cd build
cmake .. -G Ninja -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
ninja turbo1bit-infer llama-bench llama-server llama-perplexity
```

### Building the Standalone Library

```bash
# Build core TurboQuant benchmarks (independent of llama.cpp)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Running Inference

### Using the Wrapper Script (Recommended)

```bash
# Auto-detects optimal KV cache config based on available RAM
./turbo1bit run models/Bonsai-8B-gguf/Bonsai-8B.gguf "Explain quantum computing:" -n 200 -c 8192

# Run benchmarks
./turbo1bit bench models/Bonsai-8B-gguf/Bonsai-8B.gguf
```

### Direct Binary Usage

```bash
# Using turbo1bit-infer (non-interactive, with KV compression)
./bonsai-llama.cpp/build/bin/turbo1bit-infer \
    -m models/Bonsai-8B-gguf/Bonsai-8B.gguf \
    -p "Your prompt" \
    -n 200 -c 65536 \
    --ctk q4_0 --ctv q4_0 --fa

# Baseline (no compression)
./bonsai-llama.cpp/build/bin/turbo1bit-infer \
    -m models/Bonsai-8B-gguf/Bonsai-8B.gguf \
    -p "Your prompt" \
    -n 200 -c 65536 \
    --no-turbo1bit
```

### Server Mode

```bash
# OpenAI-compatible API server
./turbo1bit-server models/Bonsai-8B-gguf/Bonsai-8B.gguf --ctx 65536 --port 8080

# Test
curl http://localhost:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

## Project Architecture

```
Turbo1bit/
├── src/                              # Core TurboQuant C library
│   ├── turbo1bit_codebook.h/c        # Lloyd-Max optimal scalar quantizers
│   ├── turbo1bit_rotation.h/c        # QR orthogonal rotation + QJL projection
│   ├── turbo1bit_quantizer.h/c       # MSE and Prod quantizers
│   ├── turbo1bit_kv_cache.h/c        # Compressed KV cache manager
│   ├── turbo1bit_metal.h/m           # Metal GPU host code (macOS)
│   └── turbo1bit_metal.metal         # Metal compute shaders
├── tools/turbo1bit/                  # Standalone benchmark tools
│   ├── turbo1bit_infer.cpp           # End-to-end inference (llama.cpp)
│   ├── turbo1bit_bench.c             # Core algorithm benchmarks
│   └── turbo1bit_stress.c            # Extreme context stress test
├── turbo1bit                         # Main wrapper script
├── turbo1bit-server                  # API server wrapper
└── benchmark.sh                      # End-to-end comparison script
```

### Compression Algorithm Details

- **MSE Quantizer**: Rotates vectors via QR decomposition, quantizes each coordinate via Lloyd-Max codebook
- **Prod Quantizer**: MSE quantization at (b-1) bits + QJL sign projection for inner-product preservation
- **Value Quantization**: Group-wise quantization (configurable group size, typically 32)
- **Hybrid Cache**: Recent tokens kept at full precision (buffer_size), older tokens compressed

### Quality Thresholds

For **1-bit Bonsai models**:
- **2-bit value compression**: Lossless (output identical to baseline)
- **4-bit key compression**: Good quality (coherent text, minor rephrasings)
- **3-bit key compression**: Too aggressive (degrades to gibberish)

Standard FP16 models can use 3-bit key compression; 1-bit models require >= 4 bits.

## Benchmarking

### Memory Measurement

```bash
# Measure RSS memory via /usr/bin/time
/usr/bin/time -l ./bonsai-llama.cpp/build/bin/llama-bench \
    -m models/Bonsai-1.7B-gguf/Bonsai-1.7B.gguf \
    -p 32768 -n 1 -r 1 -ctk q4_0 -ctv q4_0 -fa 1
```

### Perplexity Testing

```bash
# WikiText-2 perplexity (standard LLM quality metric)
./bonsai-llama.cpp/build/bin/llama-perplexity \
    -m models/Bonsai-1.7B-gguf/Bonsai-1.7B.gguf \
    -f wiki.test.raw \
    -ctk q4_0 -ctv q4_0 -fa
```

### Speed Benchmarks

```bash
# Prefill + decode speed
./bonsai-llama.cpp/build/bin/llama-bench \
    -m models/Bonsai-1.7B-gguf/Bonsai-1.7B.gguf \
    -p 512,2048,8192 -n 128 -r 2 -ctk q4_0 -ctv q4_0 -fa 1
```

## Development Notes

### The `bonsai-llama.cpp` Dependency

This directory is **not tracked in git** (see `.gitignore`). Users must clone the PrismML fork separately:

```bash
git clone --branch prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git bonsai-llama.cpp
```

This fork contains Bonsai-specific 1-bit kernel implementations.

### Metal GPU Acceleration

Apple Silicon acceleration uses Metal shaders in `src/turbo1bit_metal.metal`. The shaders are compiled at runtime (no Metal toolchain required during build). Currently provides 5 GPU kernels for compressed KV attention.

### Key Flag Combinations

| Goal | Flags |
|------|-------|
| Maximum compression | `--ctk q4_0 --ctv q4_0 --fa` |
| Best quality | `--ctk q8_0 --ctv q8_0 --fa` |
| Baseline | `--no-turbo1bit` (or omit KV flags) |
| Server mode | `turbo1bit-server` auto-enables `--fa` and `--ctk q4_0 --ctv q4_0` |

### Testing Changes

When modifying compression algorithms:
1. Run `turbo1bit-bench` to verify algorithm correctness
2. Run `turbo1bit-stress` for extreme context testing
3. Compare perplexity with `llama-perplexity` on WikiText-2
4. Measure memory with `/usr/bin/time -l`

## Credits

- **Bonsai / PrismML** - 1-bit LLM models and inference
- **TurboQuant** (0xSero, ICLR 2026) - KV cache compression algorithms
- **llama.cpp** - C/C++ LLM inference engine
