# Turbo1Bit

**Run Bonsai-8B at full 65K context on an 8GB MacBook Air.**

Turbo1Bit enables KV cache compression for [PrismML's Bonsai](https://github.com/PrismML-Eng/Bonsai-demo) 1-bit LLMs. By combining Flash Attention with quantized KV storage, it reduces inference memory by up to **4.24x** — making large models fit on small hardware.

**Now with GPU acceleration support for Apple Metal and AMD ROCm (MI50/MI60)!**

## Headline Result (Measured)

**Bonsai-8B (8.2B parameters, 1-bit weights, 1.1 GB on disk)**

| Context | Without Turbo1Bit | With Turbo1Bit | Saved |
|---------|------------------|---------------|-------|
| 8K | 2,379 MiB | 1,557 MiB | 822 MiB |
| 32K | 5,891 MiB | 2,626 MiB | **3.3 GB** |
| **65K** | **10,618 MiB** | **4,000 MiB** | **6.5 GB** |

At 65K context, Bonsai-8B needs 10.4 GB — too large for 8GB hardware. With Turbo1Bit, it fits in **3.9 GB**.

## GPU Acceleration

Turbo1Bit now supports GPU acceleration for KV cache operations:

- **Apple Silicon (Metal)**: Native support on M1/M2/M3 Macs
- **AMD ROCm (HIP)**: Support for AMD GPUs including MI50/MI60

GPU acceleration provides significant speedup for:
- Attention scoring against compressed keys
- Value dequantization
- Fused decode attention with online softmax
- Matrix-vector multiply for rotation/projection

## Quick Start

### Apple Silicon (macOS)

```bash
# Clone and build
git clone https://github.com/ikantkode/Turbo1bit.git
cd Turbo1bit
git clone --branch prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git bonsai-llama.cpp
cd bonsai-llama.cpp && mkdir build && cd build
cmake .. -G Ninja -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
ninja turbo1bit-infer llama-bench llama-server
cd ../..

# Download a model
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('prism-ml/Bonsai-8B-gguf', local_dir='models/Bonsai-8B-gguf', allow_patterns='*.gguf')"

# Run with auto-optimized settings
./turbo1bit run models/Bonsai-8B-gguf/Bonsai-8B.gguf "Explain quantum computing:" -n 200 -c 8192
```

### AMD GPU (Linux/ROCm)

See the [AMD GPU Setup](#amd-gpu-setup-linux) section below for detailed instructions.

```bash
# After following AMD GPU setup (see below)
export LD_LIBRARY_PATH=/home/shakespear/bonsai/Turbo1bit/rocblas-build/install/lib:/opt/rocm/lib
cd /home/shakespear/bonsai/Turbo1bit/bonsai-llama.cpp/build

# Run server with GPU acceleration
bin/llama-server -m ../../models/Bonsai-1.7B.gguf --port 8080 --host 0.0.0.0 -c 8192 --n-gpu-layers 29
```

## AMD GPU Setup (Linux)

This guide covers setting up Turbo1Bit with AMD GPU acceleration, specifically tested on **AMD Radeon Instinct MI50 (gfx906)**.

### Prerequisites

- Ubuntu 24.04 (or similar Linux distribution)
- AMD GPU supported by ROCm (MI50, MI60, etc.)
- ROCm 7.0.0 or later installed
- CMake (3.16+), Make, GCC/Clang

### Step 1: Install ROCm

If ROCm is not already installed:

```bash
# Download and install ROCm (adjust version as needed)
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000_all.deb
sudo apt install -y ./amdgpu-install_6.0.60000_all.deb
sudo amdgpu-install --usecase=rocm,hip --no-dkms
```

### Step 2: Clone and Build Turbo1Bit

```bash
# Clone Turbo1Bit repository
git clone https://github.com/ikantkode/Turbo1bit.git
cd Turbo1bit

# Clone bonsai-llama.cpp fork
git clone --branch prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git bonsai-llama.cpp
```

### Step 3: Build Custom rocBLAS with gfx906 Support

**IMPORTANT:** The default ROCm installation may not include pre-compiled kernels for your GPU architecture (e.g., gfx906 for MI50). You need to build rocBLAS from source:

```bash
# Create build directory for rocBLAS
mkdir -p rocblas-build
cd rocblas-build

# Clone rocBLAS
git clone https://github.com/ROCm/rocBLAS.git
cd rocBLAS
git checkout develop  # or a specific release tag

# Create build directory
mkdir build && cd build

# Configure rocBLAS for your GPU architecture
cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm-7.0.0/lib/llvm/bin/clang++ \
  -DGPU_TARGETS=gfx906:xnack- \
  -DBUILD_WITH_TENSILE=OFF \
  -DCMAKE_INSTALL_PREFIX=/home/shakespear/bonsai/Turbo1bit/rocblas-build/install \
  ..

# Build rocBLAS (this will take several minutes)
make -j$(nproc)

# Install rocBLAS
make install
```

**Note:** Replace `gfx906:xnack-` with your GPU's architecture:
- MI50/MI60: `gfx906:xnack-`
- MI100: `gfx908:xnack-`
- MI200: `gfx90a:xnack-`
- MI210: `gfx90a:xnack+`

Check your GPU architecture with: `rocminfo | grep "Name:"`

### Step 4: Build llama.cpp with HIP Support

```bash
cd /home/shakespear/bonsai/Turbo1bit/bonsai-llama.cpp
rm -rf build
mkdir build && cd build

# Set library path for custom rocBLAS
export LD_LIBRARY_PATH=/home/shakespear/bonsai/Turbo1bit/rocblas-build/install/lib:$LD_LIBRARY_PATH

# Configure with HIP support
cmake .. \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/home/shakespear/bonsai/Turbo1bit/rocblas-build/install

# Build
make -j$(nproc) llama-server
```

### Step 5: Download a Model

```bash
cd /home/shakespear/bonsai/Turbo1bit
mkdir -p models

# Using Python
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('prism-ml/Bonsai-1.7B-gguf', local_dir='models', allow_patterns='*.gguf')"
```

### Step 6: Run the Server

```bash
export LD_LIBRARY_PATH=/home/shakespear/bonsai/Turbo1bit/rocblas-build/install/lib:/opt/rocm/lib
cd /home/shakespear/bonsai/Turbo1bit/bonsai-llama.cpp/build

# Start server with GPU layer offloading
bin/llama-server \
  -m ../../models/Bonsai-1.7B.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  -c 8192 \
  --n-gpu-layers 29
```

**Key Parameters:**
- `--n-gpu-layers 29`: Number of transformer layers to offload to GPU (29 = all layers for Bonsai-1.7B)
- `--host 0.0.0.0`: Listen on all network interfaces (accessible from other devices)
- `-c 8192`: Context size in tokens

### Step 7: Verify GPU Acceleration

When the server starts, you should see:

```
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack-
load_tensors: offloaded 29/29 layers to GPU
```

Check GPU usage:
```bash
rocm-smi
```

### Troubleshooting

**Issue:** `rocBLAS warning: No paths matched *gfx906*co`

**Solution:** Build rocBLAS from source as shown in Step 3. The default ROCm installation doesn't include pre-compiled kernels for all GPU architectures.

**Issue:** Server starts but inference fails with `CUBLAS_STATUS_INTERNAL_ERROR`

**Solution:** Ensure your custom-built rocBLAS is in the library path:
```bash
export LD_LIBRARY_PATH=/home/shakespear/bonsai/Turbo1bit/rocblas-build/install/lib:/opt/rocm/lib
```

**Issue:** Model layers not offloading to GPU

**Solution:**
1. Check GPU architecture support: `rocminfo | grep "Name:"`
2. Ensure `GPU_TARGETS` in rocBLAS build matches your GPU
3. Verify with smaller `--n-gpu-layers` value first

## Running Inference

### Command Line

```bash
cd /home/shakespear/bonsai/Turbo1bit/bonsai-llama.cpp/build

# Interactive mode
bin/llama-cli \
  -m ../../models/Bonsai-1.7B.gguf \
  -p "Explain quantum computing in simple terms:" \
  -n 200 \
  -c 8192 \
  --n-gpu-layers 29
```

### Web Server

```bash
# Start OpenAI-compatible API server
bin/llama-server \
  -m ../../models/Bonsai-1.7B.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  -c 8192 \
  --n-gpu-layers 29

# Test with curl
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai","messages":[{"role":"user","content":"Hello!"}]}'
```

Access the web UI at: `http://localhost:8080` or `http://<your-ip>:8080`

## How It Works

llama.cpp has built-in KV cache quantization (`--ctk`, `--ctv`) and Flash Attention (`--fa`), but Bonsai's documentation and scripts don't use either. Trying KV quantization without FA produces a cryptic error, so most users assume it's unsupported. Turbo1Bit validated that the combination works with 1-bit models and measured the quality/memory trade-offs:

```text
llama_init_from_model: quantized V cache was requested, but this requires Flash Attention
```

**Why does quantized V cache require Flash Attention?** Without FA, llama.cpp stores the V cache **transposed** — each row is a single element scattered across heads, which is incompatible with block quantization formats like Q4_0 (they need contiguous groups of 32 values). Flash Attention stores V non-transposed (contiguous per head), making quantization possible.

With `--fa on`, Q4_0/Q5_0/Q8_0 KV cache types work. This combination is undocumented in the Bonsai project and we validated quality for 1-bit models specifically.

Flash Attention also provides a **2.4x prefill speedup** as a bonus:

| Mode | Prefill (tok/s) | Decode (tok/s) |
|------|----------------|---------------|
| No FA (original) | 1,425 | 134 |
| FA + FP16 KV | **3,452** | **151** |
| FA + Q4_0 KV | **3,435** | **131** |

## Full Benchmark Results

### Bonsai-1.7B (Measured RSS via `/usr/bin/time -l`)

| Context | FP16 | Q8_0 | Q5_0 | Q4_0 |
|---------|------|------|------|------|
| 2K | 648 MiB | 543 MiB | 501 MiB | 487 MiB |
| 8K | 1,344 MiB | 924 MiB | 756 MiB | 700 MiB |
| 32K | 4,131 MiB | 2,454 MiB | 1,780 MiB | 1,555 MiB |
| 65K | 7,846 MiB | 4,489 MiB | 3,142 MiB | 2,694 MiB |

### Bonsai-8B (Measured RSS)

| Context | FP16 | Q4_0 | Saved |
|---------|------|------|-------|
| 2K | 1,592 MiB | 1,293 MiB | 299 MiB |
| 8K | 2,379 MiB | 1,557 MiB | 822 MiB |
| 32K | 5,891 MiB | 2,626 MiB | 3,265 MiB |
| 65K | 10,618 MiB | 4,000 MiB | 6,618 MiB |

### Output Quality — Perplexity (WikiText-2, Bonsai-1.7B, 20 chunks)

| Config | Perplexity | vs Baseline | Memory Saving |
|--------|-----------|-------------|---------------|
| FP16 + FA | **25.51** | — | 1x |
| Q8_0 + FA | **25.49** | -0.1% | 1.75x |
| Q5_0 + FA | **25.87** | +1.4% | 2.50x |
| Q4_0 + FA | **26.82** | +5.1% | 2.91x |

Q8_0 is statistically identical to baseline. Q4_0 adds 5% perplexity for 2.91x memory savings.

## What Turbo1Bit Includes

| Component | Description |
|-----------|-------------|
| `turbo1bit` | Simple wrapper script with auto RAM detection |
| `turbo1bit-server` | OpenAI-compatible API server with compressed KV cache |
| `turbo1bit-infer` | Non-interactive inference tool with KV compression flags |
| TurboQuant C port | Lloyd-Max codebooks, orthogonal rotation, QJL projection, group quantization |
| Metal shaders | 6 GPU kernels for compressed KV attention (Apple Silicon) |
| HIP kernels | 6 GPU kernels for compressed KV attention (AMD ROCm) |
| Quality sweep | Tested key compression at 3/4/5 bits, value compression at 2/4 bits |
| Benchmark suite | `turbo1bit-bench`, `turbo1bit-stress` for standalone KV cache testing |

### TurboQuant Compression Research

Beyond the native KV quantization, Turbo1Bit includes a C port of the [TurboQuant](https://github.com/0xSero/turboquant) (ICLR 2026) compression algorithms. Key findings for 1-bit models:

- **2-bit value compression**: Lossless — output identical to baseline
- **4-bit key compression**: Good quality — coherent text with minor rephrasings
- **3-bit key compression**: Too aggressive — output degrades to gibberish
- **Threshold**: 1-bit models need >= 4-bit keys (FP16 models can use 3-bit)

## Project Structure

```text
Turbo1bit/
├── turbo1bit                    # Simple wrapper script
├── turbo1bit-server             # OpenAI-compatible API server
├── src/                         # TurboQuant C port
│   ├── turbo1bit_codebook.h/c   # Lloyd-Max optimal codebooks
│   ├── turbo1bit_rotation.h/c   # QR rotation + QJL projection
│   ├── turbo1bit_quantizer.h/c  # MSE + Prod quantizers
│   ├── turbo1bit_kv_cache.h/c   # Compressed KV cache manager
│   ├── turbo1bit_metal.h/m      # Metal GPU host code (Apple Silicon)
│   ├── turbo1bit_hip.cpp        # HIP GPU host code (AMD ROCm)
│   └── turbo1bit_hip_kernels.hip # HIP compute shaders (AMD ROCm)
├── tools/turbo1bit/
│   ├── turbo1bit_infer.cpp      # End-to-end inference tool
│   ├── turbo1bit_bench.c        # Core algorithm benchmarks
│   └── turbo1bit_stress.c       # Extreme context stress test
├── BENCHMARKS.md                # Detailed benchmark data
├── rocblas-build/               # Custom rocBLAS build (AMD GPU)
└── CMakeLists.txt               # Standalone build
```

## GPU Kernels

Turbo1Bit includes GPU-accelerated kernels for:

1. **MSE Score**: Computes attention scores against MSE-quantized keys
2. **QJL Score**: Adds QJL residual contribution to scores
3. **Fused Attention**: Full fused decode attention with online softmax
4. **Value Dequantization**: Unpack and dequantize group-quantized values
5. **Value Quantize-Dequantize**: In-place value quantization roundtrip
6. **Matrix-Vector Multiply**: For rotation and projection operations

## Credits

- **[Bonsai / PrismML](https://github.com/PrismML-Eng/Bonsai-demo)** — 1-bit LLM models and inference
- **[TurboQuant](https://github.com/0xSero/turboquant)** by 0xSero — KV cache compression algorithms (ICLR 2026)
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — C/C++ LLM inference engine
- **[ROCm](https://github.com/ROCm/ROCm)** — AMD open-source GPU computing platform

## License

GPL-3.0
