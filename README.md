# Turbo1Bit

**Run Bonsai-8B at full 65K context on an 8GB MacBook Air.**

Turbo1Bit enables KV cache compression for [PrismML's Bonsai](https://github.com/PrismML-Eng/Bonsai-demo) 1-bit LLMs. By combining Flash Attention with quantized KV storage, it reduces inference memory by up to **4.24x** — making large models fit on small hardware.

**Now with GPU acceleration support for Apple Metal and AMD ROCm!**

---

## 🚀 How to Run

### Apple Silicon (macOS)

```bash
# 1. Clone and build
git clone https://github.com/ikantkode/Turbo1bit.git
cd Turbo1bit
git clone --branch prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git bonsai-llama.cpp
cd bonsai-llama.cpp && mkdir build && cd build
cmake .. -G Ninja -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
ninja llama-server
cd ../..

# 2. Download a model
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('prism-ml/Bonsai-1.7B-gguf', local_dir='models', allow_patterns='*.gguf')"

# 3. Run the server
cd bonsai-llama.cpp/build
bin/llama-server -m ../../models/Bonsai-1.7B.gguf --port 8080 -c 8192
```

**Access the web UI at:** http://localhost:8080

---

### AMD GPU (Linux/ROCm)

Tested on AMD Radeon Instinct MI50 (gfx906).

```bash
# 1. Install dependencies (if needed)
sudo apt update && sudo apt install -y git cmake build-essential python3-pip

# 2. Clone and build
git clone https://github.com/ikantkode/Turbo1bit.git
cd Turbo1bit
git clone --branch prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git bonsai-llama.cpp

# 3. Download a model
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('prism-ml/Bonsai-1.7B-gguf', local_dir='models', allow_patterns='*.gguf')"

# 4. Build llama.cpp with GPU support
cd bonsai-llama.cpp
mkdir build && cd build
cmake .. -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) llama-server

# 5. Run the server with GPU acceleration
bin/llama-server -m ../../models/Bonsai-1.7B.gguf --port 8080 --host 0.0.0.0 -c 8192 --n-gpu-layers 29
```

**Access the web UI at:** http://localhost:8080 or http://<your-ip>:8080

> **Note:** If you get a `rocBLAS` error about missing gfx906 kernels, see the [AMD GPU Setup Guide](#amd-gpu-setup) below.

---

## 📊 Results

### Memory Savings (Bonsai-8B)

| Context | Standard | Turbo1Bit | **Saved** |
|---------|----------|-----------|----------|
| 8K tokens | 2,379 MiB | 1,557 MiB | **822 MiB** |
| 32K tokens | 5,891 MiB | 2,626 MiB | **3.3 GB** |
| **65K tokens** | 10,618 MiB | 4,000 MiB | **6.5 GB** |

At 65K context, Bonsai-8B needs 10.6 GB — too large for 8GB hardware. With Turbo1Bit, it fits in **4 GB**.

### Quality Impact (WikiText-2 Perplexity)

| Config | Perplexity | vs Baseline |
|--------|-----------|-------------|
| FP16 + FA | **25.51** | baseline |
| Q8_0 + FA | **25.49** | -0.1% ✅ |
| Q4_0 + FA | **26.82** | +5.1% ✅ |

Q8_0 is statistically identical to baseline. Q4_0 adds minimal perplexity for 2.91x memory savings.

---

## 🖥️ Installation

### Prerequisites

- **CMake** 3.16 or later
- **C++ compiler** (GCC, Clang, or Xcode)
- **Python** 3.7+ with `pip`
- **Git**

### Platform-Specific Setup

#### macOS (Apple Silicon)

Install Xcode Command Line Tools:
```bash
xcode-select --install
```

Then follow the [Apple Silicon Quick Start](#apple-silicon-macos) above.

#### Linux with AMD GPU

**Tested on:** Ubuntu 24.04, AMD MI50 (gfx906), ROCm 7.0

**Basic Requirements:**
- ROCm installed and working
- AMD GPU supported by ROCm (MI50, MI60, MI100, etc.)

**Quick Test - Check if ROCm works:**
```bash
rocm-smi
/opt/rocm/bin/hipcc --version
```

If those commands work, you can skip to step 4 below. If you get errors about missing gfx906 kernels, continue to the full setup.

---

### AMD GPU Setup Guide (Detailed)

This section provides detailed setup instructions for AMD GPUs. If the basic quick start above worked, you can skip this.

#### Step 1: Install ROCm (if not already installed)

```bash
# Download ROCm installer
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000_all.deb

# Install ROCm
sudo apt install -y ./amdgpu-install_6.0.60000_all.deb
sudo amdgpu-install --usecase=rocm,hip --no-dkms

# Verify installation
rocm-smi
```

#### Step 2: Clone the Repository

```bash
git clone https://github.com/ikantkode/Turbo1bit.git
cd Turbo1bit
```

#### Step 3: Build Custom rocBLAS (for MI50/MI60 gfx906 support)

**Why is this needed?** The default ROCm installation doesn't include pre-compiled kernels for all GPU architectures. This step builds rocBLAS from source with support for your specific GPU.

```bash
# Create build directory
mkdir -p rocblas-build
cd rocblas-build

# Clone rocBLAS
git clone https://github.com/ROCm/rocBLAS.git
cd rocBLAS
git checkout develop

# Build rocBLAS for your GPU
mkdir build && cd build

# Configure - replace gfx906:xnack- with your GPU architecture
cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm-7.0.0/lib/llvm/bin/clang++ \
  -DGPU_TARGETS=gfx906:xnack- \
  -DBUILD_WITH_TENSILE=OFF \
  -DCMAKE_INSTALL_PREFIX=/home/shakespear/bonsai/Turbo1bit/rocblas-build/install \
  ..

# Build (takes 5-10 minutes)
make -j$(nproc)

# Install
make install
```

**GPU Architecture Mapping:**
- MI50/MI60: `gfx906:xnack-`
- MI100: `gfx908:xnack-`
- MI200: `gfx90a:xnack+`
- MI210: `gfx90a:xnack+`

Check your GPU: `rocminfo | grep "Name:"`

#### Step 4: Build llama.cpp with GPU Support

```bash
cd /home/shakespear/bonsai/Turbo1bit/bonsai-llama.cpp

# Create build directory
rm -rf build && mkdir build && cd build

# Set library path for custom rocBLAS
export LD_LIBRARY_PATH=/home/shakespear/bonsai/Turbo1bit/rocblas-build/install/lib:/opt/rocm/lib

# Configure with HIP support
cmake .. \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/home/shakespear/bonsai/Turbo1bit/rocblas-build/install

# Build
make -j$(nproc) llama-server
```

#### Step 5: Download a Model

```bash
cd /home/shakespear/bonsai/Turbo1bit
mkdir -p models

# Install huggingface_hub if needed
pip install huggingface_hub

# Download Bonsai-1.7B (or Bonsai-8B)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('prism-ml/Bonsai-1.7B-gguf', local_dir='models', allow_patterns='*.gguf')"
```

#### Step 6: Run the Server

```bash
cd /home/shakespear/bonsai/Turbo1bit/bonsai-llama.cpp/build

# Set library path
export LD_LIBRARY_PATH=/home/shakespear/bonsai/Turbo1bit/rocblas-build/install/lib:/opt/rocm/lib

# Start server
bin/llama-server \
  -m ../../models/Bonsai-1.7B.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  -c 8192 \
  --n-gpu-layers 29
```

**What the flags mean:**
- `--n-gpu-layers 29`: Offload all 29 layers to GPU (full GPU acceleration)
- `--host 0.0.0.0`: Listen on all network interfaces
- `-c 8192`: Context window of 8192 tokens

#### Step 7: Verify GPU Acceleration

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

You should see VRAM usage indicating the model is loaded on the GPU.

---

## 🔧 Usage

### Command Line Interface

```bash
cd /home/shakespear/bonsai/Turbo1bit/bonsai-llama.cpp/build

# Interactive mode
bin/llama-cli \
  -m ../../models/Bonsai-1.7B.gguf \
  -p "Explain quantum computing:" \
  -n 200 \
  -c 8192 \
  --n-gpu-layers 29
```

### Web Server (OpenAI-Compatible API)

```bash
# Start server
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

**Access the web UI at:** http://localhost:8080

---

## 🩹 Troubleshooting

### Issue: "rocBLAS warning: No paths matched *gfx906*co"

**Solution:** Build rocBLAS from source as shown in [Step 3](#step-3-build-custom-rocblas-for-mi50mi60-gfx906-support). The default ROCm installation doesn't include pre-compiled kernels for all GPU architectures.

### Issue: Server starts but inference fails

**Solution:** Ensure your custom-built rocBLAS is in the library path:
```bash
export LD_LIBRARY_PATH=/home/shakespear/bonsai/Turbo1bit/rocblas-build/install/lib:/opt/rocm/lib
```

### Issue: Model layers not offloading to GPU

**Solution:**
1. Check your GPU architecture: `rocminfo | grep "Name:"`
2. Ensure `GPU_TARGETS` in rocBLAS build matches your GPU
3. Try with fewer layers first: `--n-gpu-layers 5`

### Issue: Port 8080 already in use

**Solution:** Either stop the other process or use a different port:
```bash
bin/llama-server ... --port 8081
```

---

## 📈 How It Works

llama.cpp has built-in KV cache quantization and Flash Attention, but they need to be enabled together for 1-bit models. Turbo1Bit makes this easy:

**The Secret Sauce:**
1. **Flash Attention** stores KV cache non-transposed (contiguous)
2. **KV Quantization** compresses the cache (Q4_0, Q8_0, etc.)
3. Without FA → transposed storage → incompatible with quantization
4. With FA → contiguous storage → quantization works!

**Performance Bonus:** Flash Attention also provides a **2.4x prefill speedup**:

| Mode | Prefill (tok/s) | Decode (tok/s) |
|------|----------------|---------------|
| No FA | 1,425 | 134 |
| FA + FP16 KV | **3,452** | **151** |
| FA + Q4_0 KV | **3,435** | **131** |

---

## 🧪 Benchmarks

### Bonsai-1.7B Memory Usage (Measured RSS)

| Context | FP16 | Q8_0 | Q5_0 | Q4_0 |
|---------|------|------|------|------|
| 2K | 648 MiB | 543 MiB | 501 MiB | 487 MiB |
| 8K | 1,344 MiB | 924 MiB | 756 MiB | 700 MiB |
| 32K | 4,131 MiB | 2,454 MiB | 1,780 MiB | 1,555 MiB |
| 65K | 7,846 MiB | 4,489 MiB | 3,142 MiB | 2,694 MiB |

### Bonsai-8B Memory Usage (Measured RSS)

| Context | FP16 | Q4_0 | Saved |
|---------|------|------|-------|
| 2K | 1,592 MiB | 1,293 MiB | 299 MiB |
| 8K | 2,379 MiB | 1,557 MiB | **822 MiB** |
| 32K | 5,891 MiB | 2,626 MiB | **3.3 GB** |
| **65K** | 10,618 MiB | 4,000 MiB | **6.5 GB** |

---

## 🏗️ Project Structure

```text
Turbo1bit/
├── README.md                   # This file
├── CLAUDE.md                   # Development guide
├── CMakeLists.txt              # Main build configuration
├── turbo1bit                   # Wrapper script (auto-detects platform)
├── turbo1bit-server            # Server wrapper
├── src/                        # TurboQuant C implementation
│   ├── turbo1bit_*.h/c        # Quantization algorithms
│   ├── turbo1bit_metal.*      # Metal GPU code (Apple Silicon)
│   └── turbo1bit_hip.*        # HIP GPU code (AMD ROCm)
├── tools/turbo1bit/            # CLI tools
│   ├── turbo1bit_bench.c      # Benchmarking
│   └── turbo1bit_stress.c     # Stress testing
├── bonsai-llama.cpp/          # Forked llama.cpp
└── rocblas-build/             # Custom rocBLAS (AMD GPU)
```

---

## 🤝 Credits

- **[Bonsai / PrismML](https://github.com/PrismML-Eng/Bonsai-demo)** — 1-bit LLM models
- **[TurboQuant](https://github.com/0xSero/turboquant)** — KV cache compression algorithms (ICLR 2026)
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — C/C++ LLM inference engine
- **[ROCm](https://github.com/ROCm)** — AMD open-source GPU computing platform

---

## 📜 License

GPL-3.0
