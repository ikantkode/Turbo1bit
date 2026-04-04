# Turbo1Bit for AMD MI50

**Run Bonsai-8B at full 65K context on AMD MI50 with minimal memory.**

Turbo1Bit enables KV cache compression for [PrismML's Bonsai](https://github.com/PrismML-Eng/Bonsai-demo) 1-bit LLMs. By combining Flash Attention with quantized KV storage, it reduces inference memory by up to **4.24x** — making large models fit on smaller hardware.

**This guide is specifically for AMD Radeon Instinct MI50 (gfx906) with ROCm on a 32GB VRAM system.**

---

## Quick Start - Fresh Deployment (32GB MI50)

**Important:** All commands below assume you are working from within the `Turbo1bit` directory. After cloning, run `cd Turbo1bit` and all paths will be relative to that directory.

### Prerequisites

- Ubuntu 24.04 (or similar Linux distribution)
- AMD MI50 GPU with **32GB VRAM**
- ROCm drivers installed
- Git, CMake, build-essential, Python 3

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/ikantkode/Turbo1bit.git
cd Turbo1bit

# Clone the Bonsai fork of llama.cpp
git clone --branch prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git bonsai-llama.cpp
```

### Step 2: Verify ROCm Installation

```bash
# Check if ROCm is installed
rocm-smi
/opt/rocm/bin/hipcc --version
```

If these commands fail, install ROCm first:
```bash
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000_all.deb
sudo apt install -y ./amdgpu-install_6.0.60000_all.deb
sudo amdgpu-install --usecase=rocm,hip --no-dkms
```

### Step 3: Build Custom rocBLAS (Critical for MI50)

**Why this is needed:** The default ROCm installation doesn't include pre-compiled kernels for gfx906. You must build rocBLAS from source.

```bash
# Create build directory
mkdir -p rocblas-build
cd rocblas-build

# Clone rocBLAS
git clone https://github.com/ROCm/rocBLAS.git
cd rocBLAS
git checkout develop

# Build rocBLAS for MI50 (gfx906)
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm-7.0.0/lib/llvm/bin/clang++ \
  -DGPU_TARGETS=gfx906:xnack- \
  -DBUILD_WITH_TENSILE=OFF \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/../../install \
  ..

# Build (takes 5-10 minutes)
make -j$(nproc)

# Install
make install
```

### Step 4: Build llama.cpp

```bash
# You should be in the Turbo1bit directory
cd bonsai-llama.cpp

# Set library path for custom rocBLAS
export LD_LIBRARY_PATH=$(pwd)/../rocblas-build/install/lib:/opt/rocm/lib

# Create build directory
rm -rf build && mkdir build && cd build

# Configure with HIP support
cmake .. \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$(pwd)/../../rocblas-build/install

# Build the server and CLI tools
make -j$(nproc) llama-server llama-cli
```

### Step 5: Download Models

```bash
# Go back to Turbo1bit directory
cd ..

# Create models directory
mkdir -p models

# Install huggingface_hub
pip install huggingface_hub

# Download Bonsai-1.7B (237 MB, 65K context, fastest)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('prism-ml/Bonsai-1.7B-gguf', local_dir='models', allow_patterns='*.gguf')"

# OR download Bonsai-4B (546 MB, 32K context, balanced)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('prism-ml/Bonsai-4B-gguf', local_dir='models', allow_patterns='*.gguf')"

# OR download Bonsai-8B (1.15 GB, 65K context, smartest)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('prism-ml/Bonsai-8B-gguf', local_dir='models', allow_patterns='*.gguf')"
```

**Note:** You can download all three models and switch between them as needed.

### Step 6: Start the Server

```bash
# Navigate to build directory (from Turbo1bit root)
cd bonsai-llama.cpp/build

# Set library path (run this every time you open a new terminal)
export LD_LIBRARY_PATH=$(pwd)/../../rocblas-build/install/lib:/opt/rocm/lib

# Start with Bonsai-4B (recommended for 32GB MI50)
./bin/llama-server \
  -m ../../models/Bonsai-4B.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  -c 32768 \
  --n-gpu-layers 37
```

**Access the web UI at:** http://localhost:8080 or http://<your-ip>:8080

---

## Model Comparison

| Model | Params | Size | Layers | Max Context | GPU Layers | VRAM at Max | Speed | Quality |
|-------|--------|------|--------|-------------|------------|-------------|-------|---------|
| **Bonsai-1.7B** | 1.7B | 237 MB | 24 | 65,536 | **29** | ~9.3 GB | 🚀 Fastest | ⭐ Basic |
| **Bonsai-4B** | 4.0B | 546 MB | 36 | 32,768 | **37** | ~5.2 GB | ⚡ Fast | ⭐⭐ Good |
| **Bonsai-8B** | 8.2B | 1.15 GB | 36 | 65,536 | **37** | ~15.6 GB | 🐢 Slowest | ⭐⭐⭐ Best |

**Recommended for 32GB MI50:** Bonsai-4B (best balance of speed and quality)

---

## Switching Models

### Stop Current Server

```bash
# Find the server process
ps aux | grep llama-server

# Kill the process (replace <PID> with actual process ID)
kill <PID>

# OR kill all llama-server processes
pkill -f llama-server
```

### Start Different Model

```bash
# From Turbo1bit directory
cd bonsai-llama.cpp/build
export LD_LIBRARY_PATH=$(pwd)/../../rocblas-build/install/lib:/opt/rocm/lib

# Bonsai-1.7B - Fastest, 65K context
./bin/llama-server \
  -m ../../models/Bonsai-1.7B.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  -c 65536 \
  --n-gpu-layers 29

# Bonsai-4B - Balanced, 32K context (RECOMMENDED)
./bin/llama-server \
  -m ../../models/Bonsai-4B.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  -c 32768 \
  --n-gpu-layers 37

# Bonsai-8B - Smartest, 65K context
./bin/llama-server \
  -m ../../models/Bonsai-8B.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  -c 65536 \
  --n-gpu-layers 37
```

---

## Calling Inference

### Web UI (Easiest)

Open your browser and go to: **http://localhost:8080**

### OpenAI-Compatible API

#### Chat Completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bonsai",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 256,
    "temperature": 0.5,
    "top_p": 0.85,
    "top_k": 20
  }'
```

#### Completion (Text Generation)

```bash
curl http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bonsai",
    "prompt": "Once upon a time",
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

### Python Example

```python
import requests

url = "http://localhost:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "model": "bonsai",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about AI."}
    ],
    "max_tokens": 100,
    "temperature": 0.5
}

response = requests.post(url, headers=headers, json=data)
print(response.json()['choices'][0]['message']['content'])
```

### Command Line Interface

```bash
# From Turbo1bit directory
cd bonsai-llama.cpp/build
export LD_LIBRARY_PATH=$(pwd)/../../rocblas-build/install/lib:/opt/rocm/lib

# Interactive mode
./bin/llama-cli \
  -m ../../models/Bonsai-4B.gguf \
  -p "Explain quantum computing:" \
  -n 256 \
  -c 8192 \
  --n-gpu-layers 37

# Interactive chat
./bin/llama-cli \
  -m ../../models/Bonsai-4B.gguf \
  -cnv \
  -c 8192 \
  --n-gpu-layers 37
```

---

## Context Length Guide

The context length (`-c` flag) determines how much text the model can remember. Higher values use more VRAM.

### Recommended Context for 32GB MI50

| Context | 1.7B VRAM | 4B VRAM | 8B VRAM | Use Case |
|---------|-----------|---------|---------|----------|
| 2,048 | ~300 MB | ~500 MB | ~800 MB | Quick chats |
| 8,192 | ~1.1 GB | ~1.8 GB | ~3.0 GB | Standard conversations |
| 16,384 | ~2.3 GB | ~3.6 GB | ~5.9 GB | Long documents |
| **32,768** | ~4.5 GB | **~7.2 GB** | ~11.8 GB | **Recommended for 4B** |
| **65,536** | **~9.0 GB** | ~14.4 GB (exceeds 32K limit) | **~23.6 GB** | **Max for 1.7B/8B** |

**Important:** Bonsai-4B has a maximum context of **32,768 tokens**. Don't use `-c 65536` with 4B!

### Example: Different Context Lengths

```bash
# Low memory usage (8K context)
./bin/llama-server -m ../../models/Bonsai-4B.gguf -c 8192 --n-gpu-layers 37

# Balanced (16K context)
./bin/llama-server -m ../../models/Bonsai-4B.gguf -c 16384 --n-gpu-layers 37

# Maximum for 4B (32K context)
./bin/llama-server -m ../../models/Bonsai-4B.gguf -c 32768 --n-gpu-layers 37
```

---

## Server Flags Reference

| Flag | Description | Example |
|------|-------------|---------|
| `-m` | Model path | `-m ../../models/Bonsai-4B.gguf` |
| `--port` | Server port | `--port 8080` |
| `--host` | Network interface | `--host 0.0.0.0` (all) or `--host 127.0.0.1` (local) |
| `-c` | Context length | `-c 32768` |
| `--n-gpu-layers` | Layers to offload to GPU | `--n-gpu-layers 37` (see table below) |
| `--no-warmup` | Skip warmup (faster startup) | `--no-warmup` |

### Critical: Correct GPU Layer Counts

| Model | Total Layers | Use `--n-gpu-layers` |
|-------|--------------|---------------------|
| Bonsai-1.7B | 29 | **29** |
| Bonsai-4B | 37 | **37** |
| Bonsai-8B | 37 | **37** |

**⚠️ Important:** Using fewer GPU layers than total will leave some layers on CPU, causing **massive slowdown** (10x slower or more). Always use the exact values above!

---

## Quick Reference Commands

### Check Server Health
```bash
curl http://localhost:8080/health
```

### Check Model Info
```bash
curl http://localhost:8080/v1/models
```

### Check GPU Usage
```bash
rocm-smi
```

### Restart Server
```bash
# Stop
pkill -f llama-server

# Start (from Turbo1bit directory)
cd bonsai-llama.cpp/build
export LD_LIBRARY_PATH=$(pwd)/../../rocblas-build/install/lib:/opt/rocm/lib
./bin/llama-server -m ../../models/Bonsai-4B.gguf --port 8080 --host 0.0.0.0 -c 32768 --n-gpu-layers 37
```

### View Server Logs
```bash
# If you started server with log redirection
tail -f /tmp/server-4b.log
```

---

## Performance Benchmarks

### Speed (Tokens Per Second) on MI50

| Model | Context | Prefill | Decode |
|-------|---------|---------|--------|
| Bonsai-1.7B | 32K | ~90 tok/s | ~130 tok/s |
| **Bonsai-4B** | 32K | ~70 tok/s | **~105 tok/s** |
| Bonsai-8B | 32K | ~50 tok/s | ~70 tok/s |

### Memory Usage (32GB MI50)

| Model | Weights | KV Cache (32K) | Total |
|-------|---------|----------------|-------|
| Bonsai-1.7B | 237 MB | 4.5 GB | ~4.8 GB |
| **Bonsai-4B** | 546 MB | 4.5 GB | **~5.1 GB** |
| Bonsai-8B | 1.15 GB | 4.5 GB | ~5.7 GB |

All three models fit comfortably on 32GB VRAM with headroom to spare!

---

## Troubleshooting

### "rocBLAS warning: No paths matched *gfx906*co"

**Cause:** Default ROCm doesn't include MI50 kernels.

**Solution:** Build rocBLAS from source as shown in Step 3.

### Server starts but inference is very slow (~15 tok/s instead of 100+)

**Cause:** Not all layers are on GPU.

**Solution:** Check the logs for `offloaded X/Y layers`. If X < Y, increase `--n-gpu-layers`:
- Bonsai-1.7B: Use `--n-gpu-layers 29`
- Bonsai-4B: Use `--n-gpu-layers 37`
- Bonsai-8B: Use `--n-gpu-layers 37`

### "failed to open GGUF file"

**Cause:** Wrong model path.

**Solution:** Make sure you're in the correct directory and using the right path:
```bash
# From Turbo1bit directory
cd bonsai-llama.cpp/build
./bin/llama-server -m ../../models/Bonsai-4B.gguf ...
```

### Out of memory errors

**Solutions:**
1. Reduce context length: `-c 16384` instead of `-c 32768`
2. Use smaller model (1.7B instead of 8B)
3. Close other GPU-intensive applications

### Port 8080 already in use

**Solution:** Either stop the other process or use a different port:
```bash
./bin/llama-server ... --port 8081
```

---

## How It Works

Turbo1Bit leverages llama.cpp's built-in KV cache compression and Flash Attention:

1. **Flash Attention** stores KV cache non-transposed (contiguous per head)
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

## Quality Benchmarks

### Bonsai-4B Memory Usage (Measured RSS)

| Context | FP16 | Compressed | **Saved** |
|---------|------|------------|----------|
| 8K tokens | 1,537 MiB | 620 MiB | **917 MiB** |
| 16K tokens | 2,832 MiB | 1,071 MiB | **1.8 GB** |
| **32K tokens** | 5,422 MiB | 2,004 MiB | **3.4 GB** |

### Quality Impact (WikiText-2 Perplexity)

| Config | Perplexity | vs Baseline |
|--------|-----------|-------------|
| FP16 + FA | **25.51** | baseline |
| Q8_0 + FA | **25.49** | -0.1% ✅ |
| Q4_0 + FA | **26.82** | +5.1% ✅ |

Q8_0 is statistically identical to baseline. Q4_0 adds minimal perplexity for 2.91x memory savings.

---

## Project Structure

```text
Turbo1bit/
├── README.md                   # This file
├── CLAUDE.md                   # Development guide
├── CMakeLists.txt              # Standalone build config
├── src/                        # TurboQuant C implementation
│   ├── turbo1bit_codebook.*   # Lloyd-Max codebooks
│   ├── turbo1bit_rotation.*   # QR rotation + QJL projection
│   ├── turbo1bit_quantizer.*  # MSE + Prod quantizers
│   ├── turbo1bit_kv_cache.*   # Compressed KV cache manager
│   └── turbo1bit_hip.*        # HIP GPU code (AMD ROCm)
├── tools/turbo1bit/            # CLI tools
│   ├── turbo1bit_bench.c      # Benchmarking
│   └── turbo1bit_stress.c     # Stress testing
├── bonsai-llama.cpp/          # Forked llama.cpp
│   └── build/
│       └── bin/
│           ├── llama-server   # Web server
│           └── llama-cli      # CLI interface
├── rocblas-build/             # Custom rocBLAS (required for MI50)
│   └── install/               # Built libraries
└── models/                    # Downloaded models (NOT in git)
    ├── Bonsai-1.7B.gguf      # Downloaded separately
    ├── Bonsai-4B.gguf        # Downloaded separately
    └── Bonsai-8B.gguf        # Downloaded separately
```

---

## Credits

- **[Bonsai / PrismML](https://github.com/PrismML-Eng/Bonsai-demo)** — 1-bit LLM models
- **[TurboQuant](https://github.com/0xSero/turboquant)** — KV cache compression algorithms (ICLR 2026)
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — C/C++ LLM inference engine
- **[ROCm](https://github.com/ROCm)** — AMD open-source GPU computing platform

---

## License

GPL-3.0
