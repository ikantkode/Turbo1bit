# Turbo1Bit for AMD MI50

**Run Bonsai-8B at full 65K context on AMD MI50 with minimal memory.**

Turbo1Bit enables KV cache compression for [PrismML's Bonsai](https://github.com/PrismML-Eng/Bonsai-demo) 1-bit LLMs. By combining Flash Attention with quantized KV storage, it reduces inference memory by up to **4.24x** — making large models fit on smaller hardware.

**This guide is specifically for AMD Radeon Instinct MI50 (gfx906) with ROCm.**

---

## Quick Start - Fresh Repo Deployment

### Prerequisites

- Ubuntu 24.04 (or similar Linux distribution)
- AMD MI50 GPU with ROCm drivers installed
- Git, CMake, build-essential, Python 3

### Step 1: Clone and Setup

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
cd /path/to/Turbo1bit/bonsai-llama.cpp

# Set library path for custom rocBLAS
export LD_LIBRARY_PATH=/path/to/Turbo1bit/rocblas-build/install/lib:/opt/rocm/lib

# Create build directory
rm -rf build && mkdir build && cd build

# Configure with HIP support
cmake .. \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/Turbo1bit/rocblas-build/install

# Build the server
make -j$(nproc) llama-server
```

### Step 5: Download Models

```bash
cd /path/to/Turbo1bit
mkdir -p models

# Install huggingface_hub
pip install huggingface_hub

# Download Bonsai-1.7B (faster, less memory)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('prism-ml/Bonsai-1.7B-gguf', local_dir='models', allow_patterns='*.gguf')"

# OR download Bonsai-8B (smarter, more memory)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('prism-ml/Bonsai-8B-gguf', local_dir='models', allow_patterns='*.gguf')"
```

### Step 6: Start the Server

```bash
cd /path/to/Turbo1bit/bonsai-llama.cpp/build

# Set library path
export LD_LIBRARY_PATH=/path/to/Turbo1bit/rocblas-build/install/lib:/opt/rocm/lib

# Start server (adjust flags as needed)
./bin/llama-server \
  -m ../../models/Bonsai-1.7B.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  -c 8192 \
  --n-gpu-layers 29
```

**Access the web UI at:** http://localhost:8080 or http://<your-ip>:8080

---

## Managing Models and Context

### Switching Between Models

Stop the current server and start with a different model:

```bash
# Find and stop the running server
ps aux | grep llama-server
kill <PID>

# Start with 1.7B model (29 layers, faster, less VRAM)
./bin/llama-server \
  -m ../../models/Bonsai-1.7B.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  -c 8192 \
  --n-gpu-layers 29

# OR start with 8B model (36 layers, smarter, more VRAM)
./bin/llama-server \
  -m ../../models/Bonsai-8B.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  -c 8192 \
  --n-gpu-layers 40
```

### Adjusting Context Length

The context length (`-c` flag) determines how much text the model can remember. Higher values use more VRAM.

| Context | 1.7B VRAM | 8B VRAM | Use Case |
|---------|-----------|---------|----------|
| 2,048 | ~300 MB | ~500 MB | Quick chats, simple tasks |
| 8,192 | ~1.1 GB | ~1.8 GB | Standard conversations |
| 32,768 | ~4.5 GB | ~7.2 GB | Long documents, code analysis |
| **65,536** | ~9.0 GB | ~14.4 GB | **Maximum (entire books)** |

**Example: Run 8B with maximum context**
```bash
./bin/llama-server \
  -m ../../models/Bonsai-8B.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  -c 65536 \
  --n-gpu-layers 40
```

### Other Useful Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--port` | Change server port | `--port 8081` |
| `--host` | Network interface | `--host 0.0.0.0` (all) or `--host 127.0.0.1` (local only) |
| `--n-gpu-layers` | Layers to offload to GPU | `29` for 1.7B, `40` for 8B |
| `-c` | Context length | `8192`, `32768`, `65536` |
| `--no-warmup` | Skip warmup (faster startup) | `--no-warmup` |

---

## Quick Reference Commands

### Check Server Health
```bash
curl http://localhost:8080/health
```

### Test Inference
```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai","messages":[{"role":"user","content":"Hello!"}]}'
```

### Check GPU Usage
```bash
rocm-smi
```

### View Server Logs
```bash
# If you redirected output to a log file
tail -f /tmp/server-8b.log
```

### Restart Server with Different Settings
```bash
# Stop current server
ps aux | grep llama-server | grep -v grep | awk '{print $2}' | xargs kill

# Start with new settings
cd /path/to/Turbo1bit/bonsai-llama.cpp/build
export LD_LIBRARY_PATH=/path/to/Turbo1bit/rocblas-build/install/lib:/opt/rocm/lib
./bin/llama-server -m ../../models/Bonsai-8B.gguf --port 8080 --host 0.0.0.0 -c 65536 --n-gpu-layers 40
```

---

## Memory Usage Examples

### Bonsai-1.7B (29 layers)

| Context | Model | KV Cache | Total VRAM |
|---------|-------|----------|------------|
| 8K | 250 MB | 1.1 GB | ~1.4 GB |
| 32K | 250 MB | 4.5 GB | ~4.8 GB |
| 65K | 250 MB | 9.0 GB | ~9.3 GB |

### Bonsai-8B (36 layers)

| Context | Model | KV Cache | Total VRAM |
|---------|-------|----------|------------|
| 8K | 1.15 GB | 1.8 GB | ~3.0 GB |
| 32K | 1.15 GB | 7.2 GB | ~8.4 GB |
| 65K | 1.15 GB | 14.4 GB | ~15.6 GB |

---

## Troubleshooting

### "rocBLAS warning: No paths matched *gfx906*co"

**Cause:** Default ROCm doesn't include MI50 kernels.

**Solution:** Build rocBLAS from source as shown in Step 3.

### Server starts but inference fails

**Cause:** Wrong library path.

**Solution:**
```bash
export LD_LIBRARY_PATH=/path/to/Turbo1bit/rocblas-build/install/lib:/opt/rocm/lib
```

### Model layers not offloading to GPU

**Cause:** Mismatched GPU architecture.

**Solution:**
1. Check your GPU: `rocminfo | grep "Name:"`
2. Ensure `GPU_TARGETS=gfx906:xnack-` in rocBLAS build
3. Try with fewer layers: `--n-gpu-layers 5`

### Port 8080 already in use

**Solution:** Either stop the other process or use a different port:
```bash
./bin/llama-server ... --port 8081
```

### Out of memory errors

**Solutions:**
1. Reduce context length: `-c 32768` instead of `-c 65536`
2. Use 1.7B model instead of 8B
3. Reduce GPU layers: `--n-gpu-layers 20`

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

## Benchmarks

### Bonsai-8B Memory Usage (Measured RSS)

| Context | Standard | Compressed | **Saved** |
|---------|----------|------------|----------|
| 8K tokens | 2,379 MiB | 1,557 MiB | **822 MiB** |
| 32K tokens | 5,891 MiB | 2,626 MiB | **3.3 GB** |
| **65K tokens** | 10,618 MiB | 4,000 MiB | **6.5 GB** |

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
└── rocblas-build/             # Custom rocBLAS (required for MI50)
    └── install/               # Built libraries
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
