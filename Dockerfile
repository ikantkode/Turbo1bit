# Dockerfile for Turbo1Bit with AMD MI50 (ROCm) support
# Multi-stage build to minimize final image size

FROM rocm/dev-ubuntu-22.04:5.7 as builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    python3 \
    python3-pip \
    wget \
    libnuma-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install huggingface_hub

# Set working directory
WORKDIR /app

# ========================
# Stage 1: Build rocBLAS
# ========================

# Clone rocBLAS
RUN git clone https://github.com/ROCm/rocBLAS.git rocBLAS
WORKDIR /app/rocBLAS
RUN git checkout develop

# Build rocBLAS for MI50 (gfx906)
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
        -DGPU_TARGETS=gfx906:xnack- \
        -DBUILD_WITH_TENSILE=OFF \
        -DCMAKE_INSTALL_PREFIX=/opt/rocblas \
        .. && \
    make -j$(nproc) && \
    make install

# ========================
# Stage 2: Build llama.cpp
# ========================

WORKDIR /app

# Clone Turbo1Bit and llama.cpp
RUN git clone https://github.com/ikantkode/Turbo1bit.git turbo1bit
RUN git clone --branch prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git llama.cpp

WORKDIR /app/llama.cpp
RUN mkdir build && cd build && \
    cmake .. \
        -DGGML_HIP=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/opt/rocblas \
        -DCMAKE_LIBRARY_PATH=/opt/rocblas/lib \
        -DCMAKE_HIP_ARCHITECTURES=gfx906 && \
    make -j$(nproc) llama-server

# ========================
# Stage 3: Download Model
# ========================

WORKDIR /app/turbo1bit
RUN mkdir -p models

# Download Bonsai-1.7B model (you can change this to Bonsai-8B if desired)
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('prism-ml/Bonsai-1.7B-gguf', \
    local_dir='models', \
    allow_patterns='*.gguf')"

# ========================
# Final Stage: Runtime Image
# ========================

FROM rocm/dev-ubuntu-22.04:5.7

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libnuma1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy built components from builder
COPY --from=builder /opt/rocblas /opt/rocblas
COPY --from=builder /app/llama.cpp/build/bin /app/bin
COPY --from=builder /app/turbo1bit/models /app/models

# Set environment variables
ENV LD_LIBRARY_PATH=/opt/rocblas/lib:/opt/rocm/lib
ENV PYTHONUNBUFFERED=1

# Expose server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set default arguments
ENV MODEL_PATH=/app/models/Bonsai-1.7B.gguf
ENV GPU_LAYERS=29
ENV CONTEXT_SIZE=8192

# Entry point
ENTRYPOINT ["/app/bin/llama-server"]
CMD ["--port", "8080", \
     "--host", "0.0.0.0", \
     "-m", "/app/models/Bonsai-1.7B.gguf", \
     "-c", "8192", \
     "--n-gpu-layers", "29"]
