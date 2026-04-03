# Docker Setup for AMD MI50

Complete Docker setup for Turbo1Bit with AMD MI50 GPU acceleration.

## Quick Start

### Prerequisites

- Docker 24.0+ with ROCm support
- AMD MI50 GPU with ROCm drivers installed
- At least 8GB RAM (16GB+ recommended)

### Build and Run

```bash
# Clone the repository
git clone https://github.com/ikantkode/Turbo1bit.git
cd Turbo1bit

# Build the Docker image (this will take 20-30 minutes)
docker build -t turbo1bit-mi50 .

# Run the container
docker run -d --name turbo1bit \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -p 8080:8080 \
  turbo1bit-mi50
```

**Access the server at:** http://localhost:8080

### Using Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/Bonsai-1.7B.gguf` | Path to model file |
| `GPU_LAYERS` | `29` | Number of layers to offload to GPU |
| `CONTEXT_SIZE` | `8192` | Context window in tokens |
| `LD_LIBRARY_PATH` | `/opt/rocblas/lib:/opt/rocm/lib` | Library path for ROCm |

### Changing the Model

To use a different model:

```bash
# Option 1: Mount your model directory
docker run -d --name turbo1bit \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -p 8080:8080 \
  -v /path/to/your/models:/app/models:ro \
  -e MODEL_PATH=/app/models/Your-Model.gguf \
  turbo1bit-mi50
```

```bash
# Option 2: Build custom image
# Modify Dockerfile MODEL_PATH and rebuild
docker build -t turbo1bit-custom .
```

## Dockerfile Details

### Base Image

Uses `rocm/dev-ubuntu-22.04:5.7` which includes:
- ROCm 5.7 toolkit
- HIP compiler
- Development tools

### Multi-Stage Build

1. **Builder Stage:**
   - Installs build dependencies (cmake, ninja, etc.)
   - Builds rocBLAS from source for gfx906 (MI50)
   - Builds llama.cpp with HIP support
   - Downloads Bonsai model

2. **Runtime Stage:**
   - Starts from fresh ROCm base image
   - Copies only necessary runtime components
   - Minimal image size

### GPU Access

The container uses several mechanisms to access the GPU:

```yaml
devices:
  - /dev/kfd        # GPU access
  - /dev/dri         # GPU rendering
group_add:
  - video            # GPU group
```

## Building the Image

### Standard Build

```bash
docker build -t turbo1bit-mi50 .
```

### Build with Custom Model

```dockerfile
# Modify Dockerfile MODEL_PATH ENV
ENV MODEL_PATH=/app/models/Bonsai-8B.gguf

# Or modify the download step
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('prism-ml/Bonsai-8B-gguf', \
    local_dir='models', \
    allow_patterns='*.gguf')"
```

### Build Arguments

```bash
# Build with custom context size
docker build --build-arg CONTEXT_SIZE=16384 -t turbo1bit-mi50-large .
```

## Running the Container

### Basic Usage

```bash
docker run -d --name turbo1bit \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -p 8080:8080 \
  turbo1bit-mi50
```

### With Custom Configuration

```bash
docker run -d --name turbo1bit \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -p 8080:8080 \
  -e CONTEXT_SIZE=16384 \
  -e GPU_LAYERS=29 \
  -v /my/models:/app/models:ro \
  turbo1bit-mi50
```

### Interactive Mode

```bash
# Run in foreground to see logs
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -p 8080:8080 \
  turbo1bit-mi50
```

### Shell Access

```bash
# Get a shell inside the container
docker exec -it turbo1bit bash

# Inside container, check GPU
rocminfo
rocm-smi

# Check server
curl http://localhost:8080/health
```

## Troubleshooting

### Issue: "No GPU devices found"

**Symptoms:** Container starts but can't access GPU

**Solutions:**
```bash
# Check if ROCm is installed on host
rocm-smi

# Verify GPU is accessible
ls -la /dev/kfd /dev/dri/

# Check GPU in container
docker exec turbo1bit rocminfo
```

### Issue: "rocBLAS error: no such file or directory"

**Symptoms:** Server fails to start with rocBLAS errors

**Solution:** Verify the Docker build completed successfully:
```bash
docker logs turbo1bit
```

### Issue: "Out of memory"

**Symptoms:** Container killed during build or runtime

**Solutions:**
```bash
# Increase Docker memory limit
# In Docker Desktop: Settings > Resources > Memory

# Or build with fewer parallel jobs
docker build --build-arg MAKEFLAGS="-j4" -t turbo1bit-mi50 .
```

### Issue: "Port 8080 already in use"

**Solution:** Use a different port:
```bash
docker run -p 8081:8080 turbo1bit-mi50
```

### Issue: "Model loading failed"

**Solutions:**
```bash
# Check logs
docker logs turbo1bit

# Verify model file exists
docker exec turbo1bit ls -la /app/models/

# Check file size
docker exec turbo1bit du -sh /app/models/
```

## Monitoring

### View Logs

```bash
# Follow logs
docker logs -f turbo1bit

# Last 100 lines
docker logs --tail 100 turbo1bit
```

### Resource Usage

```bash
# Container stats
docker stats turbo1bit

# GPU usage
rocm-smi
```

### Health Check

```bash
# Check if server is responding
curl http://localhost:8080/health

# Or use Docker health check
docker inspect --format='{{.State.Health.Status}}' turbo1bit
```

## Advanced Usage

### Custom Build for Bonsai-8B

```dockerfile
# Modify download step in Dockerfile
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('prism-ml/Bonsai-8B-gguf', \
    local_dir='models', \
    allow_patterns='*.gguf')"

ENV MODEL_PATH=/app/models/Bonsai-8B.gguf
ENV GPU_LAYERS=32
ENV CONTEXT_SIZE=65536
```

### Build for Different GPU

```dockerfile
# Modify GPU_TARGETS for your GPU
# MI100: gfx908:xnack-
# MI200: gfx90a:xnack+
RUN cmake -DGPU_TARGETS=gfx908:xnack- ...
```

### Production Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  turbo1bit:
    image: turbo1bit-mi50:latest
    restart: always
    ports:
      - "8080:8080"
    environment:
      - CONTEXT_SIZE=32768
      - GPU_LAYERS=29
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          devices:
            - driver: hip
              device_ids: ['0']
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Performance Tips

1. **Pre-build images:** Build and push to registry to avoid long build times
2. **Use volumes:** Persist models across container restarts
3. **Tune context size:** Balance memory usage and model capabilities
4. **Monitor GPU usage:** Use `rocm-smi` to ensure GPU is being utilized

## Image Variants

| Tag | Description |
|-----|-------------|
| `turbo1bit-mi50:latest` | Latest stable release (Bonsai-1.7B) |
| `turbo1bit-mi50:8b` | Bonsai-8B model |
| `turbo1bit-mi50:dev` | Development version |

## Support

For issues specific to:
- **Docker setup:** Check this document first
- **GPU issues:** Verify ROCm installation with `rocm-smi`
- **Model issues:** Check model file integrity
- **General issues:** See main README.md

## License

GPL-3.0
