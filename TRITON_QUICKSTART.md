# Triton Integration - Quick Reference

## Files Overview

### Model Repository
- `agent/serving/triton/models/qwen3-vl/config.pbtxt` - Model configuration (Python backend)
- `agent/serving/triton/models/qwen3-vl/1/model.py` - Triton Python backend implementation

### Docker & Orchestration
- `Dockerfile` - FastAPI image (legacy backend)
- `agent/serving/triton/docker/Dockerfile.triton` - Multi-stage Triton image (dev/prod modes)
- `agent/serving/triton/docker/nginx.conf` - Nginx routing config
- `docker-compose.yml` - Service orchestration with GPU support

### Client & Documentation
- `agent/client/triton_client.py` - Python client for Triton
- `TRITON_GUIDE.md` - Comprehensive documentation
- `TRITON_INTEGRATION_SUMMARY.md` - Integration summary
- `agent/serving/triton/docker/requirements-triton.txt` - Dependencies

## Quick Commands

### Start Everything (Production Mode)
```bash
cd /root/workspace/lnd/aiops/apps/newbie-app
docker compose up --build
```

### Development Mode (Fast Iteration)
```bash
# Copy site-packages from host instead of pip install
BUILD_MODE=dev docker compose up --build triton-server
```

### Production Mode (Clean Build)
```bash
BUILD_MODE=prod docker compose up --build triton-server
```

### Test Services
```bash
# Triton health check
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/health/live

# Triton model metadata
curl http://localhost:8000/v2/models/qwen3-vl

# FastAPI health (legacy)
curl http://localhost:8000/health

# Test Triton Python client
python agent/client/triton_client.py
```

### View Logs
```bash
docker compose logs -f
docker compose logs triton-server  # Just Triton
docker compose logs fastapi-server  # Just FastAPI
```

### Stop Services
```bash
docker compose down
```

### Check Health Status
```bash
docker compose ps
docker inspect dsa-agent-triton --format='{{.State.Health.Status}}'
```

## Service Ports

| Service | Type | Port | URL |
|---------|------|------|-----|
| Triton | HTTP REST | 8000 | http://localhost:8000 |
| Triton | gRPC | 8001 | localhost:8001 |
| Triton | Metrics | 8002 | http://localhost:8002 |
| FastAPI (legacy) | HTTP REST | 8000 | http://localhost:8000 |
| Nginx | HTTP | 80 | http://localhost |

## Build Modes

### Development Mode (`BUILD_MODE=dev`)
- **Purpose**: Fast iteration during development
- **Method**: Copies site-packages from host via BuildKit `--from=site-packages`
- **Speed**: Very fast (no pip install, just file copy)
- **Use When**: Testing new packages, debugging, rapid prototyping
- **Caveat**: Python version mismatch (host 3.12 → container 3.10) may cause issues

### Production Mode (`BUILD_MODE=prod`)
- **Purpose**: Clean, reproducible builds
- **Method**: Installs all packages via pip during build
- **Speed**: Slower (downloads and compiles packages)
- **Use When**: Deployments, CI/CD, sharing images
- **Benefit**: No host dependencies, guaranteed compatibility

## Key Features

✅ **Triton Integration Complete:**
- Multi-stage Docker builds (dev/prod separation)
- Python backend with custom model loading
- Dynamic batching support (max_batch_size: 4)
- GPU acceleration with NVIDIA Container Toolkit
- gRPC & HTTP support
- Health checks (`/v2/health/ready`, `/v2/health/live`)
- Metrics endpoint
- BuildKit additional_contexts for site-packages

✅ **GPU Support:**
- `gpus: all` in docker-compose
- NVIDIA runtime environment variables
- Automatic device passthrough
- GPU instance_group in model config

✅ **Backward Compatible:**
- Original FastAPI server still works
- Existing clients unaffected
- Can run both services simultaneously

✅ **Production Ready:**
- Docker multi-stage builds
- Service orchestration with depends_on
- Health checks with retry logic
- Volume mounts for models and configs
- Environment-based configuration

## Next Steps (Optional)

1. **Benchmark Performance**: Compare FastAPI vs Triton inference latency
2. **Add More Models**: Create additional model configs (ONNX, TensorRT-LLM, vLLM)
3. **Scale with Replicas**: Use `docker compose up --scale triton-server=3`
4. **Add Monitoring**: Integrate Prometheus/Grafana for metrics scraping
5. **Optimize Model**: Try vLLM or TensorRT-LLM backends for faster inference
6. **Deploy to Cloud**: Push images to registry and deploy to K8s/ECS

See `TRITON_GUIDE.md` for:
- Detailed configuration options
- Performance tuning
- Troubleshooting
- Advanced usage examples
