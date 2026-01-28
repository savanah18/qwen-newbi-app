# Triton Integration Guide

## Overview

This project now includes **Triton Inference Server** integration for high-performance model serving alongside the original FastAPI server. Both services can run simultaneously, providing flexibility and scalability.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Clients                        │
│  (VS Code Extension, Web UI, REST Clients)      │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
    ┌─────▼─────┐           ┌─────▼─────┐
    │ FastAPI   │           │   Triton  │
    │ (Port     │           │   (Port   │
    │  8000)    │           │   8001)   │
    └─────┬─────┘           └─────┬─────┘
          │                       │
          └───────────┬───────────┘
                      │
            ┌─────────▼──────────┐
            │  Qwen3-VL Model    │
            │  (GPU Accelerated)  │
            └────────────────────┘
```

## Directory Structure

```
newbie-app/
├── docker-compose.yml               # Orchestration (FastAPI + Triton)
├── Dockerfile                       # FastAPI image
├── requirements.txt
│
├── agent/
│   ├── serving/                     # Original FastAPI server + Triton
│   │   ├── model_server.py
│   │   ├── inference_engine.py
│   │   ├── model_loader.py
│   │   ├── triton-models/           # NEW: Triton model repository
│   │   │   └── qwen3-vl/
│   │   │       ├── config.pbtxt
│   │   │       └── 1/
│   │   │           └── model.py
│   │   │
│   │   └── triton-docker/           # NEW: Triton Docker files
│   │       ├── Dockerfile.triton
│   │       ├── nginx.conf
│   │       ├── requirements-triton.txt
│   │       └── start-services.sh
│   └── client/
│       ├── triton_client.py         # NEW: Triton client library
│       └── web/
│
├── docs/
```

## Quick Start

### 1. Build and Start Services

```bash
cd /root/workspace/lnd/aiops/apps/newbie-app

# Make startup script executable
chmod +x triton-docker/start-services.sh

# Build and start
./triton-docker/start-services.sh
```

Or manually with docker-compose:
```bash
docker-compose build
docker-compose up -d
```

### 2. Verify Services

Check FastAPI health:
```bash
curl http://localhost:8000/health
```

Check Triton health:
```bash
curl http://localhost:8001/v2/health/live
```

### 3. Send Inference Request

**Via FastAPI (REST):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is a binary tree?",
    "image_base64": null
  }'
```

**Via Triton (HTTP):**
```bash
python agent/client/triton_client.py
```

## Usage Guide

### FastAPI Server (Original)

Still available at `http://localhost:8000`

**Endpoints:**
- `GET /health` - Server health check
- `POST /chat` - Send chat message with optional image
- `POST /load_model` - Load model with custom settings
- `POST /clear_history` - Clear conversation history
- `GET /history` - Get conversation history

**Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "What are algorithms?",
        "image_base64": None
    }
)
print(response.json())
```

### Triton Server (New)

Available at:
- **HTTP:** `http://localhost:8001`
- **gRPC:** `localhost:8002`

**Using Python Client:**
```python
from agent.client.triton_client import TritonHttpClient

client = TritonHttpClient("localhost:8001")

if client.check_health():
    response, response_time = client.chat("What is a data structure?")
    print(f"Response: {response}")
    print(f"Time: {response_time:.2f}s")
```

**Using cURL:**
```bash
# Get model metadata
curl http://localhost:8001/v2/models/qwen3-vl

# Check health
curl http://localhost:8001/v2/health/live
```

## Configuration

### Environment Variables

Both FastAPI and Triton use the same environment variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/root/workspace/lnd/aiops/vlm/Qwen/Qwen3-VL-8B-Instruct` | Model checkpoint path |
| `QUANTIZATION_TYPE` | `int4` | Quantization: none, int4, int8, awq, gptq |
| `ATTENTION_IMPL` | `sdpa` | Attention: eager, sdpa, flash_attention_2 |
| `MAX_HISTORY` | `5` | Conversation history size |
| `MAX_NEW_TOKENS` | `512` | Max tokens to generate |
| `TEMPERATURE` | `0.7` | Generation temperature |
| `TOP_P` | `0.9` | Top-p sampling parameter |

### Update Configuration

Edit `docker-compose.yml`:
```yaml
environment:
  - QUANTIZATION_TYPE=int8
  - MAX_NEW_TOKENS=256
```

Then restart:
```bash
docker-compose down
docker-compose up -d
```

## Performance Comparison

| Feature | FastAPI | Triton |
|---------|---------|--------|
| Latency | ~2-5s | ~2-5s |
| Throughput | Lower | Higher (batching) |
| Dynamic batching | ❌ | ✅ |
| Model state mgmt | Per-request | Per-instance |
| Easy debugging | ✅ | ⚠️ |
| Production-ready | ✅ | ✅ |

## Advanced Usage

### Batch Inference with Triton

```python
from agent.client.triton_client import TritonHttpClient
import concurrent.futures

client = TritonHttpClient("localhost:8001")

messages = [
    "What is a tree?",
    "What is a graph?",
    "What is a hash table?"
]

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(client.chat, msg) for msg in messages]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

for msg, (response, time) in zip(messages, results):
    print(f"Q: {msg}")
    print(f"A: {response}\n")
```

### Using gRPC for Lower Latency

```python
from agent.client.triton_client import TritonGrpcClient

client = TritonGrpcClient("localhost:8002")  # gRPC port
response, response_time = client.chat("Explain quicksort")
print(f"Response (gRPC): {response_time:.3f}s")
```

### Monitor Triton Metrics

```bash
# Prometheus metrics endpoint
curl http://localhost:8003/metrics

# Watch model load status
watch -n 1 'curl -s http://localhost:8001/v2/models/qwen3-vl | jq .'
```

## Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs -f

# Check specific service
docker-compose logs triton-server
docker-compose logs fastapi-server
```

### Out of Memory (OOM)

Reduce model size or adjust memory settings in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # Specific GPU
          capabilities: [gpu]
```

### Model not loading in Triton

1. Check model path exists and is readable
2. Verify `config.pbtxt` is properly formatted
3. Check Triton logs: `docker-compose logs triton-server`
4. Ensure model.py has no syntax errors:
   ```bash
   docker exec dsa-agent-triton python -m py_compile /models/qwen3-vl/1/model.py
   ```

### Slow inference

- **FastAPI**: Use vLLM backend
  ```bash
  LOADING_STRATEGY=vllm docker-compose up -d fastapi-server
  ```

- **Triton**: Enable dynamic batching (already configured in `config.pbtxt`)

## API Documentation

### FastAPI (REST)
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Triton
- Model metadata: `http://localhost:8001/v2/models/qwen3-vl`
- Server status: `http://localhost:8001/v2`

## Stopping Services

```bash
# Stop and remove containers
docker-compose down

# Keep volumes
docker-compose down -v

# View running containers
docker ps

# Stop specific service
docker-compose stop triton-server
```

## Next Steps

1. **Implement vLLM Backend for Triton** (optional, for higher throughput)
2. **Add ONNX Runtime backend** for faster inference on CPU
3. **Deploy to Kubernetes** using Helm charts
4. **Add monitoring** with Prometheus + Grafana
5. **Implement model versioning** in Triton

## References

- [Triton Inference Server Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/)
- [Python Backend Guide](https://github.com/triton-inference-server/python_backend)
- [Model Config Reference](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_configuration.html)
- [Triton Client Libraries](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client.html)
