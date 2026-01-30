# Triton AI Chat Assistant

Generic AI chat assistant powered by Triton Inference Server with Qwen3-VL-8B-Instruct model. Features high-performance inference with batch processing, embedding extraction, and multimodal support.

## Current Features

- **Real-time AI chat** powered by Qwen3-VL-8B-Instruct
- **Triton Inference Server** integration with HTTP/gRPC endpoints
- **Batch inference support** (up to 32 concurrent requests)
- **Dual-mode inference**: Text generation and embedding extraction
- **Multimodal support** (text + images) via base64 encoding
- **Response timing** metrics for performance tracking (model + total time)
- **Chat persistence** with clear/reset functionality
- **Server health monitoring** with model status display
- **Easy configuration** via environment variables
- **Production-ready** with comprehensive error handling

## Infrastructure

- **Backend**: Triton Inference Server (HTTP: `8000`, gRPC: `8001`, Metrics: `8002`)
- **Model**: Qwen3-VL-8B-Instruct with int4 quantization (BitsAndBytes)
- **Protocol**: Triton HTTP/REST API (gRPC available)
- **Batch Processing**: Dynamic batching with preferred sizes [16, 32]
- **Max Batch Size**: 32 concurrent requests
- **Inference Modes**:
  - `generate`: Text generation (default)
  - `embed`: 3584-dim embedding extraction with L2 normalization

## API Updates (v0.2.0)

### Triton Request Format
```json
{
  "inputs": [
    {
      "name": "message",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["Your question here"]
    },
    {
      "name": "mode",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["generate"]  // or "embed"
    }
  ],
  "outputs": [
    { "name": "response" },
    { "name": "response_time" },
    { "name": "embedding" }  // only for embed mode
  ]
}
```

### Response Format
```json
{
  "outputs": [
    {
      "name": "response",
      "datatype": "BYTES",
      "shape": [1, 1],
      "data": ["Generated text response"]
    },
    {
      "name": "response_time",
      "datatype": "FP32",
      "shape": [1, 1],
      "data": [2.45]
    },
    {
      "name": "embedding",
      "datatype": "FP32",
      "shape": [1, 3584],  // or [1, 0] for generate mode
      "data": [...]
    }
  ]
}
```

## Upcoming Features 🚀

- **Context window management** with sliding window for long conversations
- **RAG integration** for knowledge base retrieval
- **Function calling** support for tool use
- **Custom model switching** for different inference tasks
- **Performance profiling** and optimization dashboard
- **Multi-turn conversations** with state management
- **Prompt templates** library for common tasks
- **Export capabilities** for chat history and embeddings

## Commands

- `Triton AI: Start` (command id: `tritonAI.start`) — Initialize the Triton AI assistant
- `Triton AI: Open Chat Assistant` (command id: `tritonAI.openChat`) — Opens the chat panel

## Getting Started

### Prerequisites
1. **Triton Inference Server** running with qwen3-vl model
   ```bash
   cd /root/workspace/lnd/aiops/apps/newbie-app
   docker compose up triton-server
   ```
   
2. **Verify Triton is ready**:
   ```bash
   curl http://localhost:8000/v2/health/ready
   curl http://localhost:8000/v2/models/qwen3-vl/ready
   ```

3. **Server Configuration** (defaults to localhost:8000):
   - HTTP endpoint: `http://localhost:8000`
   - gRPC endpoint: `localhost:8001`
   - Metrics: `localhost:8002`

### Extension Setup
1. Navigate to extension directory:
   ```bash
   cd agent/client/extensions/vscode
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Build and launch:
   ```bash
   npm run compile
   # or for continuous build: npm run watch
   ```
   
4. Press `F5` in VS Code to launch the Extension Development Host

5. Use `Triton AI: Open Chat Assistant` command to open the chat panel

6. Verify Triton server connection - status will show in the header

## Packaging

- Install `vsce` if you want to package a `.vsix`:
  ```bash
  npm install -g @vscode/vsce
  npm run package
  ```
