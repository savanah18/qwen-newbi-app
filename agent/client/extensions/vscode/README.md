# DSA Agent - Data Structures & Algorithms

Agentic assistant for mastering data structures, algorithms, and competitive programming. Powered by Triton Inference Server with Qwen3-VL model for high-performance AI inference.

## Current Features

- **Real-time agentic chat** with AI trained on DSA patterns and techniques
- **Triton Inference Server** integration for optimal inference performance
- **Algorithm explanations** with time/space complexity analysis
- **Data structure guidance** with implementation examples
- **Code analysis** and optimization suggestions
- **Interview prep** mode for technical problem-solving
- **Response timing** metrics for performance tracking
- **Chat persistence** with clear/reset functionality
- **Server health monitoring** and model status display

## Infrastructure

- **Backend**: Triton Inference Server (HTTP endpoint: `http://localhost:8000`)
- **Model**: Qwen3-VL-8B-Instruct with int4 quantization
- **Protocol**: Triton HTTP/REST API for inference requests

## Upcoming Features 🚀

- **Code generation** for algorithm implementations
- **Problem solving agent** that generates solutions step-by-step
- **Test case generation** and validation
- **Complexity calculator** with visual analysis
- **LeetCode/HackerRank integration** for problem pulling
- **Interview simulator** with mock questions
- **Visual algorithm walkthroughs** with animation
- **RAG integration** for enhanced context retrieval

## Commands

- `DSA Agent: Start` (command id: `dsaAgent.helloWorld`)
- `DSA Agent: Open Assistant` (command id: `dsaAgent.openChat`) — opens the agentic chat panel

## Getting started

### Prerequisites
1. **Start Triton Server**:
   ```bash
   cd /root/workspace/lnd/aiops/apps/newbie-app
   docker compose up triton-server
   ```
   
2. **Verify Triton is ready**:
   ```bash
   curl http://localhost:8000/v2/health/ready
   curl http://localhost:8000/v2/models/qwen3-vl/ready
   ```

### Extension Setup
1. Install dependencies:
   ```bash
   npm install
   ```
2. Build once (or use watch):
   ```bash
   npm run compile
   # or npm run watch
   ```
3. Launch the Extension Development Host:
   - Press `F5` in VS Code, or run the **Run Extension** launch config.
   
4. Open the DSA Agent chat panel and verify Triton connection

## Packaging

- Install `vsce` if you want to package a `.vsix`:
  ```bash
  npm install -g @vscode/vsce
  npm run package
  ```
