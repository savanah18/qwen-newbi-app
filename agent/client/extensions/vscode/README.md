# DSA Agent - Data Structures & Algorithms

Agentic assistant for mastering data structures, algorithms, and competitive programming. Powered by AI, this extension helps you understand complex DSA concepts, solve problems efficiently, and prepare for technical interviews.

## Current Features

- **Real-time agentic chat** with AI trained on DSA patterns and techniques
- **Algorithm explanations** with time/space complexity analysis
- **Data structure guidance** with implementation examples
- **Code analysis** and optimization suggestions
- **Interview prep** mode for technical problem-solving
- **Response timing** metrics for performance tracking
- **Chat persistence** with clear/reset functionality
- **Server health monitoring** and model status display

## Upcoming Features 🚀

- **Code generation** for algorithm implementations
- **Problem solving agent** that generates solutions step-by-step
- **Test case generation** and validation
- **Complexity calculator** with visual analysis
- **LeetCode/HackerRank integration** for problem pulling
- **Interview simulator** with mock questions
- **Visual algorithm walkthroughs** with animation
- **Offline mode** for local model inference

## Commands

- `DSA Agent: Start` (command id: `dsaAgent.helloWorld`)
- `DSA Agent: Open Assistant` (command id: `dsaAgent.openChat`) — opens the agentic chat panel

## Getting started

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

## Packaging

- Install `vsce` if you want to package a `.vsix`:
  ```bash
  npm install -g @vscode/vsce
  npm run package
  ```
