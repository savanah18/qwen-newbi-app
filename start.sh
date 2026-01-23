#!/bin/bash
# Startup script for Qwen3-VL Chat application
# Starts both the model server and Gradio frontend with visible logs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="aiops-py312"
LOG_DIR="$SCRIPT_DIR/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

echo "================================"
echo "Qwen3-VL Chat Application"
echo "================================"
echo ""

# Check if environment exists
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "❌ Error: Conda environment '$ENV_NAME' not found"
    exit 1
fi

# Cleanup existing processes
echo "Cleaning up existing processes..."
pkill -f "python.*model_server.py" 2>/dev/null || true
pkill -f "python.*app.py" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true

# Kill any processes on ports 8000 and 7860
echo "Freeing ports 8000 and 7860..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:7860 | xargs kill -9 2>/dev/null || true

sleep 2

echo "Starting services..."
echo ""
echo "📝 Logs will be saved to: $LOG_DIR"
echo ""

# Start model server in background with log file
echo "Starting Model Server on http://localhost:8000..."
(PYTHONUNBUFFERED=1 conda run -n "$ENV_NAME" python -u "$SCRIPT_DIR/model_server.py" 2>&1 | tee -a "$LOG_DIR/model_server.log") &
MODEL_SERVER_PID=$!
echo "   PID: $MODEL_SERVER_PID"

# Wait for model server to start and show initial logs
sleep 3
echo ""
echo "--- Model Server Logs (last 10 lines) ---"
tail -10 "$LOG_DIR/model_server.log" 2>/dev/null || echo "(No logs yet)"
echo ""

# Start Gradio frontend in background with log file
echo "Starting Gradio Frontend on http://localhost:7860..."
(PYTHONUNBUFFERED=1 conda run -n "$ENV_NAME" python -u "$SCRIPT_DIR/app.py" 2>&1 | tee -a "$LOG_DIR/frontend.log") &
FRONTEND_PID=$!
echo "   PID: $FRONTEND_PID"

sleep 2
echo ""
echo "--- Frontend Logs (last 10 lines) ---"
tail -10 "$LOG_DIR/frontend.log" || true
echo ""

echo "================================"
echo "✓ Services started successfully"
echo "================================"
echo ""
echo "Frontend:  http://localhost:7860"
echo "API:       http://localhost:8000"
echo "API Docs:  http://localhost:8000/docs"
echo ""
echo "Log files:"
echo "  • Model Server: $LOG_DIR/model_server.log"
echo "  • Frontend:     $LOG_DIR/frontend.log"
echo ""
echo "View logs in real-time:"
echo "  • tail -f $LOG_DIR/model_server.log"
echo "  • tail -f $LOG_DIR/frontend.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Trap Ctrl+C to stop both processes
trap 'echo ""; echo "Stopping services..."; kill $MODEL_SERVER_PID $FRONTEND_PID 2>/dev/null; echo "Services stopped."; exit 0' INT TERM

# Keep the script running and periodically check if services are still running
while kill -0 $MODEL_SERVER_PID 2>/dev/null && kill -0 $FRONTEND_PID 2>/dev/null; do
    sleep 1
done

echo "One or more services stopped unexpectedly"
exit 1
