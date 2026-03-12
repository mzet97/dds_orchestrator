#!/bin/bash
# Start all services and run benchmarks

set -e

echo "=========================================="
echo "Starting Infrastructure for Benchmarks"
echo "=========================================="

# Variables
REPO_ROOT="/mnt/e/TI/git/tese"
LLAMA_SERVER="$REPO_ROOT/llama.cpp_dds/build/bin/llama-server"
MODEL="$REPO_ROOT/models/phi4-mini-q3_k_m.gguf"
ORCHESTRATOR_DIR="$REPO_ROOT/dds_orchestrator"
LOG_DIR="$REPO_ROOT/logs"
BENCHMARK_DIR="$REPO_ROOT/dds_orchestrator/benchmark_results"

mkdir -p "$LOG_DIR" "$BENCHMARK_DIR"

# Export DDS config
export CYCLONEDDS_URI="file://$REPO_ROOT/cyclonedds/cyclonedds-local.xml"
export LD_LIBRARY_PATH="$REPO_ROOT/llama.cpp_dds/build:$LD_LIBRARY_PATH"

# Start llama-server
echo "[1/3] Starting llama-server..."
$LLAMA_SERVER \
    -m "$MODEL" \
    -c 2048 \
    -np 1 \
    -ngl 32 \
    --port 8082 \
    2>&1 | tee "$LOG_DIR/llama_server_$$_$(date +%s).log" &

LLAMA_PID=$!
echo "llama-server PID: $LLAMA_PID"

# Wait for llama-server to start
echo "Waiting for llama-server..."
sleep 5

# Check if llama-server is running
if ! kill -0 $LLAMA_PID 2>/dev/null; then
    echo "ERROR: llama-server failed to start"
    exit 1
fi

# Start orchestrator
echo "[2/3] Starting orchestrator..."
cd "$ORCHESTRATOR_DIR"
python3 main.py --port 8080 --dds-domain 0 2>&1 | tee "$LOG_DIR/orchestrator_$$_$(date +%s).log" &

ORCH_PID=$!
echo "orchestrator PID: $ORCH_PID"

sleep 5

# Check if orchestrator is running
if ! kill -0 $ORCH_PID 2>/dev/null; then
    echo "ERROR: orchestrator failed to start"
    kill $LLAMA_PID 2>/dev/null || true
    exit 1
fi

echo "=========================================="
echo "All services started successfully!"
echo "=========================================="
echo "llama-server: PID $LLAMA_PID (port 8082)"
echo "orchestrator: PID $ORCH_PID (port 8080)"
echo ""
echo "To stop services:"
echo "  kill $LLAMA_PID"
echo "  kill $ORCH_PID"
echo "=========================================="

# Keep script running
wait
