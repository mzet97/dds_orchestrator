#!/bin/bash
# ============================================================================
# GPU Benchmark Execution Script
# Starts services and runs multi-client benchmark tests
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Paths
LLAMA_CPP_DIR="/mnt/e/TI/git/tese/llama.cpp_dds"
ORCHESTRATOR_DIR="/mnt/e/TI/git/tese/dds_orchestrator"
AGENT_DIR="/mnt/e/TI/git/tese/dds_agent"

# Model
MODEL="$LLAMA_CPP_DIR/models/tinyllama.gguf"

# Server ports
LLAMA_PORT=8082
ORCH_PORT=8080

# GPU layers (0 for CPU, 99 for GPU)
GPU_LAYERS=32

# Results
RESULTS_DIR="$SCRIPT_DIR/benchmark_results/gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# ============================================================================
# Cleanup
# ============================================================================

cleanup() {
    echo ""
    echo "[Cleanup] Stopping services..."
    pkill -f "llama-server.*$LLAMA_PORT" 2>/dev/null || true
    pkill -f "python.*main.py.*$ORCH_PORT" 2>/dev/null || true
    pkill -f "agent_llm_dds.py" 2>/dev/null || true
    sleep 2
}

trap cleanup EXIT

# ============================================================================
# Start llama-server with DDS
# ============================================================================

echo "Starting llama-server with DDS on port $LLAMA_PORT (GPU layers: $GPU_LAYERS)..."

cd "$LLAMA_CPP_DIR"

export CYCLONEDDS_URI="file://$PWD/dds/cyclonedds-local.xml"
export LD_LIBRARY_PATH="$PWD/build/bin:$LD_LIBRARY_PATH"

# Start server in background
./build/bin/llama-server \
    --enable-dds \
    --model "$MODEL" \
    --port $LLAMA_PORT \
    --ctx-size 512 \
    --parallel 8 \
    -ngl $GPU_LAYERS \
    > "$RESULTS_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo "llama-server started (PID: $SERVER_PID)"

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 30); do
    if curl --silent --fail "http://127.0.0.1:$LLAMA_PORT/health" > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 1
done

# ============================================================================
# Start orchestrator
# ============================================================================

echo "Starting orchestrator on port $ORCH_PORT..."

cd "$ORCHESTRATOR_DIR"

export PYTHONPATH="$ORCHESTRATOR_DIR:$AGENT_DIR:$PYTHONPATH"
export CYCLONEDDS_URI="file://$LLAMA_CPP_DIR/dds/cyclonedds-local.xml"

# Start orchestrator in background
python3 main.py \
    --port $ORCH_PORT \
    --dds-domain 0 \
    > "$RESULTS_DIR/orchestrator.log" 2>&1 &

ORCH_PID=$!
echo "Orchestrator started (PID: $ORCH_PID)"

# Wait for orchestrator
for i in $(seq 1 20); do
    if curl --silent --fail "http://127.0.0.1:$ORCH_PORT/health" > /dev/null 2>&1; then
        echo "Orchestrator ready!"
        break
    fi
    sleep 1
done

# ============================================================================
# Start agent
# ============================================================================

echo "Starting agent..."

cd "$AGENT_DIR/python"

export PYTHONPATH="$AGENT_DIR/python:$ORCHESTRATOR_DIR:$PYTHONPATH"
export CYCLONEDDS_URI="file://$LLAMA_CPP_DIR/dds/cyclonedds-local.xml"
export LLAMA_SERVER_PATH="$LLAMA_CPP_DIR/build/bin/llama-server"

# Start agent in background
python3 agent_llm_dds.py \
    --orchestrator-url "http://localhost:$ORCH_PORT" \
    --model-path "$MODEL" \
    --model-name "tinyllama" \
    --llama-server-port $LLAMA_PORT \
    > "$RESULTS_DIR/agent.log" 2>&1 &

AGENT_PID=$!
echo "Agent started (PID: $AGENT_PID)"

# Wait for agent to register
echo "Waiting for agent registration..."
sleep 10

# Check agents
AGENTS=$(curl -s "http://127.0.0.1:$ORCH_PORT/api/v1/agents" || echo "[]")
echo "Registered agents: $AGENTS"

# ============================================================================
# Run benchmarks
# ============================================================================

echo ""
echo "Running benchmark with 5 clients..."

cd "$ORCHESTRATOR_DIR"

python3 benchmark_orchestrator_dds.py \
    --mode sync \
    --clients 5 \
    --requests 65 \
    --warmup 3 \
    --domain 0 \
    --timeout 120 \
    --output "$RESULTS_DIR" \
    --verbose

echo ""
echo "Results saved to: $RESULTS_DIR"
