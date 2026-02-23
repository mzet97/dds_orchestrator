#!/bin/bash
# B1: Multi-Client Benchmark - Fluxo Completo
# Cliente -> Orchestrator (8080) -> Agent -> llama.cpp_dds

echo "=========================================="
echo "B1: Multi-Client Benchmark - Fluxo Completo"
echo "=========================================="

ORCH_URL="http://127.0.0.1:8080/v1/chat/completions"
NUM_RUNS=${1:-5}

echo "URL: $ORCH_URL"
echo "Runs: $NUM_RUNS"

# Warmup
echo ""
echo "[Warmup]"
for i in 1 2 3; do
    curl -s -X POST "$ORCH_URL" \
        -H "Content-Type: application/json" \
        -d '{"model":"tinyllama-1.1b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
        > /dev/null 2>&1
done
echo "Warmup done"

# Test simple prompt
echo ""
echo "[Test] Simple prompt ($NUM_RUNS runs)"
for i in $(seq 1 $NUM_RUNS); do
    start=$(date +%s%3N)
    curl -s -X POST "$ORCH_URL" \
        -H "Content-Type: application/json" \
        -d '{"model":"tinyllama-1.1b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":10}' \
        > /dev/null 2>&1
    end=$(date +%s%3N)
    latency=$((end - start))
    echo "Run $i: ${latency}ms"
done

# Test medium prompt
echo ""
echo "[Test] Medium prompt ($NUM_RUNS runs)"
for i in $(seq 1 $NUM_RUNS); do
    start=$(date +%s%3N)
    curl -s -X POST "$ORCH_URL" \
        -H "Content-Type: application/json" \
        -d '{"model":"tinyllama-1.1b","messages":[{"role":"user","content":"Explain machine learning in a few sentences."}],"max_tokens":20}' \
        > /dev/null 2>&1
    end=$(date +%s%3N)
    latency=$((end - start))
    echo "Run $i: ${latency}ms"
done

# Test complex prompt
echo ""
echo "[Test] Complex prompt ($NUM_RUNS runs)"
for i in $(seq 1 $NUM_RUNS); do
    start=$(date +%s%3N)
    curl -s -X POST "$ORCH_URL" \
        -H "Content-Type: application/json" \
        -d '{"model":"tinyllama-1.1b","messages":[{"role":"user","content":"Write a detailed technical explanation of how neural networks work, including backpropagation."}],"max_tokens":30}' \
        > /dev/null 2>&1
    end=$(date +%s%3N)
    latency=$((end - start))
    echo "Run $i: ${latency}ms"
done

echo ""
echo "B1 Complete!"
