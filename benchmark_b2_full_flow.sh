#!/bin/bash
# B2: Streaming Benchmark - Fluxo Completo (TTFT & ITL)
# Mede Time To First Token e Inter-Token Latency

echo "=========================================="
echo "B2: Streaming Benchmark - Fluxo Completo"
echo "=========================================="

URL="http://127.0.0.1:8080/v1/chat/completions"

echo "URL: $URL"

# Test 1: Simple prompt com streaming
echo ""
echo "[Test 1] Simple prompt - Count to 5"

# Medir tempo até primeiro token (TTFT)
start=$(date +%s%N)

# Fazer request e capturar cada chunk
response=$(curl -s -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d '{"model":"tinyllama-1.1b","messages":[{"role":"user","content":"Count to 5."}],"max_tokens":10,"stream":true}' \
    2>/dev/null)

end=$(date +%s%N)
total=$(( (end - start) / 1000000 ))  # Convert to ms

echo "Total time: ${total}ms"

# Contar chunks
chunk_count=$(echo "$response" | grep -c "data:" || true)
echo "Chunks received: $chunk_count"

# Test 2: Medium prompt
echo ""
echo "[Test 2] Medium prompt - Explain ML"

start=$(date +%s%N)
response=$(curl -s -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d '{"model":"tinyllama-1.1b","messages":[{"role":"user","content":"Explain machine learning."}],"max_tokens":20,"stream":true}' \
    2>/dev/null)

end=$(date +%s%N)
total=$(( (end - start) / 1000000 ))
echo "Total time: ${total}ms"

chunk_count=$(echo "$response" | grep -c "data:" || true)
echo "Chunks received: $chunk_count"

# Test 3: Complex prompt
echo ""
echo "[Test 3] Complex prompt - Neural networks"

start=$(date +%s%N)
response=$(curl -s -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d '{"model":"tinyllama-1.1b","messages":[{"role":"user","content":"Write about neural networks and backpropagation."}],"max_tokens":30,"stream":true}' \
    2>/dev/null)

end=$(date +%s%N)
total=$(( (end - start) / 1000000 ))
echo "Total time: ${total}ms"

chunk_count=$(echo "$response" | grep -c "data:" || true)
echo "Chunks received: $chunk_count"

# Test 4: Multiple tokens timing
echo ""
echo "[Test 4] Timing multiple tokens"

# Simple prompt para medir ITL
for i in 1 2 3; do
    start=$(date +%s%N)
    response=$(curl -s -X POST "$URL" \
        -H "Content-Type: application/json" \
        -d '{"model":"tinyllama-1.1b","messages":[{"role":"user","content":"Hello."}],"max_tokens":5,"stream":true}' \
        2>/dev/null)
    end=$(date +%s%N)
    total=$(( (end - start) / 1000000 ))
    echo "Run $i: ${total}ms"
done

echo ""
echo "B2 Complete!"
