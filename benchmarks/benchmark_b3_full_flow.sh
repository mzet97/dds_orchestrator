#!/bin/bash
# B3: Network Delay Benchmark - Fluxo Completo
# Testa impacto de latência de rede usando tc netem

echo "=========================================="
echo "B3: Network Delay Benchmark - Fluxo Completo"
echo "=========================================="

ORCHESTRATOR="http://127.0.0.1:8080"

# Função para adicionar delay de rede
add_delay() {
    local delay_ms=$1
    local jitter_ms=$2

    # Adicionar delay no loopback
    tc qdisc del dev lo root 2>/dev/null || true
    tc qdisc add dev lo root netem delay ${delay_ms}ms ${jitter_ms}ms 2>/dev/null

    echo "Added delay: ${delay_ms}ms +/- ${jitter_ms}ms"
}

# Função para remover delay
remove_delay() {
    tc qdisc del dev lo root 2>/dev/null || true
    echo "Removed network delay"
}

# Delay values to test
DELAYS="0 5 10 25 50 100"

echo "Testing with different network delays..."

for delay in $DELAYS; do
    if [ $delay -eq 0 ]; then
        remove_delay
        label="baseline"
    else
        add_delay $delay 5
        label="${delay}ms"
    fi

    echo ""
    echo "=== Delay: $label ==="

    # Warmup
    curl -s -X POST "$ORCHESTRATOR/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
        > /dev/null 2>&1

    # Test
    for i in 1 2 3 4 5; do
        start=$(date +%s%3N)
        curl -s -X POST "$ORCHESTRATOR/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"messages":[{"role":"user","content":"What is AI?"}],"max_tokens":10}' \
            > /dev/null 2>&1
        end=$(date +%s%3N)
        latency=$((end - start))
        echo "Run $i: ${latency}ms"
    done
done

# Cleanup
remove_delay

echo ""
echo "B3 Complete!"
