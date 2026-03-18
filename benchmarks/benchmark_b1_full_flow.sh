#!/bin/bash
# B1: Simple Prompt Benchmark - Fluxo Completo 100% DDS
# Cliente -> [DDS] -> Orchestrator -> [DDS] -> Agent -> [DDS] -> llama.cpp_dds

VM_HOST="192.168.1.60"
VM_USER="oldds"
VM_PASSWORD="Admin@123"
ORCH_PORT=8080
NUM_RUNS=${1:-5}

echo "=========================================="
echo "B1: Simple Prompt - Fluxo 100% DDS"
echo "=========================================="

# Função para executar benchmark via SSH
run_ssh() {
    python -c "
import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('$VM_HOST', username='$VM_USER', password='$VM_PASSWORD')
stdin, stdout, stderr = client.exec_command(\"$1\", timeout=120)
print(stdout.read().decode('utf-8', errors='ignore'))
client.close()
"
}

# Parar todos os processos
echo "[Setup] Parando processos existentes..."
run_ssh "pkill -9 -f llama-server || true"
run_ssh "pkill -9 -f agent_llm || true"
run_ssh "pkill -9 -f 'python.*main.py' || true"
sleep 3

# Iniciar orquestrador com DDS
echo "[Setup] Iniciando orquestrador..."
run_ssh "
cd /home/oldds/dds_orchestrator
export PYTHONPATH=/home/oldds/dds_orchestrator
export LD_LIBRARY_PATH=/home/oldds/llama.cpp_dds/build/bin:/home/oldds/cyclonedds/build-install/lib:/opt/rocm/lib
export ROCM_PATH=/opt/rocm
export CYCLONEDDS_URI=file:///home/oldds/llama.cpp_dds/dds/cyclonedds-local.xml
nohup python3 main.py --port 8080 --dds-domain 0 > /tmp/orch.log 2>&1 &
sleep 10
"

# Verificar saúde
HEALTH=$(run_ssh "curl -s http://localhost:$ORCH_PORT/health")
echo "Health: $HEALTH"

# Iniciar agente com GPU
echo "[Setup] Iniciando agente com GPU..."
run_ssh "
cd /home/oldds/dds_agent/python
export PYTHONPATH=/home/oldds/dds_agent/python:/home/oldds/dds_orchestrator
export LD_LIBRARY_PATH=/home/oldds/llama.cpp_dds/build/bin:/home/oldds/cyclonedds/build-install/lib:/opt/rocm/lib
export ROCM_PATH=/opt/rocm
export CYCLONEDDS_URI=file:///home/oldds/llama.cpp_dds/dds/cyclonedds-local.xml
nohup python3 agent_llm_dds.py --orchestrator-url http://localhost:$ORCH_PORT --model-path /home/oldds/models/phi4-mini-q3_k_m.gguf --model-name phi4-mini > /tmp/agent.log 2>&1 &
sleep 45
"

# Verificar agentes
AGENTS=$(run_ssh "curl -s http://localhost:$ORCH_PORT/api/v1/agents")
echo "Agentes: $AGENTS"

# Warmup via DDS
echo ""
echo "[Warmup]"
for i in 1 2 3; do
    run_ssh "curl -s -X POST http://127.0.0.1:$ORCH_PORT/chat -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}'" > /dev/null 2>&1
done
echo "Warmup done"

# Teste simples via DDS
echo ""
echo "[Test] Simple prompt - $NUM_RUNS runs (100% DDS)"
for i in $(seq 1 $NUM_RUNS); do
    RESULT=$(run_ssh "curl -s -w '%{time_total}' -X POST http://127.0.0.1:$ORCH_PORT/chat -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":10}'")
    LATENCY=$(echo "$RESULT" | tail -1)
    echo "Run $i: ${LATENCY}ms"
done

echo ""
echo "B1 Complete!"
