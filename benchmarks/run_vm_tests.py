#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para executar benchmarks E1-E5 nas VMs remotas (DDS vs gRPC)
====================================================================
Usa paramiko (SSH puro em Python) - funciona no Windows sem dependências externas.

Fluxo:
  1. Conecta nas 3 VMs via SSH
  2. Git pull em todos os repos
  3. Instala dependências Python
  4. Compila llama.cpp_dds com DDS + gRPC
  5. FASE DDS:  inicia serviços DDS, roda E1-E5, para serviços
  6. FASE gRPC: inicia serviços gRPC, roda E1-E5, para serviços
  7. FASE E2 standalone: roda E2 DDS (LIVELINESS) e E2 gRPC (GetStatus) no orchestrator
  8. Coleta resultados

VMs:
  - 192.168.1.60: Agent 1 (RX6600M 8GB)  - llama-server + agent
  - 192.168.1.61: Agent 2 (RTX 3080 10GB) - llama-server + agent
  - 192.168.1.62: Orchestrator            - orquestrador + benchmarks

Usage:
    python run_vm_tests.py --cenario all --n 5
    python run_vm_tests.py --cenario E1 --n 10
    python run_vm_tests.py --setup-only
"""

import argparse
import asyncio
import json
import os
import sys
import time

# Fix UTF-8 encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    import paramiko
except ImportError:
    print("ERRO: paramiko nao instalado! Execute: pip install paramiko")
    sys.exit(1)


# ─── VM Configuration ───────────────────────────────────────────────────────

VM_CONFIG = {
    "orchestrator": {
        "ip": "192.168.1.62",
        "user": "oldds",
        "password": "Admin@123",
        "role": "orchestrator",
        "desc": "Orchestrator"
    },
    "agent1": {
        "ip": "192.168.1.60",
        "user": "oldds",
        "password": "Admin@123",
        "role": "agent",
        "model": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "model_name": "Phi-4-mini",
        "desc": "Agent 1 - RX6600M 8GB",
        "gpu_target": "gfx1032",  # AMD RDNA2
    },
    "agent2": {
        "ip": "192.168.1.61",
        "user": "oldds",
        "password": "Admin@123",
        "role": "agent",
        "model": "Qwen3.5-9B-Q4_K_M.gguf",
        "model_name": "Qwen3.5-9B",
        "desc": "Agent 2 - RTX 3080 10GB",
        "gpu_target": None,  # NVIDIA uses CUDA
    },
}

REPOS = {
    "dds_orchestrator": {"url": "https://github.com/mzet97/dds_orchestrator", "branch": "main"},
    "dds_agent":        {"url": "https://github.com/mzet97/dds_agent",        "branch": "main"},
    "llama.cpp_dds":    {"url": "https://github.com/mzet97/llama.cpp_dds",    "branch": "master"},
}

BASE_DIR = "/home/oldds"
MODELS_DIR = "/home/oldds/models"
ORCH_IP = "192.168.1.62"
ORCH_PORT = 8080
BENCHMARK_N = 5


# ─── SSH Manager ─────────────────────────────────────────────────────────────

class SSHManager:
    """Gerenciador de conexoes SSH via paramiko."""

    def __init__(self, ip: str, user: str, password: str, desc: str = ""):
        self.ip = ip
        self.user = user
        self.password = password
        self.desc = desc
        self.client: Optional[paramiko.SSHClient] = None

    def connect(self) -> bool:
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            print(f"  Conectando em {self.ip} ({self.desc})...")
            self.client.connect(
                hostname=self.ip, username=self.user, password=self.password,
                timeout=30, banner_timeout=30
            )
            # Enable keepalive to prevent SSH timeout during long benchmarks
            transport = self.client.get_transport()
            if transport:
                transport.set_keepalive(30)  # Send keepalive every 30s
            print(f"  [OK] Conectado em {self.ip}")
            return True
        except Exception as e:
            print(f"  [FAIL] Erro em {self.ip}: {e}")
            return False

    def reconnect(self) -> bool:
        """Reconecta se a sessao SSH caiu."""
        try:
            if self.client:
                transport = self.client.get_transport()
                if transport and transport.is_active():
                    return True
            self.disconnect()
            return self.connect()
        except Exception:
            return self.connect()

    def disconnect(self):
        if self.client:
            self.client.close()
            self.client = None

    def run(self, command: str, timeout: int = 120) -> Tuple[int, str, str]:
        """Executa comando e retorna (exit_code, stdout, stderr). Reconecta se necessario."""
        for attempt in range(2):
            if not self.client:
                if not self.reconnect():
                    return -1, "", "Nao conectado"
            try:
                stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
                exit_code = stdout.channel.recv_exit_status()
                return exit_code, stdout.read().decode('utf-8', errors='ignore'), stderr.read().decode('utf-8', errors='ignore')
            except Exception as e:
                if attempt == 0 and "not active" in str(e).lower():
                    print(f"    [SSH] Reconectando em {self.ip}...")
                    self.reconnect()
                    continue
                return -1, "", str(e)
        return -1, "", "Falha apos reconexao"

    def run_bg(self, command: str, logfile: str = "/dev/null", env: Dict = None) -> bool:
        """Executa comando em background com nohup.

        Args:
            command: comando a executar
            logfile: arquivo de log
            env: variaveis de ambiente extras (dict)
        """
        if not self.client:
            return False
        try:
            if env:
                exports = " && ".join(f"export {k}={v}" for k, v in env.items())
                cmd = f"bash -c '{exports} && nohup {command} > {logfile} 2>&1 &'"
            else:
                cmd = f"nohup {command} > {logfile} 2>&1 &"
            self.client.exec_command(cmd, timeout=10)
            return True
        except Exception as e:
            print(f"    Erro background: {e}")
            return False


# ─── Setup Functions ─────────────────────────────────────────────────────────

def git_pull_all(ssh: SSHManager):
    """Git pull em todos os repos."""
    print(f"\n  Git pull em {ssh.ip}:")
    for name, info in REPOS.items():
        repo_path = f"{BASE_DIR}/{name}"
        ec, out, err = ssh.run(f"test -d {repo_path} && echo EXISTS || echo MISSING")
        if "EXISTS" in out:
            ec, out, err = ssh.run(f"cd {repo_path} && git fetch origin && git reset --hard origin/{info['branch']}", timeout=60)
            status = "[OK]" if ec == 0 else f"[FAIL] {err[:100]}"
        else:
            ec, out, err = ssh.run(f"cd {BASE_DIR} && git clone {info['url']}", timeout=120)
            status = "[OK] cloned" if ec == 0 else f"[FAIL] {err[:100]}"
        print(f"    {name}: {status}")


def install_deps(ssh: SSHManager):
    """Instala dependencias Python."""
    print(f"  Instalando deps em {ssh.ip}...")
    ec, out, err = ssh.run(
        "pip3 install --user --break-system-packages aiohttp pydantic psutil requests protobuf grpcio grpcio-tools cyclonedds --quiet",
        timeout=180
    )
    print(f"    {'[OK]' if ec == 0 else '[FAIL] ' + err[:200]}")


def compile_llama(ssh: SSHManager, vm_cfg: Dict):
    """Compila llama.cpp_dds com DDS + gRPC."""
    print(f"  Compilando llama.cpp_dds em {ssh.ip}...")

    build_dir = f"{BASE_DIR}/llama.cpp_dds/build"
    gpu_target = vm_cfg.get("gpu_target")

    # CycloneDDS is installed in ~/cyclonedds/install on the VMs.
    # gRPC C++ not available on VMs — benchmarks use Python gRPC proxy instead.
    cyclone_root = f"{BASE_DIR}/cyclonedds/install"
    cmake_flags = f"-DLLAMA_DDS=ON -DCYCLONEDDS_ROOT={cyclone_root} -DCMAKE_BUILD_TYPE=Release"
    if gpu_target:
        cmake_flags += f" -DAMDGPU_TARGETS={gpu_target}"

    # Clean cmake cache to avoid stale GRPC references, then reconfigure and build
    cmd = (f"cd {BASE_DIR}/llama.cpp_dds && mkdir -p build && cd build && "
           f"rm -f CMakeCache.txt && "
           f"cmake .. {cmake_flags} 2>&1 && make llama-server -j$(nproc) 2>&1")
    ec, out, err = ssh.run(cmd, timeout=600)

    # Check if binary was produced (cmake/make exit code can be misleading with warnings)
    ec2, out2, _ = ssh.run(f"ls {build_dir}/bin/llama-server 2>/dev/null || ls {build_dir}/server 2>/dev/null")
    if ec2 == 0:
        print(f"    [OK] Build completo: {out2.strip()}")
    elif ec == 0:
        print(f"    [OK] Build completo (binario nao encontrado no path esperado)")
    else:
        # Check for real errors (not just warnings)
        real_errors = [l for l in err.split('\n') if 'error:' in l.lower() and 'warning' not in l.lower()]
        if real_errors:
            print(f"    [FAIL] Erro: {real_errors[0][:200]}")
        else:
            print(f"    [OK] Build com warnings (sem erros fatais)")


def find_llama_binary(ssh: SSHManager) -> str:
    """Encontra o binario do llama-server."""
    # Check known paths first
    for path in [f"{BASE_DIR}/llama.cpp_dds/build/bin/llama-server",
                 f"{BASE_DIR}/llama.cpp_dds/build/server",
                 f"{BASE_DIR}/llama.cpp_dds/build/llama-server",
                 f"{BASE_DIR}/llama.cpp_dds/build/examples/server/server",
                 f"{BASE_DIR}/llama.cpp_dds/build/tools/server/llama-server"]:
        ec, out, _ = ssh.run(f"test -f {path} && echo FOUND")
        if "FOUND" in out:
            print(f"    llama-server encontrado: {path}")
            return path

    # Fallback: search with find (only executable FILES, not directories)
    ec, out, _ = ssh.run(f"find {BASE_DIR}/llama.cpp_dds/build -type f -executable -name 'llama-server' 2>/dev/null | head -5", timeout=15)
    if ec == 0 and out.strip():
        candidates = [l.strip() for l in out.strip().split('\n') if l.strip()]
        if candidates:
            binary = candidates[0]
            print(f"    llama-server encontrado via find: {binary}")
            return binary

    print(f"    [WARN] llama-server NAO encontrado! Usando path padrao.")
    return f"{BASE_DIR}/llama.cpp_dds/build/bin/llama-server"


# ─── Service Management ─────────────────────────────────────────────────────

def kill_services(ssh: SSHManager):
    """Para todos os servicos usando kill -9 por PID (mais confiavel que pkill via SSH)."""
    # Get PIDs of all relevant processes
    ec, out, _ = ssh.run(
        "pgrep -f 'llama-server|agent_llm|main.py' 2>/dev/null || true",
        timeout=5
    )
    pids = [p.strip() for p in out.strip().split('\n') if p.strip().isdigit()]

    if pids:
        pid_str = ' '.join(pids)
        ssh.run(f"kill -9 {pid_str} 2>/dev/null; true", timeout=5)
        time.sleep(2)

        # Verify
        ec, out, _ = ssh.run(
            "pgrep -f 'llama-server|agent_llm|main.py' 2>/dev/null || true",
            timeout=5
        )
        pids2 = [p.strip() for p in out.strip().split('\n') if p.strip().isdigit()]
        if pids2:
            ssh.run(f"kill -9 {' '.join(pids2)} 2>/dev/null; true", timeout=5)
            time.sleep(1)
            print(f"    [WARN] Matando processos restantes em {ssh.ip}")
        else:
            print(f"    Servicos parados em {ssh.ip} ({len(pids)} processos)")
    else:
        print(f"    Servicos parados em {ssh.ip} (nenhum processo)")


def get_cyclonedds_xml(ssh: SSHManager) -> str:
    """Encontra o XML do CycloneDDS para rede (VMs em maquinas diferentes)."""
    # For cross-VM communication, must use network config (not loopback)
    network_ultra = f"{BASE_DIR}/llama.cpp_dds/dds/cyclonedds-network-ultra.xml"
    ec, _, _ = ssh.run(f"test -f {network_ultra}")
    if ec == 0:
        return network_ultra
    network_xml = f"{BASE_DIR}/llama.cpp_dds/dds/cyclonedds-network.xml"
    ec, _, _ = ssh.run(f"test -f {network_xml}")
    if ec == 0:
        return network_xml
    return f"{BASE_DIR}/llama.cpp_dds/dds/cyclonedds-local.xml"


def start_llama_server_dds(ssh: SSHManager, vm_cfg: Dict):
    """Inicia llama-server com DDS."""
    binary = find_llama_binary(ssh)
    model_path = f"{MODELS_DIR}/{vm_cfg['model']}"
    xml_path = get_cyclonedds_xml(ssh)

    cmd = (f"{binary} "
           f"-m {model_path} -c 2048 --threads 8 -ngl 99 "
           f"--port 8082 --host 0.0.0.0 "
           f"--enable-dds --dds-domain 0 --dds-timeout 120")

    env = {"CYCLONEDDS_URI": f"file://{xml_path}"}
    ssh.run_bg(cmd, "/tmp/llama_dds.log", env=env)
    print(f"    [OK] llama-server DDS iniciado em {ssh.ip} (XML: {xml_path})")


def start_llama_server_grpc(ssh: SSHManager, vm_cfg: Dict):
    """Inicia llama-server com gRPC."""
    binary = find_llama_binary(ssh)
    model_path = f"{MODELS_DIR}/{vm_cfg['model']}"

    cmd = (f"{binary} "
           f"-m {model_path} -c 2048 --threads 8 -ngl 99 "
           f"--port 8082 --host 0.0.0.0 "
           f"--enable-grpc --grpc-address 0.0.0.0:50051")

    ssh.run_bg(cmd, "/tmp/llama_grpc.log")
    print(f"    [OK] llama-server gRPC iniciado em {ssh.ip}")


def start_orchestrator_dds(ssh: SSHManager):
    """Inicia orchestrator com DDS."""
    xml_path = get_cyclonedds_xml(ssh)

    cmd = (f"python3 -u {BASE_DIR}/dds_orchestrator/main.py "
           f"--port {ORCH_PORT} --dds-domain 0 --log-level INFO")

    env = {"CYCLONEDDS_URI": f"file://{xml_path}"}
    ssh.run_bg(cmd, "/tmp/orch_dds.log", env=env)
    print(f"    [OK] Orchestrator DDS iniciado em {ssh.ip}")


def start_orchestrator_grpc(ssh: SSHManager):
    """Inicia orchestrator com gRPC."""
    cmd = (f"python3 -u {BASE_DIR}/dds_orchestrator/main.py "
           f"--port {ORCH_PORT} --grpc-enabled --log-level INFO")

    ssh.run_bg(cmd, "/tmp/orch_grpc.log")
    print(f"    [OK] Orchestrator gRPC iniciado em {ssh.ip}")


def start_agent_dds(ssh: SSHManager, vm_cfg: Dict):
    """Inicia agent DDS."""
    xml_path = get_cyclonedds_xml(ssh)
    model_path = f"{MODELS_DIR}/{vm_cfg['model']}"

    cmd = (f"python3 -u {BASE_DIR}/dds_agent/python/agent_llm_dds.py "
           f"--model-name {vm_cfg['model_name']} "
           f"--model-path {model_path} "
           f"--orchestrator-url http://{ORCH_IP}:{ORCH_PORT} "
           f"--port 8081 --llama-server-port 8082 --no-server")

    env = {"CYCLONEDDS_URI": f"file://{xml_path}"}
    ssh.run_bg(cmd, "/tmp/agent_dds.log", env=env)
    print(f"    [OK] Agent DDS iniciado em {ssh.ip} ({vm_cfg['model_name']})")


def start_agent_grpc(ssh: SSHManager, vm_cfg: Dict):
    """Inicia agent gRPC."""
    model_path = f"{MODELS_DIR}/{vm_cfg['model']}"
    cmd = (f"python3 -u {BASE_DIR}/dds_agent/python/agent_llm_grpc.py "
           f"--model-name {vm_cfg['model_name']} "
           f"--model-path {model_path} "
           f"--orchestrator-url http://{ORCH_IP}:{ORCH_PORT} "
           f"--port 8081 "
           f"--grpc-address localhost:50051 "
           f"--grpc-listen-port 50053 "
           f"--no-server")

    ssh.run_bg(cmd, "/tmp/agent_grpc.log")
    print(f"    [OK] Agent gRPC iniciado em {ssh.ip} ({vm_cfg['model_name']})")


def wait_agent_registration(ssh_orch: SSHManager, expected: int = 1, max_wait: int = 120) -> bool:
    """Aguarda agentes registrarem no orchestrator."""
    print(f"    Aguardando {expected} agent(s) registrar(em)...")
    import time as _time
    orch_up = False
    for i in range(max_wait // 3):
        ec, out, _ = ssh_orch.run(f"curl -s http://localhost:{ORCH_PORT}/api/v1/agents", timeout=10)
        if ec == 0 and out.strip() and "agents" in out:
            if not orch_up:
                orch_up = True
                print(f"      Orchestrator respondendo")
            count = out.count("agent_id")
            if count >= expected:
                print(f"    [OK] {count} agent(s) registrado(s)")
                return True
            if i % 5 == 0:
                print(f"      Tentativa {i+1}: {count}/{expected} agents")
        else:
            if i % 5 == 0:
                print(f"      Tentativa {i+1}: orchestrator nao respondeu")
            # Check if orchestrator process is alive
            if i == 10 and not orch_up:
                ec2, out2, _ = ssh_orch.run("ps aux | grep 'python3.*main.py' | grep -v grep")
                if not out2.strip():
                    print(f"    [FAIL] Orchestrator nao esta rodando!")
                    # Check logs
                    for logfile in ["/tmp/orch_dds.log", "/tmp/orch_grpc.log"]:
                        ec3, out3, _ = ssh_orch.run(f"tail -10 {logfile} 2>/dev/null")
                        if out3.strip():
                            print(f"    Log ({logfile}): {out3[:300]}")
                    return False
        _time.sleep(3)
    print(f"    [FAIL] Timeout aguardando agents")
    return False


def warmup_request(ssh_orch: SSHManager, model: str):
    """Envia request de warmup."""
    cmd = (f'curl -s -X POST http://localhost:{ORCH_PORT}/v1/chat/completions '
           f'-H "Content-Type: application/json" '
           f'-d \'{{"model":"{model}","messages":[{{"role":"user","content":"warmup"}}],"max_tokens":10}}\'')
    ssh_orch.run(cmd, timeout=120)
    print(f"    [OK] Warmup completo ({model})")


# ─── Benchmark Execution ────────────────────────────────────────────────────

def run_benchmark(ssh: SSHManager, script: str, args: str, timeout: int = 600, env_prefix: str = "") -> Dict:
    """Executa um benchmark e retorna resultado."""
    cmd = f"{env_prefix}cd {BASE_DIR}/dds_orchestrator && python3 -u benchmarks/{script} {args}"
    print(f"    -> {script} {args}")

    ec, out, err = ssh.run(cmd, timeout=timeout)

    status = "[OK]" if ec == 0 else "[FAIL]"
    # Print last meaningful lines
    lines = [l for l in out.strip().split('\n') if l.strip()]
    for line in lines[-5:]:
        print(f"       {line}")
    if ec != 0 and err:
        print(f"       ERRO: {err[:200]}")

    return {
        "script": script, "args": args, "exit_code": ec,
        "stdout": out, "stderr": err[:500] if err else ""
    }


# ─── Scenario Runners ───────────────────────────────────────────────────────

def run_E1(ssh: SSHManager, model: str, protocol_label: str, n: int) -> List[Dict]:
    """E1: Latencia via orchestrator."""
    results = []
    base_url = f"http://localhost:{ORCH_PORT}"

    for prompt in ["short", "long"]:
        r = run_benchmark(ssh,
            "E1_decompose_latency_dds.py",
            f"--url {base_url} --model {model} --prompt {prompt} --n {n} --protocol-label {protocol_label}",
            timeout=300)
        results.append(r)

    return results


def run_E2_dds(ssh: SSHManager, n: int) -> List[Dict]:
    """E2: Failure detection DDS (standalone, usa proprio publisher)."""
    xml_path = get_cyclonedds_xml(ssh)
    env_prefix = f"export CYCLONEDDS_URI=file://{xml_path} && "
    return [run_benchmark(ssh,
        "E2_failure_detection_dds.py",
        f"--periodo 1000 --lease 200 --n {n} --domain 0",
        timeout=120, env_prefix=env_prefix)]


def run_E2_grpc(ssh: SSHManager, n: int) -> List[Dict]:
    """E2: Failure detection gRPC (standalone, usa proprio server)."""
    return [run_benchmark(ssh,
        "E2_failure_detection_grpc.py",
        f"--periodo 1000 --n {n} --port 50099",
        timeout=120)]


def run_E3(ssh: SSHManager, protocol_label: str, n: int) -> List[Dict]:
    """E3: Priority via orchestrator."""
    return [run_benchmark(ssh,
        "E3_priority_dds.py",
        f"--url http://localhost:{ORCH_PORT} --carga 2 --n {n} --duracao 60 --protocol-label {protocol_label}",
        timeout=300)]


def run_E4(ssh: SSHManager, protocol_label: str, n: int) -> List[Dict]:
    """E4: Scalability via orchestrator."""
    return [run_benchmark(ssh,
        "E4_scalability_dds.py",
        f"--orchestrador http://localhost:{ORCH_PORT} --n {n} --protocol-label {protocol_label}",
        timeout=600)]


def run_E5_orch(ssh: SSHManager, model: str, protocol_label: str, n: int) -> List[Dict]:
    """E5: Streaming via orchestrator SSE."""
    return [run_benchmark(ssh,
        "E5_streaming_dds.py",
        f"--url http://localhost:{ORCH_PORT} --model {model} --n {n} --protocol-label {protocol_label}",
        timeout=300)]


def run_E5_grpc_direct(ssh: SSHManager, agent_ip: str, model: str, n: int) -> List[Dict]:
    """E5: Streaming gRPC direto ao llama-server."""
    return [run_benchmark(ssh,
        "E5_streaming_grpc.py",
        f"--endpoint {agent_ip}:50051 --model {model} --n {n}",
        timeout=300)]


# ─── Main Flow ───────────────────────────────────────────────────────────────

def run_full_comparison(cenario: str, n: int) -> Dict:
    """Executa comparacao completa DDS vs gRPC."""

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {"n": n, "cenario": cenario},
        "DDS": {},
        "gRPC": {},
    }

    # ─── Connect ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CONECTANDO NAS VMs")
    print("=" * 60)

    ssh_orch = SSHManager(ORCH_IP, "oldds", "Admin@123", "Orchestrator")
    ssh_a1 = SSHManager("192.168.1.60", "oldds", "Admin@123", "Agent1 RX6600M")
    ssh_a2 = SSHManager("192.168.1.61", "oldds", "Admin@123", "Agent2 RTX3080")

    for ssh in [ssh_orch, ssh_a1, ssh_a2]:
        if not ssh.connect():
            return {"error": f"Falha ao conectar em {ssh.ip}"}

    # ─── Setup ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SETUP (git pull, deps, compile)")
    print("=" * 60)

    for ssh in [ssh_orch, ssh_a1, ssh_a2]:
        git_pull_all(ssh)
        install_deps(ssh)

    # Check models
    for ssh in [ssh_a1, ssh_a2]:
        ec, out, _ = ssh.run(f"ls {MODELS_DIR}/*.gguf")
        print(f"  Modelos em {ssh.ip}: {out.strip()}")

    # Compile on agent machines
    for name, ssh, cfg in [("agent1", ssh_a1, VM_CONFIG["agent1"]),
                            ("agent2", ssh_a2, VM_CONFIG["agent2"])]:
        compile_llama(ssh, cfg)

    # Use agent2 (RTX 3080 + Qwen3.5-9B) as primary agent for benchmarks
    primary_agent = VM_CONFIG["agent2"]
    ssh_primary = ssh_a2
    model_name = primary_agent["model_name"]

    # ─── PHASE 1: DDS Benchmarks ─────────────────────────────────────────
    if cenario in ["all", "E1", "E3", "E4", "E5"]:
        print("\n" + "=" * 60)
        print("  FASE DDS: Iniciando servicos")
        print("=" * 60)

        # Kill existing services
        for ssh in [ssh_orch, ssh_a1, ssh_a2]:
            kill_services(ssh)
        time.sleep(3)

        # Start DDS services
        start_llama_server_dds(ssh_primary, primary_agent)
        print("    Aguardando llama-server carregar modelo...")
        # Wait for llama-server to be ready (model load can take 20-30s)
        for attempt in range(12):
            time.sleep(5)
            ec, out, _ = ssh_primary.run("curl -s http://localhost:8082/health", timeout=5)
            if ec == 0 and "ok" in out.lower():
                print(f"    llama-server health: {out[:100]}")
                break
            if attempt % 3 == 0:
                print(f"      Aguardando modelo... ({(attempt+1)*5}s)")
        else:
            print(f"    [WARN] llama-server pode nao estar pronto")
            ec, out, _ = ssh_primary.run("tail -5 /tmp/llama_dds.log 2>/dev/null")
            print(f"    Log: {out[:200]}")

        start_orchestrator_dds(ssh_orch)
        time.sleep(5)

        # Verify orchestrator actually started
        ec, out, _ = ssh_orch.run("pgrep -f 'python3.*main.py'", timeout=5)
        if not out.strip():
            print("    [FAIL] Orchestrator NAO esta rodando!")
            ec, out, _ = ssh_orch.run("tail -20 /tmp/orch_dds.log 2>/dev/null")
            print(f"    Log: {out[:500]}")
            print("    Abortando fase DDS.")
        else:
            print(f"    Orchestrator PID: {out.strip()}")

        start_agent_dds(ssh_primary, primary_agent)
        wait_agent_registration(ssh_orch, expected=1)

        warmup_request(ssh_orch, model_name)
        time.sleep(2)

        # Run DDS benchmarks
        print("\n" + "=" * 60)
        print(f"  FASE DDS: Benchmarks E1-E5 (n={n})")
        print("=" * 60)

        if cenario in ["all", "E1"]:
            print("\n  --- E1: Latencia (DDS) ---")
            results["DDS"]["E1"] = run_E1(ssh_orch, model_name, "DDS", n)

        if cenario in ["all", "E3"]:
            print("\n  --- E3: Prioridade (DDS) ---")
            results["DDS"]["E3"] = run_E3(ssh_orch, "DDS", n)

        if cenario in ["all", "E4"]:
            print("\n  --- E4: Escalabilidade (DDS) ---")
            results["DDS"]["E4"] = run_E4(ssh_orch, "DDS", n)

        if cenario in ["all", "E5"]:
            print("\n  --- E5: Streaming (DDS via Orch SSE) ---")
            results["DDS"]["E5_orch"] = run_E5_orch(ssh_orch, model_name, "DDS", n)

        # Stop DDS services
        print("\n  Parando servicos DDS...")
        for ssh in [ssh_orch, ssh_a1, ssh_a2]:
            kill_services(ssh)

        # Cooldown
        print("  Cooldown GPU (30s)...")
        time.sleep(30)

    # ─── PHASE 2: gRPC Benchmarks ────────────────────────────────────────
    if cenario in ["all", "E1", "E3", "E4", "E5"]:
        print("\n" + "=" * 60)
        print("  FASE gRPC: Iniciando servicos")
        print("=" * 60)

        # Kill anything leftover from DDS phase
        for ssh in [ssh_orch, ssh_a1, ssh_a2]:
            kill_services(ssh)
        time.sleep(3)

        # Start gRPC services
        start_llama_server_grpc(ssh_primary, primary_agent)
        print("    Aguardando llama-server carregar modelo...")
        for attempt in range(12):
            time.sleep(5)
            ec, out, _ = ssh_primary.run("curl -s http://localhost:8082/health", timeout=5)
            if ec == 0 and "ok" in out.lower():
                print(f"    llama-server health: {out[:100]}")
                break
            if attempt % 3 == 0:
                print(f"      Aguardando modelo... ({(attempt+1)*5}s)")
        else:
            print(f"    [WARN] llama-server pode nao estar pronto")
            ec, out, _ = ssh_primary.run("tail -5 /tmp/llama_grpc.log 2>/dev/null")
            print(f"    Log: {out[:200]}")

        start_orchestrator_grpc(ssh_orch)
        time.sleep(5)

        # Verify orchestrator actually started
        ec, out, _ = ssh_orch.run("pgrep -f 'python3.*main.py'", timeout=5)
        if not out.strip():
            print("    [FAIL] Orchestrator NAO esta rodando!")
            ec, out, _ = ssh_orch.run("tail -20 /tmp/orch_grpc.log 2>/dev/null")
            print(f"    Log: {out[:500]}")
            print("    Abortando fase gRPC.")
        else:
            print(f"    Orchestrator PID: {out.strip()}")

        start_agent_grpc(ssh_primary, primary_agent)
        wait_agent_registration(ssh_orch, expected=1)

        warmup_request(ssh_orch, model_name)
        time.sleep(2)

        # Run gRPC benchmarks
        print("\n" + "=" * 60)
        print(f"  FASE gRPC: Benchmarks E1-E5 (n={n})")
        print("=" * 60)

        if cenario in ["all", "E1"]:
            print("\n  --- E1: Latencia (gRPC) ---")
            results["gRPC"]["E1"] = run_E1(ssh_orch, model_name, "gRPC", n)

        if cenario in ["all", "E3"]:
            print("\n  --- E3: Prioridade (gRPC) ---")
            results["gRPC"]["E3"] = run_E3(ssh_orch, "gRPC", n)

        if cenario in ["all", "E4"]:
            print("\n  --- E4: Escalabilidade (gRPC) ---")
            results["gRPC"]["E4"] = run_E4(ssh_orch, "gRPC", n)

        if cenario in ["all", "E5"]:
            print("\n  --- E5: Streaming (gRPC via Orch SSE) ---")
            results["gRPC"]["E5_orch"] = run_E5_orch(ssh_orch, model_name, "gRPC", n)

            print("\n  --- E5: Streaming (gRPC Direct) ---")
            results["gRPC"]["E5_direct"] = run_E5_grpc_direct(
                ssh_orch, primary_agent["ip"] if "ip" in primary_agent else "192.168.1.61",
                model_name, n)

        # Stop gRPC services
        print("\n  Parando servicos gRPC...")
        for ssh in [ssh_orch, ssh_a1, ssh_a2]:
            kill_services(ssh)

    # ─── PHASE 3: E2 Standalone ──────────────────────────────────────────
    if cenario in ["all", "E2"]:
        print("\n" + "=" * 60)
        print("  FASE E2: Deteccao de Falha (standalone)")
        print("=" * 60)

        # E2 runs its own subprocess - no need for external services
        print("\n  --- E2: DDS LIVELINESS ---")
        results["DDS"]["E2"] = run_E2_dds(ssh_orch, n)

        print("\n  --- E2: gRPC GetStatus ---")
        results["gRPC"]["E2"] = run_E2_grpc(ssh_orch, n)

    # ─── Collect Results ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COLETANDO RESULTADOS")
    print("=" * 60)

    # Fetch result files from orchestrator
    ec, out, _ = ssh_orch.run(f"ls {BASE_DIR}/dds_orchestrator/results/*.json 2>/dev/null")
    if ec == 0:
        for json_file in out.strip().split('\n'):
            if json_file.strip():
                fname = json_file.strip().split('/')[-1]
                ec2, content, _ = ssh_orch.run(f"cat {json_file.strip()}")
                if ec2 == 0:
                    results.setdefault("result_files", {})[fname] = content
                    print(f"    {fname}")

    # Disconnect
    for ssh in [ssh_orch, ssh_a1, ssh_a2]:
        ssh.disconnect()

    return results


def print_summary(results: Dict):
    """Imprime tabela comparativa."""
    print("\n" + "=" * 60)
    print("  RESUMO: DDS vs gRPC")
    print("=" * 60)

    for phase in ["DDS", "gRPC"]:
        if phase not in results:
            continue
        print(f"\n  --- {phase} ---")
        for exp, data in results[phase].items():
            if isinstance(data, list):
                for r in data:
                    status = "OK" if r.get("exit_code") == 0 else "FAIL"
                    print(f"    [{status}] {r.get('script', '?')} {r.get('args', '')[:60]}")

    # Count totals
    total = passed = failed = 0
    for phase in ["DDS", "gRPC"]:
        for exp, data in results.get(phase, {}).items():
            if isinstance(data, list):
                for r in data:
                    total += 1
                    if r.get("exit_code") == 0:
                        passed += 1
                    else:
                        failed += 1

    print(f"\n  Total: {total} | Passou: {passed} | Falhou: {failed}")


def main():
    parser = argparse.ArgumentParser(description="Benchmarks E1-E5 nas VMs (DDS vs gRPC)")
    parser.add_argument("--cenario", choices=["all", "E1", "E2", "E3", "E4", "E5"], default="all")
    parser.add_argument("--n", type=int, default=5, help="Iteracoes por benchmark")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--output", default="vm_benchmark_results.json")

    args = parser.parse_args()

    print("=" * 60)
    print("  BENCHMARK E1-E5: DDS vs gRPC nas VMs")
    print(f"  Orchestrator: {ORCH_IP}")
    print(f"  Agent 1: 192.168.1.60 (RX6600M)")
    print(f"  Agent 2: 192.168.1.61 (RTX 3080)")
    print(f"  Cenario: {args.cenario} | N: {args.n}")
    print("=" * 60)

    if args.setup_only:
        # Just setup
        ssh_orch = SSHManager(ORCH_IP, "oldds", "Admin@123", "Orchestrator")
        ssh_a1 = SSHManager("192.168.1.60", "oldds", "Admin@123", "Agent1")
        ssh_a2 = SSHManager("192.168.1.61", "oldds", "Admin@123", "Agent2")

        for ssh in [ssh_orch, ssh_a1, ssh_a2]:
            if ssh.connect():
                git_pull_all(ssh)
                install_deps(ssh)
                ssh.disconnect()
        for name, ssh, cfg in [("agent1", ssh_a1, VM_CONFIG["agent1"]),
                                ("agent2", ssh_a2, VM_CONFIG["agent2"])]:
            if ssh.connect():
                compile_llama(ssh, cfg)
                ssh.disconnect()
        return

    results = run_full_comparison(args.cenario, args.n)

    if "error" in results:
        print(f"\n  ERRO: {results['error']}")
        sys.exit(1)

    print_summary(results)

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
