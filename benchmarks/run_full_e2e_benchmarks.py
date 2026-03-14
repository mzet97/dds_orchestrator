#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full End-to-End Benchmark Runner — HTTP, gRPC, DDS
====================================================
Usa paramiko (SSH puro) para gerenciar VMs e subprocess para rodar
benchmarks LOCALMENTE no cliente (WSL/Linux/Mac).

Fluxo:
  1. Conecta nas 3 VMs via SSH (paramiko)
  2. Git pull em todos os repos
  3. Instala dependencias e compila llama.cpp_dds
  4. FASE HTTP:  inicia servicos HTTP, roda E1/E3/E4/E5 localmente
  5. FASE gRPC: inicia servicos gRPC, roda E1/E3/E4/E5 localmente
  6. FASE DDS:  inicia servicos DDS, roda E1/E3/E4/E5 localmente
  7. FASE E2:   roda E2 standalone no orchestrator (via SSH)
  8. Coleta resultados

VMs:
  - 192.168.1.60: Agent 1 (RX6600M 8GB)  - llama-server + agent
  - 192.168.1.61: Agent 2 (RTX 3080 10GB) - llama-server + agent
  - 192.168.1.62: Orchestrator

Cliente: maquina local (WSL) — roda benchmarks com subprocess

Usage:
    python run_full_e2e_benchmarks.py --n 5
    python run_full_e2e_benchmarks.py --n 10 --phases http,grpc,dds
    python run_full_e2e_benchmarks.py --setup-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        "desc": "Orchestrator",
    },
    "agent1": {
        "ip": "192.168.1.60",
        "user": "oldds",
        "password": "Admin@123",
        "role": "agent",
        "model": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "model_name": "Phi-4-mini",
        "desc": "Agent 1 - RX6600M 8GB",
        "gpu_target": "gfx1032",
    },
    "agent2": {
        "ip": "192.168.1.61",
        "user": "oldds",
        "password": "Admin@123",
        "role": "agent",
        "model": "Qwen3.5-9B-Q4_K_M.gguf",
        "model_name": "Qwen3.5-9B",
        "desc": "Agent 2 - RTX 3080 10GB",
        "gpu_target": None,
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
ORCH_GRPC_PORT = 50052


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
            transport = self.client.get_transport()
            if transport:
                transport.set_keepalive(30)
            print(f"  [OK] Conectado em {self.ip}")
            return True
        except Exception as e:
            print(f"  [FAIL] Erro em {self.ip}: {e}")
            return False

    def reconnect(self) -> bool:
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
                    self.reconnect()
                    continue
                return -1, "", str(e)
        return -1, "", "Falha apos reconexao"

    def run_bg(self, command: str, logfile: str = "/dev/null", env: Dict = None) -> bool:
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
    print(f"  Instalando deps em {ssh.ip}...")
    ec, out, err = ssh.run(
        "pip3 install --user --break-system-packages aiohttp pydantic psutil requests "
        "protobuf grpcio grpcio-tools cyclonedds --quiet",
        timeout=180
    )
    print(f"    {'[OK]' if ec == 0 else '[FAIL] ' + err[:200]}")


def compile_llama(ssh: SSHManager, vm_cfg: Dict):
    print(f"  Compilando llama.cpp_dds em {ssh.ip}...")
    build_dir = f"{BASE_DIR}/llama.cpp_dds/build"
    gpu_target = vm_cfg.get("gpu_target")
    cyclone_root = f"{BASE_DIR}/cyclonedds/install"
    cmake_flags = f"-DLLAMA_DDS=ON -DCYCLONEDDS_ROOT={cyclone_root} -DCMAKE_BUILD_TYPE=Release"
    if gpu_target:
        cmake_flags += f" -DAMDGPU_TARGETS={gpu_target}"

    cmd = (f"cd {BASE_DIR}/llama.cpp_dds && mkdir -p build && cd build && "
           f"rm -f CMakeCache.txt && "
           f"cmake .. {cmake_flags} 2>&1 && make llama-server -j$(nproc) 2>&1")
    ec, out, err = ssh.run(cmd, timeout=600)

    ec2, out2, _ = ssh.run(f"ls {build_dir}/bin/llama-server 2>/dev/null || ls {build_dir}/server 2>/dev/null")
    if ec2 == 0:
        print(f"    [OK] Build completo: {out2.strip()}")
    else:
        real_errors = [l for l in err.split('\n') if 'error:' in l.lower() and 'warning' not in l.lower()]
        if real_errors:
            print(f"    [FAIL] Erro: {real_errors[0][:200]}")
        else:
            print(f"    [OK] Build com warnings")


def find_llama_binary(ssh: SSHManager) -> str:
    for path in [f"{BASE_DIR}/llama.cpp_dds/build/bin/llama-server",
                 f"{BASE_DIR}/llama.cpp_dds/build/server",
                 f"{BASE_DIR}/llama.cpp_dds/build/tools/server/llama-server"]:
        ec, out, _ = ssh.run(f"test -f {path} && echo FOUND")
        if "FOUND" in out:
            return path
    ec, out, _ = ssh.run(f"find {BASE_DIR}/llama.cpp_dds/build -type f -executable -name 'llama-server' 2>/dev/null | head -1", timeout=15)
    if ec == 0 and out.strip():
        return out.strip().split('\n')[0].strip()
    return f"{BASE_DIR}/llama.cpp_dds/build/bin/llama-server"


def get_cyclonedds_xml(ssh: SSHManager) -> str:
    for xml in [f"{BASE_DIR}/llama.cpp_dds/dds/cyclonedds-network-ultra.xml",
                f"{BASE_DIR}/llama.cpp_dds/dds/cyclonedds-network.xml"]:
        ec, _, _ = ssh.run(f"test -f {xml}")
        if ec == 0:
            return xml
    return f"{BASE_DIR}/llama.cpp_dds/dds/cyclonedds-local.xml"


# ─── Service Management ─────────────────────────────────────────────────────

def kill_services(ssh: SSHManager):
    ec, out, _ = ssh.run("pgrep -f 'llama-server|agent_llm|main.py' 2>/dev/null || true", timeout=5)
    pids = [p.strip() for p in out.strip().split('\n') if p.strip().isdigit()]
    if pids:
        ssh.run(f"kill -9 {' '.join(pids)} 2>/dev/null; true", timeout=5)
        time.sleep(2)
        print(f"    Servicos parados em {ssh.ip} ({len(pids)} processos)")
    else:
        print(f"    Nenhum servico ativo em {ssh.ip}")


def wait_llama_health(ssh: SSHManager, port: int = 8082, max_wait: int = 60) -> bool:
    print(f"    Aguardando llama-server em {ssh.ip}:{port}...")
    for i in range(max_wait // 5):
        ec, out, _ = ssh.run(f"curl -s http://localhost:{port}/health", timeout=5)
        if ec == 0 and "ok" in out.lower():
            print(f"    [OK] llama-server pronto")
            return True
        if i % 3 == 0:
            print(f"      Aguardando... ({(i+1)*5}s)")
        time.sleep(5)
    print(f"    [FAIL] llama-server timeout")
    return False


def wait_agent_registration(ssh_orch: SSHManager, expected: int = 1, max_wait: int = 120) -> bool:
    print(f"    Aguardando {expected} agent(s)...")
    for i in range(max_wait // 3):
        ec, out, _ = ssh_orch.run(f"curl -s http://localhost:{ORCH_PORT}/api/v1/agents", timeout=10)
        if ec == 0 and "agent_id" in out:
            count = out.count("agent_id")
            if count >= expected:
                print(f"    [OK] {count} agent(s) registrado(s)")
                return True
        time.sleep(3)
    print(f"    [FAIL] Timeout aguardando agents")
    return False


def warmup_request(ssh_orch: SSHManager, model: str):
    cmd = (f'curl -s -X POST http://localhost:{ORCH_PORT}/v1/chat/completions '
           f'-H "Content-Type: application/json" '
           f'-d \'{{"model":"{model}","messages":[{{"role":"user","content":"warmup"}}],"max_tokens":5}}\'')
    ssh_orch.run(cmd, timeout=120)
    print(f"    [OK] Warmup completo")


# ─── Service Start Functions (per protocol) ──────────────────────────────────

def wait_orchestrator_ready(ssh_orch: SSHManager, max_wait: int = 30) -> bool:
    """Wait for orchestrator HTTP to respond."""
    for i in range(max_wait // 2):
        ec, out, _ = ssh_orch.run(f"curl -s http://localhost:{ORCH_PORT}/health", timeout=5)
        if ec == 0 and out.strip():
            print(f"    [OK] Orchestrator respondendo")
            return True
        time.sleep(2)
    print(f"    [WARN] Orchestrator pode nao estar pronto")
    return False


def start_services_http(ssh_orch, ssh_agent, agent_cfg):
    """Start HTTP-only services: orchestrator(HTTP) + agent_llm.py (starts its own llama-server)."""
    binary = find_llama_binary(ssh_agent)
    model_path = f"{MODELS_DIR}/{agent_cfg['model']}"

    # 1. Orchestrator HTTP only FIRST — explicitly disable DDS
    orch_cmd = (f"python3 -u {BASE_DIR}/dds_orchestrator/main.py "
                f"--port {ORCH_PORT} --log-level INFO")
    ssh_orch.run_bg(orch_cmd, "/tmp/orch_http.log", env={"DDS_ENABLED": "false"})
    print(f"    [OK] Orchestrator HTTP-only em {ssh_orch.ip}")
    wait_orchestrator_ready(ssh_orch)

    # 2. HTTP Agent — agent_llm.py manages its own llama-server internally.
    #    Pure HTTP: agent starts llama-server, registers with orchestrator via HTTP,
    #    receives tasks via HTTP, forwards to local llama-server via HTTP.
    agent_env = {
        "MODEL_PATH": model_path,
        "MODEL_NAME": agent_cfg["model_name"],
        "LLAMA_SERVER_PATH": binary,
        "LLAMA_SERVER_PORT": "8082",
        "AGENT_PORT": "8081",
        "ORCHESTRATOR_URL": f"http://{ORCH_IP}:{ORCH_PORT}",
        "GPU_LAYERS": "99",
    }
    agent_cmd = f"python3 -u {BASE_DIR}/dds_agent/python/agent_llm.py"
    ssh_agent.run_bg(agent_cmd, "/tmp/agent_http.log", env=agent_env)
    print(f"    [OK] Agent HTTP em {ssh_agent.ip} (manages own llama-server)")

    # Wait for llama-server (started by agent) to be healthy
    wait_llama_health(ssh_agent)


def start_services_grpc(ssh_orch, ssh_agent, agent_cfg):
    """Start gRPC services: llama-server + orchestrator(gRPC) + agent_llm_grpc."""
    binary = find_llama_binary(ssh_agent)
    model_path = f"{MODELS_DIR}/{agent_cfg['model']}"

    # 1. llama-server
    cmd = (f"{binary} -m {model_path} -c 2048 --threads 8 -ngl 99 "
           f"--port 8082 --host 0.0.0.0")
    ssh_agent.run_bg(cmd, "/tmp/llama_grpc.log")
    print(f"    [OK] llama-server em {ssh_agent.ip}")
    wait_llama_health(ssh_agent)

    # 2. Orchestrator with gRPC enabled FIRST
    orch_cmd = (f"python3 -u {BASE_DIR}/dds_orchestrator/main.py "
                f"--port {ORCH_PORT} --grpc-enabled --grpc-port {ORCH_GRPC_PORT} --log-level INFO")
    ssh_orch.run_bg(orch_cmd, "/tmp/orch_grpc.log")
    print(f"    [OK] Orchestrator gRPC em {ssh_orch.ip}")
    wait_orchestrator_ready(ssh_orch)

    # 3. gRPC agent AFTER orchestrator is ready
    agent_cmd = (f"python3 -u {BASE_DIR}/dds_agent/python/agent_llm_grpc.py "
                 f"--model-name {agent_cfg['model_name']} "
                 f"--model-path {model_path} "
                 f"--orchestrator-url http://{ORCH_IP}:{ORCH_PORT} "
                 f"--port 8081 "
                 f"--grpc-address localhost:50051 "
                 f"--grpc-listen-port 50053 "
                 f"--no-server")
    ssh_agent.run_bg(agent_cmd, "/tmp/agent_grpc.log")
    print(f"    [OK] Agent gRPC em {ssh_agent.ip}")


def start_services_dds(ssh_orch, ssh_agent, agent_cfg):
    """Start DDS services: llama-server(DDS) + orchestrator(DDS) + agent_llm_dds."""
    binary = find_llama_binary(ssh_agent)
    model_path = f"{MODELS_DIR}/{agent_cfg['model']}"
    xml_agent = get_cyclonedds_xml(ssh_agent)
    xml_orch = get_cyclonedds_xml(ssh_orch)

    # 1. llama-server with DDS enabled
    cmd = (f"{binary} -m {model_path} -c 2048 --threads 8 -ngl 99 "
           f"--port 8082 --host 0.0.0.0 "
           f"--enable-dds --dds-domain 0 --dds-timeout 120")
    ssh_agent.run_bg(cmd, "/tmp/llama_dds.log", env={"CYCLONEDDS_URI": f"file://{xml_agent}"})
    print(f"    [OK] llama-server DDS em {ssh_agent.ip}")
    wait_llama_health(ssh_agent)

    # 2. Orchestrator with DDS FIRST
    orch_cmd = (f"python3 -u {BASE_DIR}/dds_orchestrator/main.py "
                f"--port {ORCH_PORT} --dds-domain 0 --log-level INFO")
    ssh_orch.run_bg(orch_cmd, "/tmp/orch_dds.log", env={"CYCLONEDDS_URI": f"file://{xml_orch}"})
    print(f"    [OK] Orchestrator DDS em {ssh_orch.ip}")
    wait_orchestrator_ready(ssh_orch)

    # 3. DDS agent AFTER orchestrator is ready
    agent_cmd = (f"python3 -u {BASE_DIR}/dds_agent/python/agent_llm_dds.py "
                 f"--model-name {agent_cfg['model_name']} "
                 f"--model-path {model_path} "
                 f"--orchestrator-url http://{ORCH_IP}:{ORCH_PORT} "
                 f"--port 8081 --llama-server-port 8082 --no-server")
    ssh_agent.run_bg(agent_cmd, "/tmp/agent_dds.log", env={"CYCLONEDDS_URI": f"file://{xml_agent}"})
    print(f"    [OK] Agent DDS em {ssh_agent.ip}")


# ─── Local Benchmark Runner ─────────────────────────────────────────────────

def run_local_benchmark(protocol: str, model: str, n: int,
                        scenario: str = "all", extra_args: str = "") -> int:
    """Run benchmark locally using subprocess (from WSL/Linux client)."""
    script_dir = Path(__file__).parent
    script = script_dir / "e2e_benchmark_client.py"

    cmd = [
        sys.executable, str(script),
        "--protocol", protocol,
        "--url", f"http://{ORCH_IP}:{ORCH_PORT}",
        "--endpoint", f"{ORCH_IP}:{ORCH_GRPC_PORT}",
        "--domain", "0",
        "--model", model,
        "--scenario", scenario,
        "--n", str(n),
    ]

    print(f"\n    -> {protocol.upper()} benchmark (scenario={scenario}, n={n})")
    print(f"    cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(script_dir))
    return result.returncode


def run_E2_standalone(ssh_orch: SSHManager, n: int) -> Dict:
    """E2 runs standalone (its own processes). Execute on orchestrator via SSH."""
    results = {}
    xml_path = get_cyclonedds_xml(ssh_orch)
    env_prefix = f"export CYCLONEDDS_URI=file://{xml_path} && "

    print("\n  --- E2: DDS DEADLINE ---")
    cmd = f"{env_prefix}cd {BASE_DIR}/dds_orchestrator && python3 -u benchmarks/E2_failure_detection_dds.py --periodo 1000 --lease 200 --n {n} --domain 0"
    ec, out, err = ssh_orch.run(cmd, timeout=120)
    results["DDS"] = {"exit_code": ec, "stdout": out[-500:], "stderr": err[:200]}
    print(f"    {'[OK]' if ec == 0 else '[FAIL]'}")
    for line in out.strip().split('\n')[-3:]:
        print(f"      {line}")

    print("\n  --- E2: HTTP Heartbeat ---")
    cmd = f"cd {BASE_DIR}/dds_orchestrator && python3 -u benchmarks/E2_failure_detection_http.py --agent-url http://localhost:8082 --intervalo 1000 --tipo kill9 --n {n}"
    ec, out, err = ssh_orch.run(cmd, timeout=120)
    results["HTTP"] = {"exit_code": ec, "stdout": out[-500:], "stderr": err[:200]}
    print(f"    {'[OK]' if ec == 0 else '[FAIL]'}")
    for line in out.strip().split('\n')[-3:]:
        print(f"      {line}")

    print("\n  --- E2: gRPC Health ---")
    cmd = f"cd {BASE_DIR}/dds_orchestrator && python3 -u benchmarks/E2_failure_detection_grpc.py --periodo 1000 --n {n} --port 50099"
    ec, out, err = ssh_orch.run(cmd, timeout=120)
    results["gRPC"] = {"exit_code": ec, "stdout": out[-500:], "stderr": err[:200]}
    print(f"    {'[OK]' if ec == 0 else '[FAIL]'}")
    for line in out.strip().split('\n')[-3:]:
        print(f"      {line}")

    return results


# ─── Main Flow ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full E2E Benchmarks: HTTP vs gRPC vs DDS")
    parser.add_argument("--n", type=int, default=5, help="Iteracoes por benchmark")
    parser.add_argument("--phases", default="http,grpc,dds",
                        help="Fases a executar (virgula separadas)")
    parser.add_argument("--scenario", choices=["all", "E1", "E3", "E4", "E5"],
                        default="all", help="Cenario (E2 roda separado)")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--output", default="e2e_full_results.json")
    parser.add_argument("--agent", choices=["agent1", "agent2"], default="agent2",
                        help="Agent primario para benchmarks")

    args = parser.parse_args()
    phases = [p.strip().lower() for p in args.phases.split(",")]

    primary_agent_key = args.agent
    primary_agent = VM_CONFIG[primary_agent_key]
    model_name = primary_agent["model_name"]

    print("=" * 60)
    print("  FULL E2E BENCHMARK: HTTP vs gRPC vs DDS")
    print(f"  Orchestrator: {ORCH_IP}")
    print(f"  Agent: {primary_agent['ip']} ({primary_agent['desc']})")
    print(f"  Model: {model_name}")
    print(f"  Phases: {', '.join(phases)}")
    print(f"  N: {args.n} | Scenario: {args.scenario}")
    print("=" * 60)

    # ─── Connect ─────────────────────────────────────────────────────────
    print("\n  CONECTANDO NAS VMs...")
    ssh_orch = SSHManager(ORCH_IP, "oldds", "Admin@123", "Orchestrator")
    ssh_agent = SSHManager(primary_agent["ip"], "oldds", "Admin@123", primary_agent["desc"])

    for ssh in [ssh_orch, ssh_agent]:
        if not ssh.connect():
            print(f"ERRO: Falha ao conectar em {ssh.ip}")
            sys.exit(1)

    # ─── Setup ───────────────────────────────────────────────────────────
    if not args.skip_setup:
        print("\n" + "=" * 60)
        print("  SETUP (git pull, deps, compile)")
        print("=" * 60)

        for ssh in [ssh_orch, ssh_agent]:
            git_pull_all(ssh)
            install_deps(ssh)

        if not args.skip_compile:
            compile_llama(ssh_agent, primary_agent)

        # Verify models
        ec, out, _ = ssh_agent.run(f"ls {MODELS_DIR}/*.gguf")
        print(f"  Modelos em {ssh_agent.ip}: {out.strip()}")

    if args.setup_only:
        print("\n  Setup completo. Saindo.")
        for ssh in [ssh_orch, ssh_agent]:
            ssh.disconnect()
        return

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {"n": args.n, "scenario": args.scenario, "phases": phases,
                    "model": model_name, "agent": primary_agent["ip"]},
    }

    # ─── Run each protocol phase ─────────────────────────────────────────
    for phase in phases:
        print("\n" + "=" * 60)
        print(f"  FASE {phase.upper()}: Iniciando servicos")
        print("=" * 60)

        # Kill existing
        for ssh in [ssh_orch, ssh_agent]:
            kill_services(ssh)
        time.sleep(3)

        # Start services for this phase
        if phase == "http":
            start_services_http(ssh_orch, ssh_agent, primary_agent)
        elif phase == "grpc":
            start_services_grpc(ssh_orch, ssh_agent, primary_agent)
        elif phase == "dds":
            start_services_dds(ssh_orch, ssh_agent, primary_agent)

        time.sleep(5)

        # Verify orchestrator is running
        ec, out, _ = ssh_orch.run("pgrep -f 'python3.*main.py'", timeout=5)
        if not out.strip():
            print(f"    [FAIL] Orchestrator nao esta rodando!")
            for logfile in [f"/tmp/orch_{phase}.log"]:
                ec2, out2, _ = ssh_orch.run(f"tail -20 {logfile} 2>/dev/null")
                if out2.strip():
                    print(f"    Log: {out2[:500]}")
            continue
        print(f"    Orchestrator PID: {out.strip()}")

        # Wait for agent registration
        if not wait_agent_registration(ssh_orch, expected=1):
            print(f"    [FAIL] Agent nao registrou. Pulando fase {phase}.")
            # Check agent logs
            ec2, out2, _ = ssh_agent.run(f"tail -20 /tmp/agent_{phase}.log 2>/dev/null")
            if out2.strip():
                print(f"    Agent log: {out2[:500]}")
            continue

        # Warmup via HTTP (orchestrator always has HTTP endpoint)
        warmup_request(ssh_orch, model_name)
        time.sleep(2)

        # ─── Run benchmarks locally ──────────────────────────────────────
        print(f"\n  FASE {phase.upper()}: Benchmarks E1/E3/E4/E5 (n={args.n})")
        rc = run_local_benchmark(phase, model_name, args.n, args.scenario)
        results[phase.upper()] = {"exit_code": rc, "status": "OK" if rc == 0 else "FAIL"}

        # Kill services
        print(f"\n  Parando servicos {phase}...")
        for ssh in [ssh_orch, ssh_agent]:
            kill_services(ssh)

        # Cooldown
        print(f"  Cooldown GPU (20s)...")
        time.sleep(20)

    # ─── E2 Standalone ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FASE E2: Deteccao de Falha (standalone)")
    print("=" * 60)
    results["E2"] = run_E2_standalone(ssh_orch, args.n)

    # ─── Collect Results ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTADOS")
    print("=" * 60)

    # Fetch result files from local results/ dir
    results_dir = Path(__file__).parent / "results"
    if results_dir.exists():
        for f in results_dir.glob("e2e_*.json"):
            print(f"    {f.name}")

    # Disconnect
    for ssh in [ssh_orch, ssh_agent]:
        ssh.disconnect()

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Resultados salvos em: {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print("  RESUMO")
    print(f"{'='*60}")
    for phase in phases:
        status = results.get(phase.upper(), {}).get("status", "N/A")
        print(f"    {phase.upper()}: {status}")
    print(f"    E2: {', '.join(k + ':' + ('OK' if v.get('exit_code')==0 else 'FAIL') for k,v in results.get('E2',{}).items())}")


if __name__ == "__main__":
    main()
