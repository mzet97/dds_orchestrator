#!/usr/bin/env python3
"""
Deploy 5 LLM Agents on 2 GPUs for Fuzzy Benchmark
====================================================
RTX 3080 (10GB, 192.168.1.61): 3x Phi-4-mini (Agent_Fast, Agent_Quality, Agent_Balanced)
RX 6600M (8GB, 192.168.1.60):  2x Phi-4-mini (Agent_Backup_1, Agent_Backup_2)

Each agent has a different profile that the fuzzy engine uses for selection.

Usage:
    python deploy_5agents.py --start          # Start all 5 agents + orchestrator
    python deploy_5agents.py --stop           # Stop all services
    python deploy_5agents.py --status         # Check status
    python deploy_5agents.py --start --fuzzy  # Start with fuzzy enabled
"""

import argparse
import paramiko
import socket
import sys
import io
import time
from typing import Dict, List, Optional, Tuple

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = "/home/oldds"
MODELS_DIR = f"{BASE_DIR}/models"
ORCH_IP = "192.168.1.62"
ORCH_PORT = 8080

# ─── Agent Definitions ──────────────────────────────────────────────────────

AGENTS = [
    {
        "id": "agent-fast",
        "vm_ip": "192.168.1.61",
        "gpu": "rtx3080",
        "profile": "fast",
        "llama_port": 8082,
        "agent_port": 8081,
        "model": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "model_name": "Phi-4-mini",
        "context_size": 1024,
        "max_tokens": 50,
        "threads": 4,
        "desc": "Fast agent (RTX 3080, low latency)",
    },
    {
        "id": "agent-quality",
        "vm_ip": "192.168.1.61",
        "gpu": "rtx3080",
        "profile": "quality",
        "llama_port": 8083,
        "agent_port": 8091,
        "model": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "model_name": "Phi-4-mini",
        "context_size": 4096,
        "max_tokens": 500,
        "threads": 4,
        "desc": "Quality agent (RTX 3080, high quality)",
    },
    {
        "id": "agent-balanced",
        "vm_ip": "192.168.1.61",
        "gpu": "rtx3080",
        "profile": "balanced",
        "llama_port": 8084,
        "agent_port": 8092,
        "model": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "model_name": "Phi-4-mini",
        "context_size": 2048,
        "max_tokens": 200,
        "threads": 4,
        "desc": "Balanced agent (RTX 3080)",
    },
    {
        "id": "agent-backup-1",
        "vm_ip": "192.168.1.60",
        "gpu": "rx6600m",
        "profile": "backup",
        "llama_port": 8082,
        "agent_port": 8081,
        "model": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "model_name": "Phi-4-mini",
        "context_size": 2048,
        "max_tokens": 200,
        "threads": 4,
        "desc": "Backup agent 1 (RX 6600M)",
    },
    {
        "id": "agent-backup-2",
        "vm_ip": "192.168.1.60",
        "gpu": "rx6600m",
        "profile": "backup",
        "llama_port": 8083,
        "agent_port": 8091,
        "model": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "model_name": "Phi-4-mini",
        "context_size": 1024,
        "max_tokens": 100,
        "threads": 4,
        "desc": "Backup agent 2 (RX 6600M)",
    },
]


# ─── SSH Helper ──────────────────────────────────────────────────────────────

_ssh_cache: Dict[str, paramiko.SSHClient] = {}


def ssh_connect(ip: str) -> paramiko.SSHClient:
    if ip in _ssh_cache:
        try:
            t = _ssh_cache[ip].get_transport()
            if t and t.is_active():
                _ssh_cache[ip].exec_command("echo ok", timeout=3)
                return _ssh_cache[ip]
        except Exception:
            try:
                _ssh_cache[ip].close()
            except Exception:
                pass
            del _ssh_cache[ip]

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect((ip, 22))
    ssh.connect(ip, username="oldds", password="Admin@123", timeout=30, sock=sock)
    ssh.get_transport().set_keepalive(30)
    _ssh_cache[ip] = ssh
    return ssh


def ssh_run(ip: str, cmd: str, timeout: int = 30) -> Tuple[int, str, str]:
    ssh = ssh_connect(ip)
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    ec = o.channel.recv_exit_status()
    return ec, o.read().decode('utf-8', errors='replace'), e.read().decode('utf-8', errors='replace')


def ssh_bg(ip: str, cmd: str, logfile: str = "/dev/null", env: Dict = None):
    ssh = ssh_connect(ip)
    if env:
        env_str = " ".join(f"{k}={v}" for k, v in env.items())
        full = f"bash -c 'export {env_str} && nohup {cmd} > {logfile} 2>&1 &'"
    else:
        full = f"nohup {cmd} > {logfile} 2>&1 &"
    ssh.exec_command(full, timeout=10)


# ─── Deploy Functions ────────────────────────────────────────────────────────

def stop_all():
    """Stop all services on all VMs."""
    print("Stopping all services...")
    for ip in ["192.168.1.60", "192.168.1.61", "192.168.1.62"]:
        try:
            ssh_run(ip, 'pkill -9 -f "llama-server|agent_llm|main.py" 2>/dev/null; true', 5)
            print(f"  {ip}: stopped")
        except Exception as e:
            print(f"  {ip}: error ({e})")
    time.sleep(3)


def start_llama_servers():
    """Start llama-server instances on both GPUs."""
    print("\nStarting llama-servers...")

    # Group agents by VM
    vms: Dict[str, List[dict]] = {}
    for agent in AGENTS:
        vms.setdefault(agent["vm_ip"], []).append(agent)

    for vm_ip, vm_agents in vms.items():
        for agent in vm_agents:
            binary = f"{BASE_DIR}/llama.cpp_dds/build/bin/llama-server"
            model_path = f"{MODELS_DIR}/{agent['model']}"
            cmd = (f"{binary} -m {model_path} "
                   f"-c {agent['context_size']} "
                   f"--threads {agent['threads']} "
                   f"-ngl 99 --reasoning-budget 0 "
                   f"--port {agent['llama_port']} --host 0.0.0.0 "
                   f"--enable-dds --dds-domain 0 --dds-timeout 120")
            logfile = f"/tmp/llama_{agent['id']}.log"
            ssh_bg(vm_ip, cmd, logfile)
            print(f"  {vm_ip}:{agent['llama_port']} - {agent['desc']}")

    # Wait for health
    print("  Waiting for health checks...")
    for agent in AGENTS:
        for i in range(15):
            time.sleep(2)
            ec, out, _ = ssh_run(agent["vm_ip"],
                                  f"curl -s http://localhost:{agent['llama_port']}/health", 5)
            if ec == 0 and "ok" in out.lower():
                break
        else:
            print(f"  WARNING: {agent['id']} not ready after 30s")
    print("  All llama-servers started")


def start_orchestrator(fuzzy: bool = False, dds: bool = True):
    """Start orchestrator on .62."""
    print(f"\nStarting orchestrator (fuzzy={'ON' if fuzzy else 'OFF'})...")

    xml = f"{BASE_DIR}/llama.cpp_dds/dds/cyclonedds-network-ultra.xml"
    cmd = (f"python3 -u {BASE_DIR}/dds_orchestrator/main.py "
           f"--port {ORCH_PORT} --log-level INFO")
    env = {}

    if dds:
        cmd += " --dds-domain 0"
        env["CYCLONEDDS_URI"] = f"file://{xml}"
    else:
        env["DDS_ENABLED"] = "false"

    if fuzzy:
        cmd += " --fuzzy"

    ssh_bg(ORCH_IP, cmd, "/tmp/orch_fuzzy.log", env=env)
    time.sleep(8)

    # Verify
    ec, out, _ = ssh_run(ORCH_IP, f"curl -s http://localhost:{ORCH_PORT}/health", 5)
    if ec == 0 and out.strip():
        print(f"  Orchestrator ready on {ORCH_IP}:{ORCH_PORT}")
    else:
        print(f"  WARNING: Orchestrator may not be ready")


def start_agents(transport: str = "dds"):
    """Start agent proxies for all 5 agents."""
    print(f"\nStarting {len(AGENTS)} agents ({transport})...")

    xml = f"{BASE_DIR}/llama.cpp_dds/dds/cyclonedds-network-ultra.xml"

    for agent in AGENTS:
        vm_ip = agent["vm_ip"]

        if transport == "dds":
            agent_script = f"{BASE_DIR}/dds_agent/python/agent_llm_dds.py"
            cmd = (f"python3 -u {agent_script} "
                   f"--model-name {agent['model_name']} "
                   f"--model-path {MODELS_DIR}/{agent['model']} "
                   f"--orchestrator-url http://{ORCH_IP}:{ORCH_PORT} "
                   f"--port {agent['agent_port']} "
                   f"--llama-server-port {agent['llama_port']} "
                   f"--no-server")
            env = {"CYCLONEDDS_URI": f"file://{xml}"}
        elif transport == "http":
            agent_script = f"{BASE_DIR}/dds_agent/python/agent_llm.py"
            cmd = f"python3 -u {agent_script}"
            env = {
                "MODEL_PATH": f"{MODELS_DIR}/{agent['model']}",
                "MODEL_NAME": agent["model_name"],
                "LLAMA_SERVER_PATH": f"{BASE_DIR}/llama.cpp_dds/build/bin/llama-server",
                "LLAMA_SERVER_PORT": str(agent["llama_port"]),
                "AGENT_PORT": str(agent["agent_port"]),
                "ORCHESTRATOR_URL": f"http://{ORCH_IP}:{ORCH_PORT}",
                "GPU_LAYERS": "99",
                "HOSTNAME": vm_ip,
            }
        else:
            print(f"  Unknown transport: {transport}")
            continue

        logfile = f"/tmp/agent_{agent['id']}.log"
        ssh_bg(vm_ip, cmd, logfile, env=env)
        print(f"  {agent['id']} on {vm_ip}:{agent['agent_port']} ({agent['profile']})")

    # Wait for registration
    print("  Waiting for agent registration...")
    time.sleep(15)
    ec, out, _ = ssh_run(ORCH_IP, f"curl -s http://localhost:{ORCH_PORT}/api/v1/agents", 10)
    count = out.count("agent_id")
    print(f"  {count}/{len(AGENTS)} agents registered")


def check_status():
    """Check status of all services."""
    print("\n=== Status ===")

    # Orchestrator
    try:
        ec, out, _ = ssh_run(ORCH_IP, f"curl -s http://localhost:{ORCH_PORT}/health", 5)
        print(f"  Orchestrator ({ORCH_IP}): {'OK' if 'ok' in out.lower() else 'DOWN'}")
    except:
        print(f"  Orchestrator ({ORCH_IP}): UNREACHABLE")

    # Agents
    try:
        ec, out, _ = ssh_run(ORCH_IP, f"curl -s http://localhost:{ORCH_PORT}/api/v1/agents", 5)
        count = out.count("agent_id")
        print(f"  Registered agents: {count}/{len(AGENTS)}")
    except:
        print(f"  Agents: UNKNOWN")

    # llama-servers
    for agent in AGENTS:
        try:
            ec, out, _ = ssh_run(agent["vm_ip"],
                                  f"curl -s http://localhost:{agent['llama_port']}/health", 3)
            status = "OK" if "ok" in out.lower() else "DOWN"
        except:
            status = "UNREACHABLE"
        print(f"  {agent['id']:20s} llama:{agent['llama_port']} = {status}")

    # GPU usage
    try:
        _, out, _ = ssh_run("192.168.1.61",
            'echo "Admin@123" | sudo -S nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null', 5)
        print(f"  RTX 3080 VRAM: {out.strip()}")
    except:
        pass
    try:
        _, out, _ = ssh_run("192.168.1.60", "rocm-smi --showmeminfo vram 2>/dev/null | grep Used", 5)
        print(f"  RX 6600M VRAM: {out.strip()}")
    except:
        pass


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deploy 5 LLM Agents")
    parser.add_argument("--start", action="store_true", help="Start all services")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    parser.add_argument("--status", action="store_true", help="Check status")
    parser.add_argument("--fuzzy", action="store_true", help="Enable fuzzy orchestrator")
    parser.add_argument("--transport", choices=["dds", "http"], default="dds")
    parser.add_argument("--pull", action="store_true", help="Git pull before start")
    args = parser.parse_args()

    if args.stop:
        stop_all()
    elif args.status:
        check_status()
    elif args.start:
        stop_all()
        if args.pull:
            print("Git pull...")
            for ip in ["192.168.1.60", "192.168.1.61", "192.168.1.62"]:
                for repo in ["dds_orchestrator", "dds_agent"]:
                    ssh_run(ip, f"cd {BASE_DIR}/{repo} && git fetch origin && git reset --hard origin/main 2>/dev/null", 30)
            print("  Done")
        start_llama_servers()
        start_orchestrator(fuzzy=args.fuzzy, dds=(args.transport == "dds"))
        start_agents(transport=args.transport)
        check_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
