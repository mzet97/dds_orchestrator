#!/usr/bin/env python3
"""
Full E2E Benchmark: HTTP vs gRPC vs DDS
========================================
Deploys services on VMs for each protocol, runs E1-E5 from client VM (.63),
saves results, and generates comparison plots.

Each protocol runs end-to-end:
  HTTP:  client --HTTP--> orchestrator --HTTP--> agent --HTTP--> llama-server
  gRPC:  client --gRPC--> orchestrator --gRPC--> agent --gRPC--> llama-server
  DDS:   client --DDS-->  orchestrator --DDS-->  agent --DDS-->  llama-server
"""

import paramiko
import socket
import time
import io
import sys
import json
import os
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = "/home/oldds"
ORCH_IP = "192.168.1.62"
AGENT_IP = "192.168.1.61"
CLIENT_IP = "192.168.1.63"
XML = f"{BASE}/llama.cpp_dds/dds/cyclonedds-network-optimized.xml"
MODEL = "Qwen3.5-0.8B-UD-IQ2_XXS.gguf"
MODEL_NAME = "Qwen3.5-0.8B"

N = 100  # Requests per scenario


# ─── SSH ─────────────────────────────────────────────────────────────────────

_ssh_cache = {}

def ssh_connect(ip):
    if ip in _ssh_cache:
        try:
            t = _ssh_cache[ip].get_transport()
            if t and t.is_active():
                _ssh_cache[ip].exec_command("echo ok", timeout=3)
                return _ssh_cache[ip]
        except Exception:
            try: _ssh_cache[ip].close()
            except: pass
            del _ssh_cache[ip]
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect((ip, 22))
    ssh.connect(ip, username='oldds', password='Admin@123', timeout=30, sock=sock)
    ssh.get_transport().set_keepalive(30)
    _ssh_cache[ip] = ssh
    return ssh


def ssh_run(ip, cmd, timeout=60):
    ssh = ssh_connect(ip)
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    ec = o.channel.recv_exit_status()
    return ec, o.read().decode('utf-8', errors='replace'), e.read().decode('utf-8', errors='replace')


def ssh_bg(ip, cmd, logfile="/dev/null", env=None):
    ssh = ssh_connect(ip)
    if env:
        env_str = " ".join(f"{k}={v}" for k, v in env.items())
        full = f"bash -c 'export {env_str} && nohup {cmd} > {logfile} 2>&1 &'"
    else:
        full = f"nohup {cmd} > {logfile} 2>&1 &"
    ssh.exec_command(full, timeout=10)


# ─── Service Management ─────────────────────────────────────────────────────

def stop_all():
    print("  Stopping all services...")
    for ip in [AGENT_IP, ORCH_IP]:
        try:
            ssh_run(ip, 'pkill -9 -f "llama-server|agent_llm|main.py|grpc_layer" 2>/dev/null; true', 5)
        except Exception:
            pass
    time.sleep(3)


def pull_latest():
    print("Pulling latest code on all VMs...")
    for ip in [ORCH_IP, AGENT_IP, CLIENT_IP]:
        try:
            for repo in ["dds_orchestrator", "dds_agent"]:
                ssh_run(ip, f"cd {BASE}/{repo} && git fetch origin && git reset --hard origin/main 2>&1 | tail -1", 30)
            print(f"  {ip}: OK")
        except Exception as e:
            print(f"  {ip}: {e}")


def start_llama_server():
    """Start llama-server with DDS enabled (always, for all protocols)."""
    print("  Starting llama-server on .61 (RTX 3080)...")
    cmd = (f"{BASE}/llama.cpp_dds/build/bin/llama-server "
           f"-m {BASE}/models/{MODEL} "
           f"-c 1024 --threads 4 -ngl 99 --reasoning-budget 0 "
           f"--port 8082 --host 0.0.0.0 "
           f"--enable-dds --dds-domain 0 --dds-timeout 120")
    ssh_bg(AGENT_IP, cmd, "/tmp/llama_bench.log",
           {"CYCLONEDDS_URI": f"file://{XML}"})

    for i in range(20):
        time.sleep(2)
        ec, out, _ = ssh_run(AGENT_IP, "curl -s http://localhost:8082/health", 5)
        if ec == 0 and "ok" in out.lower():
            print(f"  llama-server: OK ({(i+1)*2}s)")
            return True
    print("  llama-server: FAILED")
    return False


def start_orchestrator(protocol):
    """Start orchestrator configured for the given protocol."""
    print(f"  Starting orchestrator ({protocol})...")
    cmd = f"python3 -u {BASE}/dds_orchestrator/main.py --port 8080 --log-level INFO"
    env = {}

    if protocol == "dds":
        cmd += " --dds-domain 0"
        env["CYCLONEDDS_URI"] = f"file://{XML}"
    elif protocol == "grpc":
        # gRPC needs DDS disabled, gRPC enabled
        env["DDS_ENABLED"] = "false"
        cmd += " --grpc-enabled"
    else:  # http
        env["DDS_ENABLED"] = "false"

    ssh_bg(ORCH_IP, cmd, f"/tmp/orch_{protocol}.log", env)
    time.sleep(8)

    ec, out, _ = ssh_run(ORCH_IP, "curl -s http://localhost:8080/health", 5)
    ok = ec == 0 and out.strip()
    print(f"  orchestrator: {'OK' if ok else 'FAILED'}")
    return ok


def start_agent(protocol):
    """Start agent for the given protocol."""
    print(f"  Starting agent ({protocol})...")

    if protocol == "dds":
        cmd = (f"python3 -u {BASE}/dds_agent/python/agent_llm_dds.py "
               f"--model-name {MODEL_NAME} "
               f"--model-path {BASE}/models/{MODEL} "
               f"--orchestrator-url http://{ORCH_IP}:8080 "
               f"--port 8081 --llama-server-port 8082 --no-server")
        env = {"CYCLONEDDS_URI": f"file://{XML}"}
    elif protocol == "grpc":
        cmd = (f"python3 -u {BASE}/dds_agent/python/agent_llm_grpc.py "
               f"--model-name {MODEL_NAME} "
               f"--model-path {BASE}/models/{MODEL} "
               f"--orchestrator-url http://{ORCH_IP}:8080 "
               f"--port 8081 --llama-server-port 8082 --no-server")
        env = {"HOSTNAME": AGENT_IP}
    else:  # http
        cmd = f"python3 -u {BASE}/dds_agent/python/agent_llm.py"
        env = {
            "MODEL_PATH": f"{BASE}/models/{MODEL}",
            "MODEL_NAME": MODEL_NAME,
            "LLAMA_SERVER_PATH": f"{BASE}/llama.cpp_dds/build/bin/llama-server",
            "LLAMA_SERVER_PORT": "8082",
            "AGENT_PORT": "8081",
            "ORCHESTRATOR_URL": f"http://{ORCH_IP}:8080",
            "GPU_LAYERS": "99",
            "HOSTNAME": AGENT_IP,
            "NO_SERVER": "1",
        }

    ssh_bg(AGENT_IP, cmd, f"/tmp/agent_{protocol}.log", env)
    time.sleep(12)

    ec, out, _ = ssh_run(ORCH_IP, "curl -s http://localhost:8080/api/v1/agents", 10)
    count = out.count("agent_id")
    print(f"  agents registered: {count}")
    return count > 0


def run_client_benchmark(protocol, n):
    """Run e2e_benchmark_client.py on client VM (.63)."""
    print(f"\n  Running {protocol.upper()} benchmark (n={n}) from {CLIENT_IP}...")

    if protocol == "http":
        proto_args = f"--protocol http --url http://{ORCH_IP}:8080"
    elif protocol == "grpc":
        proto_args = f"--protocol grpc --endpoint {ORCH_IP}:50052"
    elif protocol == "dds":
        proto_args = f"--protocol dds --domain 0"
    else:
        return None

    cmd = (f"cd {BASE}/dds_orchestrator/benchmarks && "
           f"CYCLONEDDS_URI=file://{XML} "
           f"python3 e2e_benchmark_client.py {proto_args} "
           f"--model {MODEL_NAME} --scenario all --n {n} --max-tokens 50")

    ec, out, err = ssh_run(CLIENT_IP, cmd, timeout=7200)
    print(out)
    if err:
        # Only show relevant errors
        for line in err.split('\n'):
            if 'Error' in line or 'Traceback' in line or 'error' in line.lower():
                print(f"  STDERR: {line}")
                break

    # Find and retrieve results file
    ec2, result_files, _ = ssh_run(CLIENT_IP,
        f"ls -t {BASE}/dds_orchestrator/benchmarks/results/e2e_{protocol}_*.json 2>/dev/null | head -1", 10)
    result_file = result_files.strip()
    if result_file:
        ec3, content, _ = ssh_run(CLIENT_IP, f"cat {result_file}", 10)
        try:
            return json.loads(content)
        except:
            pass
    return None


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"{'='*60}")
    print(f"  Full E2E Benchmark: HTTP vs gRPC vs DDS")
    print(f"  {ts} | n={N} | Model: {MODEL_NAME}")
    print(f"  RTX 3080 (.61) | Orchestrator (.62) | Client (.63)")
    print(f"{'='*60}\n")

    pull_latest()

    all_results = {}
    protocols = ["http", "grpc", "dds"]

    for protocol in protocols:
        print(f"\n{'='*60}")
        print(f"  PROTOCOL: {protocol.upper()}")
        print(f"  Path: client→{protocol}→orchestrator→{protocol}→agent→{protocol}→llama-server")
        print(f"{'='*60}")

        stop_all()

        if not start_llama_server():
            print(f"  SKIP {protocol}: llama-server failed")
            continue

        if not start_orchestrator(protocol):
            print(f"  SKIP {protocol}: orchestrator failed")
            continue

        if not start_agent(protocol):
            print(f"  SKIP {protocol}: agent failed")
            continue

        result = run_client_benchmark(protocol, N)
        if result:
            all_results[protocol] = result

    stop_all()

    # Save combined results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    combined_file = results_dir / f"e2e_combined_{ts}.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results saved to {combined_file}")

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'HTTP':>10} {'gRPC':>10} {'DDS':>10}")
    print(f"{'-'*55}")

    for scenario in ["E1", "E3", "E4", "E5"]:
        for protocol in protocols:
            r = all_results.get(protocol, {}).get(scenario)
            if not r:
                continue

            if scenario == "E1":
                for ptype in ["short", "long"]:
                    data = r.get(ptype, [])
                    lats = [d["roundtrip_ms"] for d in data if d.get("success")]
                    if lats:
                        lats.sort()
                        p50 = lats[len(lats)//2]
                        label = f"E1 {ptype} p50"
                        vals = {protocol: f"{p50:.1f}ms"}
                        # Collect other protocols
                        for p2 in protocols:
                            if p2 != protocol:
                                d2 = all_results.get(p2, {}).get("E1", {}).get(ptype, [])
                                l2 = [d["roundtrip_ms"] for d in d2 if d.get("success")]
                                if l2:
                                    l2.sort()
                                    vals[p2] = f"{l2[len(l2)//2]:.1f}ms"
                        if len(vals) == len(protocols):
                            print(f"{label:<25} {vals.get('http','N/A'):>10} {vals.get('grpc','N/A'):>10} {vals.get('dds','N/A'):>10}")
                    break  # Only print once per type

            elif scenario == "E5":
                ttfts = [d["ttft_ms"] for d in r if d.get("ttft_ms", 0) > 0]
                if ttfts:
                    ttfts.sort()
                    label = "E5 TTFT p50"
                    vals = {protocol: f"{ttfts[len(ttfts)//2]:.1f}ms"}
                    for p2 in protocols:
                        if p2 != protocol:
                            d2 = all_results.get(p2, {}).get("E5", [])
                            t2 = [d["ttft_ms"] for d in d2 if d.get("ttft_ms", 0) > 0]
                            if t2:
                                t2.sort()
                                vals[p2] = f"{t2[len(t2)//2]:.1f}ms"
                    if len(vals) == len(protocols):
                        print(f"{label:<25} {vals.get('http','N/A'):>10} {vals.get('grpc','N/A'):>10} {vals.get('dds','N/A'):>10}")
                break

    print(f"\n{'='*60}")
    print("  ALL BENCHMARKS COMPLETE")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    main()
