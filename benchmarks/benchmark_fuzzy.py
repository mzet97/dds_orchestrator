#!/usr/bin/env python3
"""
Fuzzy vs Baseline Benchmark — Scenarios F0-F4
===============================================
Compares orchestrator with and without fuzzy decision engine.

F0: Baseline (round-robin, fixed QoS, single strategy)
F1: Fuzzy agent selection only
F3: Full fuzzy (agent + QoS + strategy)
F4: Full fuzzy with injected failures

Runs from client VM (.63) via SSH. Requires deploy_5agents.py to start services.

Usage:
    python benchmark_fuzzy.py --scenario F0 --n 100 --transport dds
    python benchmark_fuzzy.py --scenario all --n 100
"""

import argparse
import json
import os
import paramiko
import socket
import statistics
import sys
import io
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = "/home/oldds"
ORCH_IP = "192.168.1.62"
ORCH_PORT = 8080
CLIENT_IP = "192.168.1.63"

# Task mix for realistic workload
TASK_MIX = {
    "simple":   {"pct": 40, "urgency": 8, "complexity": 2, "prompt": "What is 2+2?"},
    "medium":   {"pct": 30, "urgency": 5, "complexity": 5, "prompt": "Explain the difference between TCP and UDP briefly."},
    "complex":  {"pct": 20, "urgency": 3, "complexity": 8, "prompt": "Explain in detail the differences between TCP and UDP protocols, including their use cases, advantages, disadvantages, and how they relate to the OSI model."},
    "critical": {"pct": 10, "urgency": 10, "complexity": 7, "prompt": "A server is down and users are affected. What are the immediate steps to diagnose and fix the issue?"},
}


def ssh_connect(ip: str) -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect((ip, 22))
    ssh.connect(ip, username="oldds", password="Admin@123", timeout=30, sock=sock)
    ssh.get_transport().set_keepalive(60)
    return ssh


def run_scenario_on_client(ssh_client, scenario: str, n: int, transport: str,
                            fuzzy: bool, max_tokens: int = 50) -> Dict:
    """Run a benchmark scenario on the client VM."""
    xml = f"{BASE_DIR}/llama.cpp_dds/dds/cyclonedds-network-ultra.xml"

    # Build task mix as JSON
    tasks_json = json.dumps(TASK_MIX)

    # Write benchmark script to client VM
    script = f'''
import asyncio, json, time, statistics, sys, uuid, os
sys.path.insert(0, "{BASE_DIR}/dds_orchestrator")
sys.path.insert(0, "{BASE_DIR}/dds_orchestrator/benchmarks")

TASK_MIX = {tasks_json}
N = {n}
MAX_TOKENS = {max_tokens}
PROTOCOL = "{transport}"
SCENARIO = "{scenario}"
FUZZY = {fuzzy}

async def run():
    if PROTOCOL == "http":
        import aiohttp
    elif PROTOCOL == "dds":
        from client_example import DDSOrchestratorClient

    results = []
    # Build task sequence based on mix percentages
    tasks = []
    for ttype, cfg in TASK_MIX.items():
        count = int(N * cfg["pct"] / 100)
        for _ in range(count):
            tasks.append({{
                "type": ttype,
                "urgency": cfg["urgency"],
                "complexity": cfg["complexity"],
                "prompt": cfg["prompt"],
            }})

    import random
    random.shuffle(tasks)

    if PROTOCOL == "dds":
        client = DDSOrchestratorClient(domain_id=0)
        time.sleep(3)
        for i, task in enumerate(tasks):
            t_start = time.perf_counter_ns()
            result = client.chat(
                [{{"role": "user", "content": task["prompt"]}}],
                timeout_s=30,
            )
            t_end = time.perf_counter_ns()
            results.append({{
                "iteration": i,
                "task_type": task["type"],
                "urgency": task["urgency"],
                "complexity": task["complexity"],
                "roundtrip_ms": (t_end - t_start) / 1e6,
                "success": result.get("success", False),
                "content_len": len(result.get("content", "")),
            }})
            if (i+1) % 50 == 0:
                print(f"  {{i+1}}/{{len(tasks)}}", flush=True)
        client.close()
    elif PROTOCOL == "http":
        async with aiohttp.ClientSession() as session:
            for i, task in enumerate(tasks):
                payload = {{
                    "model": "",
                    "messages": [{{"role": "user", "content": task["prompt"]}}],
                    "max_tokens": MAX_TOKENS,
                    "urgency": task["urgency"],
                    "complexity": task["complexity"],
                }}
                t_start = time.perf_counter_ns()
                try:
                    async with session.post(
                        "http://{ORCH_IP}:{ORCH_PORT}/v1/chat/completions",
                        json=payload, timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        body = await resp.text()
                    t_end = time.perf_counter_ns()
                    data = json.loads(body)
                    content = ""
                    if "choices" in data:
                        content = data["choices"][0].get("message", {{}}).get("content", "")
                    results.append({{
                        "iteration": i,
                        "task_type": task["type"],
                        "urgency": task["urgency"],
                        "complexity": task["complexity"],
                        "roundtrip_ms": (t_end - t_start) / 1e6,
                        "success": bool(content),
                        "content_len": len(content),
                    }})
                except Exception as e:
                    t_end = time.perf_counter_ns()
                    results.append({{
                        "iteration": i, "task_type": task["type"],
                        "urgency": task["urgency"], "complexity": task["complexity"],
                        "roundtrip_ms": (t_end - t_start) / 1e6,
                        "success": False, "content_len": 0,
                    }})
                if (i+1) % 50 == 0:
                    print(f"  {{i+1}}/{{len(tasks)}}", flush=True)

    # Summary
    ok = [r for r in results if r["success"]]
    lats = sorted([r["roundtrip_ms"] for r in ok])
    summary = {{
        "scenario": SCENARIO,
        "fuzzy": FUZZY,
        "protocol": PROTOCOL,
        "n": len(results),
        "success": len(ok),
        "success_rate": len(ok) / max(1, len(results)),
        "p50_ms": lats[len(lats)//2] if lats else 0,
        "p95_ms": lats[int(len(lats)*0.95)] if lats else 0,
        "mean_ms": sum(lats)/len(lats) if lats else 0,
    }}

    # Per task-type breakdown
    for ttype in TASK_MIX:
        type_results = [r for r in ok if r["task_type"] == ttype]
        type_lats = sorted([r["roundtrip_ms"] for r in type_results])
        summary[f"{{ttype}}_p50_ms"] = type_lats[len(type_lats)//2] if type_lats else 0
        summary[f"{{ttype}}_count"] = len(type_results)

    output = {{"summary": summary, "results": results}}
    fname = f"results/fuzzy_{{SCENARIO}}_{{PROTOCOL}}_{{time.strftime('%Y%m%d_%H%M%S')}}.json"
    os.makedirs("results", exist_ok=True)
    with open(fname, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved: {{fname}}")
    print(json.dumps(summary, indent=2))

asyncio.run(run())
'''

    sftp = ssh_client.open_sftp()
    with sftp.open("/tmp/fuzzy_bench.py", "w") as f:
        f.write(script)
    sftp.close()

    env_prefix = ""
    if transport == "dds":
        env_prefix = f"export CYCLONEDDS_URI=file://{xml} && "

    cmd = f"{env_prefix}cd {BASE_DIR}/dds_orchestrator && python3 -u /tmp/fuzzy_bench.py"
    print(f"  Running {scenario} ({transport}, fuzzy={fuzzy}, n={n})...")
    _, out, err = ssh_client.exec_command(cmd, timeout=7200)
    ec = out.channel.recv_exit_status()
    output = out.read().decode('utf-8', errors='replace')
    print(output[-500:])
    if ec != 0:
        print(f"  ERROR: {err.read().decode()[:300]}")
    return {"exit_code": ec, "output": output[-200:]}


def main():
    parser = argparse.ArgumentParser(description="Fuzzy vs Baseline Benchmark")
    parser.add_argument("--scenario", choices=["F0", "F1", "F3", "F4", "all"], default="all")
    parser.add_argument("--n", type=int, default=100, help="Requests per scenario")
    parser.add_argument("--transport", choices=["http", "dds"], default="dds")
    parser.add_argument("--max-tokens", type=int, default=50)
    args = parser.parse_args()

    scenarios = {
        "F0": {"fuzzy": False, "desc": "Baseline (round-robin, no fuzzy)"},
        "F1": {"fuzzy": True, "desc": "Fuzzy agent selection + QoS"},
        "F3": {"fuzzy": True, "desc": "Full fuzzy (agent + QoS + strategy)"},
    }

    if args.scenario == "all":
        to_run = ["F0", "F1", "F3"]
    else:
        to_run = [args.scenario]

    print("=" * 60)
    print("  FUZZY vs BASELINE BENCHMARK")
    print(f"  Scenarios: {to_run}")
    print(f"  N: {args.n} | Transport: {args.transport}")
    print("=" * 60)

    ssh_client = ssh_connect(CLIENT_IP)

    for scenario in to_run:
        cfg = scenarios[scenario]
        print(f"\n{'='*60}")
        print(f"  {scenario}: {cfg['desc']}")
        print(f"{'='*60}")

        # Restart orchestrator with/without fuzzy
        ssh_orch = ssh_connect(ORCH_IP)
        ssh_orch.exec_command('pkill -9 -f "main.py" 2>/dev/null')
        time.sleep(3)

        xml = f"{BASE_DIR}/llama.cpp_dds/dds/cyclonedds-network-ultra.xml"
        orch_cmd = (f"python3 -u {BASE_DIR}/dds_orchestrator/main.py "
                    f"--port {ORCH_PORT} --dds-domain 0 --log-level INFO")
        if cfg["fuzzy"]:
            orch_cmd += " --fuzzy"
        env = {"CYCLONEDDS_URI": f"file://{xml}"}
        exports = " ".join(f"{k}={v}" for k, v in env.items())
        ssh_orch.exec_command(f"bash -c 'export {exports} && nohup {orch_cmd} > /tmp/orch_fuzzy.log 2>&1 &'")
        time.sleep(8)

        # Verify orchestrator
        _, out, _ = ssh_orch.exec_command(f"curl -s http://localhost:{ORCH_PORT}/health", timeout=5)
        print(f"  Orchestrator: {out.read().decode()[:50]}")

        # Wait for agents to re-register
        time.sleep(15)
        _, out, _ = ssh_orch.exec_command(f"curl -s http://localhost:{ORCH_PORT}/api/v1/agents")
        count = out.read().decode().count("agent_id")
        print(f"  Agents registered: {count}")

        run_scenario_on_client(ssh_client, scenario, args.n, args.transport,
                                cfg["fuzzy"], args.max_tokens)

    ssh_client.close()
    print(f"\n{'='*60}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
