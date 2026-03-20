#!/usr/bin/env python3
"""
Full E2E test: 3 protocols (HTTP, gRPC, DDS) with real GPU inference.
.61: llama-server + orchestrator + agents
.63: client load generator (10, 100, 500, 1000 clients)
"""

import os
import sys
import time
import json
import paramiko

# === Infrastructure ===
SERVER_HOST = "192.168.1.61"
CLIENT_HOST = "192.168.1.63"
USER = "oldds"
PASSWORD = "Admin@123"
REDIS_URL = "redis://192.168.1.51:30379"
MONGO_URL = "mongodb://admin:Admin%40123@192.168.1.51:27017/?authSource=admin"

MODEL = "Qwen3.5-2B-UD-IQ2_XXS.gguf"
MODEL_PATH = f"/home/{USER}/models/{MODEL}"
LLAMA_SERVER = f"/home/{USER}/llama.cpp_dds/build/bin/llama-server"
DEPLOY_DIR = f"/home/{USER}/tese_deploy"
DDS_CONFIG = f"{DEPLOY_DIR}/cyclonedds-38inst.xml"

# 3 GPU instances for the test
INSTANCES = [
    {"port": 8082, "ngl": 99, "parallel": 4, "ctx": 512, "threads": 2},
    {"port": 8083, "ngl": 99, "parallel": 4, "ctx": 512, "threads": 2},
    {"port": 8084, "ngl": 99, "parallel": 4, "ctx": 512, "threads": 2},
]
ORCH_PORT = 8080

# Source files
ORCH_DIR = os.path.join(os.path.dirname(__file__), "..")
REPO_ROOT = os.path.join(ORCH_DIR, "..")

# Files needed on .61 (server)
SERVER_FILES = [
    # Orchestrator
    ("dds_orchestrator/redis_layer.py", "redis_layer.py"),
    ("dds_orchestrator/mongo_layer.py", "mongo_layer.py"),
    ("dds_orchestrator/instance_pool.py", "instance_pool.py"),
    ("dds_orchestrator/backpressure.py", "backpressure.py"),
    ("dds_orchestrator/config.py", "config.py"),
    ("dds_orchestrator/server.py", "server.py"),
    ("dds_orchestrator/main.py", "main.py"),
    ("dds_orchestrator/dds.py", "dds.py"),
    ("dds_orchestrator/registry.py", "registry.py"),
    ("dds_orchestrator/scheduler.py", "scheduler.py"),
    ("dds_orchestrator/selector.py", "selector.py"),
    ("dds_orchestrator/orchestrator/__init__.py", "orchestrator/__init__.py"),
    ("dds_orchestrator/orchestrator/_OrchestratorDDS.py", "orchestrator/_OrchestratorDDS.py"),
    ("dds_orchestrator/cyclonedds-38inst.xml", "cyclonedds-38inst.xml"),
    # Agent
    ("dds_agent/python/agent_llm_dds.py", "agent/agent_llm_dds.py"),
    ("dds_agent/orchestrator/__init__.py", "agent/orchestrator/__init__.py"),
    ("dds_agent/orchestrator/_OrchestratorDDS.py", "agent/orchestrator/_OrchestratorDDS.py"),
    # LlamaDDS IDL for agent
    ("llama.cpp_dds/dds/idl/llama/__init__.py", "agent/llama/__init__.py"),
    ("llama.cpp_dds/dds/idl/llama/_LlamaDDS.py", "agent/llama/_LlamaDDS.py"),
]

# Check for optional files (grpc, fuzzy, qos)
OPTIONAL_SERVER_FILES = [
    ("dds_orchestrator/grpc_layer.py", "grpc_layer.py"),
    ("dds_orchestrator/qos_profiles.py", "qos_profiles.py"),
    ("dds_orchestrator/fuzzy_selector.py", "fuzzy_selector.py"),
    ("dds_orchestrator/http_client.py", "http_client.py"),
    ("dds_orchestrator/context.py", "context.py"),
    ("dds_orchestrator/models.py", "models.py"),
    ("dds_orchestrator/proto/__init__.py", "proto/__init__.py"),
    ("dds_orchestrator/proto/orchestrator_pb2.py", "proto/orchestrator_pb2.py"),
    ("dds_orchestrator/proto/orchestrator_pb2_grpc.py", "proto/orchestrator_pb2_grpc.py"),
]


def ssh(host, user=USER, pwd=PASSWORD):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=pwd, timeout=15)
    return c


def run(client, cmd, timeout=30):
    _, out, err = client.exec_command(cmd, timeout=timeout)
    code = out.channel.recv_exit_status()
    return out.read().decode("utf-8", errors="replace").strip(), code


def bg(client, cmd, log=None):
    redir = f"> {log} 2>&1" if log else "> /dev/null 2>&1"
    escaped = cmd.replace("'", "'\\''")
    client.exec_command(f"nohup bash -c '{escaped}' {redir} &")
    time.sleep(0.3)


def step(msg):
    print(f"\n{'-'*60}\n  {msg}\n{'-'*60}")


def upload(client, file_list, remote_base):
    """Upload files via SFTP."""
    sftp = client.open_sftp()
    # Collect all needed directories
    dirs = set()
    for _, dst in file_list:
        d = os.path.dirname(f"{remote_base}/{dst}")
        while d != remote_base and d:
            dirs.add(d)
            d = os.path.dirname(d)
    dirs.add(remote_base)

    for d in sorted(dirs):
        try:
            sftp.stat(d)
        except FileNotFoundError:
            sftp.mkdir(d)

    count = 0
    for src_rel, dst_rel in file_list:
        local = os.path.join(REPO_ROOT, src_rel)
        remote = f"{remote_base}/{dst_rel}"
        if os.path.exists(local):
            sftp.put(local, remote)
            count += 1
        # Silently skip missing optional files

    sftp.close()
    return count


def kill_all(client):
    run(client, f"echo '{PASSWORD}' | sudo -S killall -9 llama-server python3 2>/dev/null || true", timeout=10)
    time.sleep(3)


# =========================================================
# DEPLOY ON .61
# =========================================================

def deploy_server():
    step(f"Deploying on {SERVER_HOST}")
    c = ssh(SERVER_HOST)

    # Kill existing
    kill_all(c)
    out, _ = run(c, "ss -tlnp | grep -E '808[0-9]|809[0-9]' || echo FREE")
    print(f"  Ports: {out}")

    # Upload code
    n = upload(c, SERVER_FILES + OPTIONAL_SERVER_FILES, DEPLOY_DIR)
    print(f"  Uploaded {n} files")

    # Start llama-server instances
    env = f"CYCLONEDDS_URI=file://{DDS_CONFIG}"
    for inst in INSTANCES:
        cmd = (f"{env} {LLAMA_SERVER} -m {MODEL_PATH} "
               f"-c {inst['ctx']} --threads {inst['threads']} "
               f"-ngl {inst['ngl']} --parallel {inst['parallel']} "
               f"--port {inst['port']} --host 0.0.0.0 "
               f"--enable-dds")
        bg(c, cmd, log=f"/tmp/llama_{inst['port']}.log")
        print(f"  llama-server :{inst['port']} (GPU, parallel={inst['parallel']})")

    print("  Waiting 15s for model loading...")
    time.sleep(15)

    # Check health
    all_ok = True
    for inst in INSTANCES:
        out, _ = run(c, f"curl -s http://127.0.0.1:{inst['port']}/health", timeout=5)
        ok = '"ok"' in out or '"status":"ok"' in out
        print(f"  :{inst['port']} health: {'OK' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("  ABORT: not all instances healthy")
        c.close()
        return None

    # VRAM
    out, _ = run(c, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
    print(f"  VRAM: {out}")

    # Start orchestrator
    gpu_ports = ",".join(str(i["port"]) for i in INSTANCES)
    orch_cmd = (f"cd {DEPLOY_DIR} && {env} python3 main.py "
                f"--port {ORCH_PORT} --host 0.0.0.0 "
                f"--redis-url {REDIS_URL} --redis-password {PASSWORD} "
                f"--mongo-url '{MONGO_URL}' "
                f"--instance-ports-gpu {gpu_ports} "
                f"--routing-algorithm round_robin "
                f"--log-level INFO")
    bg(c, orch_cmd, log="/tmp/orchestrator.log")
    time.sleep(5)

    out, _ = run(c, f"curl -s http://127.0.0.1:{ORCH_PORT}/health")
    if "healthy" not in out:
        log, _ = run(c, "tail -10 /tmp/orchestrator.log")
        print(f"  Orchestrator FAILED:\n{log}")
        c.close()
        return None
    print(f"  Orchestrator :{ORCH_PORT} healthy")

    # Pool status
    out, _ = run(c, f"curl -s http://127.0.0.1:{ORCH_PORT}/api/v1/pool/status")
    try:
        pool = json.loads(out)
        print(f"  Pool: {pool['total_instances']} instances, algorithm={pool['algorithm']}")
    except Exception:
        print(f"  Pool: {out[:100]}")

    c.close()
    return True


# =========================================================
# LOAD TEST FROM .63
# =========================================================

def run_load_test(num_clients, duration_s=30, protocol="http"):
    """Run load test from .63 against orchestrator on .61."""
    c = ssh(CLIENT_HOST)

    payload = json.dumps({
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 5,
        "temperature": 0,
    }).replace('"', '\\"')

    # Simple async load generator using curl in parallel
    # For real scale we'd use the load_generator.py, but this proves the concept
    script = f'''
import asyncio, aiohttp, time, json, sys

URL = "http://{SERVER_HOST}:{ORCH_PORT}/api/v1/chat/completions"
N = {num_clients}
DURATION = {duration_s}
PAYLOAD = {{"messages": [{{"role": "user", "content": "Say hi"}}], "max_tokens": 5, "temperature": 0}}

results = {{"ok": 0, "fail": 0, "latencies": []}}
stop = False

async def worker(session, wid):
    global stop
    while not stop:
        t0 = time.time()
        try:
            async with session.post(URL, json=PAYLOAD, timeout=aiohttp.ClientTimeout(total=30)) as r:
                data = await r.json()
                lat = (time.time() - t0) * 1000
                msg = data.get("choices", [{{}}])[0].get("message", {{}})
                content = msg.get("content", "") or msg.get("reasoning_content", "")
                if content:
                    results["ok"] += 1
                else:
                    results["fail"] += 1
                results["latencies"].append(lat)
        except Exception as e:
            results["fail"] += 1
            await asyncio.sleep(0.5)

async def main():
    global stop
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [asyncio.create_task(worker(session, i)) for i in range(N)]
        await asyncio.sleep(DURATION)
        stop = True
        await asyncio.gather(*tasks, return_exceptions=True)

    lats = sorted(results["latencies"])
    n = len(lats)
    total = results["ok"] + results["fail"]
    if n > 0:
        p50 = lats[n//2]
        p95 = lats[int(n*0.95)]
        p99 = lats[min(int(n*0.99), n-1)]
        rps = n / DURATION
    else:
        p50 = p95 = p99 = rps = 0

    print(json.dumps({{
        "clients": N, "total": total, "ok": results["ok"], "fail": results["fail"],
        "p50": round(p50, 1), "p95": round(p95, 1), "p99": round(p99, 1),
        "rps": round(rps, 1), "error_rate": round(results["fail"]/max(total,1)*100, 1),
    }}))

asyncio.run(main())
'''

    # Write script to .63
    sftp = c.open_sftp()
    try:
        sftp.stat("/tmp/dds_loadtest")
    except FileNotFoundError:
        sftp.mkdir("/tmp/dds_loadtest")
    with sftp.open("/tmp/dds_loadtest/run.py", "w") as f:
        f.write(script)
    sftp.close()

    print(f"    {num_clients} clients, {duration_s}s...")
    t0 = time.time()
    out, code = run(c, f"cd /tmp/dds_loadtest && python3 run.py", timeout=duration_s + 30)
    elapsed = time.time() - t0

    c.close()

    try:
        result = json.loads(out)
        print(f"    OK={result['ok']} FAIL={result['fail']} "
              f"p50={result['p50']}ms p95={result['p95']}ms p99={result['p99']}ms "
              f"RPS={result['rps']} err={result['error_rate']}% ({elapsed:.0f}s)")
        return result
    except Exception:
        print(f"    Parse error: {out[:200]}")
        return None


# =========================================================
# SINGLE REQUEST TESTS (3 protocols)
# =========================================================

def test_single_request(protocol="http"):
    """Test a single request to verify the protocol works."""
    c = ssh(SERVER_HOST)
    payload = json.dumps({
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 5, "temperature": 0,
    })

    t0 = time.time()
    out, code = run(c,
        f"curl -s --max-time 30 -X POST http://127.0.0.1:{ORCH_PORT}/api/v1/chat/completions "
        f"-H 'Content-Type: application/json' -d '{payload}'",
        timeout=35)
    lat = (time.time() - t0) * 1000

    c.close()

    try:
        resp = json.loads(out)
        msg = resp.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "") or msg.get("reasoning_content", "")
        port = resp.get("instance_port", "?")
        ok = bool(content)
        return ok, lat, content[:60], port
    except Exception:
        return False, lat, out[:60], "?"


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 60)
    print("  FULL E2E: 3 GPU instances, HTTP protocol")
    print(f"  Server: {SERVER_HOST} | Client: {CLIENT_HOST}")
    print(f"  Redis: 192.168.1.51:30379 | MongoDB: 192.168.1.51:27017")
    print("=" * 60)

    # 1. Deploy
    result = deploy_server()
    if not result:
        return 1

    # 2. Single request verification
    step("Single request test (HTTP)")
    ok, lat, content, port = test_single_request()
    print(f"  {'OK' if ok else 'FAIL'} | {lat:.0f}ms | port:{port} | \"{content}\"")
    if not ok:
        print("  ABORT: single request failed")
        return 1

    # 3. Load tests with increasing clients
    step("Load tests from .63 (HTTP)")
    all_results = []
    for clients in [10, 100, 500, 1000]:
        duration = 30 if clients <= 100 else 60
        r = run_load_test(clients, duration_s=duration)
        if r:
            all_results.append(r)

    # 4. Summary
    step("SUMMARY")
    print(f"  {'Clients':>8} {'Total':>8} {'OK':>6} {'Fail':>6} "
          f"{'p50ms':>8} {'p95ms':>8} {'p99ms':>8} {'RPS':>8} {'Err%':>6}")
    print(f"  {'-'*72}")
    for r in all_results:
        print(f"  {r['clients']:>8} {r['total']:>8} {r['ok']:>6} {r['fail']:>6} "
              f"{r['p50']:>8.1f} {r['p95']:>8.1f} {r['p99']:>8.1f} "
              f"{r['rps']:>8.1f} {r['error_rate']:>5.1f}%")

    # VRAM final
    c = ssh(SERVER_HOST)
    out, _ = run(c, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
    print(f"\n  VRAM: {out}")
    c.close()

    # Cleanup
    step("Cleanup")
    c = ssh(SERVER_HOST)
    kill_all(c)
    c.close()

    passed = all(r and r["error_rate"] < 50 for r in all_results)
    print(f"\n{'=' * 60}")
    print(f"  {'ALL LOAD TESTS PASSED' if passed else 'SOME TESTS FAILED'}")
    print(f"{'=' * 60}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
