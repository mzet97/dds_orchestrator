#!/usr/bin/env python3
"""
Deploy real instances on .61, run real inference tests.
3 GPU + 3 CPU instances, 6 agents, orchestrator with Redis+MongoDB.
"""

import os
import sys
import time
import json
import paramiko

# === Config ===
VM_HOST = "192.168.1.61"
VM_USER = "oldds"
VM_PASSWORD = "Admin@123"

K8S_HOST = "192.168.1.51"
K8S_USER = "k8s1"
REDIS_URL = "redis://192.168.1.51:30379"
MONGO_URL = "mongodb://admin:Admin%40123@192.168.1.51:27017/?authSource=admin"

MODEL = "Qwen3.5-2B-UD-IQ2_XXS.gguf"
MODEL_PATH = f"/home/{VM_USER}/models/{MODEL}"
LLAMA_SERVER = f"/home/{VM_USER}/llama.cpp_dds/build/bin/llama-server"
REMOTE_DIR = f"/home/{VM_USER}/tese_deploy"
DDS_CONFIG_PATH = f"{REMOTE_DIR}/cyclonedds-38inst.xml"

# 3 GPU + 3 CPU = 6 instances (safe for 31GB RAM + 10GB VRAM)
GPU_INSTANCES = [
    {"port": 8082, "agent_port": 9082, "ngl": 99, "parallel": 4, "ctx": 512, "threads": 2},
    {"port": 8083, "agent_port": 9083, "ngl": 99, "parallel": 4, "ctx": 512, "threads": 2},
    {"port": 8084, "agent_port": 9084, "ngl": 99, "parallel": 4, "ctx": 512, "threads": 2},
]
CPU_INSTANCES = [
    {"port": 8092, "agent_port": 9092, "ngl": 0, "parallel": 2, "ctx": 256, "threads": 4},
    {"port": 8093, "agent_port": 9093, "ngl": 0, "parallel": 2, "ctx": 256, "threads": 4},
    {"port": 8094, "agent_port": 9094, "ngl": 0, "parallel": 2, "ctx": 256, "threads": 4},
]
ALL_INSTANCES = GPU_INSTANCES + CPU_INSTANCES
ORCH_PORT = 8080

# Files to copy (relative to dds_orchestrator/)
ORCH_FILES = [
    "redis_layer.py", "mongo_layer.py", "instance_pool.py", "backpressure.py",
    "config.py", "server.py", "main.py", "dds.py", "registry.py",
    "scheduler.py", "selector.py",
    "orchestrator/__init__.py", "orchestrator/_OrchestratorDDS.py",
    "cyclonedds-38inst.xml",
]
AGENT_FILES = [
    ("../dds_agent/python/agent_llm_dds.py", "agent/agent_llm_dds.py"),
    ("../dds_agent/orchestrator/__init__.py", "agent/orchestrator/__init__.py"),
    ("../dds_agent/orchestrator/_OrchestratorDDS.py", "agent/orchestrator/_OrchestratorDDS.py"),
]
# LlamaDDS IDL needed by agent
LLAMA_IDL_FILES = [
    ("../llama.cpp_dds/dds/idl/llama/__init__.py", "agent/llama/__init__.py"),
    ("../llama.cpp_dds/dds/idl/llama/_LlamaDDS.py", "agent/llama/_LlamaDDS.py"),
]


def get_client(host=VM_HOST, user=VM_USER):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=VM_PASSWORD, timeout=15)
    return c


def ssh_exec(client, cmd, timeout=30):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err, code


def ssh_bg(client, cmd, log_file=None):
    redirect = f"> {log_file} 2>&1" if log_file else "> /dev/null 2>&1"
    # Use bash -c to handle env vars in the command
    escaped = cmd.replace("'", "'\\''")
    client.exec_command(f"nohup bash -c '{escaped}' {redirect} &")
    time.sleep(0.3)


def step(msg):
    print(f"\n{'-'*60}")
    print(f"  {msg}")
    print(f"{'-'*60}")


def upload_files(client):
    """Upload orchestrator, agent, and IDL files."""
    step("Uploading code files")
    sftp = client.open_sftp()
    base = os.path.join(os.path.dirname(__file__), "..")

    # Create directories
    for d in [REMOTE_DIR, f"{REMOTE_DIR}/orchestrator", f"{REMOTE_DIR}/agent",
              f"{REMOTE_DIR}/agent/orchestrator", f"{REMOTE_DIR}/agent/llama"]:
        try:
            sftp.stat(d)
        except FileNotFoundError:
            sftp.mkdir(d)

    # Orchestrator files
    for f in ORCH_FILES:
        local = os.path.join(base, f)
        remote = f"{REMOTE_DIR}/{f}"
        if os.path.exists(local):
            sftp.put(local, remote)
            print(f"  {f}")

    # Agent files
    for src, dst in AGENT_FILES + LLAMA_IDL_FILES:
        local = os.path.join(base, src)
        remote = f"{REMOTE_DIR}/{dst}"
        if os.path.exists(local):
            sftp.put(local, remote)
            print(f"  {src} -> {dst}")
        else:
            print(f"  SKIP: {src}")

    sftp.close()


def install_deps(client):
    """Install Python dependencies."""
    step("Installing Python dependencies")
    _, _, code = ssh_exec(client, "python3 -c 'import redis; import motor; import aiohttp'", timeout=5)
    if code == 0:
        print("  Already installed")
        return

    # Try pip install
    out, err, code = ssh_exec(client,
        "python3 -m pip install --user --break-system-packages "
        "'redis[hiredis]' motor pymongo aiohttp pyyaml pydantic 2>&1",
        timeout=120)
    if code != 0:
        # Fallback: venv
        ssh_exec(client, f"python3 -m venv {REMOTE_DIR}/.venv", timeout=30)
        out, err, code = ssh_exec(client,
            f"{REMOTE_DIR}/.venv/bin/pip install "
            "'redis[hiredis]' motor pymongo aiohttp pyyaml pydantic 2>&1",
            timeout=120)
        if code != 0:
            print(f"  ERROR: {err or out}")
            return
    print("  OK")


def get_python(client):
    """Get the correct python path (venv or system)."""
    venv_python = f"{REMOTE_DIR}/.venv/bin/python"
    out, _, code = ssh_exec(client, f"test -f {venv_python} && echo YES", timeout=5)
    if "YES" in out:
        return venv_python
    return "python3"


def kill_all(client):
    """Stop all running instances."""
    step("Killing existing processes")
    ssh_exec(client, f"echo '{VM_PASSWORD}' | sudo -S killall -9 llama-server python3 2>/dev/null || true", timeout=10)
    time.sleep(3)
    # Verify ports are free
    out, _, _ = ssh_exec(client, "ss -tlnp | grep -E '808[0-9]|809[0-9]' || echo FREE")
    print(f"  Ports: {out}")


def start_instances(client):
    """Start llama-server instances."""
    step(f"Starting {len(ALL_INSTANCES)} llama-server instances")
    env = f"CYCLONEDDS_URI=file://{DDS_CONFIG_PATH}"

    for inst in GPU_INSTANCES:
        log = f"/tmp/llama_{inst['port']}.log"
        cmd = (f"{env} {LLAMA_SERVER} -m {MODEL_PATH} "
               f"-c {inst['ctx']} --threads {inst['threads']} "
               f"-ngl {inst['ngl']} --parallel {inst['parallel']} "
               f"--port {inst['port']} --host 0.0.0.0")
        ssh_bg(client, cmd, log_file=log)
        print(f"  GPU :{inst['port']} (ngl={inst['ngl']}, parallel={inst['parallel']})")

    for inst in CPU_INSTANCES:
        log = f"/tmp/llama_{inst['port']}.log"
        cmd = (f"{env} {LLAMA_SERVER} -m {MODEL_PATH} "
               f"-c {inst['ctx']} --threads {inst['threads']} "
               f"-ngl {inst['ngl']} --parallel {inst['parallel']} "
               f"--port {inst['port']} --host 0.0.0.0")
        ssh_bg(client, cmd, log_file=log)
        print(f"  CPU :{inst['port']} (threads={inst['threads']}, parallel={inst['parallel']})")

    print(f"  Waiting 10s for model loading...")
    time.sleep(10)


def wait_health(client, timeout_s=120):
    """Wait for all instances to be healthy."""
    step(f"Waiting for health (timeout {timeout_s}s)")
    healthy = set()
    start = time.time()

    while time.time() - start < timeout_s:
        for inst in ALL_INSTANCES:
            if inst["port"] in healthy:
                continue
            out, _, code = ssh_exec(client,
                f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{inst['port']}/health",
                timeout=5)
            if out.strip() == "200":
                healthy.add(inst["port"])
                elapsed = int(time.time() - start)
                print(f"  :{inst['port']} healthy ({elapsed}s)")

        if len(healthy) == len(ALL_INSTANCES):
            break
        time.sleep(3)

    failed = [i["port"] for i in ALL_INSTANCES if i["port"] not in healthy]
    if failed:
        print(f"  FAILED: {failed}")
        # Show logs of failed instances
        for port in failed[:2]:  # show max 2
            log, _, _ = ssh_exec(client, f"tail -10 /tmp/llama_{port}.log 2>/dev/null || echo 'no log'", timeout=5)
            print(f"  Log :{port}:\n{log}")
    return len(failed) == 0


def start_orchestrator(client, python):
    """Start orchestrator with Redis + MongoDB + InstancePool."""
    step("Starting orchestrator")
    gpu_ports = ",".join(str(i["port"]) for i in GPU_INSTANCES)
    cpu_ports = ",".join(str(i["port"]) for i in CPU_INSTANCES)
    env = f"CYCLONEDDS_URI=file://{DDS_CONFIG_PATH}"

    cmd = (f"cd {REMOTE_DIR} && {env} {python} main.py "
           f"--port {ORCH_PORT} --host 0.0.0.0 "
           f"--redis-url {REDIS_URL} --redis-password {VM_PASSWORD} "
           f"--mongo-url '{MONGO_URL}' "
           f"--instance-ports-gpu {gpu_ports} "
           f"--instance-ports-cpu {cpu_ports} "
           f"--routing-algorithm least_loaded "
           f"--log-level INFO "
           f"--dds-domain 0")
    ssh_bg(client, cmd, log_file="/tmp/orchestrator.log")
    time.sleep(5)

    # Verify orchestrator is up
    out, _, code = ssh_exec(client,
        f"curl -s http://127.0.0.1:{ORCH_PORT}/health", timeout=5)
    if "healthy" in out:
        print(f"  Orchestrator :{ORCH_PORT} healthy")
        return True
    else:
        print(f"  Orchestrator NOT healthy: {out}")
        # Show log
        log, _, _ = ssh_exec(client, "tail -20 /tmp/orchestrator.log", timeout=5)
        print(f"  Log:\n{log}")
        return False


def send_chat(client, prompt, max_tokens=20):
    """Send a real chat request and return the response."""
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    })
    cmd = (f"curl -s -X POST http://127.0.0.1:{ORCH_PORT}/api/v1/chat/completions "
           f"-H 'Content-Type: application/json' "
           f"-d '{payload}'")
    out, _, code = ssh_exec(client, cmd, timeout=60)
    try:
        return json.loads(out)
    except Exception:
        return {"error": out, "curl_exit": code}


def run_real_tests(client):
    """Send actual inference requests through the full stack."""
    step("Running REAL inference tests")
    results = []

    prompts = [
        ("Hello!", 10),
        ("What is DDS?", 15),
        ("Count from 1 to 5.", 15),
    ]

    for prompt, max_tok in prompts:
        t0 = time.time()
        resp = send_chat(client, prompt, max_tok)
        latency = (time.time() - t0) * 1000

        content = ""
        if "choices" in resp:
            msg = resp["choices"][0].get("message", {})
            content = msg.get("content", "") or msg.get("reasoning_content", "")
        error = resp.get("error", "")
        inst_port = resp.get("instance_port", "?")
        inst_type = resp.get("instance_type", "?")

        success = bool(content)
        results.append(success)

        status = "OK" if success else "FAIL"
        print(f"  [{status}] \"{prompt}\" -> port:{inst_port} ({inst_type}) {latency:.0f}ms")
        if content:
            preview = content.replace("\n", " ")[:80]
            print(f"         \"{preview}\"")
        if error:
            print(f"         ERROR: {error}")
        if not content and not error:
            print(f"         FULL RESPONSE: {json.dumps(resp)[:200]}")

    return results


def check_pool_status(client):
    """Check instance pool status via API."""
    step("Pool status")
    out, _, _ = ssh_exec(client,
        f"curl -s http://127.0.0.1:{ORCH_PORT}/api/v1/pool/status", timeout=5)
    try:
        data = json.loads(out)
        print(f"  Algorithm: {data.get('algorithm')}")
        print(f"  Instances: {data.get('total_instances')}")
        print(f"  Active: {data.get('active_requests')}")
        print(f"  Pressure: {data.get('pressure_level')}")
        for inst in data.get("instances", []):
            print(f"    :{inst['port']} {inst['type']} "
                  f"slots={inst['slots_used']}/{inst['slots_total']} "
                  f"latency={inst['avg_latency']:.0f}ms "
                  f"healthy={inst['healthy']}")
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Raw: {out[:200]}")


def check_vram(client):
    """Check VRAM usage."""
    out, _, _ = ssh_exec(client,
        "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader", timeout=5)
    print(f"  VRAM: {out}")


def main():
    print("=" * 60)
    print("  DEPLOY & TEST: Real GPU Inference")
    print(f"  VM: {VM_HOST} | Redis: {K8S_HOST}:30379 | MongoDB: {K8S_HOST}:27017")
    print("=" * 60)

    client = get_client()
    python = "python3"

    try:
        # 1. Kill existing
        kill_all(client)

        # 2. Upload code
        upload_files(client)

        # 3. Install deps
        install_deps(client)
        python = get_python(client)
        print(f"  Using: {python}")

        # 4. Start llama-server instances
        start_instances(client)

        # 5. Wait for health
        if not wait_health(client, timeout_s=120):
            print("\nABORT: Not all instances healthy")
            return 1

        # 6. Check VRAM
        step("VRAM check")
        check_vram(client)

        # 7. Start orchestrator
        if not start_orchestrator(client, python):
            print("\nABORT: Orchestrator failed to start")
            return 1

        # 8. Check pool status
        check_pool_status(client)

        # 9. Send REAL inference requests
        results = run_real_tests(client)

        # 10. Check pool after tests
        check_pool_status(client)

        # Summary
        step("RESULTS")
        passed = sum(results)
        total = len(results)
        check_vram(client)

        print(f"\n  {passed}/{total} inference tests passed")
        if passed == total:
            print("  ALL TESTS PASSED WITH REAL GPU INFERENCE!")
        else:
            print("  SOME TESTS FAILED")

        return 0 if passed == total else 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        step("Cleanup")
        kill_all(client)
        client.close()


if __name__ == "__main__":
    sys.exit(main())
