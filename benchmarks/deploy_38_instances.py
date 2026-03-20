#!/usr/bin/env python3
"""
Deploy 38 llama.cpp instances + 38 agents on 192.168.1.61
10 GPU instances (ports 8082-8091) + 28 CPU instances (ports 8092-8119)
"""

import argparse
import os
import sys
import time

# ===== Configuration =====

HOST = "192.168.1.61"
USER = "oldds"
PASSWORD = "Admin@123"

MODEL = "Qwen3.5-2B-UD-IQ2_XXS.gguf"
MODEL_PATH = f"/home/{USER}/models/{MODEL}"
LLAMA_SERVER = f"/home/{USER}/llama.cpp_dds/build/bin/llama-server"
DDS_CONFIG = f"/home/{USER}/llama.cpp_dds/dds/cyclonedds-38inst.xml"
AGENT_SCRIPT = f"/home/{USER}/tese/dds_agent/python/agent_llm_dds.py"
ORCHESTRATOR_DIR = f"/home/{USER}/tese/dds_orchestrator"

GPU_INSTANCES = [
    {"port": 8082 + i, "agent_port": 9082 + i, "ngl": 99,
     "parallel": 15, "threads": 2, "ctx_size": 512}
    for i in range(10)
]

CPU_INSTANCES = [
    {"port": 8092 + i, "agent_port": 9092 + i, "ngl": 0,
     "parallel": 4, "threads": 2, "ctx_size": 256}
    for i in range(28)
]

ALL_INSTANCES = GPU_INSTANCES + CPU_INSTANCES


def get_ssh_client():
    """Create paramiko SSH client."""
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    return client


def ssh_exec(client, cmd, timeout=30, check=False):
    """Execute command via SSH and return stdout."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    if check and exit_code != 0:
        raise RuntimeError(f"Command failed ({exit_code}): {cmd}\n{err}")
    return out, err, exit_code


def ssh_bg(client, cmd):
    """Run command in background via SSH (nohup + disown)."""
    bg_cmd = f"nohup {cmd} > /dev/null 2>&1 &"
    client.exec_command(bg_cmd)
    time.sleep(0.1)


def stop_all(client):
    """Kill all llama-server and agent processes."""
    print("Stopping all processes...")
    ssh_exec(client, "pkill -f llama-server || true", timeout=10)
    ssh_exec(client, "pkill -f agent_llm_dds || true", timeout=10)
    ssh_exec(client, "pkill -f 'python.*main.py' || true", timeout=10)
    time.sleep(2)
    print("All processes stopped")


def deploy_model(client):
    """Ensure model exists on remote host."""
    out, _, code = ssh_exec(client, f"test -f {MODEL_PATH} && echo EXISTS")
    if "EXISTS" in out:
        print(f"Model already exists: {MODEL_PATH}")
        return
    print(f"Model not found at {MODEL_PATH}")
    print("Please copy the model manually:")
    print(f"  scp models/{MODEL} {USER}@{HOST}:{MODEL_PATH}")
    sys.exit(1)


def build_llama_server(client):
    """Build llama-server with DDS support."""
    print("Building llama-server...")
    build_dir = f"/home/{USER}/llama.cpp_dds/build"
    ssh_exec(client, f"mkdir -p {build_dir}", timeout=10)
    ssh_exec(client,
             f"cd {build_dir} && cmake .. -DLLAMA_DDS=ON -DCMAKE_BUILD_TYPE=Release",
             timeout=60, check=True)
    ssh_exec(client, f"cd {build_dir} && make -j$(nproc)", timeout=300, check=True)
    print("Build complete")


def start_all_instances(client):
    """Start 38 llama-server instances."""
    env = f"CYCLONEDDS_URI=file://{DDS_CONFIG}"

    for inst in GPU_INSTANCES:
        cmd = (f"{env} {LLAMA_SERVER} "
               f"-m {MODEL_PATH} "
               f"-c {inst['ctx_size']} "
               f"--threads {inst['threads']} "
               f"-ngl {inst['ngl']} "
               f"--parallel {inst['parallel']} "
               f"--port {inst['port']} "
               f"--host 0.0.0.0 "
               f"--enable-dds")
        print(f"  Starting GPU instance :{inst['port']} (ngl={inst['ngl']}, parallel={inst['parallel']})")
        ssh_bg(client, cmd)

    for inst in CPU_INSTANCES:
        cmd = (f"{env} {LLAMA_SERVER} "
               f"-m {MODEL_PATH} "
               f"-c {inst['ctx_size']} "
               f"--threads {inst['threads']} "
               f"-ngl {inst['ngl']} "
               f"--parallel {inst['parallel']} "
               f"--port {inst['port']} "
               f"--host 0.0.0.0 "
               f"--enable-dds")
        print(f"  Starting CPU instance :{inst['port']} (threads={inst['threads']}, parallel={inst['parallel']})")
        ssh_bg(client, cmd)

    print(f"Started {len(ALL_INSTANCES)} llama-server instances")


def start_all_agents(client):
    """Start 38 agent processes (one per llama-server)."""
    env = f"CYCLONEDDS_URI=file://{DDS_CONFIG}"

    for inst in ALL_INSTANCES:
        agent_id = f"agent-inst-{inst['port']}"
        cmd = (f"{env} python3 {AGENT_SCRIPT} "
               f"--model-name Qwen3.5-2B "
               f"--llama-server-port {inst['port']} "
               f"--port {inst['agent_port']} "
               f"--orchestrator-url http://127.0.0.1:8080 "
               f"--no-server")
        # Set agent_id via environment
        cmd = f"AGENT_ID={agent_id} {cmd}"
        print(f"  Starting agent {agent_id} (llama:{inst['port']}, agent:{inst['agent_port']})")
        ssh_bg(client, cmd)

    print(f"Started {len(ALL_INSTANCES)} agents")


def start_orchestrator(client, redis_url="", mongo_url="",
                       routing_algorithm="least_loaded"):
    """Start orchestrator with full configuration."""
    env = f"CYCLONEDDS_URI=file://{DDS_CONFIG}"

    gpu_ports = ",".join(str(inst["port"]) for inst in GPU_INSTANCES)
    cpu_ports = ",".join(str(inst["port"]) for inst in CPU_INSTANCES)

    cmd = (f"cd {ORCHESTRATOR_DIR} && {env} python3 main.py "
           f"--port 8080 --host 0.0.0.0 "
           f"--log-level INFO")

    if redis_url:
        cmd += f" --redis-url {redis_url} --redis-password Admin@123"
    if mongo_url:
        cmd += f" --mongo-url '{mongo_url}'"
    if gpu_ports:
        cmd += f" --instance-ports-gpu {gpu_ports}"
    if cpu_ports:
        cmd += f" --instance-ports-cpu {cpu_ports}"
    cmd += f" --routing-algorithm {routing_algorithm}"

    print(f"Starting orchestrator on :8080")
    ssh_bg(client, cmd)


def verify_health(client, timeout_s=30):
    """Verify all instances are healthy."""
    print(f"Verifying health of {len(ALL_INSTANCES)} instances (timeout={timeout_s}s)...")
    healthy = set()
    start = time.time()

    while time.time() - start < timeout_s:
        for inst in ALL_INSTANCES:
            if inst["port"] in healthy:
                continue
            out, _, code = ssh_exec(
                client,
                f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{inst['port']}/health",
                timeout=5,
            )
            if out.strip() == "200":
                healthy.add(inst["port"])
        if len(healthy) == len(ALL_INSTANCES):
            break
        time.sleep(2)

    failed = [inst["port"] for inst in ALL_INSTANCES if inst["port"] not in healthy]
    print(f"Healthy: {len(healthy)}/{len(ALL_INSTANCES)}")
    if failed:
        print(f"Failed: {failed}")
    return len(failed) == 0


def verify_vram(client):
    """Check VRAM usage via nvidia-smi."""
    out, _, _ = ssh_exec(client, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits", timeout=10)
    if out:
        used, total = out.strip().split(",")
        used_mb = int(used.strip())
        total_mb = int(total.strip())
        print(f"VRAM: {used_mb}MB / {total_mb}MB ({used_mb/total_mb*100:.1f}%)")
        if used_mb > 9500:
            print("WARNING: VRAM usage > 9.5GB!")
            return False
    return True


def verify_ram(client):
    """Check RAM usage."""
    out, _, _ = ssh_exec(client, "free -h | grep Mem", timeout=10)
    if out:
        print(f"RAM: {out}")
    return True


def get_status(client):
    """Get current status of deployment."""
    print("\n=== Deployment Status ===")

    # Count processes
    out, _, _ = ssh_exec(client, "pgrep -c llama-server || echo 0", timeout=10)
    llama_count = int(out.strip())
    print(f"llama-server processes: {llama_count}")

    out, _, _ = ssh_exec(client, "pgrep -cf agent_llm_dds || echo 0", timeout=10)
    agent_count = int(out.strip())
    print(f"Agent processes: {agent_count}")

    out, _, _ = ssh_exec(client, "pgrep -cf 'python.*main.py' || echo 0", timeout=10)
    orch_count = int(out.strip())
    print(f"Orchestrator processes: {orch_count}")

    # VRAM
    verify_vram(client)
    verify_ram(client)


def full_deploy(redis_url="", mongo_url="", routing_algorithm="least_loaded"):
    """Full deployment sequence."""
    client = get_ssh_client()
    try:
        print("=" * 60)
        print("38-Instance Deployment")
        print("=" * 60)

        # 1. Stop existing
        stop_all(client)

        # 2. Check model
        deploy_model(client)

        # 3. Start llama-servers
        print("\n--- Starting llama-server instances ---")
        start_all_instances(client)

        # 4. Wait for health
        print("\n--- Waiting for instances to be ready ---")
        time.sleep(10)
        if not verify_health(client, timeout_s=120):
            print("ERROR: Not all instances healthy!")
            return False

        # 5. Start orchestrator
        print("\n--- Starting orchestrator ---")
        start_orchestrator(client, redis_url, mongo_url, routing_algorithm)
        time.sleep(3)

        # 6. Start agents
        print("\n--- Starting agents ---")
        start_all_agents(client)
        time.sleep(5)

        # 7. Verify
        print("\n--- Final verification ---")
        verify_vram(client)
        verify_ram(client)
        get_status(client)

        print("\n" + "=" * 60)
        print("Deployment complete!")
        print("=" * 60)
        return True
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="Deploy 38 llama.cpp instances")
    parser.add_argument("--deploy", action="store_true", help="Full deployment")
    parser.add_argument("--stop", action="store_true", help="Stop all processes")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--verify", action="store_true", help="Verify health only")
    parser.add_argument("--build", action="store_true", help="Build llama-server")
    parser.add_argument("--redis-url", type=str, default="redis://redis.home.arpa:6379",
                       help="Redis URL")
    parser.add_argument("--mongo-url", type=str,
                       default="mongodb://admin:Admin%40123@mongodb.home.arpa:27017/?authSource=admin",
                       help="MongoDB URL")
    parser.add_argument("--routing-algorithm", type=str, default="least_loaded",
                       choices=["round_robin", "least_loaded", "weighted_score"])

    args = parser.parse_args()

    if args.stop:
        client = get_ssh_client()
        stop_all(client)
        client.close()
    elif args.status:
        client = get_ssh_client()
        get_status(client)
        client.close()
    elif args.verify:
        client = get_ssh_client()
        verify_health(client, timeout_s=30)
        client.close()
    elif args.build:
        client = get_ssh_client()
        build_llama_server(client)
        client.close()
    elif args.deploy:
        full_deploy(args.redis_url, args.mongo_url, args.routing_algorithm)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
