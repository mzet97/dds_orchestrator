#!/usr/bin/env python3
"""
Deploy 10 llama.cpp_dds GPU instances across 2 machines + orchestrator on .62.

Topology:
  .61 (RTX 3080 10GB): 6 instances, ports 8082-8087, parallel=15
  .60 (RX 6600M 8GB):  4 instances, ports 8088-8091, parallel=10
  .62: Orchestrator
  .51: Redis + MongoDB (k8s)

All SSH via paramiko (no bash ssh from Windows).
"""

import argparse
import sys
import time

# ===== Credentials =====

USER = "oldds"
PASSWORD = "Admin@123"

# ===== Model =====

MODEL = "Qwen3.5-2B-UD-IQ2_XXS.gguf"
MODEL_NAME = "Qwen3.5-2B"

# ===== Host Configuration =====

HOSTS = {
    "192.168.1.61": {
        "gpu_type": "rtx3080",
        "vram_gb": 10,
        "vram_warn_mb": 9500,
        "build_flags": "-DLLAMA_DDS=ON -DCMAKE_BUILD_TYPE=Release",
        "vram_cmd": "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits",
        "instances": [
            {"port": 8082 + i, "ngl": 99, "parallel": 15, "ctx_size": 512}
            for i in range(5)  # 5 instances (~1.8GB VRAM each = ~9GB of 10GB)
        ],
    },
    "192.168.1.60": {
        "gpu_type": "rx6600m_vulkan",  # Vulkan backend (ROCm gfx1032 unsupported)
        "vram_gb": 8,
        "vram_warn_mb": 7500,
        "build_flags": "-DGGML_VULKAN=ON -DLLAMA_DDS=ON -DCMAKE_PREFIX_PATH=/home/oldds/cyclonedds/install -DCMAKE_BUILD_TYPE=Release",
        "vram_cmd": "cat /sys/class/drm/card0/device/mem_info_vram_used 2>/dev/null && cat /sys/class/drm/card0/device/mem_info_vram_total 2>/dev/null || echo 'N/A'",
        "extra_env": "",
        "instances": [
            {"port": 8088 + i, "ngl": 99, "parallel": 10, "ctx_size": 512}
            for i in range(4)
        ],
    },
}

ORCHESTRATOR_HOST = "192.168.1.62"

# ===== Paths (relative to /home/oldds) =====

HOME = f"/home/{USER}"
MODEL_PATH = f"{HOME}/models/{MODEL}"
LLAMA_SERVER = f"{HOME}/llama.cpp_dds/build/bin/llama-server"
BUILD_DIR = f"{HOME}/llama.cpp_dds/build"
SRC_DIR = f"{HOME}/llama.cpp_dds"
DDS_CONFIG_NAME = "cyclonedds-10inst-network.xml"
DDS_CONFIG_LOCAL = f"{HOME}/tese/dds_orchestrator/configs/{DDS_CONFIG_NAME}"
AGENT_SCRIPT = f"{HOME}/tese/dds_agent/python/agent_llm_dds.py"
ORCHESTRATOR_DIR = f"{HOME}/tese/dds_orchestrator"

# ===== Redis / MongoDB defaults =====

DEFAULT_REDIS_URL = "redis://redis.home.arpa:6379"
DEFAULT_REDIS_PASSWORD = "Admin@123"
DEFAULT_MONGO_URL = "mongodb://admin:Admin%40123@mongodb.home.arpa:27017/?authSource=admin"


def get_ssh_client(host: str):
    """Create paramiko SSH client for a given host."""
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=USER, password=PASSWORD, timeout=30)
    return client


def ssh_exec(client, cmd: str, timeout: int = 30, check: bool = False):
    """Execute command via SSH and return (stdout, stderr, exit_code)."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    if check and exit_code != 0:
        raise RuntimeError(f"Command failed ({exit_code}): {cmd}\n{err}")
    return out, err, exit_code


def ssh_bg(client, cmd: str, log_file: str = "/dev/null"):
    """Run command in background via SSH (bash -c with export + nohup)."""
    # Split env vars from command: env vars come before the binary path
    parts = cmd.split()
    env_parts = []
    cmd_parts = []
    for i, p in enumerate(parts):
        if "=" in p and not p.startswith("-") and not p.startswith("/"):
            env_parts.append(f"export {p}")
        else:
            cmd_parts = parts[i:]
            break
    env_str = " && ".join(env_parts)
    cmd_str = " ".join(cmd_parts)
    if env_str:
        bg_cmd = f'bash -c \'{env_str} && nohup {cmd_str} > {log_file} 2>&1 &\''
    else:
        bg_cmd = f'bash -c \'nohup {cmd_str} > {log_file} 2>&1 &\''
    client.exec_command(bg_cmd)
    time.sleep(0.3)


def scp_file(client, local_content: str, remote_path: str):
    """Write content to remote file via SFTP."""
    sftp = client.open_sftp()
    with sftp.file(remote_path, "w") as f:
        f.write(local_content)
    sftp.close()


# ===== Deployment Steps =====

def stop_host(host: str):
    """Kill all llama-server and agent processes on a host."""
    print(f"  [{host}] Stopping processes...")
    client = get_ssh_client(host)
    ssh_exec(client, "pkill -f llama-server || true", timeout=10)
    ssh_exec(client, "pkill -f agent_llm_dds || true", timeout=10)
    ssh_exec(client, "pkill -f 'python.*main.py' || true", timeout=10)
    client.close()
    time.sleep(1)


def stop_all():
    """Stop all processes on all hosts."""
    print("Stopping all processes...")
    for host in HOSTS:
        stop_host(host)
    stop_host(ORCHESTRATOR_HOST)
    time.sleep(2)
    print("All processes stopped")


def verify_model(host: str):
    """Ensure model file exists on remote host."""
    client = get_ssh_client(host)
    out, _, code = ssh_exec(client, f"test -f {MODEL_PATH} && echo EXISTS")
    client.close()
    if "EXISTS" not in out:
        print(f"  [{host}] ERROR: Model not found at {MODEL_PATH}")
        print(f"  Copy manually: scp models/{MODEL} {USER}@{host}:{MODEL_PATH}")
        return False
    print(f"  [{host}] Model OK: {MODEL_PATH}")
    return True


def build_llama_server(host: str, build_flags: str):
    """Build llama-server with DDS support on a host."""
    print(f"  [{host}] Building llama-server ({build_flags})...")
    client = get_ssh_client(host)
    ssh_exec(client, f"mkdir -p {BUILD_DIR}", timeout=10)
    ssh_exec(client, f"cd {BUILD_DIR} && cmake .. {build_flags}", timeout=120, check=True)
    ssh_exec(client, f"cd {BUILD_DIR} && make -j$(nproc)", timeout=600, check=True)
    client.close()
    print(f"  [{host}] Build complete")


def deploy_dds_config(host: str):
    """Ensure DDS config exists on remote host."""
    client = get_ssh_client(host)
    # Check if the config directory exists, create if needed
    ssh_exec(client, f"mkdir -p {HOME}/tese/dds_orchestrator/configs", timeout=5)
    # Read local config and push
    config_path = "E:/TI/git/tese/dds_orchestrator/configs/cyclonedds-10inst-network.xml"
    try:
        with open(config_path, "r") as f:
            content = f.read()
        scp_file(client, content, DDS_CONFIG_LOCAL)
        print(f"  [{host}] DDS config deployed: {DDS_CONFIG_LOCAL}")
    except FileNotFoundError:
        print(f"  [{host}] WARNING: Local DDS config not found at {config_path}")
    client.close()


def start_instances(host: str, instances: list[dict], extra_env: str = ""):
    """Start llama-server instances on a host."""
    client = get_ssh_client(host)
    env = f"CYCLONEDDS_URI=file://{DDS_CONFIG_LOCAL}"
    if extra_env:
        env = f"{extra_env} {env}"

    for inst in instances:
        cmd = (
            f"{env} {LLAMA_SERVER} "
            f"-m {MODEL_PATH} "
            f"-c {inst['ctx_size']} "
            f"--threads 2 "
            f"-ngl {inst['ngl']} "
            f"--parallel {inst['parallel']} "
            f"--port {inst['port']} "
            f"--host 0.0.0.0 "
            f"--enable-dds"
        )
        log = f"/tmp/llama_{inst['port']}.log"
        print(f"  [{host}] Starting instance :{inst['port']} "
              f"(ngl={inst['ngl']}, parallel={inst['parallel']})")
        ssh_bg(client, cmd, log_file=log)

    client.close()
    print(f"  [{host}] Started {len(instances)} instances")


def start_agents(host: str, instances: list[dict]):
    """Start one DDS agent per llama-server instance on a host."""
    client = get_ssh_client(host)
    env = f"CYCLONEDDS_URI=file://{DDS_CONFIG_LOCAL}"

    for inst in instances:
        agent_id = f"agent-inst-{inst['port']}"
        agent_port = inst["port"] + 1000  # 9082-9091
        cmd = (
            f"AGENT_ID={agent_id} {env} python3 {AGENT_SCRIPT} "
            f"--model-name {MODEL_NAME} "
            f"--llama-server-port {inst['port']} "
            f"--port {agent_port} "
            f"--orchestrator-url http://{ORCHESTRATOR_HOST}:8080 "
            f"--no-server"
        )
        log = f"/tmp/agent_{inst['port']}.log"
        print(f"  [{host}] Starting agent {agent_id} "
              f"(llama:{inst['port']}, agent:{agent_port})")
        ssh_bg(client, cmd, log_file=log)

    client.close()
    print(f"  [{host}] Started {len(instances)} agents")


def start_orchestrator(redis_url: str, mongo_url: str,
                       routing_algorithm: str = "least_loaded"):
    """Start orchestrator on .62."""
    client = get_ssh_client(ORCHESTRATOR_HOST)
    env = f"CYCLONEDDS_URI=file://{DDS_CONFIG_LOCAL}"

    # Collect all GPU instance ports across all hosts
    all_gpu_ports = []
    all_hostnames = {}  # port -> hostname mapping
    for host, cfg in HOSTS.items():
        for inst in cfg["instances"]:
            all_gpu_ports.append(str(inst["port"]))
            all_hostnames[inst["port"]] = host

    gpu_ports_str = ",".join(all_gpu_ports)

    # Build hostname mapping as comma-separated host:port pairs
    host_map_str = ",".join(
        f"{all_hostnames[int(p)]}:{p}" for p in all_gpu_ports
    )

    cmd = (
        f"cd {ORCHESTRATOR_DIR} && {env} python3 main.py "
        f"--port 8080 --host 0.0.0.0 "
        f"--log-level INFO "
        f"--instance-ports-gpu {gpu_ports_str} "
        f"--routing-algorithm {routing_algorithm}"
    )

    if redis_url:
        cmd += f" --redis-url {redis_url} --redis-password {DEFAULT_REDIS_PASSWORD}"
    if mongo_url:
        cmd += f" --mongo-url '{mongo_url}'"

    # Pass hostname mapping via env var (orchestrator reads it)
    cmd = f"INSTANCE_HOST_MAP='{host_map_str}' {cmd}"

    print(f"  [{ORCHESTRATOR_HOST}] Starting orchestrator on :8080")
    print(f"  Ports: {gpu_ports_str}")
    print(f"  Algorithm: {routing_algorithm}")
    ssh_bg(client, cmd, log_file="/tmp/orchestrator.log")
    client.close()


def verify_health(timeout_s: int = 120) -> bool:
    """Verify all instances are healthy via HTTP health check."""
    print(f"\nVerifying health of all instances (timeout={timeout_s}s)...")
    all_instances = []
    for host, cfg in HOSTS.items():
        for inst in cfg["instances"]:
            all_instances.append((host, inst["port"]))

    healthy = set()
    start = time.time()

    while time.time() - start < timeout_s:
        for host, port in all_instances:
            if (host, port) in healthy:
                continue
            try:
                client = get_ssh_client(host)
                out, _, code = ssh_exec(
                    client,
                    f"curl -s -o /dev/null -w '%{{http_code}}' "
                    f"http://127.0.0.1:{port}/health",
                    timeout=5,
                )
                client.close()
                if out.strip() == "200":
                    healthy.add((host, port))
                    print(f"  [{host}:{port}] HEALTHY")
            except Exception:
                pass

        if len(healthy) == len(all_instances):
            break
        time.sleep(3)

    failed = [(h, p) for h, p in all_instances if (h, p) not in healthy]
    print(f"\nHealthy: {len(healthy)}/{len(all_instances)}")
    if failed:
        print(f"Failed: {failed}")
    return len(failed) == 0


def verify_vram(host: str, cfg: dict):
    """Check VRAM usage on a host."""
    client = get_ssh_client(host)
    if cfg["gpu_type"] == "rtx3080":
        out, _, _ = ssh_exec(client, cfg["vram_cmd"], timeout=10)
        if out and "," in out:
            used, total = out.strip().split(",")
            used_mb = int(used.strip())
            total_mb = int(total.strip())
            pct = used_mb / total_mb * 100
            print(f"  [{host}] VRAM: {used_mb}MB / {total_mb}MB ({pct:.1f}%)")
            if used_mb > cfg["vram_warn_mb"]:
                print(f"  [{host}] WARNING: VRAM > {cfg['vram_warn_mb']}MB!")
                client.close()
                return False
    else:
        # AMD: read sysfs
        out, _, _ = ssh_exec(client, cfg["vram_cmd"], timeout=10)
        if out and out != "N/A":
            lines = out.strip().split("\n")
            if len(lines) >= 2:
                used_bytes = int(lines[0].strip())
                total_bytes = int(lines[1].strip())
                used_mb = used_bytes // (1024 * 1024)
                total_mb = total_bytes // (1024 * 1024)
                pct = used_mb / max(total_mb, 1) * 100
                print(f"  [{host}] VRAM: {used_mb}MB / {total_mb}MB ({pct:.1f}%)")
                if used_mb > cfg["vram_warn_mb"]:
                    print(f"  [{host}] WARNING: VRAM > {cfg['vram_warn_mb']}MB!")
                    client.close()
                    return False
            else:
                print(f"  [{host}] VRAM: (unable to parse)")
        else:
            print(f"  [{host}] VRAM: (sysfs not available)")
    client.close()
    return True


def get_status():
    """Get current status of all hosts."""
    print("\n" + "=" * 60)
    print("Deployment Status")
    print("=" * 60)

    for host, cfg in HOSTS.items():
        print(f"\n--- {host} ({cfg['gpu_type']}) ---")
        try:
            client = get_ssh_client(host)
            out, _, _ = ssh_exec(client, "pgrep -c llama-server || echo 0", timeout=10)
            print(f"  llama-server processes: {out.strip()}")
            out, _, _ = ssh_exec(client, "pgrep -cf agent_llm_dds || echo 0", timeout=10)
            print(f"  Agent processes: {out.strip()}")
            out, _, _ = ssh_exec(client, "free -h | grep Mem", timeout=10)
            print(f"  RAM: {out.strip()}")
            client.close()
            verify_vram(host, cfg)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n--- {ORCHESTRATOR_HOST} (orchestrator) ---")
    try:
        client = get_ssh_client(ORCHESTRATOR_HOST)
        out, _, _ = ssh_exec(client, "pgrep -cf 'python.*main.py' || echo 0", timeout=10)
        print(f"  Orchestrator processes: {out.strip()}")
        out, _, _ = ssh_exec(client, "free -h | grep Mem", timeout=10)
        print(f"  RAM: {out.strip()}")
        client.close()
    except Exception as e:
        print(f"  ERROR: {e}")


def full_deploy(redis_url: str, mongo_url: str, routing_algorithm: str,
                skip_build: bool = False):
    """Full deployment sequence."""
    print("=" * 60)
    print("10-Instance Multi-Host Deployment")
    print("=" * 60)
    total_instances = sum(len(cfg["instances"]) for cfg in HOSTS.values())
    total_slots = sum(
        inst["parallel"]
        for cfg in HOSTS.values()
        for inst in cfg["instances"]
    )
    print(f"Instances: {total_instances} (slots: {total_slots})")
    for host, cfg in HOSTS.items():
        n = len(cfg["instances"])
        ports = [str(inst["port"]) for inst in cfg["instances"]]
        print(f"  {host} ({cfg['gpu_type']}): {n} instances, ports {','.join(ports)}")
    print()

    # 1. Stop existing
    stop_all()

    # 2. Check models
    print("\n--- Verifying models ---")
    for host in HOSTS:
        if not verify_model(host):
            return False

    # 3. Build (if not skipped)
    if not skip_build:
        print("\n--- Building llama-server ---")
        for host, cfg in HOSTS.items():
            build_llama_server(host, cfg["build_flags"])

    # 3.5 Ensure socket buffer limits on all hosts
    print("\n--- Configuring socket buffers ---")
    for host in list(HOSTS.keys()) + [ORCHESTRATOR_HOST]:
        try:
            client = get_ssh_client(host)
            for sysctl in [
                "echo Admin@123 | sudo -S sysctl -w net.core.rmem_max=8388608",
                "echo Admin@123 | sudo -S sysctl -w net.core.wmem_max=4194304",
            ]:
                ssh_exec(client, sysctl, timeout=10)
            client.close()
            print(f"  [{host}] Socket buffers OK")
        except Exception as e:
            print(f"  [{host}] WARNING: {e}")

    # 4. Deploy DDS config
    print("\n--- Deploying DDS config ---")
    for host in HOSTS:
        deploy_dds_config(host)
    deploy_dds_config(ORCHESTRATOR_HOST)

    # 5. Start llama-servers
    print("\n--- Starting llama-server instances ---")
    for host, cfg in HOSTS.items():
        start_instances(host, cfg["instances"], extra_env=cfg.get("extra_env", ""))

    # 6. Wait for health
    print("\n--- Waiting for instances to be ready ---")
    time.sleep(10)
    if not verify_health(timeout_s=120):
        print("ERROR: Not all instances healthy!")
        return False

    # 7. Start orchestrator on .62
    print("\n--- Starting orchestrator on .62 ---")
    start_orchestrator(redis_url, mongo_url, routing_algorithm)
    time.sleep(5)

    # 8. Verify orchestrator health
    print("\n--- Verifying orchestrator ---")
    try:
        client = get_ssh_client(ORCHESTRATOR_HOST)
        out, _, code = ssh_exec(
            client,
            f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:8080/health",
            timeout=10,
        )
        client.close()
        if out.strip() == "200":
            print(f"  [{ORCHESTRATOR_HOST}:8080] Orchestrator HEALTHY")
        else:
            print(f"  [{ORCHESTRATOR_HOST}:8080] Orchestrator returned {out}")
    except Exception as e:
        print(f"  Orchestrator health check failed: {e}")

    # 9. Start agents
    print("\n--- Starting agents ---")
    for host, cfg in HOSTS.items():
        start_agents(host, cfg["instances"])
    time.sleep(5)

    # 10. Final status
    print("\n--- Final verification ---")
    get_status()

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print(f"Orchestrator: http://{ORCHESTRATOR_HOST}:8080")
    print(f"Instances: {total_instances} GPU ({total_slots} slots)")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Deploy 10 llama.cpp_dds instances across .60 and .61"
    )
    parser.add_argument("--deploy", action="store_true", help="Full deployment")
    parser.add_argument("--stop", action="store_true", help="Stop all processes")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--verify", action="store_true", help="Verify health only")
    parser.add_argument("--build", action="store_true", help="Build llama-server on all hosts")
    parser.add_argument("--skip-build", action="store_true",
                       help="Skip build step during deploy")
    parser.add_argument(
        "--redis-url", type=str, default=DEFAULT_REDIS_URL,
        help="Redis URL",
    )
    parser.add_argument(
        "--mongo-url", type=str, default=DEFAULT_MONGO_URL,
        help="MongoDB URL",
    )
    parser.add_argument(
        "--routing-algorithm", type=str, default="least_loaded",
        choices=["round_robin", "least_loaded", "weighted_score"],
    )

    args = parser.parse_args()

    if args.stop:
        stop_all()
    elif args.status:
        get_status()
    elif args.verify:
        verify_health(timeout_s=30)
    elif args.build:
        print("Building llama-server on all hosts...")
        for host, cfg in HOSTS.items():
            build_llama_server(host, cfg["build_flags"])
    elif args.deploy:
        full_deploy(args.redis_url, args.mongo_url,
                    args.routing_algorithm, args.skip_build)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
