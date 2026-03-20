#!/usr/bin/env python3
"""
Run tests remotely on k8s VM (192.168.1.51) via paramiko SSH.
Copies test files, installs deps, and runs pytest against real Redis+MongoDB.
"""

import os
import sys
import time
import paramiko

# === Config ===
VM_HOST = "192.168.1.51"
VM_USER = "k8s1"
VM_PASSWORD = "Admin@123"
VM_TEST_DIR = "/tmp/dds_test_38inst"

REDIS_URL = "redis://127.0.0.1:30379"
REDIS_PASSWORD = "Admin@123"
MONGO_URL = "mongodb://admin:Admin%40123@127.0.0.1:27017/?authSource=admin"

# Files to copy
LOCAL_BASE = os.path.join(os.path.dirname(__file__), "..")
FILES_TO_COPY = [
    # Core modules
    "redis_layer.py",
    "mongo_layer.py",
    "instance_pool.py",
    "backpressure.py",
    "config.py",
    "server.py",
    "main.py",
    "dds.py",
    "registry.py",
    "scheduler.py",
    "selector.py",
    # IDL
    "orchestrator/__init__.py",
    "orchestrator/_OrchestratorDDS.py",
    # DDS config
    "cyclonedds-38inst.xml",
    # Benchmark scripts
    "benchmarks/__init__.py",
    "benchmarks/deploy_38_instances.py",
    "benchmarks/register_1000_agents.py",
    "benchmarks/load_generator.py",
    "benchmarks/run_full_benchmark.py",
    "benchmarks/generate_38inst_plots.py",
    # Tests
    "tests/__init__.py",
    "tests/test_38inst.py",
    "tests/test_38inst_e2e.py",
]

# Extra files from other parts of the repo (relative to repo root)
EXTRA_FILES = [
    ("llama.cpp_dds/dds/cyclonedds-38inst.xml", "llama_dds/cyclonedds-38inst.xml"),
    ("llama.cpp_dds/dds/idl/OrchestratorDDS.idl", "llama_dds/OrchestratorDDS.idl"),
    ("dds_agent/orchestrator/__init__.py", "dds_agent_orch/__init__.py"),
    ("dds_agent/orchestrator/_OrchestratorDDS.py", "dds_agent_orch/_OrchestratorDDS.py"),
]


def get_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(VM_HOST, username=VM_USER, password=VM_PASSWORD, timeout=15)
    return client


def ssh_exec(client, cmd, timeout=60):
    """Execute command and return (stdout, stderr, exit_code)."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out, err, exit_code


def copy_files(client):
    """Copy test files to remote VM via SFTP."""
    sftp = client.open_sftp()

    # Create directories
    dirs = [
        VM_TEST_DIR,
        f"{VM_TEST_DIR}/orchestrator",
        f"{VM_TEST_DIR}/tests",
        f"{VM_TEST_DIR}/benchmarks",
        f"{VM_TEST_DIR}/llama_dds",
        f"{VM_TEST_DIR}/dds_agent_orch",
    ]
    for d in dirs:
        try:
            sftp.stat(d)
        except FileNotFoundError:
            sftp.mkdir(d)

    # Create __init__.py in root
    with sftp.open(f"{VM_TEST_DIR}/__init__.py", "w") as f:
        f.write("")

    # Create benchmarks __init__.py if not in list
    bench_init = os.path.join(LOCAL_BASE, "benchmarks", "__init__.py")
    if not os.path.exists(bench_init):
        with sftp.open(f"{VM_TEST_DIR}/benchmarks/__init__.py", "w") as f:
            f.write("")

    for rel_path in FILES_TO_COPY:
        local_path = os.path.join(LOCAL_BASE, rel_path)
        remote_path = f"{VM_TEST_DIR}/{rel_path}"

        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {rel_path}")
            continue

        sftp.put(local_path, remote_path)
        print(f"  COPY: {rel_path}")

    # Copy extra files from repo root
    repo_root = os.path.join(LOCAL_BASE, "..")
    for src_rel, dst_rel in EXTRA_FILES:
        local_path = os.path.join(repo_root, src_rel)
        remote_path = f"{VM_TEST_DIR}/{dst_rel}"
        if os.path.exists(local_path):
            sftp.put(local_path, remote_path)
            print(f"  COPY: {src_rel} -> {dst_rel}")

    sftp.close()


def patch_test_urls(client):
    """Patch test files to use localhost URLs (tests run on the k8s VM itself)."""
    # On the k8s VM, Redis is at localhost:30379 (NodePort) and MongoDB at localhost:27017
    sed_cmd = (
        f"sed -i "
        f"'s|redis://192.168.1.51:30379|redis://127.0.0.1:30379|g; "
        f"s|192.168.1.51:27017|127.0.0.1:27017|g' "
        f"{VM_TEST_DIR}/tests/test_38inst.py {VM_TEST_DIR}/tests/test_38inst_e2e.py"
    )
    ssh_exec(client, sed_cmd)
    print("  Patched test URLs to localhost")


def install_deps(client):
    """Install Python dependencies on remote VM."""
    print("\nInstalling dependencies...")
    deps = "redis[hiredis] motor pymongo pytest pytest-asyncio aiohttp pyyaml pydantic"

    # First ensure pip is installed
    _, _, pip_check = ssh_exec(client, "python3 -m pip --version", timeout=10)
    if pip_check != 0:
        print("  pip not found, installing via apt...")
        out, err, code = ssh_exec(client,
            f"echo '{VM_PASSWORD}' | sudo -S apt-get update -qq && "
            f"echo '{VM_PASSWORD}' | sudo -S apt-get install -y -qq python3-pip python3-venv",
            timeout=120)
        if code != 0:
            print(f"  apt install failed, trying get-pip.py...")
            ssh_exec(client, "curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --user", timeout=60)

    # Create a venv to avoid --break-system-packages issues
    venv = f"{VM_TEST_DIR}/.venv"
    ssh_exec(client, f"python3 -m venv {venv}", timeout=30)
    out, err, code = ssh_exec(client, f"{venv}/bin/pip install {deps}", timeout=120)
    if code != 0:
        print(f"  ERROR: pip install failed (exit code {code})")
        for line in (err or out).strip().split("\n")[-10:]:
            print(f"    {line}")
        return False
    else:
        print("  Dependencies OK")
        return True


def run_tests(client):
    """Run pytest on the remote VM."""
    print(f"\n{'='*60}")
    print("Running tests on VM...")
    print(f"{'='*60}\n")

    venv = f"{VM_TEST_DIR}/.venv"
    cmd = (
        f"cd {VM_TEST_DIR} && "
        f"PYTHONPATH={VM_TEST_DIR} "
        f"{venv}/bin/python -m pytest tests/test_38inst.py tests/test_38inst_e2e.py "
        f"-v --tb=short 2>&1"
    )
    out, err, code = ssh_exec(client, cmd, timeout=180)
    print(out)
    if err.strip():
        print(err)

    return code


def cleanup(client):
    """Remove test files from remote VM."""
    ssh_exec(client, f"rm -rf {VM_TEST_DIR}")
    print("Cleaned up remote test files")


def main():
    print(f"Connecting to {VM_USER}@{VM_HOST}...")
    client = get_client()
    print("Connected!\n")

    try:
        # Check Python
        out, _, _ = ssh_exec(client, "python3 --version")
        print(f"Remote Python: {out.strip()}")

        # Copy files
        print("\nCopying files...")
        copy_files(client)
        patch_test_urls(client)

        # Install deps
        if not install_deps(client):
            print("Failed to install dependencies. Aborting.")
            return 1

        # Run tests
        exit_code = run_tests(client)

        print(f"\n{'='*60}")
        if exit_code == 0:
            print("ALL TESTS PASSED ON VM!")
        else:
            print(f"TESTS FAILED (exit code {exit_code})")
        print(f"{'='*60}")

        return exit_code

    finally:
        cleanup(client)
        client.close()


if __name__ == "__main__":
    sys.exit(main())
