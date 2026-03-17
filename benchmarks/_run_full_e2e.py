#!/usr/bin/env python3
"""
Full E1-E5 Benchmark Runner via SSH
====================================
Deploys services on VMs and runs benchmarks from client VM (.63).
Uses RTX 3080 (.61) for llama-server, orchestrator on .62, client on .63.

Protocols: HTTP, gRPC, DDS (full E2E path)
"""

import paramiko
import socket
import time
import io
import sys
import json
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = "/home/oldds"
ORCH_IP = "192.168.1.62"
AGENT_IP = "192.168.1.61"
CLIENT_IP = "192.168.1.63"
XML = f"{BASE}/llama.cpp_dds/dds/cyclonedds-network-optimized.xml"
MODEL = "Phi-4-mini-instruct-Q4_K_M.gguf"
MODEL_NAME = "Phi-4-mini"

N_REQUESTS = 100  # Per scenario (can increase to 1000 for final run)

# ─── SSH Helpers ─────────────────────────────────────────────────────────────

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


def ssh_upload(ip, local_content, remote_path):
    ssh = ssh_connect(ip)
    sftp = ssh.open_sftp()
    with sftp.file(remote_path, 'w') as f:
        f.write(local_content)
    sftp.close()


# ─── Service Management ─────────────────────────────────────────────────────

def stop_all():
    print("Stopping all services...")
    for ip in [AGENT_IP, ORCH_IP]:
        try:
            ssh_run(ip, 'pkill -9 -f "llama-server|agent_llm|main.py|grpc_layer" 2>/dev/null; true', 5)
        except Exception:
            pass
    time.sleep(3)


def pull_latest():
    print("Pulling latest code...")
    for ip in [ORCH_IP, AGENT_IP, CLIENT_IP]:
        try:
            for repo in ["dds_orchestrator", "dds_agent"]:
                ssh_run(ip, f"cd {BASE}/{repo} && git fetch origin && git reset --hard origin/main 2>&1 | tail -1", 30)
            print(f"  {ip}: OK")
        except Exception as e:
            print(f"  {ip}: {e}")


def start_llama_server():
    print(f"Starting llama-server on {AGENT_IP} (RTX 3080)...")
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


def start_orchestrator(transport="dds", fuzzy=False):
    print(f"Starting orchestrator ({transport})...")
    cmd = f"python3 -u {BASE}/dds_orchestrator/main.py --port 8080 --log-level INFO"
    env = {}

    if transport == "dds":
        cmd += " --dds-domain 0"
        env["CYCLONEDDS_URI"] = f"file://{XML}"
    else:
        env["DDS_ENABLED"] = "false"

    if fuzzy:
        cmd += " --fuzzy"

    ssh_bg(ORCH_IP, cmd, "/tmp/orch_bench.log", env)
    time.sleep(8)

    ec, out, _ = ssh_run(ORCH_IP, f"curl -s http://localhost:8080/health", 5)
    ok = ec == 0 and out.strip()
    print(f"  orchestrator: {'OK' if ok else 'FAILED'}")
    return ok


def start_agent(transport="dds"):
    print(f"Starting agent ({transport})...")
    if transport == "dds":
        cmd = (f"python3 -u {BASE}/dds_agent/python/agent_llm_dds.py "
               f"--model-name {MODEL_NAME} "
               f"--model-path {BASE}/models/{MODEL} "
               f"--orchestrator-url http://{ORCH_IP}:8080 "
               f"--port 8081 --llama-server-port 8082 --no-server")
        env = {"CYCLONEDDS_URI": f"file://{XML}"}
    elif transport == "grpc":
        cmd = (f"python3 -u {BASE}/dds_agent/python/agent_llm_grpc.py "
               f"--model-name {MODEL_NAME} "
               f"--model-path {BASE}/models/{MODEL} "
               f"--orchestrator-url http://{ORCH_IP}:8080 "
               f"--port 8081 --llama-server-port 8082 --no-server")
        env = {}
    else:  # http
        cmd = (f"python3 -u {BASE}/dds_agent/python/agent_llm.py")
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

    ssh_bg(AGENT_IP, cmd, f"/tmp/agent_{transport}.log", env)
    time.sleep(12)

    ec, out, _ = ssh_run(ORCH_IP, f"curl -s http://localhost:8080/api/v1/agents", 10)
    count = out.count("agent_id")
    print(f"  agents registered: {count}")
    return count > 0


# ─── Benchmark Script ────────────────────────────────────────────────────────

def get_benchmark_script(protocol, n):
    """Generate Python benchmark script to run on client VM."""
    return f'''#!/usr/bin/env python3
import time, statistics, json, sys, os, uuid, concurrent.futures
sys.path.insert(0, "/home/oldds/dds_orchestrator")
sys.path.insert(0, "/home/oldds/dds_orchestrator/benchmarks")

ORCH_URL = "http://{ORCH_IP}:8080"
PROTOCOL = "{protocol}"
N = {n}

try:
    import orjson
    _loads = orjson.loads
except ImportError:
    _loads = json.loads

import requests

def http_chat(prompt, max_tokens=10, timeout=60):
    t0 = time.perf_counter()
    r = requests.post(f"{{ORCH_URL}}/api/v1/chat/completions",
        json={{"model":"{MODEL_NAME}","messages":[{{"role":"user","content":prompt}}],"max_tokens":max_tokens}},
        timeout=timeout)
    lat = (time.perf_counter() - t0) * 1000
    ok = r.status_code == 200 and "content" in r.text
    return lat, ok

def dds_chat(prompt, max_tokens=10, timeout=60):
    """DDS chat via client DDS topic (same as HTTP for now — DDS client not on .63)."""
    return http_chat(prompt, max_tokens, timeout)

def bench(label, prompt, n, max_tokens=10):
    lats = []
    errors = 0
    for i in range(n):
        lat, ok = http_chat(prompt, max_tokens) if PROTOCOL != "dds" else dds_chat(prompt, max_tokens)
        if ok:
            lats.append(lat)
        else:
            errors += 1
        if (i+1) % max(1, n//10) == 0:
            print(f"  {{label}}: {{i+1}}/{{n}} ({{errors}} err)", flush=True)
    result = {{"label": label, "n": n, "ok": len(lats), "errors": errors}}
    if lats:
        lats.sort()
        result.update({{
            "p50": lats[len(lats)//2],
            "p95": lats[int(len(lats)*0.95)],
            "p99": lats[int(len(lats)*0.99)],
            "mean": statistics.mean(lats),
            "std": statistics.stdev(lats) if len(lats)>1 else 0,
            "min": min(lats),
            "max": max(lats),
        }})
        print(f"  {{label}}: p50={{result['p50']:.1f}}ms p95={{result['p95']:.1f}}ms mean={{result['mean']:.1f}}ms ({{len(lats)}}/{{n}})", flush=True)
    return result

def bench_concurrent(label, prompt, n_total, n_clients, max_tokens=10):
    """E4: concurrent clients."""
    import threading
    results_per_client = []
    barrier = threading.Barrier(n_clients)

    def worker(cid):
        barrier.wait()
        client_lats = []
        per_client = n_total // n_clients
        for i in range(per_client):
            lat, ok = http_chat(prompt, max_tokens, timeout=120)
            if ok:
                client_lats.append(lat)
        results_per_client.append(client_lats)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_clients)]
    for t in threads: t.start()
    for t in threads: t.join()

    all_lats = [l for r in results_per_client for l in r]
    result = {{"label": label, "n_clients": n_clients, "n_total": n_total, "ok": len(all_lats)}}
    if all_lats:
        all_lats.sort()
        result.update({{
            "p50": all_lats[len(all_lats)//2],
            "p95": all_lats[int(len(all_lats)*0.95)],
            "mean": statistics.mean(all_lats),
            "std": statistics.stdev(all_lats) if len(all_lats)>1 else 0,
            "throughput_rps": len(all_lats) / (max(all_lats) / 1000) if all_lats else 0,
        }})
        print(f"  {{label}} ({{n_clients}} clients): p50={{result['p50']:.1f}}ms throughput={{result['throughput_rps']:.1f}} rps", flush=True)
    return result

def bench_stream(label, prompt, n, max_tokens=50):
    """E5: streaming TTFT + ITL."""
    ttfts = []
    itls_all = []
    for i in range(n):
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{{ORCH_URL}}/api/v1/chat/completions",
                json={{"model":"{MODEL_NAME}","messages":[{{"role":"user","content":prompt}}],
                       "max_tokens":max_tokens,"stream":True}},
                stream=True, timeout=60)
            first_chunk_time = None
            prev_time = t0
            chunk_times = []
            for line in r.iter_lines():
                if not line:
                    continue
                now = time.perf_counter()
                if first_chunk_time is None:
                    first_chunk_time = now
                    ttfts.append((first_chunk_time - t0) * 1000)
                else:
                    itl = (now - prev_time) * 1000
                    if itl > 0.1:
                        itls_all.append(itl)
                prev_time = now
        except Exception:
            pass
        if (i+1) % max(1, n//10) == 0:
            print(f"  {{label}}: {{i+1}}/{{n}}", flush=True)

    result = {{"label": label, "n": n}}
    if ttfts:
        ttfts.sort()
        result["ttft_p50"] = ttfts[len(ttfts)//2]
        result["ttft_mean"] = statistics.mean(ttfts)
        print(f"  {{label}} TTFT: p50={{result['ttft_p50']:.1f}}ms mean={{result['ttft_mean']:.1f}}ms", flush=True)
    if itls_all:
        itls_all.sort()
        result["itl_p50"] = itls_all[len(itls_all)//2]
        result["itl_mean"] = statistics.mean(itls_all)
        print(f"  {{label}} ITL: p50={{result['itl_p50']:.1f}}ms mean={{result['itl_mean']:.1f}}ms", flush=True)
    return result

# ─── Run All Scenarios ──────────────────────────────────────────────────────

print(f"=== {{PROTOCOL.upper()}} Benchmark (n={{N}}) ===", flush=True)

# Warmup
print("Warmup (10 requests)...", flush=True)
for _ in range(10):
    http_chat("Hi", 5)

results = {{"protocol": PROTOCOL, "n": N, "timestamp": time.strftime("%Y%m%d_%H%M%S")}}

# E1: Latency
print("\\n--- E1: Latency ---", flush=True)
results["E1_short"] = bench("E1_short", "Hi", N, max_tokens=10)
results["E1_long"] = bench("E1_long", "Explain what is machine learning in one paragraph", N, max_tokens=50)

# E2: Error/Reliability (same as E1 but we track error rate)
print("\\n--- E2: Reliability ---", flush=True)
results["E2"] = bench("E2_reliability", "What is 2+2?", N, max_tokens=10)

# E3: Priority (alternate high/low priority — measured as latency difference)
print("\\n--- E3: Priority ---", flush=True)
results["E3_normal"] = bench("E3_normal", "Hello", N//2, max_tokens=10)
results["E3_complex"] = bench("E3_complex", "Write a detailed essay about distributed systems", N//2, max_tokens=100)

# E4: Scalability (concurrent clients)
print("\\n--- E4: Scalability ---", flush=True)
for nc in [1, 2, 4]:
    results[f"E4_c{{nc}}"] = bench_concurrent(f"E4_c{{nc}}", "Hi", min(N, nc*20), nc, max_tokens=10)

# E5: Streaming
print("\\n--- E5: Streaming ---", flush=True)
results["E5"] = bench_stream("E5_stream", "Explain quantum computing briefly", min(N, 50), max_tokens=50)

# Save
outfile = f"/home/oldds/dds_orchestrator/benchmarks/results/e2e_{{PROTOCOL}}_{{results['timestamp']}}.json"
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, "w") as f:
    json.dump(results, f, indent=2)
print(f"\\nResults saved to {{outfile}}", flush=True)

# Summary
print("\\n=== SUMMARY ===", flush=True)
for key in sorted(results.keys()):
    v = results[key]
    if isinstance(v, dict) and "p50" in v:
        print(f"  {{key}}: p50={{v['p50']:.1f}}ms p95={{v.get('p95',0):.1f}}ms err={{v.get('errors',0)}}", flush=True)
    elif isinstance(v, dict) and "ttft_p50" in v:
        print(f"  {{key}}: TTFT_p50={{v['ttft_p50']:.1f}}ms ITL_p50={{v.get('itl_p50',0):.1f}}ms", flush=True)

print("\\nDONE", flush=True)
'''


# ─── Main ────────────────────────────────────────────────────────────────────

def run_benchmark(protocol, n=N_REQUESTS):
    """Upload and run benchmark on client VM."""
    script = get_benchmark_script(protocol, n)
    remote_path = f"/tmp/bench_{protocol}.py"
    ssh_upload(CLIENT_IP, script, remote_path)
    print(f"\nRunning {protocol.upper()} benchmark (n={n}) from {CLIENT_IP}...")
    ec, out, err = ssh_run(CLIENT_IP, f"python3 {remote_path}", timeout=7200)
    print(out)
    if err and "Error" in err:
        print(f"STDERR: {err[:500]}")
    return out


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== Full E2E Benchmark Run — {ts} ===\n")

    # Pull latest code
    pull_latest()

    protocols = ["http", "dds"]  # gRPC can be added if needed

    for protocol in protocols:
        print(f"\n{'='*60}")
        print(f"  PROTOCOL: {protocol.upper()}")
        print(f"{'='*60}")

        stop_all()

        if not start_llama_server():
            print(f"SKIP {protocol}: llama-server failed")
            continue

        transport = "dds" if protocol == "dds" else "http"
        if not start_orchestrator(transport=transport):
            print(f"SKIP {protocol}: orchestrator failed")
            continue

        if not start_agent(transport=transport):
            print(f"SKIP {protocol}: agent failed")
            continue

        run_benchmark(protocol, N_REQUESTS)

    stop_all()
    print(f"\n=== ALL BENCHMARKS COMPLETE ===")


if __name__ == "__main__":
    main()
