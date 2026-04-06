#!/usr/bin/env python3
"""
E6: Scalability Benchmark — 29 instances (5 GPU + 24 CPU), 1000 agents
Load ramp: 100, 500, 1000, 5000, 10000 clients
Duration: 60s per scenario (1h not feasible in single run, adjust as needed)

Infrastructure:
  .61 (192.168.1.61): 29 llama-server + orchestrator + Redis/MongoDB
  .63 (192.168.1.63): load generator clients

Model: Qwen3.5-2B-UD-IQ2_XXS.gguf (733MB IQ2_XXS)
VRAM budget: 5 GPU x 1.82GB = 9.1GB / 10GB
RAM budget: 24 CPU x 733MB = 17.6GB / 31GB
"""

import os
import sys
import time
import json
import paramiko

# === Infrastructure ===
SERVER = "192.168.1.61"
CLIENT = "192.168.1.63"
USER = "oldds"
PWD = "Admin@123"
REDIS = "redis://192.168.1.51:30379"
MONGO = "mongodb://admin:Admin%40123@192.168.1.51:27017/?authSource=admin"

MODEL_PATH = f"/home/{USER}/models/Qwen3.5-2B-UD-IQ2_XXS.gguf"
LLAMA = f"/home/{USER}/llama.cpp_dds/build/bin/llama-server"
DEPLOY = f"/home/{USER}/tese_deploy"
DDS_CFG = f"{DEPLOY}/cyclonedds-38inst.xml"
ORCH_PORT = 8080

# 4 GPU (parallel=15) + 25 CPU (parallel=4) = 29 instances
# GPU: 4 x 1.82GB = 7.3GB VRAM (safe in 10GB)
# CPU: -ngl 0 --device cpu to avoid touching GPU
GPU_INSTANCES = [{"port": 8082 + i, "ngl": 99, "parallel": 15, "ctx": 512, "threads": 2} for i in range(4)]
CPU_INSTANCES = [{"port": 8092 + i, "ngl": 0,  "parallel": 4,  "ctx": 256, "threads": 2} for i in range(25)]
ALL = GPU_INSTANCES + CPU_INSTANCES

# k6-style ramp: stages with target VUs and duration
# Each stage ramps linearly from previous target to new target
# Total: ~1 hour
RAMP_STAGES = [
    {"target": 10,    "duration": 60},    # 0->10 in 1min (warmup)
    {"target": 100,   "duration": 120},   # 10->100 in 2min
    {"target": 100,   "duration": 300},   # hold 100 for 5min
    {"target": 500,   "duration": 120},   # 100->500 in 2min
    {"target": 500,   "duration": 300},   # hold 500 for 5min
    {"target": 1000,  "duration": 120},   # 500->1000 in 2min
    {"target": 1000,  "duration": 300},   # hold 1000 for 5min
    {"target": 5000,  "duration": 180},   # 1000->5000 in 3min
    {"target": 5000,  "duration": 300},   # hold 5000 for 5min
    {"target": 10000, "duration": 300},   # 5000->10000 in 5min
    {"target": 10000, "duration": 600},   # hold 10000 for 10min
    {"target": 0,     "duration": 60},    # ramp down
]
SAMPLE_INTERVAL = 30  # report metrics every 30s

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

SERVER_FILES = [
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
    ("dds_orchestrator/grpc_layer.py", "grpc_layer.py"),
    ("dds_orchestrator/qos_profiles.py", "qos_profiles.py"),
    ("dds_orchestrator/http_client.py", "http_client.py"),
    ("dds_orchestrator/context.py", "context.py"),
    ("dds_orchestrator/models.py", "models.py"),
]


def ssh(host):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=USER, password=PWD, timeout=15)
    return c


def run(c, cmd, timeout=30):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    code = out.channel.recv_exit_status()
    return out.read().decode("utf-8", errors="replace").strip(), code


def bg(c, cmd, log=None):
    redir = f"> {log} 2>&1" if log else "> /dev/null 2>&1"
    escaped = cmd.replace("'", "'\\''")
    c.exec_command(f"nohup bash -c '{escaped}' {redir} &")
    time.sleep(0.2)


def step(msg):
    print(f"\n{'-'*70}\n  {msg}\n{'-'*70}")


def upload(c, file_list, remote_base):
    sftp = c.open_sftp()
    dirs = set()
    for _, dst in file_list:
        d = os.path.dirname(f"{remote_base}/{dst}")
        while d and d != remote_base:
            dirs.add(d)
            d = os.path.dirname(d)
    dirs.add(remote_base)
    for d in sorted(dirs):
        try:
            sftp.stat(d)
        except FileNotFoundError:
            sftp.mkdir(d)
    n = 0
    for src, dst in file_list:
        local = os.path.join(REPO_ROOT, src)
        if os.path.exists(local):
            sftp.put(local, f"{remote_base}/{dst}")
            n += 1
    sftp.close()
    return n


def kill_all(c):
    run(c, f"echo '{PWD}' | sudo -S killall -9 llama-server python3 2>/dev/null || true", timeout=10)
    time.sleep(4)


# =====================================================================
# DEPLOY 29 INSTANCES + ORCHESTRATOR ON .61
# =====================================================================

def deploy():
    step(f"DEPLOY: 29 instances on {SERVER}")
    c = ssh(SERVER)

    kill_all(c)
    out, _ = run(c, "ss -tlnp | grep -E '808|809' || echo FREE")
    print(f"  Ports: {out[:80]}")

    n = upload(c, SERVER_FILES, DEPLOY)
    print(f"  Uploaded {n} files")

    env = f"CYCLONEDDS_URI=file://{DDS_CFG}"

    # Start GPU instances
    for inst in GPU_INSTANCES:
        cmd = (f"{env} {LLAMA} -m {MODEL_PATH} "
               f"-c {inst['ctx']} --threads {inst['threads']} "
               f"-ngl {inst['ngl']} --parallel {inst['parallel']} "
               f"--port {inst['port']} --host 0.0.0.0")
        bg(c, cmd, log=f"/tmp/llama_{inst['port']}.log")
    print(f"  Started {len(GPU_INSTANCES)} GPU instances (:{GPU_INSTANCES[0]['port']}-:{GPU_INSTANCES[-1]['port']})")

    # Start CPU instances (CUDA_VISIBLE_DEVICES= to force CPU-only)
    for inst in CPU_INSTANCES:
        cmd = (f"CUDA_VISIBLE_DEVICES= {env} {LLAMA} -m {MODEL_PATH} "
               f"-c {inst['ctx']} --threads {inst['threads']} "
               f"-ngl 0 --parallel {inst['parallel']} "
               f"--port {inst['port']} --host 0.0.0.0")
        bg(c, cmd, log=f"/tmp/llama_{inst['port']}.log")
    print(f"  Started {len(CPU_INSTANCES)} CPU instances (:{CPU_INSTANCES[0]['port']}-:{CPU_INSTANCES[-1]['port']})")

    print("  Waiting 45s for model loading (CPU warmup is slow)...")
    time.sleep(45)

    # Health check all
    healthy = 0
    failed_ports = []
    for inst in ALL:
        out, _ = run(c, f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{inst['port']}/health", timeout=5)
        if out.strip() == "200":
            healthy += 1
        else:
            failed_ports.append(inst["port"])
    print(f"  Healthy: {healthy}/{len(ALL)}")
    if failed_ports:
        print(f"  Failed: {failed_ports[:5]}...")
        # Retry failed ones after more wait
        time.sleep(15)
        for port in failed_ports:
            out, _ = run(c, f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/health", timeout=5)
            if out.strip() == "200":
                healthy += 1
        print(f"  After retry: {healthy}/{len(ALL)}")

    if healthy < len(ALL) * 0.8:  # Accept 80% healthy
        print(f"  ABORT: too few healthy instances ({healthy}/{len(ALL)})")
        c.close()
        return False

    # VRAM
    out, _ = run(c, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
    print(f"  VRAM: {out}")

    # RAM
    out, _ = run(c, "free -h | grep Mem")
    print(f"  RAM:  {out}")

    # Start orchestrator
    gpu_ports = ",".join(str(i["port"]) for i in GPU_INSTANCES)
    cpu_ports = ",".join(str(i["port"]) for i in CPU_INSTANCES)
    orch_cmd = (f"cd {DEPLOY} && {env} python3 main.py "
                f"--port {ORCH_PORT} --host 0.0.0.0 "
                f"--redis-url {REDIS} --redis-password {PWD} "
                f"--mongo-url '{MONGO}' "
                f"--instance-ports-gpu {gpu_ports} "
                f"--instance-ports-cpu {cpu_ports} "
                f"--routing-algorithm least_loaded "
                f"--log-level INFO")
    bg(c, orch_cmd, log="/tmp/orchestrator.log")
    time.sleep(5)

    out, _ = run(c, f"curl -s http://127.0.0.1:{ORCH_PORT}/health")
    if "healthy" not in out:
        log, _ = run(c, "tail -10 /tmp/orchestrator.log")
        print(f"  Orchestrator FAILED:\n{log}")
        c.close()
        return False

    out, _ = run(c, f"curl -s http://127.0.0.1:{ORCH_PORT}/api/v1/pool/status")
    try:
        pool = json.loads(out)
        print(f"  Orchestrator OK: {pool['total_instances']} instances, "
              f"algorithm={pool['algorithm']}, pressure={pool.get('pressure_level','?')}")
    except Exception:
        print(f"  Orchestrator OK")

    c.close()
    return True


# =====================================================================
# K6-STYLE RAMP LOAD GENERATOR ON .63
# =====================================================================

def run_ramp():
    """k6-style ramp: gradual VU scaling with per-interval metrics."""
    c = ssh(CLIENT)

    stages_json = json.dumps(RAMP_STAGES)
    total_duration = sum(s["duration"] for s in RAMP_STAGES)

    script = f'''
import asyncio, aiohttp, time, json, sys, math

URL = "http://{SERVER}:{ORCH_PORT}/api/v1/chat/completions"
STAGES = {stages_json}
SAMPLE_INTERVAL = {SAMPLE_INTERVAL}
PAYLOAD = {{"messages": [{{"role": "user", "content": "Explain DDS in one sentence."}}], "max_tokens": 10, "temperature": 0}}

# Shared state
ok = 0
fail = 0
lats = []
interval_ok = 0
interval_fail = 0
interval_lats = []
active_vus = 0
cancel_flags = []  # one asyncio.Event per worker, set = stop

async def worker(session, stop_event):
    global ok, fail, interval_ok, interval_fail
    while not stop_event.is_set():
        t0 = time.time()
        try:
            async with session.post(URL, json=PAYLOAD, timeout=aiohttp.ClientTimeout(total=30)) as r:
                data = await r.json()
                lat = (time.time() - t0) * 1000
                msg = data.get("choices", [{{}}])[0].get("message", {{}})
                content = msg.get("content", "") or msg.get("reasoning_content", "")
                if r.status == 200 and content:
                    ok += 1
                    interval_ok += 1
                else:
                    fail += 1
                    interval_fail += 1
                lats.append(lat)
                interval_lats.append(lat)
        except Exception:
            fail += 1
            interval_fail += 1
            await asyncio.sleep(0.1)

def compute_target(elapsed):
    """Compute target VUs at a given time (k6-style linear interpolation)."""
    t = 0
    prev_target = 0
    for stage in STAGES:
        dur = stage["duration"]
        target = stage["target"]
        if elapsed <= t + dur:
            # Linear interpolation within this stage
            progress = (elapsed - t) / dur if dur > 0 else 1
            return int(prev_target + (target - prev_target) * progress)
        t += dur
        prev_target = target
    return 0  # past all stages

async def main():
    global active_vus, interval_ok, interval_fail, interval_lats

    conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []       # list of (asyncio.Task, asyncio.Event)
        t_start = time.time()
        total_dur = sum(s["duration"] for s in STAGES)
        next_sample = t_start + SAMPLE_INTERVAL

        while True:
            elapsed = time.time() - t_start
            if elapsed >= total_dur:
                break

            target = compute_target(elapsed)

            # Scale UP
            while active_vus < target:
                ev = asyncio.Event()
                t = asyncio.create_task(worker(session, ev))
                tasks.append((t, ev))
                active_vus += 1

            # Scale DOWN
            while active_vus > target and tasks:
                t, ev = tasks.pop()
                ev.set()
                active_vus -= 1

            # Sample interval
            if time.time() >= next_sample:
                s = sorted(interval_lats)
                n = len(s)
                itotal = interval_ok + interval_fail
                mins = int(elapsed) // 60
                secs = int(elapsed) % 60
                result = {{
                    "time": f"{{mins:02d}}:{{secs:02d}}",
                    "vus": active_vus,
                    "total": itotal,
                    "ok": interval_ok,
                    "fail": interval_fail,
                    "p50": round(s[n//2], 1) if n else 0,
                    "p95": round(s[int(n*0.95)], 1) if n else 0,
                    "p99": round(s[min(int(n*0.99), n-1)], 1) if n else 0,
                    "rps": round(n / SAMPLE_INTERVAL, 1),
                    "error_rate": round(interval_fail / max(itotal, 1) * 100, 1),
                }}
                print(json.dumps(result), flush=True)
                interval_ok = 0
                interval_fail = 0
                interval_lats = []
                next_sample += SAMPLE_INTERVAL

            await asyncio.sleep(1)  # check every 1s

        # Stop all remaining workers
        for t, ev in tasks:
            ev.set()
        await asyncio.gather(*[t for t, _ in tasks], return_exceptions=True)

        # Final summary
        s = sorted(lats)
        n = len(s)
        total = ok + fail
        summary = {{
            "type": "summary",
            "total": total, "ok": ok, "fail": fail,
            "p50": round(s[n//2], 1) if n else 0,
            "p95": round(s[int(n*0.95)], 1) if n else 0,
            "p99": round(s[min(int(n*0.99), n-1)], 1) if n else 0,
            "rps": round(n / {total_duration}, 1),
            "error_rate": round(fail / max(total, 1) * 100, 1),
            "duration_s": {total_duration},
        }}
        print(json.dumps(summary), flush=True)

asyncio.run(main())
'''

    sftp = c.open_sftp()
    try:
        sftp.stat("/tmp/dds_e6")
    except FileNotFoundError:
        sftp.mkdir("/tmp/dds_e6")
    with sftp.open("/tmp/dds_e6/ramp.py", "w") as f:
        f.write(script)
    sftp.close()

    print(f"  k6-style ramp ({len(RAMP_STAGES)} stages, {total_duration//60}min total):")
    prev = 0
    for s in RAMP_STAGES:
        arrow = f"{prev}->{s['target']}" if prev != s['target'] else f"hold {s['target']}"
        print(f"    {arrow} ({s['duration']}s)")
        prev = s['target']

    # Run — read results line by line as they stream
    _, stdout, _ = c.exec_command(
        f"cd /tmp/dds_e6 && python3 ramp.py",
        timeout=total_duration + 120,
    )

    intervals = []
    summary = None
    print(f"\n  {'Time':>6} {'VUs':>6} {'OK':>7} {'Fail':>7} "
          f"{'p50ms':>8} {'p95ms':>8} {'RPS':>7} {'Err%':>6}")
    print(f"  {'-'*62}")

    for line in stdout:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if r.get("type") == "summary":
                summary = r
            else:
                intervals.append(r)
                print(f"  {r['time']:>6} {r['vus']:>6} {r['ok']:>7} {r['fail']:>7} "
                      f"{r['p50']:>8.1f} {r['p95']:>8.1f} {r['rps']:>7.1f} {r['error_rate']:>5.1f}%")
        except json.JSONDecodeError:
            pass

    c.close()
    return intervals, summary


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("  E6: SCALABILITY BENCHMARK")
    print(f"  29 instances (5 GPU + 24 CPU) | Qwen3.5-2B IQ2_XXS | RTX 3080")
    print(f"  Server: {SERVER} | Client: {CLIENT}")
    print(f"  Ramp: {len(RAMP_STAGES)} stages, max {max(s['target'] for s in RAMP_STAGES)} VUs")
    print("=" * 70)

    # 1. Deploy
    if not deploy():
        return 1

    # 2. Warmup
    step("Warmup (5 requests)")
    c = ssh(SERVER)
    for i in range(5):
        payload = json.dumps({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5, "temperature": 0})
        out, _ = run(c, f"curl -s --max-time 10 -X POST http://127.0.0.1:{ORCH_PORT}/api/v1/chat/completions "
                        f"-H 'Content-Type: application/json' -d '{payload}'", timeout=15)
        try:
            r = json.loads(out)
            msg = r.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "") or msg.get("reasoning_content", "")
            print(f"  #{i+1}: port={r.get('instance_port','?')} \"{content[:40]}\"")
        except Exception:
            print(f"  #{i+1}: {out[:60]}")
    c.close()

    # 3. Run k6-style ramp from .63
    step("K6-STYLE RAMP")
    intervals, summary = run_ramp()
    all_results = intervals

    # 4. Summary
    step("E6 RESULTS SUMMARY")
    if summary:
        print(f"  Total requests: {summary['total']}")
        print(f"  OK: {summary['ok']}  Fail: {summary['fail']}")
        print(f"  p50={summary['p50']}ms  p95={summary['p95']}ms  p99={summary['p99']}ms")
        print(f"  RPS={summary['rps']}  Error rate={summary['error_rate']}%")
    elif not all_results:
        print("  No results collected!")

    # VRAM/RAM
    c = ssh(SERVER)
    out, _ = run(c, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
    print(f"\n  VRAM: {out}")
    out, _ = run(c, "free -h | grep Mem")
    print(f"  RAM:  {out}")
    c.close()

    # Save results to file
    save_data = {
        "experiment": "E6", "instances": len(ALL),
        "stages": RAMP_STAGES,
        "intervals": all_results,
        "summary": summary,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_file = os.path.join(os.path.dirname(__file__), "..", "results", "E6_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {out_file}")

    # Cleanup
    step("Cleanup")
    c = ssh(SERVER)
    kill_all(c)
    c.close()

    print(f"\n{'='*70}")
    print(f"  E6 BENCHMARK COMPLETE")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
