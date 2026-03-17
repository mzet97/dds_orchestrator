#!/usr/bin/env python3
"""Full E1 DDS benchmark n=1000 from client VM .63"""
import paramiko
import socket
import io
import sys
import json
import time
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BENCH_SCRIPT = r'''
import time, statistics, requests, json

ORCH = "http://192.168.1.62:8080"

def bench(prompt, n, label, max_tokens=10):
    lats = []
    errors = 0
    for i in range(n):
        t0 = time.time()
        try:
            r = requests.post(f"{ORCH}/api/v1/chat/completions",
                json={"model":"Phi-4-mini","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens},
                timeout=60)
            lat = (time.time()-t0)*1000
            if r.status_code == 200 and "content" in r.text:
                lats.append(lat)
            else:
                errors += 1
        except Exception:
            errors += 1
        if (i+1) % 100 == 0:
            print(f"  {label}: {i+1}/{n} ({errors} errors)")

    result = {"label": label, "n": n, "ok": len(lats), "errors": errors}
    if lats:
        lats.sort()
        result["p50"] = lats[len(lats)//2]
        result["p95"] = lats[int(len(lats)*0.95)]
        result["p99"] = lats[int(len(lats)*0.99)]
        result["mean"] = statistics.mean(lats)
        result["std"] = statistics.stdev(lats) if len(lats)>1 else 0
        result["min"] = min(lats)
        result["max"] = max(lats)
        result["latencies"] = lats
        print(f"  {label}: p50={result['p50']:.1f}ms p95={result['p95']:.1f}ms mean={result['mean']:.1f}ms std={result['std']:.1f}ms ({len(lats)}/{n} ok)")
    return result

# Warmup
print("Warmup (10 requests)...")
for _ in range(10):
    requests.post(f"{ORCH}/api/v1/chat/completions",
        json={"model":"Phi-4-mini","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}, timeout=30)

results = {}

print("\nE1 Short (max_tokens=10, n=1000):")
results["short"] = bench("Hi", 1000, "short", 10)

print("\nE1 Long (max_tokens=50, n=1000):")
results["long"] = bench("Explain what is machine learning in one paragraph", 1000, "long", 50)

# Save results
for k in results:
    if "latencies" in results[k]:
        del results[k]["latencies"]  # Don't print raw latencies

print("\n=== RESULTS ===")
print(json.dumps(results, indent=2))
'''


def ssh_run(ip, cmd, timeout=30):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect((ip, 22))
    ssh.connect(ip, username='oldds', password='Admin@123', timeout=30, sock=sock)
    ssh.get_transport().set_keepalive(30)
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    ec = o.channel.recv_exit_status()
    out = o.read().decode('utf-8', errors='replace')
    err = e.read().decode('utf-8', errors='replace')
    ssh.close()
    return ec, out, err


# Upload and run
print("Uploading benchmark to .63...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(30)
sock.connect(('192.168.1.63', 22))
ssh.connect('192.168.1.63', username='oldds', password='Admin@123', timeout=30, sock=sock)
sftp = ssh.open_sftp()
with sftp.file('/tmp/bench_e1_full.py', 'w') as f:
    f.write(BENCH_SCRIPT)
sftp.close()
ssh.close()

print(f"Running E1 n=1000 from .63 at {datetime.now().strftime('%H:%M:%S')}...")
ec, out, err = ssh_run('192.168.1.63', 'python3 /tmp/bench_e1_full.py', 7200)
print(out)
if err:
    print(f"STDERR: {err[:500]}")
