#!/usr/bin/env python3
"""Quick E1 DDS benchmark from client VM .63"""
import paramiko
import socket
import time
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BENCH_SCRIPT = r'''
import time, statistics, requests

ORCH = "http://192.168.1.62:8080"

def bench(prompt, n, label, max_tokens=10):
    lats = []
    for i in range(n):
        t0 = time.time()
        r = requests.post(f"{ORCH}/api/v1/chat/completions",
            json={"model":"Phi-4-mini","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens},
            timeout=30)
        lat = (time.time()-t0)*1000
        if r.status_code == 200 and "content" in r.text:
            lats.append(lat)
        if (i+1) % 20 == 0:
            print(f"  {label}: {i+1}/{n}")
    if lats:
        lats.sort()
        p50 = lats[len(lats)//2]
        mean = statistics.mean(lats)
        std = statistics.stdev(lats) if len(lats)>1 else 0
        print(f"  {label}: p50={p50:.1f}ms mean={mean:.1f}ms std={std:.1f}ms ({len(lats)}/{n} ok)")
    return lats

# Warmup
print("Warmup...")
for _ in range(5):
    requests.post(f"{ORCH}/api/v1/chat/completions",
        json={"model":"Phi-4-mini","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}, timeout=30)

print("\nE1 Short (max_tokens=10):")
short = bench("Hi", 100, "short", 10)

print("\nE1 Long (max_tokens=50):")
long = bench("Explain what is machine learning in one paragraph", 100, "long", 50)

print("\nDONE")
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


# Upload and run benchmark script on .63
print("Uploading benchmark to .63...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(30)
sock.connect(('192.168.1.63', 22))
ssh.connect('192.168.1.63', username='oldds', password='Admin@123', timeout=30, sock=sock)

sftp = ssh.open_sftp()
with sftp.file('/tmp/bench_e1.py', 'w') as f:
    f.write(BENCH_SCRIPT)
sftp.close()
ssh.close()

print("Running E1 benchmark from .63...")
ec, out, err = ssh_run('192.168.1.63', 'python3 /tmp/bench_e1.py', 600)
print(out)
if err:
    print(f"STDERR: {err[:300]}")
print("Benchmark complete!")
