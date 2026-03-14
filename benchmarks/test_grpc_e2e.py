#!/usr/bin/env python3
"""Test full gRPC E2E: client -> orch -> agent -> llama."""
import paramiko
import socket
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def ssh_connect(ip):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect((ip, 22))
    ssh.connect(ip, username='oldds', password='Admin@123', timeout=30, sock=sock)
    ssh.get_transport().set_keepalive(30)
    return ssh


def run(ssh, cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    ec = o.channel.recv_exit_status()
    return ec, o.read().decode('utf-8', errors='replace'), e.read().decode('utf-8', errors='replace')


def main():
    ssh62 = ssh_connect('192.168.1.62')
    ssh61 = ssh_connect('192.168.1.61')

    # Kill all
    print("Killing all services...")
    for s in [ssh62, ssh61]:
        run(s, 'pkill -9 -f "main.py|agent_llm|llama-server" 2>/dev/null; true', 5)
    time.sleep(3)

    # Git pull
    for s, ip in [(ssh62, '.62'), (ssh61, '.61')]:
        for repo in ['dds_orchestrator', 'dds_agent']:
            _, o, _ = run(s, f'cd /home/oldds/{repo} && git fetch origin && git reset --hard origin/main 2>&1 | tail -1', 30)
            print(f"  {ip} {repo}: {o.strip()[:80]}")

    # Regen stubs on .61
    run(ssh61,
        'cd /home/oldds/dds_agent/python && '
        'python3 -m grpc_tools.protoc -I /home/oldds/dds_orchestrator/proto '
        '--python_out=. --grpc_python_out=. /home/oldds/dds_orchestrator/proto/orchestrator.proto && '
        'python3 -m grpc_tools.protoc -I /home/oldds/dds_orchestrator/benchmarks/proto '
        '--python_out=. --grpc_python_out=. /home/oldds/dds_orchestrator/benchmarks/proto/llama_service.proto'
        , 30)
    # Regen on .62
    run(ssh62,
        'cd /home/oldds/dds_orchestrator && '
        'python3 -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/orchestrator.proto'
        , 30)

    # 1. Start llama-server on .61
    print("\nStarting llama-server on .61...")
    ssh61.exec_command(
        'nohup /home/oldds/llama.cpp_dds/build/bin/llama-server '
        '-m /home/oldds/models/Qwen3.5-9B-Q4_K_M.gguf '
        '-c 2048 --threads 8 -ngl 99 --port 8082 --host 0.0.0.0 '
        '> /tmp/llama.log 2>&1 &'
    )
    for i in range(12):
        time.sleep(5)
        _, o, _ = run(ssh61, 'curl -s http://localhost:8082/health', 5)
        if 'ok' in o.lower():
            print("  llama-server ready")
            break

    # 2. Start orchestrator on .62 FIRST
    print("Starting orchestrator on .62...")
    ssh62.exec_command(
        'nohup python3 -u /home/oldds/dds_orchestrator/main.py '
        '--port 8080 --grpc-enabled --grpc-port 50052 --log-level INFO '
        '> /tmp/orch.log 2>&1 &'
    )
    time.sleep(8)
    _, o, _ = run(ssh62, 'ss -tlnp | grep 50052', 5)
    print(f"  Port 50052: {o.strip() or 'NOT LISTENING'}")

    # 3. Start agent gRPC on .61 AFTER orchestrator
    print("Starting agent gRPC on .61...")
    ssh61.exec_command(
        'export HOSTNAME=192.168.1.61 && '
        'nohup python3 -u /home/oldds/dds_agent/python/agent_llm_grpc.py '
        '--model-name Qwen3.5-9B '
        '--model-path /home/oldds/models/Qwen3.5-9B-Q4_K_M.gguf '
        '--orchestrator-url http://192.168.1.62:8080 '
        '--port 8081 --grpc-address localhost:50051 --grpc-listen-port 50053 '
        '--no-server > /tmp/agent.log 2>&1 &'
    )
    time.sleep(10)

    # Check registration
    _, o, _ = run(ssh62, 'curl -s http://localhost:8080/api/v1/agents')
    registered = 'agent_id' in o
    print(f"  Agent registered: {registered}")
    if not registered:
        _, o, _ = run(ssh61, 'tail -5 /tmp/agent.log', 5)
        print(f"  Agent log: {o.strip()}")

    # Warmup
    print("\nWarmup...")
    run(ssh62,
        'curl -s -X POST http://localhost:8080/v1/chat/completions '
        '-H "Content-Type: application/json" '
        '-d \'{"model":"Qwen3.5-9B","messages":[{"role":"user","content":"warmup"}],"max_tokens":5}\''
        , 120)
    print("  Done")

    # FULL E2E TEST: gRPC client from .62 to orch to agent
    print("\n=== FULL gRPC E2E TEST ===")
    print("client(.62) -> gRPC -> orch(.62:50052) -> gRPC -> agent(.61:50053) -> HTTP -> llama")
    ec, o, e = run(ssh62,
        'cd /home/oldds/dds_orchestrator && timeout 30 python3 -c "'
        'import sys; sys.path.insert(0,chr(112)+chr(114)+chr(111)+chr(116)+chr(111)); '
        'import grpc, time, json; '
        'from proto import orchestrator_pb2 as pb2, orchestrator_pb2_grpc as g; '
        'ch=grpc.insecure_channel(chr(108)+chr(111)+chr(99)+chr(97)+chr(108)+chr(104)+chr(111)+chr(115)+chr(116)+chr(58)+chr(53)+chr(48)+chr(48)+chr(53)+chr(50)); '
        'st=g.ClientOrchestratorServiceStub(ch); '
        'req=pb2.ClientChatRequest(request_id=chr(116)+chr(101)+chr(115)+chr(116),model=chr(81),max_tokens=10,timeout_ms=25000); '
        'm=req.messages.add(); m.role=chr(117)+chr(115)+chr(101)+chr(114); m.content=chr(87)+chr(104)+chr(97)+chr(116)+chr(32)+chr(105)+chr(115)+chr(32)+chr(50)+chr(43)+chr(50)+chr(63); '
        't0=time.time(); '
        'try:\\n'
        '    r=st.Chat(req,timeout=25)\\n'
        '    print(json.dumps(dict(time_s=round(time.time()-t0,2),success=r.success,content=r.content[:200],error=r.error_message)))\\n'
        'except Exception as e:\\n'
        '    print(json.dumps(dict(time_s=round(time.time()-t0,2),error=str(e)[:300])))\\n'
        'ch.close()'
        '" 2>&1', 40)
    print(f"  Result: {o.strip()}")
    if e.strip():
        print(f"  Stderr: {e.strip()[:300]}")

    # Check logs
    _, o, _ = run(ssh62, 'grep -i "gRPC calling\\|gRPC response\\|Error" /tmp/orch.log 2>/dev/null | tail -5', 5)
    print(f"\nOrch log: {o.strip()}")
    _, o, _ = run(ssh61, 'tail -5 /tmp/agent.log', 5)
    print(f"Agent log: {o.strip()}")

    ssh62.close()
    ssh61.close()


if __name__ == "__main__":
    main()
