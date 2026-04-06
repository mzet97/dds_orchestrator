#!/usr/bin/env python3
"""Debug gRPC cross-VM connectivity."""
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

    # Start llama-server
    print("Starting llama-server on .61...")
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

    # Regenerate stubs on agent from orchestrator proto
    print("Regenerating proto stubs on .61...")
    ec, o, e = run(ssh61,
        'cd /home/oldds/dds_agent/python && '
        'python3 -m grpc_tools.protoc -I /home/oldds/dds_orchestrator/proto '
        '--python_out=. --grpc_python_out=. '
        '/home/oldds/dds_orchestrator/proto/orchestrator.proto 2>&1', 30)
    print(f"  stubs: {'OK' if ec == 0 else 'FAIL ' + o[:200]}")

    # Also generate llama stubs
    run(ssh61,
        'cd /home/oldds/dds_agent/python && '
        'python3 -m grpc_tools.protoc -I /home/oldds/dds_orchestrator/benchmarks/proto '
        '--python_out=. --grpc_python_out=. '
        '/home/oldds/dds_orchestrator/benchmarks/proto/llama_service.proto 2>&1', 30)

    # Start agent gRPC on .61
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
    time.sleep(8)

    # Verify agent gRPC port
    _, o, _ = run(ssh61, 'ss -tlnp | grep 50053', 5)
    print(f"  Agent port 50053: {o.strip() or 'NOT LISTENING'}")

    # TEST 1: From .61 itself (local)
    print("\nTEST 1: Local gRPC call on .61 (localhost:50053)...")
    ec, o, e = run(ssh61,
        'cd /home/oldds/dds_agent/python && timeout 15 python3 -c "'
        'import grpc, time, json, orchestrator_pb2 as pb2, orchestrator_pb2_grpc as g; '
        'ch=grpc.insecure_channel(chr(108)+chr(111)+chr(99)+chr(97)+chr(108)+chr(104)+chr(111)+chr(115)+chr(116)+chr(58)+chr(53)+chr(48)+chr(48)+chr(53)+chr(51)); '
        'st=g.OrchestratorAgentServiceStub(ch); '
        'req=pb2.AgentTaskRequest(task_id=chr(116),requester_id=chr(116),task_type=chr(99),timeout_ms=10000); '
        'm=req.messages.add(); m.role=chr(117); m.content=chr(104)+chr(105); '
        't0=time.time(); r=st.SubmitTask(req,timeout=10); '
        'print(json.dumps(dict(time=round(time.time()-t0,3),success=r.success,content=r.content[:80],error=r.error_message)))'
        '" 2>&1', 20)
    print(f"  Result: {o.strip()}")

    # TEST 2: From .62 to .61 (cross-VM)
    print("\nTEST 2: Cross-VM gRPC call from .62 to .61:50053...")
    # First regenerate stubs on .62 too
    run(ssh62,
        'cd /home/oldds/dds_orchestrator && '
        'python3 -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/orchestrator.proto 2>&1', 30)

    ec, o, e = run(ssh62,
        'cd /home/oldds/dds_orchestrator && timeout 15 python3 -c "'
        'import sys; sys.path.insert(0,chr(112)+chr(114)+chr(111)+chr(116)+chr(111)); '
        'import grpc, time, json; '
        'from proto import orchestrator_pb2 as pb2, orchestrator_pb2_grpc as g; '
        'ch=grpc.insecure_channel(chr(49)+chr(57)+chr(50)+chr(46)+chr(49)+chr(54)+chr(56)+chr(46)+chr(49)+chr(46)+chr(54)+chr(49)+chr(58)+chr(53)+chr(48)+chr(48)+chr(53)+chr(51)); '
        'st=g.OrchestratorAgentServiceStub(ch); '
        'req=pb2.AgentTaskRequest(task_id=chr(116),requester_id=chr(116),task_type=chr(99),timeout_ms=10000); '
        'm=req.messages.add(); m.role=chr(117); m.content=chr(104)+chr(105); '
        't0=time.time(); r=st.SubmitTask(req,timeout=10); '
        'print(json.dumps(dict(time=round(time.time()-t0,3),success=r.success,content=r.content[:80],error=r.error_message)))'
        '" 2>&1', 20)
    print(f"  Result: {o.strip()}")
    if e.strip():
        print(f"  Stderr: {e.strip()[:300]}")

    # Check agent log after tests
    _, o, _ = run(ssh61, 'tail -10 /tmp/agent.log 2>/dev/null', 5)
    print(f"\nAgent log:\n{o.strip()}")

    ssh62.close()
    ssh61.close()


if __name__ == "__main__":
    main()
