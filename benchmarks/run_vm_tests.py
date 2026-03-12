#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para executar benchmarks E1-E5 nas VMs remotas
=====================================================
Usa paramiko (SSH puro em Python) - funciona no Windows sem dependencias externas

Usage:
    python run_vm_tests.py --cenario all
    python run_vm_tests.py --cenario E1 --n 20
    python run_vm_tests.py --setup-only
"""

import argparse
import asyncio
import json
import os
import sys
import time

# Fix UTF-8 encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Tentar importar paramiko
try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False
    print("ERRO: paramiko não instalado!")
    print("  Instale com: pip install paramiko")
    sys.exit(1)


# Configuração das VMs
VM_CONFIG = {
    "orchestrator": {
        "ip": "192.168.1.62",
        "user": "oldds",
        "password": "Admin@123",
        "role": "orchestrator",
        "description": "Orchestrator - sem GPU"
    },
    "agent1": {
        "ip": "192.168.1.60",
        "user": "oldds",
        "password": "Admin@123",
        "role": "agent",
        "model": "Phi-4-mini-reasoning-Q4_K_M.gguf",
        "description": "Agent 1 - RX6600M 8GB"
    },
    "agent2": {
        "ip": "192.168.1.61",
        "user": "oldds",
        "password": "Admin@123",
        "role": "agent",
        "model": "Qwen3.5-9B-Q4_K_M.gguf",
        "description": "Agent 2 - RTX 3080 10GB"
    }
}

REPOS = [
    "https://github.com/mzet97/dds_orchestrator",
    "https://github.com/mzet97/dds_agent",
    "https://github.com/mzet97/llama.cpp_dds"
]

# Branch mapping (some repos use master instead of main)
REPO_BRANCHES = {
    "llama.cpp_dds": "master"
}

MODELS_DIR = "/home/oldds/models/"


class SSHManager:
    """Gerenciador de conexões SSH."""

    def __init__(self, ip: str, user: str, password: str):
        self.ip = ip
        self.user = user
        self.password = password
        self.client: Optional[paramiko.SSHClient] = None

    def connect(self) -> bool:
        """Estabelece conexão SSH."""
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            print(f"  Conectando em {self.ip}...")
            self.client.connect(
                hostname=self.ip,
                username=self.user,
                password=self.password,
                timeout=30,
                banner_timeout=30
            )
            print(f"  [OK] Conectado em {self.ip}")
            return True
        except Exception as e:
            print(f"  [FAIL] Erro ao conectar em {self.ip}: {e}")
            return False

    def disconnect(self):
        """Fecha conexão SSH."""
        if self.client:
            self.client.close()
            self.client = None

    def execute(self, command: str, timeout: int = 120, get_pty: bool = False) -> Tuple[int, str, str]:
        """Executa comando SSH e retorna (returncode, stdout, stderr)."""
        if not self.client:
            return -1, "", "Não conectado"

        try:
            stdin, stdout, stderr = self.client.exec_command(
                command,
                timeout=timeout,
                get_pty=get_pty
            )

            # Aguardar comando terminar
            exit_code = stdout.channel.recv_exit_status()

            stdout_str = stdout.read().decode('utf-8', errors='ignore')
            stderr_str = stderr.read().decode('utf-8', errors='ignore')

            return exit_code, stdout_str, stderr_str

        except Exception as e:
            return -1, "", str(e)

    def execute_background(self, command: str) -> bool:
        """Executa comando em background (sem esperar resposta)."""
        if not self.client:
            return False

        try:
            # Usar nohup para executar em background
            cmd = f"nohup {command} > /tmp/vm_test.log 2>&1 &"
            self.client.exec_command(cmd, timeout=10)
            return True
        except Exception as e:
            print(f"    Erro ao executar em background: {e}")
            return False


async def git_pull_repos(ssh: SSHManager, base_dir: str = "/home/oldds") -> Dict:
    """Faz git pull nos repositórios."""
    results = {}

    for repo in REPOS:
        repo_name = repo.split("/")[-1]
        repo_path = f"{base_dir}/{repo_name}"

        print(f"\n    -> {repo_name}")

        # Verificar se diretório existe
        exit_code, stdout, stderr = ssh.execute(f"ls -d {repo_path}")

        if exit_code == 0:
            # CD e git pull
            print(f"      Atualizando repositório...")
            # Get the correct branch
            branch = REPO_BRANCHES.get(repo_name, "main")
            exit_code, stdout, stderr = ssh.execute(f"cd {repo_path} && git pull origin {branch}")

            if exit_code == 0:
                print(f"      [OK] Pull OK")
                results[repo_name] = {"status": "updated", "output": stdout}
            else:
                print(f"      [FAIL] Erro no pull: {stderr[:200]}")
                results[repo_name] = {"status": "error", "error": stderr}
        else:
            # Clone repositório
            print(f"      Clonando repositório...")
            exit_code, stdout, stderr = ssh.execute(f"cd {base_dir} && git clone {repo}")

            if exit_code == 0:
                print(f"      [OK] Clone OK")
                results[repo_name] = {"status": "cloned", "output": stdout}
            else:
                print(f"      [FAIL] Erro no clone: {stderr[:200]}")
                results[repo_name] = {"status": "error", "error": stderr}

    return results


async def check_models(ssh: SSHManager) -> Dict:
    """Verifica modelos disponíveis."""
    print(f"\n    Modelos em {ssh.ip}:")

    exit_code, stdout, stderr = ssh.execute(f"ls -la {MODELS_DIR}")

    if exit_code == 0:
        print(f"    {stdout}")
        return {"status": "ok", "models": stdout}
    else:
        print(f"    [FAIL] Erro: {stderr}")
        return {"status": "error", "error": stderr}


async def install_dependencies(ssh: SSHManager) -> Dict:
    """Instala dependências Python."""
    print(f"\n    Instalando dependências...")

    # Instalar pip packages necessários
    cmd = "pip3 install --user aiohttp pydantic psutil requests protobuf grpcio --quiet"
    exit_code, stdout, stderr = ssh.execute(cmd, timeout=180)

    if exit_code == 0:
        print(f"      [OK] Dependências instaladas")
        return {"status": "ok"}
    else:
        print(f"      [FAIL] Erro: {stderr[:200]}")
        return {"status": "error", "error": stderr}


async def setup_orchestrator(ssh: SSHManager) -> Dict:
    """Configura o orchestrator."""
    print(f"\n=== Configurando Orchestrator em {ssh.ip} ===")

    # Git pull
    await git_pull_repos(ssh)

    # Instalar dependências
    await install_dependencies(ssh)

    # Verificar modelos
    await check_models(ssh)

    # Verificar se orchestrator está rodando
    exit_code, stdout, stderr = ssh.execute("ps aux | grep 'python.*main.py' | grep -v grep")

    if not stdout.strip():
        # Iniciar orchestrator
        print("\n    Iniciando orchestrator...")
        cmd = "cd /home/oldds/dds_orchestrator && nohup python3 main.py --port 8080 --dds-domain 0 > /tmp/orchestrator.log 2>&1 &"
        ssh.execute_background(cmd)
        await asyncio.sleep(3)
        print("    [OK] Orchestrator iniciado")
    else:
        print("\n    [OK] Orchestrator já está rodando")

    return {"status": "ready"}


async def setup_agent(ssh: SSHManager, vm_config: Dict) -> Dict:
    """Configura o agent."""
    model = vm_config.get("model", "Phi-4-mini-reasoning-Q4_K_M.gguf")

    print(f"\n=== Configurando Agent em {ssh.ip} (model: {model}) ===")

    # Git pull
    await git_pull_repos(ssh)

    # Instalar dependências
    await install_dependencies(ssh)

    # Verificar modelos
    await check_models(ssh)

    # Compilar llama.cpp_dds se necessário
    print("\n    Verificando build do llama.cpp_dds...")
    exit_code, stdout, stderr = ssh.execute("ls /home/oldds/llama.cpp_dds/build/server 2>/dev/null && echo 'EXISTS' || echo 'NOT_FOUND'")

    if "NOT_FOUND" in stdout:
        print("    Compilando llama.cpp_dds (isso pode levar alguns minutos)...")
        cmd = "cd /home/oldds/llama.cpp_dds && mkdir -p build && cd build && cmake .. -DLLAMA_DDS=ON -DCMAKE_BUILD_TYPE=Release && make -j4"
        exit_code, stdout, stderr = ssh.execute(cmd, timeout=600)

        if exit_code == 0:
            print("    [OK] Compilação concluída")
        else:
            print(f"    [FAIL] Erro na compilação: {stderr[:300]}")
    else:
        print("    [OK] Build já existe")

    # Verificar se agent está rodando
    exit_code, stdout, stderr = ssh.execute("ps aux | grep 'python.*agent_llm' | grep -v grep")

    if not stdout.strip():
        # Iniciar agent
        print("\n    Iniciando agent...")
        model_path = f"{MODELS_DIR}{model}"
        # Primeiro verificar se llama-server está rodando
        exit_code, stdout, stderr = ssh.execute("ps aux | grep 'llama-server' | grep -v grep")

        if not stdout.strip():
            # Iniciar llama-server
            cmd = f"cd /home/oldds/llama.cpp_dds/build && nohup ./server -m '{model_path}' -ngl 99 --port 8082 > /tmp/llama-server.log 2>&1 &"
            ssh.execute_background(cmd)
            await asyncio.sleep(3)
            print("    [OK] llama-server iniciado")

        # Iniciar agent
        cmd = f"cd /home/oldds/dds_agent/python && nohup python3 agent_llm.py --model-path '{model_path}' --orchestrator-url http://192.168.1.62:8080 > /tmp/agent.log 2>&1 &"
        ssh.execute_background(cmd)
        await asyncio.sleep(2)
        print("    [OK] Agent iniciado")
    else:
        print("\n    [OK] Agent já está rodando")

    return {"status": "ready"}


async def run_benchmark(ssh: SSHManager, benchmark: str, args: str = "", timeout: int = 600) -> Dict:
    """Executa um benchmark específico."""
    print(f"\n    Executando {benchmark} {args}...")

    cmd = f"cd /home/oldds/dds_orchestrator/benchmarks && python3 {benchmark}.py {args}"

    exit_code, stdout, stderr = ssh.execute(cmd, timeout=timeout)

    result = {
        "benchmark": benchmark,
        "args": args,
        "exit_code": exit_code,
        "stdout": stdout[:2000] if stdout else "",
        "stderr": stderr[:1000] if stderr else ""
    }

    if exit_code == 0:
        print(f"      [OK] Sucesso")
    else:
        print(f"      [FAIL] Erro (código: {exit_code})")
        if stderr:
            print(f"        Erro: {stderr[:300]}")

    return result


async def run_scenario_E1(ssh: SSHManager, n: int = 50) -> List[Dict]:
    """Executa testes E1 - Decomposição de latência."""
    results = []

    tests = [
        ("E1_decompose_latency_dds", f"--model phi4-mini --prompt short --n {n}"),
        ("E1_decompose_latency_dds", f"--model phi4-mini --prompt long --n {n}"),
        ("E1_decompose_latency_http", f"--model phi4-mini --prompt short --n {n}"),
        ("E1_decompose_latency_http", f"--model phi4-mini --prompt long --n {n}"),
    ]

    for benchmark, args in tests:
        result = await run_benchmark(ssh, benchmark, args)
        results.append(result)
        await asyncio.sleep(2)

    return results


async def run_scenario_E2(ssh: SSHManager, n: int = 10) -> List[Dict]:
    """Executa testes E2 - Detecção de falha."""
    results = []

    tests = [
        ("E2_failure_detection_dds", f"--periodo 1000 --tipo kill9 --n {n}"),
        ("E2_failure_detection_dds", f"--periodo 5000 --tipo kill9 --n {n}"),
        ("E2_failure_detection_dds", f"--periodo 10000 --tipo kill9 --n {n}"),
    ]

    for benchmark, args in tests:
        result = await run_benchmark(ssh, benchmark, args)
        results.append(result)
        await asyncio.sleep(2)

    return results


async def run_scenario_E3(ssh: SSHManager, n: int = 30) -> List[Dict]:
    """Executa testes E3 - Priorização."""
    results = []

    test = ("E3_priority_dds", f"--carga 10 --n {n}")

    result = await run_benchmark(ssh, test[0], test[1])
    results.append(result)

    return results


async def run_scenario_E4(ssh: SSHManager, n: int = 20) -> List[Dict]:
    """Executa testes E4 - Escalabilidade."""
    results = []

    tests = [
        ("E4_scalability_dds", f"--clientes 1 --n {n}"),
        ("E4_scalability_dds", f"--clientes 2 --n {n}"),
        ("E4_scalability_dds", f"--clientes 4 --n {n}"),
        ("E4_scalability_dds", f"--clientes 8 --n {n}"),
    ]

    for benchmark, args in tests:
        result = await run_benchmark(ssh, benchmark, args)
        results.append(result)
        await asyncio.sleep(2)

    return results


async def run_scenario_E5(ssh: SSHManager, n: int = 10) -> List[Dict]:
    """Executa testes E5 - Streaming."""
    results = []

    tests = [
        ("E5_streaming_dds", f"--model phi4-mini --n {n}"),
        ("E5_streaming_dds", f"--model qwen3.5-9b --n {n}"),
    ]

    for benchmark, args in tests:
        result = await run_benchmark(ssh, benchmark, args)
        results.append(result)
        await asyncio.sleep(2)

    return results


async def run_all_scenarios(cenario: str, n: int) -> Dict:
    """Executa todos os cenários de teste."""
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": {},
        "summary": {}
    }

    orchestrator = VM_CONFIG["orchestrator"]
    agent1 = VM_CONFIG["agent1"]
    agent2 = VM_CONFIG["agent2"]

    # Conectar nas VMs
    print("\n" + "=" * 60)
    print("  CONECTANDO NAS VMs")
    print("=" * 60)

    ssh_orch = SSHManager(orchestrator["ip"], orchestrator["user"], orchestrator["password"])
    ssh_agent1 = SSHManager(agent1["ip"], agent1["user"], agent1["password"])
    ssh_agent2 = SSHManager(agent2["ip"], agent2["user"], agent2["password"])

    # Conectar
    if not ssh_orch.connect():
        return {"error": "Falha ao conectar no orchestrator"}
    if not ssh_agent1.connect():
        return {"error": "Falha ao conectar no agent1"}
    if not ssh_agent2.connect():
        return {"error": "Falha ao conectar no agent2"}

    # Setup
    print("\n" + "=" * 60)
    print("  SETUP DAS VMs")
    print("=" * 60)

    await setup_orchestrator(ssh_orch)
    await setup_agent(ssh_agent1, agent1)
    await setup_agent(ssh_agent2, agent2)

    # Aguardar startup
    print("\n    Aguardando 5s para serviços inicializarem...")
    await asyncio.sleep(5)

    # Executar cenários
    print("\n" + "=" * 60)
    print("  EXECUTANDO BENCHMARKS")
    print("=" * 60)

    if cenario in ["all", "E1"]:
        print("\n### CENÁRIO E1: Decomposição de Latência ###")
        all_results["scenarios"]["E1"] = await run_scenario_E1(ssh_orch, n)

    if cenario in ["all", "E2"]:
        print("\n### CENÁRIO E2: Detecção de Falha ###")
        all_results["scenarios"]["E2"] = await run_scenario_E2(ssh_orch, min(n, 10))

    if cenario in ["all", "E3"]:
        print("\n### CENÁRIO E3: Priorização ###")
        all_results["scenarios"]["E3"] = await run_scenario_E3(ssh_orch)

    if cenario in ["all", "E4"]:
        print("\n### CENÁRIO E4: Escalabilidade ###")
        all_results["scenarios"]["E4"] = await run_scenario_E4(ssh_orch, min(n, 20))

    if cenario in ["all", "E5"]:
        print("\n### CENÁRIO E5: Streaming ###")
        all_results["scenarios"]["E5"] = await run_scenario_E5(ssh_orch, min(n, 10))

    # Fechar conexões
    ssh_orch.disconnect()
    ssh_agent1.disconnect()
    ssh_agent2.disconnect()

    # Resumo
    total = 0
    passed = 0
    failed = 0

    for scenario, results in all_results["scenarios"].items():
        for r in results:
            total += 1
            if r["exit_code"] == 0:
                passed += 1
            else:
                failed += 1

    all_results["summary"] = {
        "total": total,
        "passed": passed,
        "failed": failed
    }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Executa benchmarks E1-E5 nas VMs")
    parser.add_argument("--cenario", choices=["all", "E1", "E2", "E3", "E4", "E5"], default="all",
                        help="Cenário a executar (default: all)")
    parser.add_argument("--n", type=int, default=20,
                        help="Número de iterações (default: 20)")
    parser.add_argument("--setup-only", action="store_true",
                        help="Apenas configurar VMs sem executar benchmarks")
    parser.add_argument("--output", default="vm_benchmark_results.json",
                        help="Arquivo de saída para resultados")

    args = parser.parse_args()

    print("=" * 60)
    print("  BENCHMARK E1-E5 NAS VMs")
    print("  Orchestrator: 192.168.1.62")
    print("  Agent 1: 192.168.1.60 (RX6600M)")
    print("  Agent 2: 192.168.1.61 (RTX 3080)")
    print("=" * 60)

    if not HAS_PARAMIKO:
        print("\nERRO: paramiko não está instalado!")
        print("  Execute: pip install paramiko")
        sys.exit(1)

    # Executar benchmarks
    if args.setup_only:
        print("\nModo: Setup apenas")
        # Implementar setup apenas
        print("  (Setup completo executado automaticamente durante os testes)")
    else:
        print(f"\nModo: Execução completa")
        print(f"Cenário: {args.cenario}")
        print(f"Iterações: {args.n}")

        results = asyncio.run(run_all_scenarios(args.cenario, args.n))

        # Salvar resultados
        print("\n" + "=" * 60)
        print("  RESULTADOS")
        print("=" * 60)

        if "error" in results:
            print(f"  ERRO: {results['error']}")
        else:
            summary = results.get("summary", {})
            print(f"  Total: {summary.get('total', 0)}")
            print(f"  Passou: {summary.get('passed', 0)}")
            print(f"  Falhou: {summary.get('failed', 0)}")

            print("\n  Detalhes por cenário:")
            for scenario, scenario_results in results.get("scenarios", {}).items():
                print(f"\n  {scenario}:")
                for r in scenario_results:
                    status = "[OK]" if r["exit_code"] == 0 else "[FAIL]"
                    print(f"    {status} {r['benchmark']} {r['args']}")

        # Salvar em JSON
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n  Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
