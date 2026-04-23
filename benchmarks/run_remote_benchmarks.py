#!/usr/bin/env python3
"""
Script para executar benchmarks E1-E5 nas VMs remotas
=====================================================
SSH automation para deployment e execução de testes

Usage:
    python run_remote_benchmarks.py --vms 192.168.1.60,192.168.1.61,192.168.1.62
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict
import json


# Configuração das VMs
VM_CONFIG = {
    "orchestrator": {
        "ip": "192.168.1.62",
        "user": "oldds",
        "password": "Admin@123",
        "role": "orchestrator"
    },
    "agent1": {
        "ip": "192.168.1.60",
        "user": "oldds",
        "password": "Admin@123",
        "role": "agent",
        "model": "Qwen3.5-0.8B-reasoning-Q4_K_M.gguf"
    },
    "agent2": {
        "ip": "192.168.1.61",
        "user": "oldds",
        "password": "Admin@123",
        "role": "agent",
        "model": "Qwen3.5-9B-Q4_K_M.gguf"
    }
}

REPOS = [
    "https://github.com/mzet97/dds_orchestrator",
    "https://github.com/mzet97/dds_agent",
    "https://github.com/mzet97/llama.cpp_dds"
]

MODELS_DIR = "/home/oldds/models/"


def get_ssh_command(vm_ip: str, user: str, password: str, command: str) -> List[str]:
    """Retorna comando SSH usando sshpass."""
    # sshpass -p 'password' ssh user@ip 'command'
    return ["sshpass", "-p", password, "ssh", "-o", "StrictHostKeyChecking=no",
            f"{user}@{vm_ip}", command]


async def run_ssh_command(ip: str, user: str, password: str, command: str) -> Dict:
    """Executa comando SSH na VM."""
    import subprocess

    full_cmd = f"sshpass -p '{password}' ssh -o StrictHostKeyChecking=no {user}@{ip} '{command}'"

    print(f"  [SSH] {user}@{ip}: {command[:80]}...")

    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout", "returncode": -1}
    except Exception as e:
        return {"success": False, "error": str(e), "returncode": -1}


async def git_pull_repos(vm_ip: str, user: str, password: str, base_dir: str = "/home/oldds") -> Dict:
    """Faz git pull nos repositórios."""
    results = {}

    for repo in REPOS:
        repo_name = repo.split("/")[-1]
        repo_path = f"{base_dir}/{repo_name}"

        print(f"\n  -> {repo_name}")

        # Verificar se diretório existe
        cmd_check = f"ls -d {repo_path}"
        result = await run_ssh_command(vm_ip, user, password, cmd_check)

        if result.get("success") and repo_path in result.get("stdout", ""):
            # CD e git pull
            cmd_pull = f"cd {repo_path} && git pull origin main"
            result = await run_ssh_command(vm_ip, user, password, cmd_pull)
            print(f"     Pull: {'OK' if result.get('success') else 'FALHOU'}")
            if not result.get("success"):
                print(f"     Error: {result.get('stderr', result.get('error', ''))[:200]}")
            results[repo_name] = result
        else:
            # Clone repositório
            cmd_clone = f"cd {base_dir} && git clone {repo}"
            result = await run_ssh_command(vm_ip, user, password, cmd_clone)
            print(f"     Clone: {'OK' if result.get('success') else 'FALHOU'}")
            if not result.get("success"):
                print(f"     Error: {result.get('stderr', result.get('error', ''))[:200]}")
            results[repo_name] = result

    return results


async def check_models(vm_ip: str, user: str, password: str) -> Dict:
    """Verifica modelos disponíveis."""
    cmd = f"ls -la {MODELS_DIR}"
    result = await run_ssh_command(vm_ip, user, password, cmd)

    if result.get("success"):
        print(f"\n  Modelos em {vm_ip}:")
        print(f"  {result['stdout']}")

    return result


async def install_dependencies(vm_ip: str, user: str, password: str) -> Dict:
    """Instala dependências Python."""
    print(f"\n  Instalando dependências...")

    # Instalar pip packages necessários
    cmd = "pip install aiohttp pydantic psutil requests protobuf grpcio"

    result = await run_ssh_command(vm_ip, user, password, cmd)
    print(f"     {'OK' if result.get('success') else 'FALHOU'}")

    return result


async def run_benchmark_on_vm(vm_ip: str, user: str, password: str, benchmark: str,
                               args: str = "") -> Dict:
    """Executa um benchmark específico na VM."""
    print(f"\n  Executando {benchmark}...")

    cmd = f"cd /home/oldds/dds_orchestrator/benchmarks && python {benchmark}.py {args}"

    result = await run_ssh_command(vm_ip, user, password, cmd)
    print(f"     {'OK' if result.get('success') else 'FALHOU'}")

    if result.get("stdout"):
        print(f"     Output: {result['stdout'][:500]}")

    return result


async def setup_orchestrator(vm_config: Dict) -> Dict:
    """Configura e inicia o orchestrator."""
    ip = vm_config["ip"]
    user = vm_config["user"]
    password = vm_config["password"]

    print(f"\n=== Configurando Orchestrator em {ip} ===")

    # Git pull
    await git_pull_repos(ip, user, password)

    # Instalar dependências
    await install_dependencies(ip, user, password)

    # Verificar modelos
    await check_models(ip, user, password)

    # Verificar se orchestrator está rodando
    cmd_check = "ps aux | grep 'python.*main.py' | grep -v grep"
    result = await run_ssh_command(ip, user, password, cmd_check)

    if not result.get("stdout"):
        # Iniciar orchestrator
        print("  Iniciando orchestrator...")
        cmd_start = "cd /home/oldds/dds_orchestrator && nohup python main.py --port 8080 --dds-domain 0 > orchestrator.log 2>&1 &"
        await run_ssh_command(ip, user, password, cmd_start)

    return {"status": "ready"}


async def setup_agent(vm_config: Dict, agent_num: int) -> Dict:
    """Configura e inicia o agent."""
    ip = vm_config["ip"]
    user = vm_config["user"]
    password = vm_config["password"]
    model = vm_config.get("model", "Qwen3.5-0.8B-reasoning-Q4_K_M.gguf")

    print(f"\n=== Configurando Agent {agent_num} em {ip} (model: {model}) ===")

    # Git pull
    await git_pull_repos(ip, user, password)

    # Instalar dependências
    await install_dependencies(ip, user, password)

    # Verificar modelos
    await check_models(ip, user, password)

    # Compilar llama.cpp_dds se necessário
    print("  Verificando build do llama.cpp_dds...")
    cmd_build = "cd /home/oldds/llama.cpp_dds && ls build/server 2>/dev/null || (mkdir -p build && cd build && cmake .. -DLLAMA_DDS=ON && make -j4)"
    await run_ssh_command(ip, user, password, cmd_build)

    # Verificar se agent está rodando
    cmd_check = "ps aux | grep 'python.*agent_llm.py' | grep -v grep"
    result = await run_ssh_command(ip, user, password, cmd_check)

    if not result.get("stdout"):
        # Iniciar agent
        print("  Iniciando agent...")
        model_path = f"{MODELS_DIR}{model}"
        cmd_start = f"cd /home/oldds/dds_agent/python && nohup python agent_llm.py --model-path '{model_path}' --orchestrator-url http://192.168.1.62:8080 > agent.log 2>&1 &"
        await run_ssh_command(ip, user, password, cmd_start)

    return {"status": "ready"}


async def run_all_benchmarks() -> Dict:
    """Executa todos os benchmarks E1-E5."""
    results = {}

    # Configurar cada VM
    orchestrator = VM_CONFIG["orchestrator"]
    agent1 = VM_CONFIG["agent1"]
    agent2 = VM_CONFIG["agent2"]

    # Setup
    await setup_orchestrator(orchestrator)
    await setup_agent(agent1, 1)
    await setup_agent(agent2, 2)

    # Aguardar startup
    print("\n  Aguardando 5s para serviços inicializarem...")
    await asyncio.sleep(5)

    # Executar benchmarks
    benchmarks = [
        ("E1_decompose_latency_dds", ""),
        ("E1_decompose_latency_http", ""),
        ("E2_failure_detection_dds", "--intervalo 1000 --n 5"),
        ("E2_failure_detection_http", "--intervalo 1000 --n 5"),
        ("E3_priority_dds", "--carga 10 --duracao 30 --n 5"),
        ("E3_priority_http", "--carga 10 --duracao 30 --n 5"),
        ("E4_scalability_dds", "--agentes http://192.168.1.60:8082,http://192.168.1.61:8082 --clientes 4 --n 10"),
        ("E4_scalability_http", "--agentes http://192.168.1.60:8082,http://192.168.1.61:8082 --clientes 4 --n 10"),
        ("E5_streaming_dds", "--model qwen3.5-0.8b --n 5"),
        ("E5_streaming_http", "--model qwen3.5-0.8b --n 5"),
    ]

    orch_ip = orchestrator["ip"]
    orch_user = orchestrator["user"]
    orch_password = orchestrator["password"]

    for benchmark_name, args in benchmarks:
        print(f"\n{'='*60}")
        print(f"  Executando: {benchmark_name}")
        print(f"{'='*60}")

        result = await run_benchmark_on_vm(orch_ip, orch_user, orch_password, benchmark_name, args)
        results[benchmark_name] = result

        await asyncio.sleep(2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Executa benchmarks E1-E5 nas VMs")
    parser.add_argument("--vms", help="IPs das VMs (separados por vírgula)")
    parser.add_argument("--benchmark", help="Executar apenas um benchmark específico")
    parser.add_argument("--setup-only", action="store_true", help="Apenas configurar VMs")

    args = parser.parse_args()

    print("=" * 60)
    print("  BENCHMARK E1-E5 REMOTO")
    print("  VMs: 192.168.1.60, 192.168.1.61, 192.168.1.62")
    print("=" * 60)

    # Verificar sshpass
    import subprocess
    try:
        subprocess.run(["sshpass", "-V"], capture_output=True, check=True)
    except:
        print("ERRO: sshpass não instalado!")
        print("  Instale com: apt install sshpass (Linux) ou brew install sshpass (Mac)")
        sys.exit(1)

    # Executar benchmarks
    if args.setup_only:
        print("\nModo: Setup apenas")
        # Implementar setup
    else:
        print("\nModo: Execução completa")
        results = asyncio.run(run_all_benchmarks())

        # Salvar resultados
        print("\n" + "=" * 60)
        print("  RESULTADOS")
        print("=" * 60)

        for name, result in results.items():
            status = "OK" if result.get("success") else "FALHOU"
            print(f"  {name}: {status}")

        # Salvar em JSON
        with open("remote_benchmarks_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n  Resultados salvos em: remote_benchmarks_results.json")


if __name__ == "__main__":
    main()
