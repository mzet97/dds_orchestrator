#!/usr/bin/env python3
"""
E4: Escalabilidade Multi-Agente Distribuído
==============================================
Mede throughput e latência com múltiplos clientes e agentes
100% REAL - usa servidor LLM real em múltiplas VMs

Usage:
    python E4_scalability_dds.py --agentes 2 --clientes 8 --n 50
"""

import argparse
import asyncio
import json
import time
import statistics
import psutil
from pathlib import Path
from typing import Dict, List
import aiohttp


class ScalabilityBenchmark:
    """Benchmark de escalabilidade multi-agente."""

    def __init__(self, agentes: List[str], clientes: int):
        self.agentes = agentes  # Lista de URLs dos agentes
        self.clientes = clientes

    async def client_request(self, session: aiohttp.ClientSession, agent_url: str) -> Dict:
        """Executa uma requisição REAL e mede latência."""
        start = time.perf_counter()

        try:
            async with session.post(
                f"{agent_url}/chat",
                json={"messages": [{"role": "user", "content": "O que é 2+2?"}]},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                await resp.text()
                end = time.perf_counter()
                latency = (end - start) * 1000  # ms

                return {
                    "success": True,
                    "latency_ms": latency,
                    "status": resp.status
                }
        except Exception as e:
            end = time.perf_counter()
            return {
                "success": False,
                "latency_ms": (end - start) * 1000,
                "error": str(e)
            }

    async def run_concurrent_clients(self, n: int) -> List[Dict]:
        """Executa N requisições concorrentes de múltiplos clientes."""
        results = []

        async with aiohttp.ClientSession() as session:
            # Criar tasks para todos os clientes
            tasks = []
            for _ in range(n):
                # Cada cliente escolhe um agente aleatório
                import random
                agent = random.choice(self.agentes)
                task = self.client_request(session, agent)
                tasks.append(task)

            # Executar todos concurrently
            results = await asyncio.gather(*tasks)

        return results

    def get_orchestrator_resources(self) -> Dict:
        """Captura uso de CPU e memória do orchestrator."""
        # Encontrar processo do orchestrator
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            try:
                if 'python' in proc.info['name'].lower():
                    return {
                        "cpu_percent": proc.cpu_percent(),
                        "memory_mb": proc.info['memory_info'].rss / 1024 / 1024 if proc.info.get('memory_info') else 0
                    }
            except:
                pass
        return {"cpu_percent": 0, "memory_mb": 0}


async def run_benchmark(args):
    """Executa benchmark de escalabilidade."""

    # URLs dos agentes (VMs reais)
    agentes = args.agentes.split(",") if args.agentes else ["http://localhost:8082"]

    benchmark = ScalabilityBenchmark(agentes=agentes, clientes=args.clientes)

    results = []

    print(f"E4: Escalabilidade Multi-Agente")
    print(f"Agentes: {len(agentes)}")
    print(f"Clientes simultâneos: {args.clientes}")
    print(f"Requisições por cliente: {args.n}")
    print("-" * 50)

    # Executar teste
    client_results = await benchmark.run_concurrent_clients(args.clientes * args.n)

    # Calcular métricas
    latencies = [r["latency_ms"] for r in client_results if r.get("success")]
    successes = sum(1 for r in client_results if r.get("success"))

    # Recursos do orchestrator
    resources = benchmark.get_orchestrator_resources()

    summary = {
        "protocol": "DDS",
        "num_agentes": len(agentes),
        "num_clientes": args.clientes,
        "total_requests": len(client_results),
        "successful_requests": successes,
        "throughput_req_s": successes / (max(latencies) / 1000) if latencies else 0,
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0,
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 2),
        "latency_p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0, 2),
        "cpu_orchestrator_pct": round(resources.get("cpu_percent", 0), 2),
        "mem_orchestrator_mb": round(resources.get("memory_mb", 0), 2)
    }

    # Salvar CSV
    csv_file = f"results/E4_scalability_{len(agentes)}ag_{args.clientes}cl.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write("cliente,latency_ms,success\n")
        for i, r in enumerate(client_results):
            f.write(f"{i},{r['latency_ms']},{1 if r.get('success') else 0}\n")

    # Salvar JSON
    json_file = f"results/E4_scalability_{len(agentes)}ag_{args.clientes}cl_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"Throughput: {summary['throughput_req_s']:.2f} req/s")
    print(f"Latência p50: {summary['latency_p50_ms']:.2f}ms")
    print(f"Latência p95: {summary['latency_p95_ms']:.2f}ms")
    print(f"CPU: {summary['cpu_orchestrator_pct']:.1f}%")
    print(f"Memória: {summary['mem_orchestrator_mb']:.1f}MB")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E4: Escalabilidade Multi-Agente")
    parser.add_argument("--agentes", default="http://localhost:8082", help="URLs dos agentes (separados por vírgula)")
    parser.add_argument("--clientes", type=int, default=1, help="Número de clientes simultâneos")
    parser.add_argument("--n", type=int, default=50, help="Requisições por cliente")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
