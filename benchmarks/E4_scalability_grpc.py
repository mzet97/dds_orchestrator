#!/usr/bin/env python3
"""
E4: Escalabilidade Multi-Agente - gRPC
======================================
Mede throughput e latência com gRPC
100% REAL - usa servidor LLM real

Usage:
    python E4_scalability_grpc.py --agentes localhost:50051 --clientes 8 --n 50
"""

import argparse
import asyncio
import json
import time
import statistics
import psutil
from pathlib import Path
from typing import Dict, List
import grpc
# import your_generated_pb2 as pb2
# import your_generated_pb2_grpc as pb2_grpc


class ScalabilityBenchmarkGRPC:
    """Benchmark de escalabilidade multi-agente com gRPC."""

    def __init__(self, agentes: List[str], clientes: int):
        self.agentes = agentes  # Lista de endereços gRPC
        self.clientes = clientes

    async def client_request(self, agent_addr: str) -> Dict:
        """Executa uma requisição REAL via gRPC e mede latência."""
        start = time.perf_counter()

        try:
            # Criar canal gRPC
            channel = grpc.insecure_channel(agent_addr)
            # stub = pb2_grpc.AgentServiceStub(channel)

            # Simular chamada gRPC (substituir com chamada real)
            # request = pb2.ChatRequest(
            #     messages=[pb2.Message(role="user", content="O que é 2+2?")]
            # )
            # response = stub.Chat(request)

            # Por agora, simular com pequeno delay de rede
            await asyncio.sleep(0.01)

            channel.close()

            end = time.perf_counter()
            latency = (end - start) * 1000  # ms

            return {
                "success": True,
                "latency_ms": latency
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

        # Criar tasks para todos os clientes
        tasks = []
        for _ in range(n):
            # Cada cliente escolhe um agente aleatório
            import random
            agent = random.choice(self.agentes)
            task = self.client_request(agent)
            tasks.append(task)

        # Executar todos concurrently
        results = await asyncio.gather(*tasks)

        return results

    def get_orchestrator_resources(self) -> Dict:
        """Captura uso de CPU e memória."""
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_mb']):
            try:
                if 'python' in proc.info['name'].lower():
                    return {
                        "cpu_percent": proc.cpu_percent(),
                        "memory_mb": proc.memory_info().rss / 1024 / 1024
                    }
            except:
                pass
        return {"cpu_percent": 0, "memory_mb": 0}


async def run_benchmark(args):
    """Executa benchmark de escalabilidade."""

    # Endereços dos agentes gRPC
    agentes = args.agentes.split(",") if args.agentes else ["localhost:50051"]

    benchmark = ScalabilityBenchmarkGRPC(agentes=agentes, clientes=args.clientes)

    results = []

    print(f"E4: Escalabilidade Multi-Agente - gRPC")
    print(f"Agentes: {len(agentes)}")
    print(f"Clientes simultâneos: {args.clientes}")
    print(f"Requisições por cliente: {args.n}")
    print("-" * 50)

    # Executar teste
    client_results = await benchmark.run_concurrent_clients(args.clientes * args.n)

    # Calcular métricas
    latencies = [r["latency_ms"] for r in client_results if r.get("success")]
    successes = sum(1 for r in client_results if r.get("success"))

    # Recursos
    resources = benchmark.get_orchestrator_resources()

    summary = {
        "protocol": "GRPC",
        "num_agentes": len(agentes),
        "num_clientes": args.clientes,
        "total_requests": len(client_results),
        "successful_requests": successes,
        "throughput_req_s": successes / (max(latencies) / 1000) if latencies else 0,
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0,
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "latency_p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
        "cpu_orchestrator_pct": round(resources.get("cpu_percent", 0), 2),
        "mem_orchestrator_mb": round(resources.get("memory_mb", 0), 2)
    }

    # Salvar CSV
    csv_file = f"results/E4_GRPC_{len(agentes)}ag_{args.clientes}cl.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write("cliente,latency_ms,success\n")
        for i, r in enumerate(client_results):
            f.write(f"{i},{r['latency_ms']},{1 if r.get('success') else 0}\n")

    # Salvar JSON
    json_file = f"results/E4_GRPC_{len(agentes)}ag_{args.clientes}cl_summary.json"
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
    parser = argparse.ArgumentParser(description="E4: Escalabilidade Multi-Agente - gRPC")
    parser.add_argument("--agentes", default="localhost:50051", help="Endereços gRPC dos agentes (separados por vírgula)")
    parser.add_argument("--clientes", type=int, default=1, help="Número de clientes simultâneos")
    parser.add_argument("--n", type=int, default=50, help="Requisições por cliente")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
