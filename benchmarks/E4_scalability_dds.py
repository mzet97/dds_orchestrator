#!/usr/bin/env python3
"""
E4: Escalabilidade Multi-Agente Distribuído - DDS
==================================================
Avalia throughput e latência com múltiplos clientes e agentes em ambiente distribuído.

Design experimental (conforme dissertação):
  Fase A: 1 agente (VM1 ou VM2), clientes = 1, 2, 4, 8
  Fase B: 2 agentes (VM1 + VM2), clientes = 1, 2, 4, 8
  N = 50 requisições por cliente por configuração

Ambiente distribuído (inter-VM via Proxmox):
  VM1: 192.168.1.60  - Agente AMD RX6600M (Phi-4-mini)
  VM2: 192.168.1.61  - Agente NVIDIA RTX 3080 (Qwen3.5-9B)
  VM3: 192.168.1.62  - Orquestrador (sem GPU)

Métricas: throughput (req/s), latência p50/p95/p99, CPU/memória do orquestrador.

Usage:
    # Fase A: 1 agente
    python E4_scalability_dds.py \\
        --orchestrador http://192.168.1.62:8080 \\
        --agentes http://192.168.1.60:8082 \\
        --n 50

    # Fase B: 2 agentes
    python E4_scalability_dds.py \\
        --orchestrador http://192.168.1.62:8080 \\
        --agentes http://192.168.1.60:8082,http://192.168.1.61:8082 \\
        --n 50
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

CLIENT_COUNTS = [1, 2, 4, 8]  # configurações de clientes conforme dissertação


class ScalabilityBenchmarkDDS:
    """Benchmark de escalabilidade multi-agente com DDS."""

    def __init__(self, orchestrador_url: str, agentes: List[str]):
        self.orchestrador_url = orchestrador_url
        self.agentes = agentes

    async def _single_request(self, session: aiohttp.ClientSession) -> Dict:
        """Executa uma requisição via orquestrador (DDS routing interno)."""
        start = time.perf_counter()
        try:
            async with session.post(
                f"{self.orchestrador_url}/v1/chat/completions",
                json={
                    "model": "phi4-mini",
                    "messages": [{"role": "user", "content": "O que e 2+2?"}],
                    "max_tokens": 20
                },
                timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                await resp.text()
                end = time.perf_counter()
                return {
                    "success": resp.status == 200,
                    "latency_ms": (end - start) * 1000,
                    "status": resp.status
                }
        except Exception as e:
            end = time.perf_counter()
            return {
                "success": False,
                "latency_ms": (end - start) * 1000,
                "error": str(e)
            }

    async def run_concurrent(self, num_clientes: int, n_per_client: int):
        """Executa num_clientes × n_per_client requisições concorrentes.

        Returns: (results, wall_time_s) tuple
        """
        total = num_clientes * n_per_client

        wall_start = time.perf_counter()
        async with aiohttp.ClientSession() as session:
            tasks = [self._single_request(session) for _ in range(total)]
            results = await asyncio.gather(*tasks)
        wall_end = time.perf_counter()

        return list(results), wall_end - wall_start

    def get_orchestrador_resources(self) -> Dict:
        """Captura CPU e memória do processo orquestrador."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                cmdline = " ".join(proc.info.get('cmdline') or [])
                if 'main.py' in cmdline or 'orchestrat' in cmdline.lower():
                    return {
                        "cpu_percent": proc.cpu_percent(interval=0.1),
                        "memory_mb": proc.info['memory_info'].rss / 1024 / 1024
                    }
            except Exception:
                pass
        # Fallback: sistema
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_mb": psutil.virtual_memory().used / 1024 / 1024
        }


def compute_summary(protocol: str, phase: str, num_agentes: int,
                    num_clientes: int, results: List[Dict],
                    resources: Dict, wall_time_s: float = 0) -> Dict:
    """Calcula métricas estatísticas de uma rodada."""
    latencies = sorted([r["latency_ms"] for r in results if r.get("success")])
    successes = len(latencies)
    total = len(results)

    if not latencies:
        return {"error": "nenhuma requisição bem-sucedida"}

    # Throughput: requisições bem-sucedidas / wall-clock time
    throughput = successes / wall_time_s if wall_time_s > 0 else 0

    return {
        "protocol": protocol,
        "phase": phase,
        "num_agentes": num_agentes,
        "num_clientes": num_clientes,
        "total_requests": total,
        "successful_requests": successes,
        "throughput_req_s": round(throughput, 3),
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
        "latency_p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
        "cpu_orchestrador_pct": round(resources.get("cpu_percent", 0), 2),
        "mem_orchestrador_mb": round(resources.get("memory_mb", 0), 2)
    }


async def run_benchmark(args):
    """Executa benchmark E4 completo (Fase A e Fase B)."""

    agentes = [a.strip() for a in args.agentes.split(",") if a.strip()]
    num_agentes = len(agentes)
    phase = "A" if num_agentes == 1 else "B" if num_agentes == 2 else f"{num_agentes}ag"

    protocol_label = args.protocol_label
    benchmark = ScalabilityBenchmarkDDS(
        orchestrador_url=args.orchestrador,
        agentes=agentes
    )

    print(f"E4: Escalabilidade Multi-Agente - DDS")
    print(f"Orquestrador: {args.orchestrador}")
    print(f"Agentes ({num_agentes}): {agentes}")
    print(f"Fase: {phase}")
    print(f"Configurações de clientes: {CLIENT_COUNTS}")
    print(f"Requisições por cliente: {args.n}")
    print("=" * 60)

    all_summaries = []
    Path("results").mkdir(exist_ok=True)

    # Iterar sobre cada configuração de clientes
    for num_clientes in CLIENT_COUNTS:
        print(f"\n--- Fase {phase}: {num_agentes} agente(s), {num_clientes} cliente(s) ---")

        resources_before = benchmark.get_orchestrador_resources()

        # Executar rodada
        results, wall_time_s = await benchmark.run_concurrent(num_clientes, args.n)

        resources_after = benchmark.get_orchestrador_resources()
        resources = {
            "cpu_percent": max(resources_before["cpu_percent"], resources_after["cpu_percent"]),
            "memory_mb": resources_after["memory_mb"]
        }

        summary = compute_summary(
            protocol=protocol_label,
            phase=phase,
            num_agentes=num_agentes,
            num_clientes=num_clientes,
            results=results,
            resources=resources,
            wall_time_s=wall_time_s,
        )
        all_summaries.append(summary)

        print(f"  Throughput: {summary.get('throughput_req_s', 0):.2f} req/s")
        print(f"  Latência p50: {summary.get('latency_p50_ms', 0):.2f}ms  "
              f"p95: {summary.get('latency_p95_ms', 0):.2f}ms  "
              f"p99: {summary.get('latency_p99_ms', 0):.2f}ms")
        print(f"  Sucesso: {summary.get('successful_requests', 0)}/{summary.get('total_requests', 0)}")

        # Salvar CSV desta rodada
        csv_file = f"results/E4_{protocol_label}_fase{phase}_{num_agentes}ag_{num_clientes}cl.csv"
        with open(csv_file, "w") as f:
            f.write("latency_ms,success\n")
            for r in results:
                f.write(f"{r['latency_ms']},{1 if r.get('success') else 0}\n")

        # Pausa entre configurações para estabilizar
        await asyncio.sleep(2.0)

    # Salvar JSON consolidado
    json_file = f"results/E4_{protocol_label}_fase{phase}_{num_agentes}ag_summary.json"
    with open(json_file, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Resumo Fase {phase} ({num_agentes} agente(s)):")
    print(f"{'Clientes':>8} {'p50(ms)':>10} {'p95(ms)':>10} {'req/s':>8}")
    for s in all_summaries:
        if "error" not in s:
            print(f"{s['num_clientes']:>8} {s['latency_p50_ms']:>10.1f} "
                  f"{s['latency_p95_ms']:>10.1f} {s['throughput_req_s']:>8.2f}")

    print(f"\nJSON: {json_file}")
    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="E4: Escalabilidade Multi-Agente - DDS")
    parser.add_argument("--orchestrador", default="http://localhost:8080",
                        help="URL do orquestrador DDS")
    parser.add_argument("--agentes", default="http://localhost:8082",
                        help="URLs dos agentes separados por vírgula. "
                             "1 agente = Fase A, 2 agentes = Fase B")
    parser.add_argument("--n", type=int, default=50,
                        help="Requisições por cliente por configuração")
    parser.add_argument("--protocol-label", default="DDS", help="Label do protocolo para nomes de arquivos")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
