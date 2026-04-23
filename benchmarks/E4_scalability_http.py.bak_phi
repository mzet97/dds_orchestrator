#!/usr/bin/env python3
"""
E4: Escalabilidade Multi-Agente - HTTP
====================================
Avalia throughput e latência com múltiplos clientes (HTTP direto ao agente).

Design experimental (conforme dissertação):
  Fase A: 1 agente, clientes = 1, 2, 4, 8
  Fase B: 2 agentes, clientes = 1, 2, 4, 8
  N = 50 requisições por cliente por configuração

Diferença para E4_DDS:
  - HTTP: requisições vão diretamente ao agente (sem orquestrador DDS)
  - DDS: requisições passam pelo orquestrador que roteia via DDS

Usage:
    python E4_scalability_http.py \\
        --agentes http://192.168.1.60:8082,http://192.168.1.61:8082 \\
        --n 50
"""

import argparse
import asyncio
import json
import time
import statistics
import psutil
import random
from pathlib import Path
from typing import Dict, List
import aiohttp

CLIENT_COUNTS = [1, 2, 4, 8]


async def run_benchmark(args):
    """Executa benchmark E4 completo (Fase A e Fase B)."""

    agentes = [a.strip() for a in args.agentes.split(",") if a.strip()]
    num_agentes = len(agentes)
    phase = "A" if num_agentes == 1 else "B" if num_agentes == 2 else f"{num_agentes}ag"

    print(f"E4: Escalabilidade - HTTP")
    print(f"Agentes ({num_agentes}): {agentes}")
    print(f"Fase: {phase}")
    print(f"Configurações de clientes: {CLIENT_COUNTS}")
    print(f"Requisições por cliente: {args.n}")
    print("=" * 60)

    async def client_request(session: aiohttp.ClientSession, agent_url: str) -> Dict:
        start = time.perf_counter()
        try:
            async with session.post(
                f"{agent_url}/v1/chat/completions",
                json={
                    "model": "phi4-mini",
                    "messages": [{"role": "user", "content": "O que e 2+2?"}],
                    "max_tokens": 20
                },
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                await resp.text()
                end = time.perf_counter()
                return {"success": resp.status == 200, "latency_ms": (end - start) * 1000}
        except Exception as e:
            end = time.perf_counter()
            return {"success": False, "latency_ms": (end - start) * 1000, "error": str(e)}

    all_summaries = []
    Path("results").mkdir(exist_ok=True)

    for num_clientes in CLIENT_COUNTS:
        print(f"\n--- Fase {phase}: {num_agentes} agente(s), {num_clientes} cliente(s) ---")

        total = num_clientes * args.n

        async with aiohttp.ClientSession() as session:
            tasks = [
                client_request(session, random.choice(agentes))
                for _ in range(total)
            ]
            results = await asyncio.gather(*tasks)

        latencies = sorted([r["latency_ms"] for r in results if r.get("success")])
        successes = len(latencies)

        if not latencies:
            print("  ERRO: nenhuma requisição bem-sucedida")
            all_summaries.append({"phase": phase, "num_clientes": num_clientes, "error": "no data"})
            continue

        throughput = successes / (latencies[-1] / 1000.0) if latencies[-1] > 0 else 0

        resources = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_mb": psutil.virtual_memory().used / 1024 / 1024
        }

        summary = {
            "protocol": "HTTP",
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
            "cpu_pct": round(resources["cpu_percent"], 1),
            "mem_mb": round(resources["memory_mb"], 1)
        }
        all_summaries.append(summary)

        print(f"  Throughput: {summary['throughput_req_s']:.2f} req/s")
        print(f"  Latência p50: {summary['latency_p50_ms']:.2f}ms  "
              f"p95: {summary['latency_p95_ms']:.2f}ms  "
              f"p99: {summary['latency_p99_ms']:.2f}ms")
        print(f"  Sucesso: {successes}/{total}")

        # Salvar CSV desta rodada
        csv_file = f"results/E4_HTTP_fase{phase}_{num_agentes}ag_{num_clientes}cl.csv"
        with open(csv_file, "w") as f:
            f.write("latency_ms,success\n")
            for r in results:
                f.write(f"{r['latency_ms']},{1 if r.get('success') else 0}\n")

        await asyncio.sleep(2.0)

    # Salvar JSON consolidado
    json_file = f"results/E4_HTTP_fase{phase}_{num_agentes}ag_summary.json"
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
    parser = argparse.ArgumentParser(description="E4: Escalabilidade - HTTP")
    parser.add_argument("--agentes", default="http://localhost:8082",
                        help="URLs dos agentes separados por vírgula. "
                             "1 agente = Fase A, 2 agentes = Fase B")
    parser.add_argument("--n", type=int, default=50,
                        help="Requisições por cliente por configuração")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
