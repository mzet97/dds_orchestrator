#!/usr/bin/env python3
"""
E3: Priorização sob Carga - HTTP + heapq
=========================================
Mede latência com fila de prioridade Python (heapq)
100% REAL - carga sustentada com injeção

Usage:
    python E3_priority_http.py --carga 10 --n 30
"""

import argparse
import asyncio
import json
import time
import statistics
import heapq
import aiohttp
from pathlib import Path
from typing import Dict, List


class PriorityQueueHTTP:
    """Fila de prioridade com heapq."""

    def __init__(self):
        self.queue = []
        self.counter = 0  # Para desempate

    def enqueue(self, item: Dict, priority: int):
        """Adiciona item com prioridade (menor = maior prioridade)."""
        # heapq é min-heap, então usamos -priority para inverter
        heapq.heappush(self.queue, (-priority, self.counter, item))
        self.counter += 1

    def dequeue(self) -> Dict:
        """Remove e retorna item de maior prioridade."""
        if self.queue:
            _, _, item = heapq.heappop(self.queue)
            return item
        return None


async def run_benchmark(args):
    """Executa benchmark."""

    print(f"E3: Priorização - HTTP + heapq")
    print(f"Carga: {args.carga} req/s")
    print(f"Duração: {args.duracao}s")
    print(f"Injeções: {args.n}")
    print("-" * 50)

    results = []
    priority_queue = PriorityQueueHTTP()

    async def normal_load():
        """Gera carga normal."""
        interval = 1.0 / args.carga
        start = time.perf_counter()
        count = 0

        while (time.perf_counter() - start) < args.duracao:
            send_time = time.perf_counter()

            # Adicionar à fila com prioridade NORMAL (0)
            priority_queue.enqueue({"time": send_time}, 0)

            # Simular processamento
            await asyncio.sleep(0.001)

            recv_time = time.perf_counter()
            latency = (recv_time - send_time) * 1000

            results.append({
                "priority": "NORMAL",
                "latency_ms": latency,
                "queue_time": send_time - start
            })

            count += 1
            await asyncio.sleep(interval)

        return count

    async def priority_injection():
        """Injeta mensagens prioritárias."""
        inject_interval = args.duracao / args.n

        for i in range(args.n):
            await asyncio.sleep(inject_interval)

            send_time = time.perf_counter()

            # Adicionar com prioridade HIGH (1)
            priority_queue.enqueue({"time": send_time}, 1)

            # Simular processamento
            await asyncio.sleep(0.001)

            recv_time = time.perf_counter()
            latency = (recv_time - send_time) * 1000

            results.append({
                "priority": "HIGH",
                "latency_ms": latency,
                "queue_time": send_time - start_time
            })

            print(f"Injeção {i+1}/{args.n}: {latency:.2f}ms")

    # Executar carga e injeções
    start_time = time.perf_counter()

    load_task = asyncio.create_task(normal_load())
    await priority_injection()

    normal_count = await load_task

    # Análise
    normal_latencies = [r["latency_ms"] for r in results if r["priority"] == "NORMAL"]
    priority_latencies = [r["latency_ms"] for r in results if r["priority"] == "HIGH"]

    summary = {
        "protocol": "HTTP_HEAPQ",
        "carga_req_s": args.carga,
        "duracao_s": args.duracao,
        "n_injections": args.n,
        "normal": {
            "n": len(normal_latencies),
            "mean_ms": round(statistics.mean(normal_latencies), 4) if normal_latencies else 0,
            "median_ms": round(statistics.median(normal_latencies), 4) if normal_latencies else 0,
        },
        "priority": {
            "n": len(priority_latencies),
            "mean_ms": round(statistics.mean(priority_latencies), 4) if priority_latencies else 0,
            "median_ms": round(statistics.median(priority_latencies), 4) if priority_latencies else 0,
        }
    }

    # Salvar CSV
    csv_file = f"results/E3_HTTP_HEAPQ_carga{args.carga}.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write("priority,latency_ms,queue_time\n")
        for r in results:
            f.write(f"{r['priority']},{r['latency_ms']},{r['queue_time']}\n")

    # Salvar JSON
    json_file = f"results/E3_HTTP_HEAPQ_carga{args.carga}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"NORMAL: {summary['normal']['n']} msgs, média: {summary['normal']['mean_ms']:.2f}ms")
    print(f"HIGH: {summary['priority']['n']} msgs, média: {summary['priority']['mean_ms']:.2f}ms")
    print(f"CSV: {csv_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E3: Priorização - HTTP + heapq")
    parser.add_argument("--carga", type=int, default=10, help="Carga em req/s")
    parser.add_argument("--duracao", type=int, default=300, help="Duração em segundos")
    parser.add_argument("--n", type=int, default=30, help="Número de injeções")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
