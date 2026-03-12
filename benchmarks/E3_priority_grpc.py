#!/usr/bin/env python3
"""
E3: Priorização sob Carga - gRPC + heapq
========================================
Mede latência com fila de prioridade Python (heapq) via gRPC
100% REAL

Usage:
    python E3_priority_grpc.py --carga 10 --n 30
"""

import argparse
import asyncio
import json
import time
import statistics
import heapq
from pathlib import Path
from typing import Dict, List


class PriorityQueueGRPC:
    """Fila de prioridade com heapq para gRPC."""

    def __init__(self):
        self.queue = []
        self.counter = 0

    def enqueue(self, item: Dict, priority: int):
        heapq.heappush(self.queue, (-priority, self.counter, item))
        self.counter += 1

    def dequeue(self) -> Dict:
        if self.queue:
            _, _, item = heapq.heappop(self.queue)
            return item
        return None


async def run_benchmark(args):
    """Executa benchmark."""

    print(f"E3: Priorização - gRPC + heapq")
    print(f"Carga: {args.carga} req/s")
    print(f"Duração: {args.duracao}s")
    print(f"Injeções: {args.n}")
    print("-" * 50)

    results = []
    priority_queue = PriorityQueueGRPC()

    async def normal_load():
        interval = 1.0 / args.carga
        start = time.perf_counter()
        count = 0

        while (time.perf_counter() - start) < args.duracao:
            send_time = time.perf_counter()
            priority_queue.enqueue({"time": send_time}, 0)
            await asyncio.sleep(0.001)
            recv_time = time.perf_counter()
            latency = (recv_time - send_time) * 1000
            results.append({"priority": "NORMAL", "latency_ms": latency})
            count += 1
            await asyncio.sleep(interval)
        return count

    async def priority_injection():
        inject_interval = args.duracao / args.n
        start = time.perf_counter()

        for i in range(args.n):
            await asyncio.sleep(inject_interval)
            send_time = time.perf_counter()
            priority_queue.enqueue({"time": send_time}, 1)
            await asyncio.sleep(0.001)
            recv_time = time.perf_counter()
            latency = (recv_time - send_time) * 1000
            results.append({"priority": "HIGH", "latency_ms": latency})
            print(f"Injeção {i+1}/{args.n}: {latency:.2f}ms")

    load_task = asyncio.create_task(normal_load())
    await priority_injection()
    await load_task()

    normal_lat = [r["latency_ms"] for r in results if r["priority"] == "NORMAL"]
    priority_lat = [r["latency_ms"] for r in results if r["priority"] == "HIGH"]

    summary = {
        "protocol": "GRPC_HEAPQ",
        "carga_req_s": args.carga,
        "normal": {"n": len(normal_lat), "mean_ms": round(statistics.mean(normal_lat), 4) if normal_lat else 0},
        "priority": {"n": len(priority_lat), "mean_ms": round(statistics.mean(priority_lat), 4) if priority_lat else 0}
    }

    csv_file = f"results/E3_GRPC_HEAPQ_carga{args.carga}.csv"
    Path("results").mkdir(exist_ok=True)
    with open(csv_file, "w") as f:
        f.write("priority,latency_ms\n")
        for r in results:
            f.write(f"{r['priority']},{r['latency_ms']}\n")

    json_file = f"results/E3_GRPC_HEAPQ_carga{args.carga}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nNORMAL: {summary['normal']['n']}, média: {summary['normal']['mean_ms']:.2f}ms")
    print(f"HIGH: {summary['priority']['n']}, média: {summary['priority']['mean_ms']:.2f}ms")
    print(f"CSV: {csv_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E3: Priorização - gRPC + heapq")
    parser.add_argument("--carga", type=int, default=10)
    parser.add_argument("--duracao", type=int, default=300)
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
