#!/usr/bin/env python3
"""
E4: Escalabilidade Multi-Agente - HTTP
====================================
Mede throughput e latência com HTTP

Usage:
    python E4_scalability_http.py --clientes 8 --n 50
"""

import argparse
import asyncio
import json
import time
import statistics
import psutil
import aiohttp
from pathlib import Path
from typing import Dict, List


async def run_benchmark(args):
    """Executa benchmark."""

    agentes = args.agentes.split(",") if args.agentes else ["http://localhost:8082"]

    print(f"E4: Escalabilidade - HTTP")
    print(f"Agentes: {len(agentes)}")
    print(f"Clientes: {args.clientes}")
    print(f"Requisições: {args.n}")
    print("-" * 50)

    results = []

    async def client_request(session, agent_url):
        start = time.perf_counter()
        try:
            async with session.post(
                f"{agent_url}/chat",
                json={"messages": [{"role": "user", "content": "hi"}]},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                await resp.text()
                end = time.perf_counter()
                return {"success": True, "latency_ms": (end - start) * 1000}
        except Exception as e:
            end = time.perf_counter()
            return {"success": False, "latency_ms": (end - start) * 1000}

    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(args.clientes * args.n):
            import random
            agent = random.choice(agentes)
            tasks.append(client_request(session, agent))

        results = await asyncio.gather(*tasks)

    latencies = [r["latency_ms"] for r in results if r.get("success")]

    # Recursos
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().used / 1024 / 1024

    summary = {
        "protocol": "HTTP",
        "num_agentes": len(agentes),
        "num_clientes": args.clientes,
        "total_requests": len(results),
        "throughput_req_s": len(results) / (max(latencies)/1000) if latencies else 0,
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0,
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
        "latency_p99_ms": round(sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0,
        "cpu_pct": round(cpu, 1),
        "mem_mb": round(mem, 1)
    }

    csv_file = f"results/E4_HTTP_{len(agentes)}ag_{args.clientes}cl.csv"
    Path("results").mkdir(exist_ok=True)
    with open(csv_file, "w") as f:
        f.write("latency_ms,success\n")
        for r in results:
            f.write(f"{r['latency_ms']},{1 if r.get('success') else 0}\n")

    json_file = f"results/E4_HTTP_{len(agentes)}ag_{args.clientes}cl_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nThroughput: {summary['throughput_req_s']:.1f} req/s")
    print(f"p50: {summary['latency_p50_ms']:.1f}ms")
    print(f"CSV: {csv_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E4: Escalabilidade - HTTP")
    parser.add_argument("--agentes", default="http://localhost:8082")
    parser.add_argument("--clientes", type=int, default=1)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
