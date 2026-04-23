#!/usr/bin/env python3
"""
E4: Multi-client scalability — full DDS native.

Spins N concurrent DDS client tasks (each one its own DomainParticipant)
publishing on `client/request` and consuming `client/response`. Measures
per-client latency and aggregate throughput.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dds-automation" / "bench"))
from bench_dds_native_client import DDSClient

CLIENT_COUNTS = [1, 2, 4, 8]


async def one_client_loop(client_id: int, n_per_client: int, results: list):
    c = DDSClient(domain_id=0, client_id=f"e4-c{client_id}")
    c.bind_loop(asyncio.get_running_loop())
    await c.wait_for_discovery(timeout_s=2)
    for _ in range(n_per_client):
        try:
            wall, resp = await c.request(
                [{"role": "user", "content": "Hi"}],
                max_tokens=1, temperature=0.0, timeout_s=120,
            )
            results.append({
                "client": client_id,
                "latency_ms": wall,
                "success": resp is not None and getattr(resp, "success", True),
            })
        except Exception as e:
            results.append({"client": client_id, "latency_ms": 0, "success": False, "error": str(e)})


async def run_benchmark(args):
    print("E4: Escalabilidade - DDS NATIVE")
    print(f"Client counts: {CLIENT_COUNTS}, N per client: {args.n}")
    print("=" * 60)

    all_summaries = []
    Path("results").mkdir(exist_ok=True)

    for num_clientes in CLIENT_COUNTS:
        print(f"\n--- {num_clientes} client(s) ---")
        results: list = []
        wall_start = time.perf_counter()
        await asyncio.gather(*[
            one_client_loop(i, args.n, results) for i in range(num_clientes)
        ])
        wall_dur = time.perf_counter() - wall_start

        successes = [r for r in results if r["success"]]
        latencies = sorted(r["latency_ms"] for r in successes)
        n_succ = len(latencies)
        if not latencies:
            print("  ALL FAILED")
            continue
        p = lambda q: latencies[min(int(len(latencies) * q), len(latencies) - 1)]
        summary = {
            "protocol": "DDS_NATIVE",
            "phase": "A",
            "num_agentes": 1,
            "num_clientes": num_clientes,
            "total_requests": len(results),
            "successful_requests": n_succ,
            "throughput_req_s": round(n_succ / wall_dur, 3),
            "latency_p50_ms": round(p(0.50), 2),
            "latency_p95_ms": round(p(0.95), 2),
            "latency_p99_ms": round(p(0.99), 2),
            "latency_mean_ms": round(statistics.mean(latencies), 2),
            "latency_stdev_ms": round(statistics.stdev(latencies), 2) if n_succ > 1 else 0,
        }
        all_summaries.append(summary)

        print(f"  Throughput: {summary['throughput_req_s']:.2f} req/s")
        print(f"  p50: {summary['latency_p50_ms']:.1f}ms p95: {summary['latency_p95_ms']:.1f}ms p99: {summary['latency_p99_ms']:.1f}ms")

        csv_file = f"results/E4_DDS_NATIVE_faseA_1ag_{num_clientes}cl.csv"
        with open(csv_file, "w") as f:
            f.write("client,latency_ms,success\n")
            for r in results:
                f.write(f"{r['client']},{r['latency_ms']},{1 if r['success'] else 0}\n")

    json_file = "results/E4_DDS_NATIVE_faseA_1ag_summary.json"
    with open(json_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nJSON: {json_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000, help="N requests per client (v3: N=1000)")
    args = p.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
