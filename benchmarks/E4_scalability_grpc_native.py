#!/usr/bin/env python3
"""E4: Scalability — full gRPC native via ClientOrchestratorService."""

import argparse
import asyncio
import grpc
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proto"))
import orchestrator_pb2 as pb2
import orchestrator_pb2_grpc as pb2_grpc

CLIENT_COUNTS = [1, 2, 4, 8]


async def one_client(stub, client_id, n_per_client, results):
    for i in range(n_per_client):
        t0 = time.perf_counter()
        try:
            req = pb2.ClientChatRequest(
                request_id=f"e4-c{client_id}-{i}",
                model="phi4-mini",
                messages=[pb2.ChatMessage(role="user", content="Hi")],
                max_tokens=1, temperature=0.0, priority=5, timeout_ms=120000,
            )
            resp = await stub.Chat(req, timeout=120)
            wall = (time.perf_counter() - t0) * 1000
            results.append({"client": client_id, "latency_ms": wall, "success": bool(resp.success)})
        except Exception as e:
            results.append({"client": client_id, "latency_ms": 0, "success": False, "error": str(e)})


async def main_async(args):
    print("E4: Escalabilidade - gRPC NATIVE (full)")
    print(f"Client counts: {CLIENT_COUNTS}, N per client: {args.n}")
    print("=" * 60)

    all_summaries = []
    Path("results").mkdir(exist_ok=True)

    for num in CLIENT_COUNTS:
        print(f"\n--- {num} client(s) ---")
        # Each client gets its own channel + stub
        channels = [grpc.aio.insecure_channel(args.orch) for _ in range(num)]
        stubs = [pb2_grpc.ClientOrchestratorServiceStub(c) for c in channels]
        results: list = []
        wall_start = time.perf_counter()
        await asyncio.gather(*[one_client(stubs[i], i, args.n, results) for i in range(num)])
        wall_dur = time.perf_counter() - wall_start
        for c in channels:
            await c.close()

        successes = [r for r in results if r["success"]]
        latencies = sorted(r["latency_ms"] for r in successes)
        if not latencies:
            print("  ALL FAILED"); continue
        p = lambda q: latencies[min(int(len(latencies) * q), len(latencies) - 1)]
        s = {
            "protocol": "GRPC_NATIVE_FULL",
            "phase": "A",
            "num_agentes": 1,
            "num_clientes": num,
            "total_requests": len(results),
            "successful_requests": len(latencies),
            "throughput_req_s": round(len(latencies) / wall_dur, 3),
            "latency_p50_ms": round(p(0.50), 2),
            "latency_p95_ms": round(p(0.95), 2),
            "latency_p99_ms": round(p(0.99), 2),
            "latency_mean_ms": round(statistics.mean(latencies), 2),
            "latency_stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
        }
        all_summaries.append(s)
        print(f"  Throughput: {s['throughput_req_s']:.2f} req/s")
        print(f"  p50: {s['latency_p50_ms']:.1f}ms p95: {s['latency_p95_ms']:.1f}ms p99: {s['latency_p99_ms']:.1f}ms")

        with open(f"results/E4_GRPC_NATIVE_FULL_faseA_1ag_{num}cl.csv", "w") as f:
            f.write("client,latency_ms,success\n")
            for r in results:
                f.write(f"{r['client']},{r['latency_ms']},{1 if r['success'] else 0}\n")

    with open("results/E4_GRPC_NATIVE_FULL_faseA_1ag_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orch", default="localhost:50052")
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
