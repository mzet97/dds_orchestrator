#!/usr/bin/env python3
"""E3: Priority injection — full gRPC native via ClientOrchestratorService."""

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


async def main_async(args):
    channel = grpc.aio.insecure_channel(args.orch)
    stub = pb2_grpc.ClientOrchestratorServiceStub(channel)

    print("E3: Priorizacao - gRPC NATIVE (full)")
    print(f"Carga: {args.carga} req/s NORMAL  Duracao: {args.duracao}s  N: {args.n}")
    print(f"Intervalo: {args.duracao / args.n:.1f}s")
    print("-" * 50)

    normal_lat = []
    high_lat = []
    stop = asyncio.Event()

    async def bg():
        interval = 1.0 / args.carga
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                req = pb2.ClientChatRequest(
                    request_id=f"bg-{int(t0*1000)}",
                    model="qwen3.5-0.8b",
                    messages=[pb2.ChatMessage(role="user", content="Hi")],
                    max_tokens=1, temperature=0.0, priority=5, timeout_ms=10000,
                )
                resp = await stub.Chat(req, timeout=10)
                if resp.success:
                    normal_lat.append((time.perf_counter() - t0) * 1000)
            except Exception:
                pass
            sleep_for = interval - (time.perf_counter() - t0)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    bg_task = asyncio.create_task(bg())
    inject_int = args.duracao / args.n
    for i in range(args.n):
        await asyncio.sleep(inject_int)
        if stop.is_set():
            break
        try:
            t0 = time.perf_counter()
            req = pb2.ClientChatRequest(
                request_id=f"hi-{i}",
                model="qwen3.5-0.8b",
                messages=[pb2.ChatMessage(role="user", content="Hi")],
                max_tokens=1, temperature=0.0, priority=10, timeout_ms=10000,
            )
            resp = await stub.Chat(req, timeout=10)
            wall = (time.perf_counter() - t0) * 1000
            if resp.success:
                high_lat.append(wall)
                print(f"Injecao {i+1}/{args.n}: latencia={wall:.2f}ms")
        except Exception as e:
            print(f"Injecao {i+1}/{args.n}: ERR {e}")

    stop.set()
    await asyncio.sleep(0.3)
    bg_task.cancel()

    def stats(lst):
        if not lst:
            return {}
        s = sorted(lst)
        return {
            "n": len(s),
            "mean_ms": round(statistics.mean(s), 2),
            "median_ms": round(statistics.median(s), 2),
            "stdev_ms": round(statistics.stdev(s), 2) if len(s) > 1 else 0,
            "p95_ms": round(s[min(int(len(s) * 0.95), len(s) - 1)], 2),
        }

    summary = {
        "protocol": "GRPC_NATIVE_FULL_PRIORITY",
        "carga_req_s": args.carga,
        "duracao_s": args.duracao,
        "normal": stats(normal_lat),
        "priority_high": stats(high_lat),
        "priority_advantage_ms": round(
            (statistics.median(normal_lat) if normal_lat else 0)
            - (statistics.median(high_lat) if high_lat else 0), 2),
    }

    Path("results").mkdir(exist_ok=True)
    with open(f"results/E3_GRPC_NATIVE_FULL_carga{args.carga}.csv", "w") as f:
        f.write("type,latency_ms\n")
        for v in normal_lat: f.write(f"NORMAL,{v}\n")
        for v in high_lat: f.write(f"HIGH,{v}\n")
    with open(f"results/E3_GRPC_NATIVE_FULL_carga{args.carga}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"NORMAL n={summary['normal'].get('n',0)} mediana={summary['normal'].get('median_ms',0)}ms p95={summary['normal'].get('p95_ms',0)}ms")
    print(f"HIGH   n={summary['priority_high'].get('n',0)} mediana={summary['priority_high'].get('median_ms',0)}ms")
    print(f"Advantage: {summary['priority_advantage_ms']}ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orch", default="localhost:50052")
    p.add_argument("--carga", type=int, default=5)
    # v3 forces N=1000 HIGH injections; duration scales at ~carga req/s
    p.add_argument("--duracao", type=int, default=10000)
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
