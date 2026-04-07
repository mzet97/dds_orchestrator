#!/usr/bin/env python3
"""
E3: Priority injection — full DDS native.

Background load (NORMAL priority) at `carga` req/s, periodic HIGH priority
injections, all via the DDS client/request topic with TRANSPORT_PRIORITY.
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

NORMAL_PRIORITY = 5
HIGH_PRIORITY = 10


async def run_benchmark(args):
    client = DDSClient(domain_id=args.domain)
    client.bind_loop(asyncio.get_running_loop())
    await client.wait_for_discovery()

    print(f"E3: Priorizacao - DDS NATIVE")
    print(f"Carga: {args.carga} req/s NORMAL")
    print(f"Duracao: {args.duracao}s")
    print(f"Injecoes HIGH: {args.n}")
    print(f"Intervalo entre injecoes: {args.duracao / args.n:.1f}s")
    print("-" * 50)

    normal_latencies: list[float] = []
    high_latencies: list[float] = []
    stop_event = asyncio.Event()

    async def background_load():
        interval = 1.0 / args.carga
        while not stop_event.is_set():
            t0 = asyncio.get_running_loop().time()
            try:
                wall, resp = await client.request(
                    [{"role": "user", "content": "Hi"}],
                    max_tokens=1, temperature=0.0, priority=NORMAL_PRIORITY,
                    timeout_s=10,
                )
                if resp:
                    normal_latencies.append(wall)
            except Exception:
                pass
            elapsed = asyncio.get_running_loop().time() - t0
            sleep_for = interval - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    bg_task = asyncio.create_task(background_load())

    inject_interval = args.duracao / args.n
    for i in range(args.n):
        await asyncio.sleep(inject_interval)
        if stop_event.is_set():
            break
        try:
            wall, resp = await client.request(
                [{"role": "user", "content": "Hi"}],
                max_tokens=1, temperature=0.0, priority=HIGH_PRIORITY,
                timeout_s=10,
            )
            if resp:
                high_latencies.append(wall)
                print(f"Injecao {i+1}/{args.n}: latencia={wall:.2f}ms (OK)")
        except Exception as e:
            print(f"Injecao {i+1}/{args.n}: ERR {e}")

    stop_event.set()
    await asyncio.sleep(0.5)
    bg_task.cancel()

    def stats(lst, name):
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
        "protocol": "DDS_NATIVE_PRIORITY",
        "carga_req_s": args.carga,
        "duracao_s": args.duracao,
        "n_injections": len(high_latencies),
        "normal": stats(normal_latencies, "NORMAL"),
        "priority_high": stats(high_latencies, "HIGH"),
        "priority_advantage_ms": round(
            (statistics.median(normal_latencies) if normal_latencies else 0)
            - (statistics.median(high_latencies) if high_latencies else 0),
            2,
        ),
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/E3_DDS_NATIVE_PRIORITY_carga{}.csv".format(args.carga), "w") as f:
        f.write("type,latency_ms\n")
        for v in normal_latencies: f.write(f"NORMAL,{v}\n")
        for v in high_latencies: f.write(f"HIGH,{v}\n")
    with open("results/E3_DDS_NATIVE_PRIORITY_carga{}_summary.json".format(args.carga), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Resultados:")
    print(f"NORMAL  (n={summary['normal'].get('n',0)}): mediana={summary['normal'].get('median_ms',0)}ms p95={summary['normal'].get('p95_ms',0)}ms")
    print(f"HIGH    (n={summary['priority_high'].get('n',0)}): mediana={summary['priority_high'].get('median_ms',0)}ms")
    print(f"Vantagem HIGH sobre NORMAL: {summary['priority_advantage_ms']}ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--carga", type=int, default=5)
    p.add_argument("--duracao", type=int, default=30)
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
