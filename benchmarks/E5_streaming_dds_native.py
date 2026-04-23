#!/usr/bin/env python3
"""
E5: Streaming — full DDS native.

Client publishes ClientRequest with stream=True; orchestrator forwards
agent chunks back via client/response (one DDS sample per token).
Measures TTFT (time to first token) and ITL (inter-token latency).
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


async def measure_one(client: DDSClient, prompt: str, max_tokens: int):
    ttft = None
    chunk_times: list[float] = []
    n_tokens = 0
    last_t = None
    async for content, t_rel_ms, is_final, _sample in client.stream_request(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.0, timeout_s=120,
    ):
        if content:
            n_tokens += 1
            if ttft is None:
                ttft = t_rel_ms
                last_t = t_rel_ms
            else:
                chunk_times.append(t_rel_ms - last_t)
                last_t = t_rel_ms
        if is_final:
            break
    return ttft, chunk_times, n_tokens


async def run_benchmark(args):
    client = DDSClient(domain_id=args.domain)
    client.bind_loop(asyncio.get_running_loop())
    await client.wait_for_discovery()

    print("E5: Streaming - DDS NATIVE")
    print(f"Modelo: {args.model}  N: {args.n}")
    print("-" * 50)

    ttfts: list[float] = []
    itls_per_iter: list[float] = []  # mean of each iter
    all_itls: list[float] = []
    token_counts: list[int] = []

    for i in range(args.n):
        try:
            ttft, itls, n_tokens = await measure_one(client, "Conte uma historia curta sobre o oceano.", 200)
        except Exception as e:
            print(f"Iter {i+1}: ERR {e}")
            continue
        if ttft is None:
            print(f"Iter {i+1}: no tokens")
            continue
        ttfts.append(ttft)
        token_counts.append(n_tokens)
        if itls:
            itls_per_iter.append(statistics.mean(itls))
            all_itls.extend(itls)
        else:
            itls_per_iter.append(0)
        print(f"Iter {i+1}/{args.n}: TTFT={ttft:.1f}ms ITL_mean={itls_per_iter[-1]:.2f}ms tokens={n_tokens}")

    if not ttfts:
        print("Nada coletado.")
        sys.exit(1)

    summary = {
        "protocol": "DDS_NATIVE_STREAM",
        "model": args.model,
        "n": len(ttfts),
        "ttft": {
            "mean_ms": round(statistics.mean(ttfts), 2),
            "median_ms": round(statistics.median(ttfts), 2),
            "stdev_ms": round(statistics.stdev(ttfts), 2) if len(ttfts) > 1 else 0,
        },
        "itl_mean": {
            "mean_ms": round(statistics.mean(itls_per_iter), 4),
            "median_ms": round(statistics.median(itls_per_iter), 4),
            "stdev_ms": round(statistics.stdev(itls_per_iter), 4) if len(itls_per_iter) > 1 else 0,
        },
        "itl_p99": {
            "mean_ms": round(sorted(all_itls)[int(len(all_itls) * 0.99)], 2) if all_itls else 0,
        },
        "tokens": {"mean": round(statistics.mean(token_counts), 0)},
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/E5_DDS_NATIVE_streaming_qwen3.5-0.8b.csv", "w") as f:
        f.write("iteration,ttft_ms,itl_mean_ms,tokens\n")
        for i, (t, m, n) in enumerate(zip(ttfts, itls_per_iter, token_counts), start=1):
            f.write(f"{i},{t},{m},{n}\n")
    with open("results/E5_DDS_NATIVE_streaming_qwen3.5-0.8b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"TTFT medio: {summary['ttft']['mean_ms']}ms")
    print(f"ITL medio:  {summary['itl_mean']['mean_ms']}ms")
    print(f"ITL p99:    {summary['itl_p99']['mean_ms']}ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--model", default="qwen3.5-0.8b")
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
