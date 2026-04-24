#!/usr/bin/env python3
"""
E1: Latency decomposition — full DDS native (no HTTP).

Client publishes ClientRequest on `client/request` and listens on
`client/response`. The orchestrator routes via DDS to the agent which
talks DDS to llama-server. End-to-end full DDS.

Path measured:
  client DDS → orch DDS → agent DDS → llama-server (DDS C++)
            ← orch DDS ← agent DDS ←
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

# Reuse the native DDS client wrapper
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dds-automation" / "bench"))
from bench_dds_native_client import DDSClient


async def run_benchmark(args):
    client = DDSClient(domain_id=args.domain)
    client.bind_loop(asyncio.get_running_loop())
    await client.wait_for_discovery()

    prompts = {
        "short": "O que e 2+2?",
        "long": "Explique detalhadamente a teoria da relatividade geral de Albert Einstein.",
    }
    prompt = prompts.get(args.prompt_type, prompts["short"])

    results = []
    print(f"E1: Decomposicao de Latencia - DDS NATIVE (no HTTP hop)")
    print(f"Modelo: {args.model}  Prompt: {args.prompt_type}  N: {args.n}")
    print("-" * 50)

    for i in range(args.n):
        # T1: serialize the messages list (mirrors HTTP variant)
        t1_start = time.perf_counter_ns()
        messages = [{"role": "user", "content": prompt}]
        _ = json.dumps(messages)
        t1_end = time.perf_counter_ns()
        T1 = (t1_end - t1_start) / 1e6

        try:
            wall_ms, resp = await client.request(messages, max_tokens=50, temperature=0.0, timeout_s=60)
        except Exception as e:
            print(f"Iter {i+1} ERR: {e}")
            continue
        if not resp:
            print(f"Iter {i+1} TIMEOUT")
            continue

        T6_start = time.perf_counter_ns()
        # Deserialization is essentially free for IDL — measure as 0
        T6 = (time.perf_counter_ns() - T6_start) / 1e6

        T_rtt = wall_ms
        T4 = float(getattr(resp, "processing_time_ms", 0))
        T_transport = max(T_rtt - T4, 0)

        results.append({
            "iteration": i + 1,
            "T1_serialization_ms": T1,
            "T2_transport_send_ms": T_transport / 2,
            "T3_queue_ms": 0,
            "T4_inference_ms": T4,
            "T5_transport_return_ms": T_transport / 2,
            "T6_deserialization_ms": T6,
            "T_total_ms": T_rtt,
            "T_round_trip_ms": T_rtt,
            "transport_overhead_pct": (T_transport / T_rtt * 100) if T_rtt > 0 else 0,
        })

        if (i + 1) % 10 == 0:
            print(f"Iter {i+1}/{args.n}: T_total={T_rtt:.1f}ms T4={T4:.1f}ms")

    if not results:
        print("No results.")
        sys.exit(1)

    metrics = ["T1_serialization_ms", "T2_transport_send_ms", "T3_queue_ms",
               "T4_inference_ms", "T5_transport_return_ms", "T6_deserialization_ms",
               "T_total_ms", "T_round_trip_ms", "transport_overhead_pct"]
    summary = {"protocol": "DDS_NATIVE", "model": args.model,
               "prompt_type": args.prompt_type, "n": len(results)}
    for m in metrics:
        vals = sorted(r[m] for r in results)
        summary[f"{m}_mean"] = round(statistics.mean(vals), 4)
        summary[f"{m}_median"] = round(statistics.median(vals), 4)
        summary[f"{m}_stdev"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0
        summary[f"{m}_min"] = round(min(vals), 4)
        summary[f"{m}_max"] = round(max(vals), 4)

    Path("results").mkdir(exist_ok=True)
    csv_file = f"results/E1_DDS_NATIVE_{args.model}_{args.prompt_type}.csv"
    with open(csv_file, "w") as f:
        f.write(",".join(["iteration"] + metrics) + "\n")
        for r in results:
            f.write(",".join(str(r.get(m, 0)) for m in ["iteration"] + metrics) + "\n")
    json_file = f"results/E1_DDS_NATIVE_{args.model}_{args.prompt_type}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"T_total medio: {summary['T_total_ms_mean']:.2f}ms")
    print(f"T4 inference:  {summary['T4_inference_ms_mean']:.2f}ms")
    print(f"Overhead:      {summary['transport_overhead_pct_mean']:.2f}%")
    print(f"\nCSV: {csv_file}\nJSON: {json_file}")


def main():
    p = argparse.ArgumentParser(description="E1 — DDS NATIVE")
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--model", default="qwen3.5-0.8b")
    p.add_argument("--prompt", dest="prompt_type", choices=["short", "long"], default="short")
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
