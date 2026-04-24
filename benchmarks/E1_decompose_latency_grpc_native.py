#!/usr/bin/env python3
"""
E1: Latency decomposition — full gRPC native (client → orch → agent → llama).

Uses ClientOrchestratorService.Chat exposed by the orchestrator on
`localhost:50052`. The orchestrator forwards via gRPC to the agent,
which forwards via gRPC to the llama-server (LLAMA_GRPC=ON).
"""

import argparse
import grpc
import json
import statistics
import sys
import time
from pathlib import Path

# Make orchestrator pb2 stubs importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proto"))
import orchestrator_pb2 as pb2
import orchestrator_pb2_grpc as pb2_grpc


def main():
    p = argparse.ArgumentParser(description="E1 — gRPC NATIVE (full)")
    p.add_argument("--orch", default="localhost:50052")
    p.add_argument("--model", default="qwen3.5-0.8b")
    p.add_argument("--prompt", dest="prompt_type", choices=["short", "long"], default="short")
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()

    prompts = {
        "short": "O que e 2+2?",
        "long": "Explique detalhadamente a teoria da relatividade geral de Albert Einstein.",
    }
    prompt = prompts[args.prompt_type]

    channel = grpc.insecure_channel(args.orch)
    stub = pb2_grpc.ClientOrchestratorServiceStub(channel)

    print(f"E1: Decomposicao de Latencia - gRPC NATIVE (full)")
    print(f"Orch: {args.orch}  Modelo: {args.model}  N: {args.n}")
    print("-" * 50)

    results = []
    for i in range(args.n):
        # T1: serialize the request (proto encode)
        t1_start = time.perf_counter_ns()
        req = pb2.ClientChatRequest(
            request_id=f"e1-{i}",
            model=args.model,
            messages=[pb2.ChatMessage(role="user", content=prompt)],
            max_tokens=50, temperature=0.0,
            priority=5, timeout_ms=60000,
        )
        T1 = (time.perf_counter_ns() - t1_start) / 1e6

        # T_rtt
        t_rtt_start = time.perf_counter_ns()
        try:
            resp = stub.Chat(req, timeout=60)
        except Exception as e:
            print(f"Iter {i+1} ERR: {e}")
            continue
        T_rtt = (time.perf_counter_ns() - t_rtt_start) / 1e6

        T6_start = time.perf_counter_ns()
        _ = resp.content
        T6 = (time.perf_counter_ns() - T6_start) / 1e6

        T4 = float(resp.processing_time_ms or 0)
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
        print("No results")
        sys.exit(1)

    metrics = ["T1_serialization_ms", "T2_transport_send_ms", "T3_queue_ms",
               "T4_inference_ms", "T5_transport_return_ms", "T6_deserialization_ms",
               "T_total_ms", "T_round_trip_ms", "transport_overhead_pct"]
    summary = {"protocol": "GRPC_NATIVE_FULL", "model": args.model,
               "prompt_type": args.prompt_type, "n": len(results)}
    for m in metrics:
        vals = sorted(r[m] for r in results)
        summary[f"{m}_mean"] = round(statistics.mean(vals), 4)
        summary[f"{m}_median"] = round(statistics.median(vals), 4)
        summary[f"{m}_stdev"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0
        summary[f"{m}_min"] = round(min(vals), 4)
        summary[f"{m}_max"] = round(max(vals), 4)

    Path("results").mkdir(exist_ok=True)
    csv_file = f"results/E1_GRPC_NATIVE_FULL_{args.model}_{args.prompt_type}.csv"
    with open(csv_file, "w") as f:
        f.write(",".join(["iteration"] + metrics) + "\n")
        for r in results:
            f.write(",".join(str(r.get(m, 0)) for m in ["iteration"] + metrics) + "\n")
    json_file = f"results/E1_GRPC_NATIVE_FULL_{args.model}_{args.prompt_type}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"T_total medio: {summary['T_total_ms_mean']:.2f}ms")
    print(f"T4 inference:  {summary['T4_inference_ms_mean']:.2f}ms")
    print(f"\nCSV: {csv_file}\nJSON: {json_file}")


if __name__ == "__main__":
    main()
