#!/usr/bin/env python3
"""E5: Streaming — full gRPC native via ClientOrchestratorService.StreamChat."""

import argparse
import grpc
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proto"))
import orchestrator_pb2 as pb2
import orchestrator_pb2_grpc as pb2_grpc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orch", default="localhost:50052")
    p.add_argument("--model", default="phi4-mini")
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()

    channel = grpc.insecure_channel(args.orch)
    stub = pb2_grpc.ClientOrchestratorServiceStub(channel)

    print("E5: Streaming - gRPC NATIVE (full)")
    print(f"Orch: {args.orch}  Modelo: {args.model}  N: {args.n}")
    print("-" * 50)

    ttfts = []
    itl_means = []
    all_itls = []
    token_counts = []

    for i in range(args.n):
        req = pb2.ClientChatRequest(
            request_id=f"e5-{i}",
            model=args.model,
            messages=[pb2.ChatMessage(role="user", content="Conte uma historia curta sobre o oceano.")],
            max_tokens=200, temperature=0.0, priority=5, timeout_ms=120000,
        )
        ttft = None
        prev = None
        itls = []
        n_tokens = 0
        t0 = time.perf_counter()
        try:
            for chunk in stub.StreamChat(req, timeout=120):
                t = (time.perf_counter() - t0) * 1000
                if chunk.content:
                    n_tokens += 1
                    if ttft is None:
                        ttft = t
                        prev = t
                    else:
                        itls.append(t - prev)
                        prev = t
                if chunk.is_final:
                    break
        except Exception as e:
            print(f"Iter {i+1} ERR: {e}")
            continue
        if ttft is None:
            print(f"Iter {i+1}: no tokens"); continue
        ttfts.append(ttft)
        token_counts.append(n_tokens)
        itl_mean = statistics.mean(itls) if itls else 0
        itl_means.append(itl_mean)
        all_itls.extend(itls)
        print(f"Iter {i+1}/{args.n}: TTFT={ttft:.1f}ms ITL_mean={itl_mean:.2f}ms tokens={n_tokens}")

    if not ttfts:
        print("Nada coletado"); sys.exit(1)

    summary = {
        "protocol": "GRPC_NATIVE_FULL_STREAM",
        "model": args.model,
        "n": len(ttfts),
        "ttft": {
            "mean_ms": round(statistics.mean(ttfts), 2),
            "median_ms": round(statistics.median(ttfts), 2),
            "stdev_ms": round(statistics.stdev(ttfts), 2) if len(ttfts) > 1 else 0,
        },
        "itl_mean": {
            "mean_ms": round(statistics.mean(itl_means), 4),
            "median_ms": round(statistics.median(itl_means), 4),
            "stdev_ms": round(statistics.stdev(itl_means), 4) if len(itl_means) > 1 else 0,
        },
        "itl_p99": {
            "mean_ms": round(sorted(all_itls)[int(len(all_itls) * 0.99)], 2) if all_itls else 0,
        },
        "tokens": {"mean": round(statistics.mean(token_counts), 0)},
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/E5_GRPC_NATIVE_FULL_streaming_phi4-mini.csv", "w") as f:
        f.write("iteration,ttft_ms,itl_mean_ms,tokens\n")
        for i, (t, m, n) in enumerate(zip(ttfts, itl_means, token_counts), 1):
            f.write(f"{i},{t},{m},{n}\n")
    with open("results/E5_GRPC_NATIVE_FULL_streaming_phi4-mini_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTTFT medio: {summary['ttft']['mean_ms']}ms")
    print(f"ITL medio:  {summary['itl_mean']['mean_ms']}ms")


if __name__ == "__main__":
    main()
