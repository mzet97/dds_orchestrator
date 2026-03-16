#!/usr/bin/env python3
"""
High-Concurrency Scalability Test — HTTP, gRPC, DDS
=====================================================
Tests throughput and latency with 1 to 1000 concurrent clients.

Each client creates its own connection and sends requests in parallel.
Measures: throughput (req/s), p50, p95, p99 latency.

Usage:
    python e2e_scalability_test.py --protocol http --url http://192.168.1.62:8080 \
        --clients 1,10,50,100,200,500,1000 --requests-per-client 5

Designed to run from client VM (.63) via SSH.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List


PROMPTS = {
    "short": "Say hello",
}


# ─── HTTP Client (one per "virtual client") ──────────────────────────────────

async def http_single_request(url: str, model: str, max_tokens: int = 10):
    """Single HTTP request — creates own session."""
    import aiohttp
    t_start = time.perf_counter_ns()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{url}/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": "Say hello"}],
                      "max_tokens": max_tokens},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                body = await resp.text()
        t_end = time.perf_counter_ns()
        data = json.loads(body)
        content = ""
        if "choices" in data:
            content = data["choices"][0].get("message", {}).get("content", "")
        return {"roundtrip_ms": (t_end - t_start) / 1e6, "success": bool(content), "content_len": len(content)}
    except Exception as e:
        t_end = time.perf_counter_ns()
        return {"roundtrip_ms": (t_end - t_start) / 1e6, "success": False, "error": str(e)[:100]}


# ─── gRPC Client ─────────────────────────────────────────────────────────────

def grpc_single_request_sync(endpoint: str, model: str, max_tokens: int = 10):
    """Single gRPC request — creates own channel."""
    import grpc
    proto_dir = str(Path(__file__).parent.parent / "proto")
    if proto_dir not in sys.path:
        sys.path.insert(0, proto_dir)
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from proto import orchestrator_pb2 as pb2
    from proto import orchestrator_pb2_grpc as pb2_grpc

    t_start = time.perf_counter_ns()
    try:
        channel = grpc.insecure_channel(endpoint)
        stub = pb2_grpc.ClientOrchestratorServiceStub(channel)
        req = pb2.ClientChatRequest(
            request_id=str(uuid.uuid4()), model=model,
            messages=[pb2.ChatMessage(role="user", content="Say hello")],
            max_tokens=max_tokens, timeout_ms=30000,
        )
        resp = stub.Chat(req, timeout=30)
        channel.close()
        t_end = time.perf_counter_ns()
        return {"roundtrip_ms": (t_end - t_start) / 1e6, "success": resp.success,
                "content_len": len(resp.content)}
    except Exception as e:
        t_end = time.perf_counter_ns()
        return {"roundtrip_ms": (t_end - t_start) / 1e6, "success": False, "error": str(e)[:100]}


async def grpc_single_request(endpoint: str, model: str, max_tokens: int = 10):
    """Async wrapper for sync gRPC call."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, grpc_single_request_sync, endpoint, model, max_tokens)


# ─── DDS Client ──────────────────────────────────────────────────────────────

_dds_client = None

def _get_dds_client(domain_id: int = 0):
    """Shared DDS client (pub/sub is naturally multi-consumer)."""
    global _dds_client
    if _dds_client is None:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from client_example import DDSOrchestratorClient
        _dds_client = DDSOrchestratorClient(domain_id=domain_id)
        time.sleep(3)  # Discovery
    return _dds_client


async def dds_single_request(domain_id: int, model: str, max_tokens: int = 10):
    """Single DDS request via shared client."""
    loop = asyncio.get_running_loop()
    client = _get_dds_client(domain_id)
    result = await loop.run_in_executor(
        None, client.chat,
        [{"role": "user", "content": "Say hello"}], 15.0
    )
    return {
        "roundtrip_ms": result.get("dds_roundtrip_ms", 0),
        "success": result.get("success", False),
        "content_len": len(result.get("content", "")),
    }


# ─── Scalability Runner ──────────────────────────────────────────────────────

async def run_scalability(protocol: str, num_clients: int, requests_per_client: int,
                          url: str = "", endpoint: str = "", domain: int = 0,
                          model: str = "", max_tokens: int = 10):
    """Run scalability test with given number of concurrent clients."""

    async def client_worker(client_id: int):
        """Single client worker — sends requests_per_client requests sequentially."""
        results = []
        for _ in range(requests_per_client):
            if protocol == "http":
                r = await http_single_request(url, model, max_tokens)
            elif protocol == "grpc":
                r = await grpc_single_request(endpoint, model, max_tokens)
            elif protocol == "dds":
                r = await dds_single_request(domain, model, max_tokens)
            results.append(r)
        return results

    # Run all clients concurrently
    t_start = time.time()
    tasks = [client_worker(i) for i in range(num_clients)]
    all_results_nested = await asyncio.gather(*tasks, return_exceptions=True)
    t_total = time.time() - t_start

    # Flatten results
    all_results = []
    errors = 0
    for r in all_results_nested:
        if isinstance(r, Exception):
            errors += 1
        else:
            all_results.extend(r)

    latencies = [r["roundtrip_ms"] for r in all_results if r.get("success")]
    success_count = len(latencies)
    total_requests = num_clients * requests_per_client

    if latencies:
        s = sorted(latencies)
        p50 = s[len(s) // 2]
        p95 = s[int(len(s) * 0.95)]
        p99 = s[int(len(s) * 0.99)]
        mean = statistics.mean(latencies)
    else:
        p50 = p95 = p99 = mean = 0

    throughput = success_count / t_total if t_total > 0 else 0

    return {
        "num_clients": num_clients,
        "requests_per_client": requests_per_client,
        "total_requests": total_requests,
        "success": success_count,
        "errors": total_requests - success_count,
        "throughput_rps": round(throughput, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "mean_ms": round(mean, 2),
        "total_time_s": round(t_total, 2),
    }


async def main_async(args):
    client_counts = [int(x.strip()) for x in args.clients.split(",")]

    print(f"\n{'='*60}")
    print(f"  Scalability Test: {args.protocol.upper()}")
    print(f"  Clients: {client_counts}")
    print(f"  Requests/client: {args.requests_per_client}")
    print(f"  Model: {args.model}")
    print(f"{'='*60}\n")

    # Warmup
    print("Warmup...", flush=True)
    if args.protocol == "http":
        await http_single_request(args.url, args.model)
    elif args.protocol == "grpc":
        await grpc_single_request(args.endpoint, args.model)
    elif args.protocol == "dds":
        await dds_single_request(args.domain, args.model)
    print("Done.\n")

    all_results = {
        "protocol": args.protocol.upper(),
        "model": args.model,
        "requests_per_client": args.requests_per_client,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "results": [],
    }

    print(f"{'Clients':>8s} {'Success':>10s} {'Throughput':>12s} {'p50':>8s} {'p95':>8s} {'p99':>8s} {'Time':>8s}")
    print("-" * 70)

    for nc in client_counts:
        r = await run_scalability(
            args.protocol, nc, args.requests_per_client,
            url=args.url, endpoint=args.endpoint, domain=args.domain,
            model=args.model, max_tokens=args.max_tokens,
        )
        all_results["results"].append(r)
        print(f"{nc:>8d} {r['success']:>5d}/{r['total_requests']:<4d} "
              f"{r['throughput_rps']:>9.1f} r/s "
              f"{r['p50_ms']:>7.0f}ms {r['p95_ms']:>7.0f}ms {r['p99_ms']:>7.0f}ms "
              f"{r['total_time_s']:>6.1f}s")

    # Save
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    fname = f"scalability_{args.protocol}_{all_results['timestamp']}.json"
    with open(results_dir / fname, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to results/{fname}")


def main():
    parser = argparse.ArgumentParser(description="High-Concurrency Scalability Test")
    parser.add_argument("--protocol", choices=["http", "grpc", "dds"], required=True)
    parser.add_argument("--url", default="http://192.168.1.62:8080")
    parser.add_argument("--endpoint", default="192.168.1.62:50052")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--model", default="Phi-4-mini")
    parser.add_argument("--clients", default="1,10,50,100,200,500",
                        help="Comma-separated client counts")
    parser.add_argument("--requests-per-client", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
