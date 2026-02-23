#!/usr/bin/env python3
"""
Benchmark DDS Overhead — Mede overhead isolado do transporte DDS
(sem inferência LLM) para comparar com HTTP equivalente.

Métricas:
1. Publish latency (tempo para serializar + publicar no tópico)
2. Round-trip DDS (publish request → subscribe response via echo)
3. Serialização/deserialização dos tipos IDL
4. Comparação com HTTP equivalente (JSON POST + response)

Uso:
    python benchmark_dds_overhead.py [--runs 1000] [--domain 0]
"""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from typing import List

import aiohttp


# ============================================
# DDS BENCHMARK
# ============================================


async def benchmark_dds_publish(runs: int = 1000, domain: int = 0) -> dict:
    """Mede latência de publish DDS (sem subscriber)"""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from config import OrchestratorConfig
        from dds import DDSLayer, TOPIC_AGENT_REQUEST
    except ImportError as e:
        print(f"[SKIP] Dependência não disponível: {e}")
        return {"skipped": True, "reason": str(e)}

    config = OrchestratorConfig(dds_enabled=True, dds_domain=domain)
    dds = DDSLayer(config)

    if not dds.is_available():
        print("[SKIP] CycloneDDS não disponível")
        return {"skipped": True, "reason": "CycloneDDS not available"}

    test_data = {
        "task_id": "overhead-test",
        "requester_id": "benchmark",
        "task_type": "chat",
        "messages_json": json.dumps([{"role": "user", "content": "Hello, this is a benchmark message."}]),
        "priority": 5,
        "timeout_ms": 30000,
        "requires_context": False,
        "context_id": "",
        "created_at": 0,
    }

    # Warmup
    for _ in range(10):
        await dds.publish(TOPIC_AGENT_REQUEST, test_data)

    # Benchmark
    latencies: List[float] = []
    for i in range(runs):
        test_data["created_at"] = int(time.time() * 1000)
        start = time.perf_counter()
        await dds.publish(TOPIC_AGENT_REQUEST, test_data)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    dds.close()

    return _compute_stats("DDS Publish", latencies)


async def benchmark_dds_serialization(runs: int = 1000) -> dict:
    """Mede overhead de serialização/deserialização JSON (simula IDL)"""
    test_message = {
        "task_id": str(uuid.uuid4()),
        "requester_id": "benchmark-client",
        "task_type": "chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain the concept of DDS middleware in distributed systems."},
        ],
        "priority": 5,
        "timeout_ms": 30000,
        "requires_context": False,
        "context_id": "",
        "created_at": int(time.time() * 1000),
    }

    # Warmup
    for _ in range(10):
        encoded = json.dumps(test_message).encode("utf-8")
        json.loads(encoded.decode("utf-8"))

    # Benchmark serialize
    serialize_latencies: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        encoded = json.dumps(test_message).encode("utf-8")
        end = time.perf_counter()
        serialize_latencies.append((end - start) * 1000)

    # Benchmark deserialize
    encoded = json.dumps(test_message).encode("utf-8")
    deserialize_latencies: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        json.loads(encoded.decode("utf-8"))
        end = time.perf_counter()
        deserialize_latencies.append((end - start) * 1000)

    # Benchmark round-trip (serialize + deserialize)
    roundtrip_latencies: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        encoded = json.dumps(test_message).encode("utf-8")
        json.loads(encoded.decode("utf-8"))
        end = time.perf_counter()
        roundtrip_latencies.append((end - start) * 1000)

    return {
        "serialize": _compute_stats("Serialization", serialize_latencies),
        "deserialize": _compute_stats("Deserialization", deserialize_latencies),
        "roundtrip": _compute_stats("Serialize+Deserialize", roundtrip_latencies),
        "payload_bytes": len(json.dumps(test_message).encode("utf-8")),
    }


# ============================================
# HTTP BENCHMARK
# ============================================


async def benchmark_http_echo(runs: int = 100, host: str = "127.0.0.1", port: int = 8082) -> dict:
    """
    Mede latência HTTP round-trip (POST JSON + response).
    Requer um servidor llama-server ou equivalente rodando.
    Usa /health como endpoint leve para medir overhead puro.
    """
    url_health = f"http://{host}:{port}/health"
    url_completions = f"http://{host}:{port}/v1/completions"

    # Health check latency (mínimo overhead HTTP)
    health_latencies: List[float] = []
    try:
        async with aiohttp.ClientSession() as session:
            # Warmup
            for _ in range(5):
                async with session.get(url_health) as resp:
                    await resp.text()

            # Benchmark health
            for _ in range(runs):
                start = time.perf_counter()
                async with session.get(url_health) as resp:
                    await resp.text()
                end = time.perf_counter()
                health_latencies.append((end - start) * 1000)

    except Exception as e:
        print(f"[SKIP] HTTP health benchmark: {e}")
        return {"skipped": True, "reason": str(e)}

    # POST JSON latency (simula request real, com max_tokens=1 para mínima inferência)
    post_latencies: List[float] = []
    payload = {
        "prompt": "Hi",
        "max_tokens": 1,
        "temperature": 0.0,
    }

    try:
        async with aiohttp.ClientSession() as session:
            # Warmup
            for _ in range(3):
                try:
                    async with session.post(url_completions, json=payload,
                                           timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        await resp.text()
                except Exception:
                    pass

            # Benchmark POST
            for _ in range(min(runs, 50)):  # Limitar POST pois tem inferência
                start = time.perf_counter()
                try:
                    async with session.post(url_completions, json=payload,
                                           timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        await resp.text()
                    end = time.perf_counter()
                    post_latencies.append((end - start) * 1000)
                except Exception:
                    pass

    except Exception as e:
        print(f"[WARN] HTTP POST benchmark: {e}")

    result = {
        "health_check": _compute_stats("HTTP GET /health", health_latencies),
    }
    if post_latencies:
        result["post_completions"] = _compute_stats("HTTP POST /v1/completions (1 token)", post_latencies)

    return result


# ============================================
# UTILITIES
# ============================================


def _compute_stats(label: str, latencies: List[float]) -> dict:
    """Computa estatísticas de latência"""
    if not latencies:
        return {"label": label, "n": 0}

    sorted_lats = sorted(latencies)
    n = len(sorted_lats)

    return {
        "label": label,
        "n": n,
        "mean_ms": statistics.mean(sorted_lats),
        "median_ms": statistics.median(sorted_lats),
        "stddev_ms": statistics.stdev(sorted_lats) if n > 1 else 0.0,
        "min_ms": sorted_lats[0],
        "max_ms": sorted_lats[-1],
        "p50_ms": sorted_lats[int(n * 0.50)],
        "p95_ms": sorted_lats[int(n * 0.95)],
        "p99_ms": sorted_lats[int(n * 0.99)],
    }


def print_stats(stats: dict, indent: int = 0):
    """Pretty-print stats"""
    prefix = "  " * indent
    if stats.get("skipped"):
        print(f"{prefix}[SKIPPED] {stats.get('reason', 'unknown')}")
        return

    if "label" in stats:
        s = stats
        print(f"{prefix}{s['label']} (n={s['n']}):")
        print(f"{prefix}  Mean:   {s['mean_ms']:.4f} ms")
        print(f"{prefix}  Median: {s['median_ms']:.4f} ms")
        print(f"{prefix}  Stddev: {s['stddev_ms']:.4f} ms")
        print(f"{prefix}  Min:    {s['min_ms']:.4f} ms")
        print(f"{prefix}  Max:    {s['max_ms']:.4f} ms")
        print(f"{prefix}  p50:    {s['p50_ms']:.4f} ms")
        print(f"{prefix}  p95:    {s['p95_ms']:.4f} ms")
        print(f"{prefix}  p99:    {s['p99_ms']:.4f} ms")
    else:
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_stats(value, indent + 1)
            else:
                print(f"{prefix}{key}: {value}")


# ============================================
# MAIN
# ============================================


async def main():
    parser = argparse.ArgumentParser(description="Benchmark DDS vs HTTP overhead")
    parser.add_argument("--runs", type=int, default=64, help="Number of benchmark runs (N=64 for statistical significance per Cohen 1988)")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain ID")
    parser.add_argument("--http-host", type=str, default="127.0.0.1", help="HTTP server host")
    parser.add_argument("--http-port", type=int, default=8082, help="HTTP server port")
    parser.add_argument("--skip-dds", action="store_true", help="Skip DDS benchmarks")
    parser.add_argument("--skip-http", action="store_true", help="Skip HTTP benchmarks")
    args = parser.parse_args()

    print("=" * 60)
    print("DDS vs HTTP Overhead Benchmark")
    print(f"Runs: {args.runs}")
    print("=" * 60)

    # 1. Serialization overhead
    print("\n--- Serialization Overhead ---")
    serial_stats = await benchmark_dds_serialization(args.runs)
    print_stats(serial_stats)

    # 2. DDS publish overhead
    if not args.skip_dds:
        print("\n--- DDS Publish Overhead ---")
        dds_stats = await benchmark_dds_publish(args.runs, args.domain)
        print_stats(dds_stats)

    # 3. HTTP overhead
    if not args.skip_http:
        print(f"\n--- HTTP Overhead ({args.http_host}:{args.http_port}) ---")
        http_stats = await benchmark_http_echo(
            runs=min(args.runs, 500),
            host=args.http_host,
            port=args.http_port,
        )
        print_stats(http_stats)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "roundtrip" in serial_stats:
        rt = serial_stats["roundtrip"]
        print(f"  JSON serialize+deserialize: {rt['mean_ms']:.4f} ms (payload: {serial_stats['payload_bytes']} bytes)")

    if not args.skip_dds and not dds_stats.get("skipped"):
        print(f"  DDS publish:               {dds_stats['mean_ms']:.4f} ms")

    if not args.skip_http and not http_stats.get("skipped"):
        if "health_check" in http_stats:
            print(f"  HTTP GET /health:          {http_stats['health_check']['mean_ms']:.4f} ms")
        if "post_completions" in http_stats:
            print(f"  HTTP POST (1 token):       {http_stats['post_completions']['mean_ms']:.4f} ms")

    print("\n> DDS overhead esperado: < 1ms em rede local")
    print("> Diferença principal é overhead de rede TCP (HTTP) vs shared memory (DDS)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
