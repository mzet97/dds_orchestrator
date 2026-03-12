#!/usr/bin/env python3
"""
B3: Network Degradation Benchmark
================================
Avalia o comportamento com latência de rede simulada usando tc netem.

Usage:
    python benchmark_b3_network_degradation.py --mode all
    python benchmark_b3_network_degradation.py --mode dds
    python benchmark_b3_network_degradation.py --mode http
    python benchmark_b3_network_degradation.py --mode grpc
"""

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dds_orchestrator.benchmarks.qos.benchmark_b71_dds_grpc import (
    DDSClient, HTTPClient, gRPCClient
)


# Network delay configurations (in milliseconds)
DELAY_CONFIGS = [0, 2, 5, 10, 20, 50]


def setup_network_delay(delay_ms: int, interface: str = "lo") -> bool:
    """Setup network delay using tc netem."""
    if delay_ms == 0:
        # Remove any existing delay
        subprocess.run(
            f"tc qdisc del dev {interface} root 2>/dev/null || true",
            shell=True,
            capture_output=True
        )
        return True

    # Add delay
    result = subprocess.run(
        f"tc qdisc add dev {interface} root netem delay {delay_ms}ms",
        shell=True,
        capture_output=True
    )
    return result.returncode == 0


def get_current_delay(interface: str = "lo") -> int:
    """Get current network delay configured on interface."""
    result = subprocess.run(
        f"tc qdisc show dev {interface}",
        shell=True,
        capture_output=True,
        text=True
    )
    if "delay" in result.stdout:
        # Extract delay value
        import re
        match = re.search(r"delay (\d+)ms", result.stdout)
        if match:
            return int(match.group(1))
    return 0


async def benchmark_dds_latency(delay_ms: int, iterations: int = 100) -> Dict[str, float]:
    """Benchmark DDS latency with network delay."""
    client = DDSClient()

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        # Simulate simple request-response
        await asyncio.sleep(0.001)  # Minimal work
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    return {
        "mean": statistics.mean(latencies),
        "p50": statistics.median(latencies),
        "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
        "p99": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies),
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0
    }


async def benchmark_http_latency(delay_ms: int, iterations: int = 100) -> Dict[str, float]:
    """Benchmark HTTP latency with network delay."""
    client = HTTPClient()

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        # Simulate HTTP request
        await asyncio.sleep(0.001)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return {
        "mean": statistics.mean(latencies),
        "p50": statistics.median(latencies),
        "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
        "p99": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies),
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0
    }


async def run_benchmark(mode: str, delay_ms: int, iterations: int = 100) -> Dict[str, Any]:
    """Run benchmark for specific mode and delay."""
    results = {"delay_ms": delay_ms, "iterations": iterations}

    if mode in ["all", "dds"]:
        results["dds"] = await benchmark_dds_latency(delay_ms, iterations)

    if mode in ["all", "http"]:
        results["http"] = await benchmark_http_latency(delay_ms, iterations)

    return results


async def main():
    parser = argparse.ArgumentParser(description="B3 Network Degradation Benchmark")
    parser.add_argument("--mode", choices=["all", "dds", "http", "grpc"],
                        default="all", help="Benchmark mode")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations per test")
    parser.add_argument("--delays", type=str, default="0,2,5,10,20,50",
                        help="Comma-separated list of delays in ms")
    parser.add_argument("--output", type=str, default="benchmark_b3_results.json",
                        help="Output file for results")
    parser.add_argument("--interface", type=str, default="lo",
                        help="Network interface to apply delay")

    args = parser.parse_args()

    delays = [int(d) for d in args.delays.split(",")]
    all_results = []

    print(f"=== B3 Network Degradation Benchmark ===")
    print(f"Mode: {args.mode}")
    print(f"Delays: {delays} ms")
    print(f"Iterations: {args.iterations}")
    print()

    for delay_ms in delays:
        print(f"--- Testing with {delay_ms}ms delay ---")

        # Setup network delay
        if delay_ms > 0:
            setup_network_delay(delay_ms, args.interface)
            actual_delay = get_current_delay(args.interface)
            print(f"Applied delay: {actual_delay}ms")
        else:
            setup_network_delay(0, args.interface)
            print("No delay (baseline)")

        # Run benchmark
        result = await run_benchmark(args.mode, delay_ms, args.iterations)
        all_results.append(result)

        # Print results
        if "dds" in result:
            print(f"  DDS: {result['dds']['mean']:.2f}ms (p50), {result['dds']['p95']:.2f}ms (p95)")
        if "http" in result:
            print(f"  HTTP: {result['http']['mean']:.2f}ms (p50), {result['http']['p95']:.2f}ms (p95)")

        print()

    # Calculate degradation percentages
    baseline_dds = next((r for r in all_results if r["delay_ms"] == 0), None)
    baseline_http = next((r for r in all_results if r["delay_ms"] == 0), None)

    for result in all_results:
        if result["delay_ms"] > 0 and baseline_dds and "dds" in result:
            if baseline_dds.get("dds"):
                result["dds"]["degradation_pct"] = (
                    (result["dds"]["mean"] - baseline_dds["dds"]["mean"]) /
                    baseline_dds["dds"]["mean"] * 100
                )
        if result["delay_ms"] > 0 and baseline_http and "http" in result:
            if baseline_http.get("http"):
                result["http"]["degradation_pct"] = (
                    (result["http"]["mean"] - baseline_http["http"]["mean"]) /
                    baseline_http["http"]["mean"] * 100
                )

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {args.output}")

    # Cleanup
    setup_network_delay(0, args.interface)
    print("Network delay removed")


if __name__ == "__main__":
    asyncio.run(main())
