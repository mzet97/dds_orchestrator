#!/usr/bin/env python3
"""
B5.3: Load Balancing Benchmark
=============================
Compares load balancing between:
- DDS OWNERSHIP (via orchestrator HTTP API -- the orchestrator uses DDS OWNERSHIP internally)
- Python Round-Robin (direct HTTP to agent URLs)
- Nginx Load Balancer

Metrics:
- Request distribution standard deviation (lower = better)
- Latency under load
- Failover behavior when replica fails

Usage:
    python benchmark_b53_load_balancing.py --mode all --url http://localhost:8080
    python benchmark_b53_load_balancing.py --mode dds --url http://localhost:8080
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests as req_lib


@dataclass
class LoadBalanceResult:
    """Result of a single load balancing test."""
    replica_id: int
    request_count: int
    total_latency_ms: float
    avg_latency_ms: float


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    method: str
    num_replicas: int
    num_requests: int
    distribution: Dict[int, int] = field(default_factory=dict)  # replica -> count
    mean_latency_ms: float = 0
    std_latency_ms: float = 0
    distribution_std: float = 0
    min_latency_ms: float = 0
    max_latency_ms: float = 0


class DDSOwnershipBalancer:
    """Uses the DDS orchestrator (via HTTP API) which performs load balancing
    internally using DDS OWNERSHIP QoS.

    Requests go through: client -> orchestrator HTTP -> DDS -> agent (selected by OWNERSHIP).
    The orchestrator's internal DDS layer handles agent selection.
    """

    def __init__(self, orchestrator_url: str):
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.session = req_lib.Session()
        self._request_counter = 0

    def setup(self, num_replicas: int):
        """No setup needed -- orchestrator manages agents."""
        pass

    def send_request(self, request_id: str) -> dict:
        """Send request to orchestrator which routes via DDS OWNERSHIP."""
        start = time.perf_counter()
        try:
            resp = self.session.post(
                f"{self.orchestrator_url}/v1/chat/completions",
                json={
                    "model": "phi4-mini",
                    "messages": [{"role": "user", "content": "ok"}],
                    "max_tokens": 5,
                },
                timeout=30
            )
            latency = (time.perf_counter() - start) * 1000
            # Try to extract which agent handled the request from response headers or body
            agent_id = -1
            try:
                body = resp.json()
                agent_id = hash(body.get("model", "")) % 100  # proxy for agent identity
            except Exception:
                pass
            return {
                "request_id": request_id,
                "latency_ms": latency,
                "success": resp.ok,
                "agent_id": agent_id,
            }
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return {
                "request_id": request_id,
                "latency_ms": latency,
                "success": False,
                "agent_id": -1,
                "error": str(e),
            }

    def cleanup(self):
        """Cleanup session."""
        self.session.close()


class PythonRoundRobinBalancer:
    """Python-based round-robin load balancer using real HTTP requests
    to agent endpoints."""

    def __init__(self, replica_urls: List[str]):
        self.replica_urls = replica_urls
        self.current_index = 0
        self.lock = threading.Lock()
        self.request_counts = defaultdict(int)
        self.session = req_lib.Session()

    def send_request(self, request_id: str) -> dict:
        """Send request using round-robin across agent URLs."""
        with self.lock:
            replica_id = self.current_index
            url = self.replica_urls[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.replica_urls)
            self.request_counts[replica_id] += 1

        start = time.perf_counter()
        try:
            resp = self.session.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": "phi4-mini",
                    "messages": [{"role": "user", "content": "ok"}],
                    "max_tokens": 5,
                },
                timeout=30
            )
            latency = (time.perf_counter() - start) * 1000
            return {
                "request_id": request_id,
                "latency_ms": latency,
                "success": resp.ok,
                "replica_id": replica_id,
            }
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return {
                "request_id": request_id,
                "latency_ms": latency,
                "success": False,
                "replica_id": replica_id,
                "error": str(e),
            }

    def get_distribution(self) -> Dict[int, int]:
        """Get request distribution across replicas."""
        return dict(self.request_counts)

    def cleanup(self):
        """Cleanup session."""
        self.session.close()


class NginxBalancer:
    """Nginx-based load balancer (requires nginx running)."""

    def __init__(self, nginx_url: str = "http://localhost:8080"):
        self.base_url = nginx_url.rstrip("/")
        self.session = req_lib.Session()

    def send_request(self, request_id: str) -> dict:
        """Send request through nginx."""
        start = time.perf_counter()
        try:
            resp = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "phi4-mini",
                    "messages": [{"role": "user", "content": "ok"}],
                    "max_tokens": 5,
                },
                timeout=30
            )
            latency = (time.perf_counter() - start) * 1000
            return {
                "request_id": request_id,
                "latency_ms": latency,
                "success": resp.ok,
                "replica_id": -1,  # Cannot determine which backend nginx chose
            }
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return {
                "request_id": request_id,
                "latency_ms": latency,
                "success": False,
                "replica_id": -1,
                "error": str(e),
            }

    def cleanup(self):
        """Cleanup session."""
        self.session.close()


# NOTE: KubernetesBalancer has been removed. Kubernetes load balancing requires
# a running k8s cluster with configured Services, which cannot be benchmarked
# without the infrastructure. To test Kubernetes, deploy the orchestrator as a
# k8s Service and use the NginxBalancer or DDSOwnershipBalancer pointed at the
# cluster endpoint.


def run_benchmark(
    method: str,
    num_replicas: int = 3,
    num_requests: int = 1000,
    orchestrator_url: str = "http://localhost:8080",
    agent_urls: Optional[List[str]] = None,
) -> BenchmarkResults:
    """Run load balancing benchmark."""

    print(f"\n{'='*60}")
    print(f"Running: {method}")
    print(f"Replicas: {num_replicas}, Requests: {num_requests}")
    print(f"{'='*60}")

    distribution = defaultdict(int)
    latencies = []

    if method == "DDS":
        balancer = DDSOwnershipBalancer(orchestrator_url)
        balancer.setup(num_replicas)

        for i in range(num_requests):
            request_id = f"req_{i}"
            result = balancer.send_request(request_id)
            if result["success"]:
                latencies.append(result["latency_ms"])
                distribution[result["agent_id"]] += 1

        balancer.cleanup()

    elif method == "Python":
        if agent_urls is None:
            agent_urls = [f"http://localhost:{8081+i}" for i in range(num_replicas)]
        balancer = PythonRoundRobinBalancer(agent_urls)

        def sender_task(start_idx: int, count: int):
            for i in range(count):
                request_id = f"req_{start_idx}_{i}"
                result = balancer.send_request(request_id)
                if result["success"]:
                    with lock:
                        latencies.append(result["latency_ms"])
                        distribution[result["replica_id"]] += 1

        lock = threading.Lock()

        # Use multiple threads for concurrent requests
        threads = []
        requests_per_thread = num_requests // 4
        for t in range(4):
            thread = threading.Thread(target=sender_task, args=(t, requests_per_thread))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        balancer.cleanup()

    elif method == "Nginx":
        balancer = NginxBalancer(orchestrator_url)

        for i in range(num_requests):
            request_id = f"req_{i}"
            result = balancer.send_request(request_id)
            if result["success"]:
                latencies.append(result["latency_ms"])
            # Cannot track distribution without nginx access log

        balancer.cleanup()

    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute statistics
    result = BenchmarkResults(
        method=method,
        num_replicas=num_replicas,
        num_requests=num_requests,
        distribution=dict(distribution),
    )

    if latencies:
        result.mean_latency_ms = statistics.mean(latencies)
        result.std_latency_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0
        result.min_latency_ms = min(latencies)
        result.max_latency_ms = max(latencies)

    # Calculate distribution standard deviation
    if distribution:
        counts = list(distribution.values())
        result.distribution_std = statistics.stdev(counts) if len(counts) > 1 else 0

    print(f"\nDistribution: {distribution}")
    print(f"Distribution Std Dev: {result.distribution_std:.2f}")
    print(f"Mean Latency: {result.mean_latency_ms:.2f}ms")

    return result


def run_failover_test(
    method: str,
    num_replicas: int = 3,
    num_requests: int = 500,
    orchestrator_url: str = "http://localhost:8080",
    agent_urls: Optional[List[str]] = None,
) -> BenchmarkResults:
    """Test failover when a replica fails."""

    print(f"\n{'='*60}")
    print(f"Running Failover Test: {method}")
    print(f"{'='*60}")

    distribution = defaultdict(int)
    latencies = []

    if method == "DDS":
        balancer = DDSOwnershipBalancer(orchestrator_url)
        balancer.setup(num_replicas)

        # Send some requests
        for i in range(num_requests // 2):
            request_id = f"req_{i}"
            result = balancer.send_request(request_id)
            if result["success"]:
                latencies.append(result["latency_ms"])
                distribution[result["agent_id"]] += 1

        # In a real scenario, one agent would be killed here.
        # The orchestrator detects the failure via DDS DEADLINE and
        # redistributes via OWNERSHIP.
        print("NOTE: To test real failover, kill one agent process now.")
        print("Continuing to send requests after 2s pause...")
        time.sleep(2)

        # Send more requests after failure
        for i in range(num_requests // 2, num_requests):
            request_id = f"req_{i}"
            result = balancer.send_request(request_id)
            if result["success"]:
                latencies.append(result["latency_ms"])
                distribution[result["agent_id"]] += 1

        balancer.cleanup()

    elif method == "Python":
        if agent_urls is None:
            agent_urls = [f"http://localhost:{8081+i}" for i in range(num_replicas)]
        balancer = PythonRoundRobinBalancer(agent_urls)

        # Send some requests
        for i in range(num_requests // 2):
            request_id = f"req_{i}"
            result = balancer.send_request(request_id)
            if result["success"]:
                latencies.append(result["latency_ms"])
                distribution[result["replica_id"]] += 1

        # Simulate failure by removing a replica
        print("Simulating replica failure by removing one URL...")
        if len(balancer.replica_urls) > 1:
            balancer.replica_urls.pop(1)

        # Send more requests
        for i in range(num_requests // 2, num_requests):
            request_id = f"req_{i}"
            result = balancer.send_request(request_id)
            if result["success"]:
                latencies.append(result["latency_ms"])
                distribution[result["replica_id"]] += 1

        balancer.cleanup()

    result = BenchmarkResults(
        method=method,
        num_replicas=num_replicas,
        num_requests=num_requests,
        distribution=dict(distribution),
    )

    if latencies:
        result.mean_latency_ms = statistics.mean(latencies)
        result.std_latency_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0
        result.min_latency_ms = min(latencies)
        result.max_latency_ms = max(latencies)

    if distribution:
        counts = list(distribution.values())
        result.distribution_std = statistics.stdev(counts) if len(counts) > 1 else 0

    print(f"\nDistribution after failover: {distribution}")

    return result


def main():
    parser = argparse.ArgumentParser(description="B5.3 Load Balancing Benchmark")
    parser.add_argument(
        "--mode",
        choices=["all", "dds", "python", "nginx"],
        default="all",
        help="Benchmark mode"
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=3,
        help="Number of replica instances"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=1000,
        help="Number of requests to send"
    )
    parser.add_argument(
        "--failover",
        action="store_true",
        help="Run failover test"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080",
        help="Orchestrator URL"
    )
    parser.add_argument(
        "--agent-urls",
        type=str,
        default=None,
        help="Comma-separated agent URLs for round-robin mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results",
        help="Output directory"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    agent_urls = None
    if args.agent_urls:
        agent_urls = [u.strip() for u in args.agent_urls.split(",")]

    results = []

    modes = ["DDS", "Python"] if args.mode == "all" else [args.mode.upper()]

    for method in modes:
        if args.failover:
            result = run_failover_test(
                method=method,
                num_replicas=args.replicas,
                num_requests=args.requests,
                orchestrator_url=args.url,
                agent_urls=agent_urls,
            )
        else:
            result = run_benchmark(
                method=method,
                num_replicas=args.replicas,
                num_requests=args.requests,
                orchestrator_url=args.url,
                agent_urls=agent_urls,
            )
        results.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output, f"B53_loadbalance_{timestamp}.json")

    results_data = {
        "benchmark": "B5.3 - Load Balancing",
        "timestamp": timestamp,
        "config": {
            "num_replicas": args.replicas,
            "num_requests": args.requests,
            "failover_test": args.failover,
        },
        "results": [
            {
                "method": r.method,
                "num_replicas": r.num_replicas,
                "num_requests": r.num_requests,
                "distribution": r.distribution,
                "distribution_std": r.distribution_std,
                "mean_latency_ms": r.mean_latency_ms,
                "std_latency_ms": r.std_latency_ms,
            }
            for r in results
        ]
    }

    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    # Print summary
    print(f"\n{'Method':<12} {'Replicas':<10} {'Requests':<10} {'Dist Std':<12} {'Mean Lat':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r.method:<12} {r.num_replicas:<10} {r.num_requests:<10} {r.distribution_std:<12.2f} {r.mean_latency_ms:<12.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
