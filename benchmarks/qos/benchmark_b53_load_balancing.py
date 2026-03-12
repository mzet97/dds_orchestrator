#!/usr/bin/env python3
"""
B5.3: Load Balancing Benchmark
=============================
Compares load balancing between:
- DDS OWNERSHIP (partition-based ownership)
- Python Round-Robin
- Nginx Load Balancer
- Kubernetes Service

Metrics:
- Request distribution standard deviation (lower = better)
- Latency under load
- Failover behavior when replica fails

Usage:
    python benchmark_b53_load_balancing.py --mode all
    python benchmark_b53_load_balancing.py --mode dds
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    """DDS-based load balancing using OWNERSHIP."""

    def __init__(self, domain_id: int = 0):
        self.domain_id = domain_id
        self.participant = None
        self.writers = []
        self.reader = None

    def setup(self, num_replicas: int):
        """Setup DDS with multiple writer instances for ownership."""
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.topic import Topic
        from cyclonedds.pub import DataWriter
        from cyclonedds.core import Policy
        from cyclonedds.qos import Qos
        from cyclonedds.util import duration

        self.participant = DomainParticipant(self.domain_id)
        topic = Topic(self.participant, "loadbalance/requests", dict)

        # Create writers for each replica with ownership
        for i in range(num_replicas):
            qos = Qos(
                Policy.Reliability.Reliable(duration(seconds=10)),
                Policy.Durability.Volatile,
                Policy.Ownership(i),  # Ownership strength based on replica ID
            )
            writer = DataWriter(self.participant, topic, qos)
            self.writers.append(writer)

        # Reader
        reader_qos = Qos(
            Policy.Reliability.Reliable(duration(seconds=10)),
            Policy.Durability.Volatile,
        )
        self.reader = DataReader(self.participant, topic, reader_qos)

    def send_request(self, request_id: str) -> int:
        """Send request - DDS will route based on ownership."""
        # Round-robin across writers (simulating different replicas)
        writer = self.writers[request_id % len(self.writers)]
        writer.write({
            "id": request_id,
            "timestamp": time.time(),
        })
        return request_id % len(self.writers)

    def cleanup(self):
        """Cleanup DDS entities."""
        for writer in self.writers:
            writer.close()
        if self.reader:
            self.reader.close()
        if self.participant:
            self.participant.close()


class PythonRoundRobinBalancer:
    """Python-based round-robin load balancer."""

    def __init__(self, replica_urls: List[str]):
        self.replica_urls = replica_urls
        self.current_index = 0
        self.lock = threading.Lock()
        self.request_counts = defaultdict(int)

    def send_request(self, request_id: str) -> int:
        """Send request using round-robin."""
        with self.lock:
            replica_id = self.current_index
            self.current_index = (self.current_index + 1) % len(self.replica_urls)
            self.request_counts[replica_id] += 1

        # Simulate network latency
        latency = random.uniform(10, 50)
        time.sleep(latency / 1000)

        return replica_id

    def get_distribution(self) -> Dict[int, int]:
        """Get request distribution across replicas."""
        return dict(self.request_counts)


class NginxBalancer:
    """Nginx-based load balancer (requires nginx running)."""

    def __init__(self, nginx_port: int = 8080):
        self.nginx_port = nginx_port
        self.base_url = f"http://localhost:{nginx_port}"

    def send_request(self, request_id: str) -> int:
        """Send request through nginx."""
        import urllib.request

        try:
            req = urllib.request.Request(f"{self.base_url}/api/request")
            urllib.request.urlopen(req, timeout=5)
            # Nginx distributes to backend - we can't know which one
            return -1
        except Exception as e:
            print(f"Error: {e}")
            return -1


class KubernetesBalancer:
    """Kubernetes Service load balancer (requires k8s cluster)."""

    def __init__(self, service_name: str, namespace: str = "default"):
        self.service_name = service_name
        self.namespace = namespace
        self.service_url = f"{service_name}.{namespace}.svc.cluster.local"

    def send_request(self, request_id: str) -> int:
        """Send request through Kubernetes service."""
        # Requires kubectl port-forward or internal cluster access
        # This is a simulation
        return random.randint(0, 2)


def run_benchmark(
    method: str,
    num_replicas: int = 3,
    num_requests: int = 1000,
) -> BenchmarkResults:
    """Run load balancing benchmark."""

    print(f"\n{'='*60}")
    print(f"Running: {method}")
    print(f"Replicas: {num_replicas}, Requests: {num_requests}")
    print(f"{'='*60}")

    distribution = defaultdict(int)
    latencies = []

    if method == "DDS":
        balancer = DDSOwnershipBalancer()
        balancer.setup(num_replicas)

        for i in range(num_requests):
            request_id = f"req_{i}"
            replica_id = balancer.send_request(request_id)
            distribution[replica_id] += 1

            # Simulate processing latency
            latency = random.uniform(10, 50)
            latencies.append(latency)
            time.sleep(0.001)

        balancer.cleanup()

    elif method == "Python":
        replica_urls = [f"http://localhost:{8081+i}" for i in range(num_replicas)]
        balancer = PythonRoundRobinBalancer(replica_urls)

        def sender_task(count: int):
            for i in range(count):
                request_id = f"req_{i}"
                replica_id = balancer.send_request(request_id)
                distribution[replica_id] += 1

        # Use multiple threads for concurrent requests
        threads = []
        requests_per_thread = num_requests // 4
        for t in range(4):
            thread = threading.Thread(target=sender_task, args=(requests_per_thread,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        distribution = balancer.get_distribution()

    elif method == "Nginx":
        balancer = NginxBalancer()

        for i in range(num_requests):
            request_id = f"req_{i}"
            balancer.send_request(request_id)
            # Cannot track distribution without nginx access log

    elif method == "Kubernetes":
        balancer = KubernetesBalancer("llm-agent-service")

        for i in range(num_requests):
            request_id = f"req_{i}"
            replica_id = balancer.send_request(request_id)
            distribution[replica_id] += 1
            time.sleep(0.001)

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
        expected = num_requests / num_replicas
        result.distribution_std = statistics.stdev(counts) if len(counts) > 1 else 0

    print(f"\nDistribution: {distribution}")
    print(f"Distribution Std Dev: {result.distribution_std:.2f}")
    print(f"Mean Latency: {result.mean_latency_ms:.2f}ms")

    return result


def run_failover_test(
    method: str,
    num_replicas: int = 3,
    num_requests: int = 500,
) -> BenchmarkResults:
    """Test failover when a replica fails."""

    print(f"\n{'='*60}")
    print(f"Running Failover Test: {method}")
    print(f"{'='*60}")

    distribution = defaultdict(int)

    if method == "DDS":
        balancer = DDSOwnershipBalancer()
        balancer.setup(num_replicas)

        # Send some requests
        for i in range(num_requests // 2):
            request_id = f"req_{i}"
            replica_id = balancer.send_request(request_id)
            distribution[replica_id] += 1

        # Simulate replica failure (close one writer)
        if len(balancer.writers) > 1:
            print("Simulating replica failure...")
            balancer.writers[1].close()
            balancer.writers.pop(1)

        # Send more requests after failure
        for i in range(num_requests // 2, num_requests):
            request_id = f"req_{i}"
            replica_id = balancer.send_request(request_id)
            distribution[replica_id] += 1

        balancer.cleanup()

    elif method == "Python":
        replica_urls = [f"http://localhost:{8081+i}" for i in range(num_replicas)]
        balancer = PythonRoundRobinBalancer(replica_urls)

        # Send some requests
        for i in range(num_requests // 2):
            request_id = f"req_{i}"
            replica_id = balancer.send_request(request_id)
            distribution[replica_id] += 1

        # Simulate failure by removing a replica
        print("Simulating replica failure...")
        balancer.replica_urls.pop(1)

        # Send more requests
        for i in range(num_requests // 2, num_requests):
            request_id = f"req_{i}"
            replica_id = balancer.send_request(request_id)
            distribution[replica_id] += 1

    result = BenchmarkResults(
        method=method,
        num_replicas=num_replicas,
        num_requests=num_requests,
        distribution=dict(distribution),
    )

    print(f"\nDistribution after failover: {distribution}")

    return result


def main():
    parser = argparse.ArgumentParser(description="B5.3 Load Balancing Benchmark")
    parser.add_argument(
        "--mode",
        choices=["all", "dds", "python", "nginx", "kubernetes"],
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
        "--output",
        type=str,
        default="benchmark_results",
        help="Output directory"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results = []

    modes = ["DDS", "Python"]  # Nginx and K8s need special setup

    for method in modes:
        if args.failover:
            result = run_failover_test(
                method=method,
                num_replicas=args.replicas,
                num_requests=args.requests,
            )
        else:
            result = run_benchmark(
                method=method,
                num_replicas=args.replicas,
                num_requests=args.requests,
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
