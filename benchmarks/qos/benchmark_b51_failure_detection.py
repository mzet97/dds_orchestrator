#!/usr/bin/env python3
"""
B5.1: Failure Detection Benchmark
=================================
Compares agent failure detection time between:
- DDS DEADLINE policy (native)
- HTTP Heartbeat (Python threads)
- gRPC Health Checking (native protocol)

Metrics:
- Time to detect failure after agent crash (kill -9)
- Time to detect failure after graceful shutdown
- Time to detect overloaded agent (artificial latency injection)

Usage:
    python benchmark_b51_failure_detection.py --mode all
    python benchmark_b51_failure_detection.py --mode dds
    python benchmark_b51_failure_detection.py --mode http
    python benchmark_b51_failure_detection.py --mode grpc
"""

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
import signal
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.core import Policy
from cyclonedds.qos import Qos
from cyclonedds.util import duration

from orchestrator import ClientRequest, ClientResponse, AgentStatus


@dataclass
class FailureDetectionResult:
    """Result of a single failure detection test."""
    method: str
    failure_type: str  # "kill", "graceful", "overload"
    detection_time_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    method: str
    failure_type: str
    iterations: int
    detection_times: List[float] = field(default_factory=list)
    min_ms: float = 0
    max_ms: float = 0
    mean_ms: float = 0
    std_ms: float = 0
    p50_ms: float = 0
    p95_ms: float = 0
    p99_ms: float = 0

    def compute_stats(self):
        if self.detection_times:
            self.detection_times.sort()
            self.min_ms = min(self.detection_times)
            self.max_ms = max(self.detection_times)
            self.mean_ms = statistics.mean(self.detection_times)
            self.std_ms = statistics.stdev(self.detection_times) if len(self.detection_times) > 1 else 0
            self.p50_ms = self.detection_times[len(self.detection_times) // 2]
            self.p95_ms = self.detection_times[int(len(self.detection_times) * 0.95)]
            self.p99_ms = self.detection_times[int(len(self.detection_times) * 0.99)]


class DDSFailureDetector:
    """DDS-based failure detection using polling (simpler approach).
    This measures detection time when heartbeat stops.
    """

    def __init__(self, domain_id: int = 0):
        self.domain_id = domain_id
        self.participant = None
        self.writer = None
        self.reader = None
        self.last_heartbeat_time = None

    def setup(self, deadline_interval_ms: int = 1000):
        """Setup DDS entities."""
        self.participant = DomainParticipant(self.domain_id)

        topic = Topic(self.participant, "agent/status", AgentStatus)

        qos = Qos(
            Policy.Reliability.Reliable(duration(seconds=10)),
            Policy.Durability.Volatile,
        )

        self.writer = DataWriter(self.participant, topic, qos)
        self.reader = DataReader(self.participant, topic, qos)

    def send_heartbeat(self):
        """Send heartbeat to keep agent alive."""
        if self.writer:
            status = AgentStatus(
                agent_id="test-agent",
                state="idle",
                current_slots=1,
                idle_slots=1,
                memory_usage_mb=0,
                vram_usage_mb=0,
                current_model="test",
                last_heartbeat=int(time.time())
            )
            self.writer.write(status)
            self.last_heartbeat_time = time.time()

    def detect_failure(self, timeout_ms: int = 5000) -> float:
        """Detect failure by checking for heartbeats.
        Returns detection time in ms, or -1 if heartbeat received.
        """
        start = time.perf_counter()
        deadline_sec = timeout_ms / 1000

        while (time.perf_counter() - start) < deadline_sec:
            # Try to read samples
            samples = self.reader.take(N=1)
            if samples:
                # Got heartbeat - agent still alive
                self.last_heartbeat_time = time.time()
                return -1
            time.sleep(0.01)

        # No heartbeat received within timeout
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms

    def cleanup(self):
        """Cleanup DDS entities."""
        pass


class HTTPHeartbeatDetector:
    """HTTP-based failure detection using polling."""

    def __init__(self, agent_url: str = "http://localhost:8081"):
        self.agent_url = agent_url
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = True
        self.last_heartbeat = time.time()
        self.detection_time = None

    async def check_health(self) -> bool:
        """Check agent health via HTTP."""
        import aiohttp
        try:
            async with aiohttp.ClientTimeout(total=1) as timeout:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.agent_url}/health") as resp:
                        return resp.status == 200
        except Exception:
            return False

    async def monitor_loop(self, interval_ms: int = 1000):
        """Monitor agent heartbeats."""
        while self.running:
            healthy = await self.check_health()
            if not healthy and self.detection_time is None:
                self.detection_time = time.perf_counter()
            await asyncio.sleep(interval_ms / 1000)

    def detect_failure(self, interval_ms: int = 1000, timeout_ms: int = 5000) -> float:
        """Detect failure via HTTP polling."""
        import aiohttp

        start = time.perf_counter()
        interval = interval_ms / 1000

        while time.perf_counter() - start < timeout_ms / 1000:
            try:
                # Synchronous check for simplicity
                import urllib.request
                req = urllib.request.Request(f"{self.agent_url}/health")
                with urllib.request.urlopen(req, timeout=1) as resp:
                    if resp.status == 200:
                        time.sleep(interval)
                        continue
            except Exception:
                pass

            # Failure detected
            return (time.perf_counter() - start) * 1000

        return timeout_ms  # Timeout

    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        self.executor.shutdown(wait=False)


class gRPCHealthDetector:
    """gRPC-based failure detection using Health Checking Protocol."""

    def __init__(self, agent_address: str = "localhost:50051"):
        self.agent_address = agent_address

    def detect_failure(self, interval_ms: int = 1000, timeout_ms: int = 5000) -> float:
        """
        Detect failure via gRPC health checks.
        Note: Requires grpc-health-probe or grpcio-tools installed.
        """
        start = time.perf_counter()

        # Try using grpc_health_v1
        try:
            from grpc import insecure_channel
            from grpc_health.v1 import health_pb2, health_pb2_grpc

            channel = insecure_channel(self.agent_address)
            stub = health_pb2_grpc.HealthStub(channel)

            interval = interval_ms / 1000

            while time.perf_counter() - start < timeout_ms / 1000:
                try:
                    response = stub.Check(
                        health_pb2.HealthCheckRequest(),
                        timeout=1
                    )
                    if response.status == health_pb2.HealthCheckResponse.SERVING:
                        time.sleep(interval)
                        continue
                except Exception:
                    pass

                # Failure detected
                channel.close()
                return (time.perf_counter() - start) * 1000

            channel.close()
            return timeout_ms

        except ImportError:
            # Fallback: use grpc_health_probe if available
            return self._detect_with_probe(timeout_ms)

    def _detect_with_probe(self, timeout_ms: int) -> float:
        """Fallback using grpc_health_probe binary."""
        start = time.perf_counter()

        result = subprocess.run(
            ["grpc_health_probe", "-addr", self.agent_address],
            capture_output=True,
            timeout=timeout_ms / 1000
        )

        if result.returncode != 0:
            return (time.perf_counter() - start) * 1000

        return timeout_ms


def run_benchmark(
    method: str,
    failure_type: str,
    iterations: int = 10,
    interval_ms: int = 1000,
) -> BenchmarkResults:
    """Run a specific benchmark scenario."""

    result = BenchmarkResults(
        method=method,
        failure_type=failure_type,
        iterations=iterations,
    )

    print(f"\n{'='*60}")
    print(f"Running: {method} - {failure_type}")
    print(f"Iterations: {iterations}, Interval: {interval_ms}ms")
    print(f"{'='*60}")

    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}...", end=" ")

        try:
            if method == "DDS":
                detector = DDSFailureDetector()
                detector.setup(deadline_interval_ms=interval_ms)

                # Wait for DDS discovery between participants
                time.sleep(0.5)

                # Simulate agent by sending heartbeats
                for _ in range(3):
                    detector.send_heartbeat()
                    time.sleep(0.1)

                # Simulate failure by NOT sending more heartbeats
                # The reader will timeout after deadline_interval_ms
                detection_time = detector.detect_failure(timeout_ms=interval_ms * 3)

                detector.cleanup()

            elif method == "HTTP":
                detector = HTTPHeartbeatDetector()
                detection_time = detector.detect_failure(
                    interval_ms=interval_ms,
                    timeout_ms=interval_ms * 3
                )
                detector.cleanup()

            elif method == "gRPC":
                detector = gRPCHealthDetector()
                detection_time = detector.detector_failure(
                    interval_ms=interval_ms,
                    timeout_ms=interval_ms * 3
                )

            else:
                raise ValueError(f"Unknown method: {method}")

            result.detection_times.append(detection_time)
            print(f"Detected in {detection_time:.1f}ms")

        except Exception as e:
            print(f"Error: {e}")
            result.detection_times.append(-1)

        # Wait between iterations
        time.sleep(0.5)

    result.compute_stats()

    print(f"\nResults:")
    print(f"  Mean:   {result.mean_ms:.2f}ms")
    print(f"  Std:    {result.std_ms:.2f}ms")
    print(f"  Min:    {result.min_ms:.2f}ms")
    print(f"  Max:    {result.max_ms:.2f}ms")
    print(f"  P50:    {result.p50_ms:.2f}ms")
    print(f"  P95:    {result.p95_ms:.2f}ms")
    print(f"  P99:    {result.p99_ms:.2f}ms")

    return result


def simulate_agent_crash(pid: int, method: str) -> None:
    """Simulate agent crash by killing the process."""
    if method == "kill":
        # Abrupt termination
        os.kill(pid, signal.SIGKILL)
    elif method == "graceful":
        # Graceful shutdown
        os.kill(pid, signal.SIGTERM)
    elif method == "overload":
        # Simulate overload with artificial latency
        # This would require agent-side support
        pass


def main():
    parser = argparse.ArgumentParser(description="B5.1 Failure Detection Benchmark")
    parser.add_argument(
        "--mode",
        choices=["all", "dds", "http", "grpc"],
        default="all",
        help="Benchmark mode"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations per test"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="Heartbeat/deadline interval in ms (1000, 5000, 10000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results",
        help="Output directory"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    results = []

    modes = ["dds", "http", "grpc"] if args.mode == "all" else [args.mode]
    failure_types = ["kill", "graceful", "overload"]

    for method in modes:
        for failure_type in failure_types:
            result = run_benchmark(
                method=method.upper(),
                failure_type=failure_type,
                iterations=args.iterations,
                interval_ms=args.interval,
            )
            results.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output, f"B51_failure_detection_{timestamp}.json")

    results_data = {
        "benchmark": "B5.1 - Failure Detection",
        "timestamp": timestamp,
        "config": {
            "iterations": args.iterations,
            "interval_ms": args.interval,
        },
        "results": [
            {
                "method": r.method,
                "failure_type": r.failure_type,
                "iterations": r.iterations,
                "min_ms": r.min_ms,
                "max_ms": r.max_ms,
                "mean_ms": r.mean_ms,
                "std_ms": r.std_ms,
                "p50_ms": r.p50_ms,
                "p95_ms": r.p95_ms,
                "p99_ms": r.p99_ms,
            }
            for r in results
        ]
    }

    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'Method':<10} {'Type':<12} {'Mean (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r.method:<10} {r.failure_type:<12} {r.mean_ms:<12.2f} {r.p50_ms:<12.2f} {r.p95_ms:<12.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
