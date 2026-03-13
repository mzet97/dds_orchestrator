#!/usr/bin/env python3
"""
B5.2: Message Prioritization Benchmark
===================================
Compares message prioritization between:
- DDS TRANSPORT_PRIORITY (native)
- Python Priority Queue (multiple queues with round-robin)
- Redis Priority Queue (sorted sets)

NOTA: TRANSPORT_PRIORITY e uma dica para o transporte de rede.
Em localhost/loopback, pode nao haver diferenca mensuravel de ordenacao.
Este benchmark pode mostrar ruido estatistico em vez de diferenca real
quando executado em loopback.

Metrics:
- Latency of high-priority message from send to processing start
- Latency degradation for low-priority messages under load
- Priority inversion prevention

Usage:
    python benchmark_b52_message_priority.py --mode all
    python benchmark_b52_message_priority.py --mode dds
"""

import argparse
import asyncio
import json
import os
import queue
import random
import statistics
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import IDL types
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.core import Policy
from cyclonedds.qos import Qos
from cyclonedds.util import duration
import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


# Define simple IDL type for priority queue messages
@dataclass
@annotate.final
@annotate.autoid("sequential")
class PriorityMessage(idl.IdlStruct, typename="priority.Message"):
    message_id: str
    priority: types.int32
    payload: str
    timestamp: types.int64


@dataclass
class PriorityResult:
    """Result of a single priority test."""
    priority: int  # 0 = highest, 2 = lowest
    send_time: float
    process_time: float
    latency_ms: float
    queue_position: int


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    method: str
    priority_level: int
    num_messages: int
    latencies: List[float] = field(default_factory=list)
    mean_ms: float = 0
    std_ms: float = 0
    min_ms: float = 0
    max_ms: float = 0
    p50_ms: float = 0
    p95_ms: float = 0
    p99_ms: float = 0

    def compute_stats(self):
        if self.latencies:
            self.latencies.sort()
            self.min_ms = min(self.latencies)
            self.max_ms = max(self.latencies)
            self.mean_ms = statistics.mean(self.latencies)
            self.std_ms = statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0
            self.p50_ms = statistics.median(self.latencies)
            if len(self.latencies) >= 20:
                self.p95_ms = statistics.quantiles(self.latencies, n=20)[18]
            else:
                self.p95_ms = max(self.latencies)
            if len(self.latencies) >= 100:
                self.p99_ms = statistics.quantiles(self.latencies, n=100)[98]
            else:
                self.p99_ms = max(self.latencies)


# ============================================================================
# DDS Priority Implementation
# ============================================================================

class DDSPriorityQueue:
    """DDS-based priority queue using TRANSPORT_PRIORITY.

    NOTA: TRANSPORT_PRIORITY is a hint to the network transport layer.
    On localhost/loopback, the OS may not honor priority differentiation.
    This benchmark measures whether DDS delivers any observable ordering
    benefit vs. application-level queuing.
    """

    def __init__(self, domain_id: int = 0):
        self.domain_id = domain_id
        self.participant = None
        self.writers = {}  # priority -> writer
        self.reader = None
        self.messages_received = []
        self.lock = threading.Lock()

    def setup(self):
        """Setup DDS entities with different priority writers."""
        print("NOTA: TRANSPORT_PRIORITY e uma dica para o transporte de rede.")
        print("Em localhost/loopback, pode nao haver diferenca mensuravel de ordenacao.")
        print("Este benchmark pode mostrar ruido estatistico em vez de diferenca real.")
        print()

        self.participant = DomainParticipant(self.domain_id)
        topic = Topic(self.participant, "priority/queue", PriorityMessage)

        # Create writers with different priorities (0=highest, 2=lowest)
        for priority in range(3):
            qos = Qos(
                Policy.Reliability.Reliable(duration(seconds=10)),
                Policy.Durability.Volatile,
                Policy.TransportPriority(priority),  # Higher = more priority
            )
            self.writers[priority] = DataWriter(self.participant, topic, qos)

        # Reader for processing
        self.reader_qos = Qos(
            Policy.Reliability.Reliable(duration(seconds=10)),
            Policy.Durability.Volatile,
        )
        self.reader = DataReader(self.participant, topic, self.reader_qos)

    def send(self, priority: int, message_id: str):
        """Send message with specified priority."""
        writer = self.writers.get(priority)
        if writer:
            msg = PriorityMessage(
                message_id=message_id,
                priority=priority,
                payload="x" * 100,  # 100 bytes payload
                timestamp=int(time.time() * 1000)
            )
            writer.write(msg)

    def receive(self, timeout_ms: int = 1000) -> Optional[PriorityMessage]:
        """Receive next message."""
        deadline = time.time() + (timeout_ms / 1000)
        while time.time() < deadline:
            try:
                samples = self.reader.take(N=1)
                if samples:
                    return samples[0]
            except Exception:
                pass
            time.sleep(0.001)
        return None

    def cleanup(self):
        """Cleanup DDS entities."""
        for writer in self.writers.values():
            writer.close()
        if self.reader:
            self.reader.close()
        if self.participant:
            self.participant.close()


# ============================================================================
# Python Priority Queue Implementation
# ============================================================================

class PythonPriorityQueue:
    """Python-based priority queue with multiple queues and round-robin."""

    def __init__(self):
        self.queues = [queue.Queue() for _ in range(3)]  # 3 priority levels
        self.processing = []
        self.lock = threading.Lock()
        self.message_times = {}  # message_id -> send_time

    def send(self, priority: int, message_id: str):
        """Send message with specified priority."""
        self.message_times[message_id] = time.perf_counter()
        self.queues[priority].put(PriorityMessage(
            message_id=message_id,
            priority=priority,
            payload="x" * 100,
            timestamp=int(time.time() * 1000),
        ))

    def receive(self, timeout_ms: int = 1000) -> Optional[PriorityMessage]:
        """Receive next message using round-robin across priorities."""
        start = time.perf_counter()
        timeout = timeout_ms / 1000

        while time.perf_counter() - start < timeout:
            # Check highest priority first, then round-robin
            for priority in range(3):
                try:
                    msg = self.queues[priority].get_nowait()
                    return msg
                except queue.Empty:
                    continue
            time.sleep(0.001)

        return None

    def get_queue_depth(self, priority: int) -> int:
        """Get current queue depth for a priority level."""
        return self.queues[priority].qsize()

    def cleanup(self):
        """Cleanup resources."""
        for q in self.queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass


# ============================================================================
# Redis Priority Queue Implementation
# ============================================================================

class RedisPriorityQueue:
    """Redis-based priority queue using sorted sets."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        self.message_times = {}
        self.key = "priority:queue"

    def setup(self):
        """Setup Redis connection."""
        try:
            import redis
            self.redis = redis.from_url(self.redis_url, decode_responses=True)
            # Clear existing queue
            self.redis.delete(self.key)
        except ImportError:
            print("Warning: redis-py not installed. Using simulation mode.")
            self.redis = None

    def send(self, priority: int, message_id: str):
        """Send message with specified priority using sorted set."""
        self.message_times[message_id] = time.perf_counter()

        if self.redis:
            # Use negative priority for ascending sort (lower = higher priority)
            # Add timestamp to break ties within same priority
            score = -priority * 1000000 + time.time()
            self.redis.zadd(self.key, {message_id: score})
        else:
            # Simulation mode
            pass

    def receive(self, timeout_ms: int = 1000) -> Optional[PriorityMessage]:
        """Receive highest priority message."""
        if not self.redis:
            return None

        try:
            # Pop highest priority (lowest score)
            result = self.redis.zpopmin(self.key, count=1)
            if result:
                message_id, score = result[0]
                return PriorityMessage(
                    message_id=message_id,
                    priority=-int(score // 1000000),
                    payload="x" * 100,
                    timestamp=int(time.time() * 1000),
                )
        except Exception:
            pass

        return None

    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        if self.redis:
            return self.redis.zcard(self.key)
        return 0

    def cleanup(self):
        """Cleanup resources."""
        if self.redis:
            self.redis.delete(self.key)
            self.redis.close()


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(
    method: str,
    num_high_priority: int = 10,
    num_low_priority: int = 90,
    processing_delay_ms: int = 50,
) -> List[BenchmarkResults]:
    """Run priority benchmark for a specific method.

    NOTE: processing_delay_ms is NOT included in the latency metric.
    Latency measures only the time from send to receive (delivery time).
    The processing delay simulates work between receives to build up queue pressure.
    """

    results_by_priority = {0: [], 1: [], 2: []}

    print(f"\n{'='*60}")
    print(f"Running: {method}")
    print(f"Messages: {num_high_priority} high, {num_low_priority} low priority")
    print(f"Processing delay: {processing_delay_ms}ms (not included in latency)")
    print(f"{'='*60}")

    if method == "DDS":
        queue_impl = DDSPriorityQueue()
        queue_impl.setup()
    elif method == "Python":
        queue_impl = PythonPriorityQueue()
    elif method == "Redis":
        queue_impl = RedisPriorityQueue()
        queue_impl.setup()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Send high priority messages first
    high_priority_ids = [f"high_{i}" for i in range(num_high_priority)]
    low_priority_ids = [f"low_{i}" for i in range(num_low_priority)]

    # Shuffle to simulate real traffic
    all_ids = high_priority_ids + low_priority_ids
    random.shuffle(all_ids)

    # Send all messages
    send_times = {}
    for msg_id in all_ids:
        priority = 0 if msg_id.startswith("high") else 2
        send_times[msg_id] = time.perf_counter()
        queue_impl.send(priority, msg_id)

    # Receive and measure latencies
    received = 0
    total = len(all_ids)

    while received < total:
        msg = queue_impl.receive(timeout_ms=1000)
        if msg:
            # Use message_id consistently (PriorityMessage dataclass attribute)
            msg_id = msg.message_id
            if msg_id in send_times:
                # Measure delivery latency only (excludes processing delay)
                latency_ms = (time.perf_counter() - send_times[msg_id]) * 1000
                priority = msg.priority
                results_by_priority[priority].append(latency_ms)
                received += 1

            # Simulate processing delay to build up queue pressure
            # This is NOT included in the latency metric above
            time.sleep(processing_delay_ms / 1000)
        else:
            break  # timeout, no more messages

    queue_impl.cleanup()

    # Compute stats for each priority level
    results = []
    for priority in range(3):
        if results_by_priority[priority]:
            result = BenchmarkResults(
                method=method,
                priority_level=priority,
                num_messages=len(results_by_priority[priority]),
                latencies=results_by_priority[priority]
            )
            result.compute_stats()
            results.append(result)

            print(f"\nPriority {priority}:")
            print(f"  Mean: {result.mean_ms:.2f}ms")
            print(f"  P50:  {result.p50_ms:.2f}ms")
            print(f"  P95:  {result.p95_ms:.2f}ms")

    return results


def run_with_load(
    method: str,
    num_high_priority: int = 10,
    num_low_priority: int = 90,
    num_concurrent_senders: int = 4,
) -> List[BenchmarkResults]:
    """Run benchmark with concurrent senders to simulate load."""

    results_by_priority = {0: [], 1: [], 2: []}

    print(f"\n{'='*60}")
    print(f"Running: {method} with {num_concurrent_senders} concurrent senders")
    print(f"{'='*60}")

    if method == "DDS":
        queue_impl = DDSPriorityQueue()
        queue_impl.setup()
    elif method == "Python":
        queue_impl = PythonPriorityQueue()
    elif method == "Redis":
        queue_impl = RedisPriorityQueue()
        queue_impl.setup()
    else:
        raise ValueError(f"Unknown method: {method}")

    send_times = {}
    lock = threading.Lock()

    def sender_task(start_idx: int, count: int):
        """Send messages in a thread."""
        for i in range(count):
            priority = 0 if i % 10 == 0 else 2  # 10% high priority
            msg_id = f"sender_{start_idx}_msg_{i}"

            with lock:
                send_times[msg_id] = time.perf_counter()

            queue_impl.send(priority, msg_id)
            time.sleep(0.01)  # Small delay between messages

    # Start sender threads
    messages_per_sender = (num_high_priority + num_low_priority) // num_concurrent_senders
    threads = []
    for s in range(num_concurrent_senders):
        t = threading.Thread(
            target=sender_task,
            args=(s, messages_per_sender)
        )
        threads.append(t)
        t.start()

    # Wait for all sends to complete
    for t in threads:
        t.join()

    # Receive messages
    received = 0
    total = num_high_priority + num_low_priority

    while received < total:
        msg = queue_impl.receive(timeout_ms=5000)
        if msg:
            # Use message_id consistently (PriorityMessage dataclass attribute)
            msg_id = msg.message_id
            if msg_id in send_times:
                latency_ms = (time.perf_counter() - send_times[msg_id]) * 1000
                priority = msg.priority
                results_by_priority[priority].append(latency_ms)
                received += 1
        else:
            break  # timeout, no more messages

        time.sleep(0.01)

    queue_impl.cleanup()

    # Compute stats
    results = []
    for priority in range(3):
        if results_by_priority[priority]:
            result = BenchmarkResults(
                method=method,
                priority_level=priority,
                num_messages=len(results_by_priority[priority]),
                latencies=results_by_priority[priority]
            )
            result.compute_stats()
            results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="B5.2 Message Prioritization Benchmark")
    parser.add_argument(
        "--mode",
        choices=["all", "dds", "python", "redis"],
        default="all",
        help="Benchmark mode"
    )
    parser.add_argument(
        "--high-priority",
        type=int,
        default=10,
        help="Number of high priority messages"
    )
    parser.add_argument(
        "--low-priority",
        type=int,
        default=90,
        help="Number of low priority messages"
    )
    parser.add_argument(
        "--senders",
        type=int,
        default=1,
        help="Number of concurrent senders"
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

    modes = ["DDS", "Python", "Redis"] if args.mode == "all" else [args.mode.upper()]

    for method in modes:
        if args.senders > 1:
            result = run_with_load(
                method=method,
                num_high_priority=args.high_priority,
                num_low_priority=args.low_priority,
                num_concurrent_senders=args.senders,
            )
        else:
            result = run_benchmark(
                method=method,
                num_high_priority=args.high_priority,
                num_low_priority=args.low_priority,
            )
        results.extend(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output, f"B52_priority_{timestamp}.json")

    results_data = {
        "benchmark": "B5.2 - Message Prioritization",
        "timestamp": timestamp,
        "config": {
            "high_priority": args.high_priority,
            "low_priority": args.low_priority,
            "concurrent_senders": args.senders,
        },
        "results": [
            {
                "method": r.method,
                "priority_level": r.priority_level,
                "num_messages": r.num_messages,
                "mean_ms": r.mean_ms,
                "std_ms": r.std_ms,
                "min_ms": r.min_ms,
                "max_ms": r.max_ms,
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
    print(f"\n{'Method':<10} {'Priority':<10} {'Count':<8} {'Mean (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r.method:<10} {r.priority_level:<10} {r.num_messages:<8} {r.mean_ms:<12.2f} {r.p50_ms:<12.2f} {r.p95_ms:<12.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
