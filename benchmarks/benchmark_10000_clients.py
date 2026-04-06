#!/usr/bin/env python3
"""
10,000-client load test against 10-instance deployment.
Run from .63 (client machine) targeting orchestrator on .62.

Supports distributed mode: split 10,000 clients across N worker
processes coordinated via Redis (signal_ready, wait_start, push_result,
collect_results from redis_layer.py).

Topology:
  .61 (RTX 3080 10GB): 6 instances, ports 8082-8087, parallel=15
  .60 (RX 6600M 8GB):  4 instances, ports 8088-8091, parallel=10
  .62: Orchestrator on port 8080
  .51: Redis + MongoDB (k8s)

Usage:
  # Single-machine (all 10,000 clients from one process):
  python benchmark_10000_clients.py

  # Distributed (coordinator launches, workers join):
  # On coordinator:
  python benchmark_10000_clients.py --distributed --num-workers 4 --coordinator
  # On each worker:
  python benchmark_10000_clients.py --distributed --num-workers 4 --worker-id worker-1
  python benchmark_10000_clients.py --distributed --num-workers 4 --worker-id worker-2
  ...
"""
import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))
from load_generator import LoadConfig, LoadGenerator

ORCHESTRATOR = "http://192.168.1.62:8080"
MONGO_URL = "mongodb://admin:Admin%40123@mongodb.home.arpa:27017/?authSource=admin"
REDIS_URL = "redis://redis.home.arpa:6379"
REDIS_PASSWORD = "Admin@123"

TOTAL_CLIENTS = 10000
DURATION_S = 180
RAMP_UP_S = 30
WARMUP_REQUESTS = 1000
SCENARIO = "S5_10000"


async def run_single(args):
    """Run all 10,000 clients from a single process."""
    for protocol in args.protocols:
        for algorithm in args.algorithms:
            print(f"\n{'='*60}")
            print(f"  {SCENARIO} | {protocol} | {algorithm} | {TOTAL_CLIENTS} clients")
            print(f"{'='*60}")

            config = LoadConfig(
                orchestrator_url=args.url,
                num_clients=TOTAL_CLIENTS,
                protocol=protocol,
                duration_s=DURATION_S,
                ramp_up_s=RAMP_UP_S,
                prompt="Explain DDS middleware in one sentence.",
                max_tokens=50,
                scenario=SCENARIO,
                warmup_requests=WARMUP_REQUESTS,
                algorithm=algorithm,
            )
            gen = LoadGenerator(config)
            result = await gen.run()

            await _save_results(args, result, protocol, algorithm)


async def run_distributed_worker(args):
    """Run as a distributed worker: take a slice of clients, coordinate via Redis."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from redis_layer import RedisStateManager

    redis = RedisStateManager(url=args.redis_url, password=args.redis_password)
    await redis.connect()

    num_workers = args.num_workers
    worker_id = args.worker_id
    clients_per_worker = math.ceil(TOTAL_CLIENTS / num_workers)

    print(f"Worker {worker_id}: {clients_per_worker} clients (of {TOTAL_CLIENTS} total)")

    for protocol in args.protocols:
        for algorithm in args.algorithms:
            print(f"\n  Worker {worker_id}: {protocol}/{algorithm} - signalling ready")

            # Signal ready and wait for coordinator start signal
            await redis.signal_ready(f"{worker_id}_{protocol}_{algorithm}")

            print(f"  Worker {worker_id}: waiting for start signal...")
            started = await redis.wait_start(timeout_s=120)
            if not started:
                print(f"  Worker {worker_id}: timeout waiting for start, aborting")
                continue

            print(f"  Worker {worker_id}: GO - running {clients_per_worker} clients")

            config = LoadConfig(
                orchestrator_url=args.url,
                num_clients=clients_per_worker,
                protocol=protocol,
                duration_s=DURATION_S,
                ramp_up_s=RAMP_UP_S,
                prompt="Explain DDS middleware in one sentence.",
                max_tokens=50,
                scenario=SCENARIO,
                warmup_requests=WARMUP_REQUESTS // num_workers,
                algorithm=algorithm,
            )
            gen = LoadGenerator(config)
            result = await gen.run()

            # Push partial result to Redis for coordinator to collect
            partial = {
                "worker_id": worker_id,
                "protocol": protocol,
                "algorithm": algorithm,
                "stats": result.stats,
                "num_clients": clients_per_worker,
                "total_requests": result.total_requests,
                "latencies_sample": result.latencies[:500],
                "errors_sample": result.errors[:50],
            }
            await redis.push_result(partial)
            print(f"  Worker {worker_id}: result pushed to Redis")

    await redis.close()


async def run_distributed_coordinator(args):
    """Coordinate distributed workers: wait for ready, signal start, collect results."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from redis_layer import RedisStateManager

    redis = RedisStateManager(url=args.redis_url, password=args.redis_password)
    await redis.connect()

    num_workers = args.num_workers

    for protocol in args.protocols:
        for algorithm in args.algorithms:
            print(f"\n{'='*60}")
            print(f"  COORDINATOR: {SCENARIO} | {protocol} | {algorithm}")
            print(f"  Waiting for {num_workers} workers to be ready...")
            print(f"{'='*60}")

            # Wait for all workers to signal ready
            deadline = time.time() + 120
            while time.time() < deadline:
                ready = await redis.get_ready_count()
                if ready >= num_workers:
                    break
                print(f"  Ready: {ready}/{num_workers}, waiting...")
                await asyncio.sleep(2)
            else:
                ready = await redis.get_ready_count()
                print(f"  WARNING: Only {ready}/{num_workers} workers ready, proceeding anyway")

            # Signal all workers to start
            print(f"  Signalling START to {num_workers} workers")
            for _ in range(num_workers):
                await redis.signal_start()

            # Collect results from all workers
            print(f"  Collecting results (timeout={DURATION_S + RAMP_UP_S + 60}s)...")
            partials = await redis.collect_results(
                count=num_workers,
                timeout_s=DURATION_S + RAMP_UP_S + 60,
            )

            if not partials:
                print(f"  ERROR: No results collected!")
                continue

            # Merge partial results
            merged = _merge_partial_results(partials, protocol, algorithm)
            print(f"\n  MERGED RESULTS ({len(partials)} workers):")
            print(f"  Total requests: {merged['stats'].get('total_requests', 0)}")
            print(f"  Error rate:     {merged['stats'].get('error_rate', 0)*100:.1f}%")
            print(f"  Throughput:     {merged['stats'].get('throughput_rps', 0):.1f} req/s")
            print(f"  p50:            {merged['stats'].get('p50', 0):.1f} ms")
            print(f"  p95:            {merged['stats'].get('p95', 0):.1f} ms")
            print(f"  p99:            {merged['stats'].get('p99', 0):.1f} ms")

            # Save merged results
            if args.save_mongo:
                try:
                    from mongo_layer import MongoMetricsStore

                    store = MongoMetricsStore(args.save_mongo)
                    await store.connect()
                    run_data = {
                        "run_id": f"{SCENARIO}_{protocol}_{algorithm}_{time.strftime('%Y%m%d_%H%M%S')}",
                        "experiment": "10inst_10000agents_distributed",
                        "scenario": SCENARIO,
                        "protocol": protocol,
                        "algorithm": algorithm,
                        "num_clients": TOTAL_CLIENTS,
                        "num_workers": num_workers,
                        "num_instances": 10,
                        "topology": {"rtx3080": 6, "rx6600m": 4},
                        "results": merged["stats"],
                        "worker_results": partials,
                        "timestamp": time.time(),
                    }
                    await store.save_run(run_data)
                    await store.close()
                    print(f"  Saved to MongoDB: {run_data['run_id']}")
                except Exception as e:
                    print(f"  MongoDB save failed: {e}")

            # Save to file
            fname = f"results_10000_{protocol}_{algorithm}_distributed.json"
            with open(fname, "w") as f:
                json.dump(merged, f)
            print(f"  Saved: {fname}")

    await redis.close()


def _merge_partial_results(partials: list[dict], protocol: str, algorithm: str) -> dict:
    """Merge partial results from multiple workers into a single result."""
    all_latencies = []
    total_requests = 0
    total_errors = 0

    for p in partials:
        all_latencies.extend(p.get("latencies_sample", []))
        total_requests += p.get("total_requests", 0)
        stats = p.get("stats", {})
        total_errors += stats.get("error_count", 0)

    # Compute merged percentile stats from the sampled latencies
    sorted_lat = sorted(all_latencies) if all_latencies else []
    n = len(sorted_lat)

    if n > 0:
        stats = {
            "p50": sorted_lat[int(n * 0.50)],
            "p95": sorted_lat[int(n * 0.95)],
            "p99": sorted_lat[min(int(n * 0.99), n - 1)],
            "mean": sum(sorted_lat) / n,
            "max": max(sorted_lat),
            "min": min(sorted_lat),
            "throughput_rps": total_requests / max(DURATION_S, 1),
            "total_requests": total_requests,
            "error_count": total_errors,
            "error_rate": total_errors / max(total_requests, 1),
        }
    else:
        stats = {
            "p50": 0, "p95": 0, "p99": 0, "mean": 0, "max": 0, "min": 0,
            "throughput_rps": 0, "total_requests": total_requests,
            "error_count": total_errors,
            "error_rate": total_errors / max(total_requests, 1),
        }

    return {
        "scenario": SCENARIO,
        "protocol": protocol,
        "algorithm": algorithm,
        "num_clients": TOTAL_CLIENTS,
        "num_workers": len(partials),
        "stats": stats,
        "worker_results": partials,
    }


async def _save_results(args, result, protocol: str, algorithm: str):
    """Save results to MongoDB and local file."""
    if args.save_mongo:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from mongo_layer import MongoMetricsStore

            store = MongoMetricsStore(args.save_mongo)
            await store.connect()
            run_data = {
                "run_id": f"{SCENARIO}_{protocol}_{algorithm}_{time.strftime('%Y%m%d_%H%M%S')}",
                "experiment": "10inst_10000agents",
                "scenario": SCENARIO,
                "protocol": protocol,
                "algorithm": algorithm,
                "num_clients": TOTAL_CLIENTS,
                "num_instances": 10,
                "num_agents": TOTAL_CLIENTS,
                "topology": {"rtx3080": 6, "rx6600m": 4},
                "results": result.stats,
                "timestamp": time.time(),
            }
            await store.save_run(run_data)
            await store.close()
            print(f"Saved to MongoDB: {run_data['run_id']}")
        except Exception as e:
            print(f"MongoDB save failed: {e}")

    # Save to file
    fname = f"results_10000_{protocol}_{algorithm}.json"
    with open(fname, "w") as f:
        json.dump(
            {
                "scenario": result.scenario,
                "protocol": protocol,
                "algorithm": algorithm,
                "stats": result.stats,
                "latencies": result.latencies[:1000],
            },
            f,
        )
    print(f"Saved: {fname}")


def main():
    parser = argparse.ArgumentParser(description="10,000-client benchmark")
    parser.add_argument("--url", default=ORCHESTRATOR)
    parser.add_argument(
        "--protocols", nargs="+", default=["http", "dds", "grpc"]
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["least_loaded", "round_robin", "weighted_score"],
    )
    parser.add_argument("--save-mongo", default=MONGO_URL)

    # Distributed mode options
    parser.add_argument(
        "--distributed", action="store_true",
        help="Enable distributed mode (split clients across workers via Redis)",
    )
    parser.add_argument(
        "--coordinator", action="store_true",
        help="Run as coordinator (waits for workers, signals start, collects results)",
    )
    parser.add_argument(
        "--worker-id", type=str, default="",
        help="Worker identifier (required in distributed worker mode)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of distributed workers (default: 4, each gets 2500 clients)",
    )
    parser.add_argument("--redis-url", default=REDIS_URL)
    parser.add_argument("--redis-password", default=REDIS_PASSWORD)

    args = parser.parse_args()

    if args.distributed:
        if args.coordinator:
            print(f"Running as COORDINATOR for {args.num_workers} workers")
            asyncio.run(run_distributed_coordinator(args))
        elif args.worker_id:
            print(f"Running as WORKER {args.worker_id}")
            asyncio.run(run_distributed_worker(args))
        else:
            parser.error("--distributed requires either --coordinator or --worker-id")
    else:
        print(f"Running single-process mode: {TOTAL_CLIENTS} clients")
        asyncio.run(run_single(args))


if __name__ == "__main__":
    main()
