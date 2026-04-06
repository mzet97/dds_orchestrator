#!/usr/bin/env python3
"""
Distributed Load Generator
Generates N concurrent async clients against the orchestrator.
Runs on client machine (.63) targeting orchestrator on .61.
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import aiohttp


@dataclass
class LoadConfig:
    orchestrator_url: str = "http://192.168.1.61:8080"
    num_clients: int = 1000
    protocol: str = "http"          # http | grpc | dds
    duration_s: int = 60
    ramp_up_s: int = 10
    prompt: str = "Explain DDS in one sentence."
    max_tokens: int = 50
    scenario: str = "S3"
    warmup_requests: int = 50
    algorithm: str = "least_loaded"


@dataclass
class LoadResult:
    scenario: str = ""
    protocol: str = ""
    algorithm: str = ""
    num_clients: int = 0
    stats: dict = field(default_factory=dict)
    latencies: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    duration_s: float = 0
    timestamp: str = ""
    total_requests: int = 0


class LoadGenerator:
    """Load generator with gradual ramp-up and metrics collection."""

    def __init__(self, config: LoadConfig):
        self.config = config
        self._stop = False
        self._warmup_done = asyncio.Event()

    async def run(self) -> LoadResult:
        """Run the load test and return results."""
        print(f"Load Generator: {self.config.num_clients} clients, "
              f"{self.config.duration_s}s duration, "
              f"ramp-up {self.config.ramp_up_s}s")
        print(f"Target: {self.config.orchestrator_url}")
        print(f"Scenario: {self.config.scenario}, Protocol: {self.config.protocol}")

        # Set routing algorithm if specified
        if self.config.algorithm:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.put(
                        f"{self.config.orchestrator_url}/api/v1/routing/algorithm",
                        json={"algorithm": self.config.algorithm},
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status == 200:
                            print(f"Algorithm set to: {self.config.algorithm}")
            except Exception:
                pass

        all_latencies = []
        all_errors = []
        warmup_count = [0]

        # Warmup phase
        if self.config.warmup_requests > 0:
            print(f"\nWarmup: {self.config.warmup_requests} requests...")
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=50)
            ) as session:
                warmup_tasks = [
                    self._single_request(session, -1)
                    for _ in range(self.config.warmup_requests)
                ]
                await asyncio.gather(*warmup_tasks, return_exceptions=True)
            print("Warmup complete")

        # Main load phase with ramp-up
        print(f"\nStarting load test...")
        t_start = time.time()

        # Create client tasks with staggered start
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for client_id in range(self.config.num_clients):
                # Calculate start delay for ramp-up
                if self.config.ramp_up_s > 0:
                    delay = (client_id / self.config.num_clients) * self.config.ramp_up_s
                else:
                    delay = 0
                tasks.append(
                    asyncio.create_task(
                        self._client_loop(session, client_id, delay,
                                          all_latencies, all_errors)
                    )
                )

            # Wait for duration
            await asyncio.sleep(self.config.ramp_up_s + self.config.duration_s)
            self._stop = True

            # Wait for all clients to finish
            await asyncio.gather(*tasks, return_exceptions=True)

        t_end = time.time()
        actual_duration = t_end - t_start

        # Compute stats
        stats = self._compute_stats(all_latencies)
        stats["total_requests"] = len(all_latencies) + len(all_errors)
        stats["error_count"] = len(all_errors)
        stats["error_rate"] = len(all_errors) / max(stats["total_requests"], 1)
        stats["actual_duration_s"] = actual_duration

        result = LoadResult(
            scenario=self.config.scenario,
            protocol=self.config.protocol,
            algorithm=self.config.algorithm,
            num_clients=self.config.num_clients,
            stats=stats,
            latencies=all_latencies,
            errors=all_errors[:100],  # Cap error list
            duration_s=actual_duration,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_requests=stats["total_requests"],
        )

        self._print_results(result)
        return result

    async def _client_loop(self, session, client_id, start_delay,
                           latencies, errors):
        """Single client loop: sends requests until stopped."""
        await asyncio.sleep(start_delay)

        while not self._stop:
            try:
                latency = await self._single_request(session, client_id)
                latencies.append(latency)
            except Exception as e:
                errors.append({
                    "client_id": client_id,
                    "error": str(e),
                    "timestamp": time.time(),
                })
                # Backoff on errors
                await asyncio.sleep(0.5)

    async def _single_request(self, session, client_id):
        """Send a single request and return latency in ms."""
        t0 = time.time()
        payload = {
            "messages": [{"role": "user", "content": self.config.prompt}],
            "max_tokens": self.config.max_tokens,
            "scenario": self.config.scenario,
        }

        try:
            async with session.post(
                f"{self.config.orchestrator_url}/api/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status == 429:
                    # Rate limited - backoff
                    await asyncio.sleep(1)
                    raise RuntimeError("Rate limited (429)")
                elif resp.status == 503:
                    raise RuntimeError("No capacity (503)")
                elif resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")

                await resp.json()
                return (time.time() - t0) * 1000  # ms
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout")

    def _compute_stats(self, latencies):
        """Compute percentile statistics."""
        if not latencies:
            return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "max": 0,
                    "min": 0, "throughput_rps": 0}

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        return {
            "p50": sorted_lat[int(n * 0.50)],
            "p95": sorted_lat[int(n * 0.95)],
            "p99": sorted_lat[min(int(n * 0.99), n - 1)],
            "mean": statistics.mean(sorted_lat),
            "max": max(sorted_lat),
            "min": min(sorted_lat),
            "throughput_rps": n / max(self.config.duration_s, 1),
        }

    def _print_results(self, result: LoadResult):
        """Print results summary."""
        s = result.stats
        print(f"\n{'='*60}")
        print(f"Results: {result.scenario} | {result.protocol} | "
              f"{result.algorithm} | {result.num_clients} clients")
        print(f"{'='*60}")
        print(f"Total Requests: {s.get('total_requests', 0)}")
        print(f"Errors:         {s.get('error_count', 0)} "
              f"({s.get('error_rate', 0)*100:.1f}%)")
        print(f"Throughput:     {s.get('throughput_rps', 0):.1f} req/s")
        print(f"Latency p50:    {s.get('p50', 0):.1f} ms")
        print(f"Latency p95:    {s.get('p95', 0):.1f} ms")
        print(f"Latency p99:    {s.get('p99', 0):.1f} ms")
        print(f"Latency mean:   {s.get('mean', 0):.1f} ms")
        print(f"Latency max:    {s.get('max', 0):.1f} ms")
        print(f"Duration:       {result.duration_s:.1f}s")
        print(f"{'='*60}")


async def run_load_test(args):
    config = LoadConfig(
        orchestrator_url=args.url,
        num_clients=args.clients,
        protocol=args.protocol,
        duration_s=args.duration,
        ramp_up_s=args.ramp_up,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        scenario=args.scenario,
        warmup_requests=args.warmup,
        algorithm=args.algorithm,
    )

    gen = LoadGenerator(config)
    result = await gen.run()

    # Save to MongoDB
    if args.save_mongo:
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from mongo_layer import MongoMetricsStore
            store = MongoMetricsStore(args.save_mongo)
            await store.connect()
            run_data = {
                "run_id": f"{result.scenario}_{result.protocol}_{result.algorithm}_{result.timestamp}",
                "experiment": args.experiment or "load_test",
                "scenario": result.scenario,
                "protocol": result.protocol,
                "algorithm": result.algorithm,
                "num_clients": result.num_clients,
                "config": asdict(config),
                "results": result.stats,
                "timestamp": time.time(),
            }
            run_id = await store.save_run(run_data)
            print(f"Saved to MongoDB: {run_id}")
            await store.close()
        except Exception as e:
            print(f"MongoDB save failed: {e}")

    # Save latencies to file
    if args.save_file:
        with open(args.save_file, "w") as f:
            json.dump({
                "scenario": result.scenario,
                "protocol": result.protocol,
                "algorithm": result.algorithm,
                "num_clients": result.num_clients,
                "stats": result.stats,
                "latencies": result.latencies,
                "errors": result.errors,
            }, f)
        print(f"Results saved to {args.save_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Load Generator")
    parser.add_argument("--url", type=str, default="http://192.168.1.61:8080")
    parser.add_argument("--clients", type=int, default=1000)
    parser.add_argument("--protocol", type=str, default="http",
                       choices=["http", "grpc", "dds"])
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--ramp-up", type=int, default=10)
    parser.add_argument("--prompt", type=str, default="Explain DDS in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--scenario", type=str, default="S3")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--algorithm", type=str, default="least_loaded",
                       choices=["round_robin", "least_loaded", "weighted_score"])
    parser.add_argument("--save-mongo", type=str, default="",
                       help="MongoDB URL to save results")
    parser.add_argument("--save-file", type=str, default="",
                       help="File path to save results JSON")
    parser.add_argument("--experiment", type=str, default="",
                       help="Experiment name for MongoDB")

    args = parser.parse_args()
    asyncio.run(run_load_test(args))


if __name__ == "__main__":
    main()
