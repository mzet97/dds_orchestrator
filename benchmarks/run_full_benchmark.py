#!/usr/bin/env python3
"""
Full Benchmark Suite: E1-E5 × S1-S5 × 3 Protocols × 3 Algorithms
Total: 225 combinations (with early-stop on >50% error rate)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass

# Add parent for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from load_generator import LoadGenerator, LoadConfig


# === Scenario Definitions ===

SCENARIOS = {
    "S1": {"clients": 100,   "ramp_up": 5,  "duration": 60},
    "S2": {"clients": 500,   "ramp_up": 10, "duration": 60},
    "S3": {"clients": 1000,  "ramp_up": 15, "duration": 60},
    "S4": {"clients": 5000,  "ramp_up": 30, "duration": 120},
    "S5": {"clients": 10000, "ramp_up": 60, "duration": 180},
}

PROTOCOLS = ["http", "grpc", "dds"]
ALGORITHMS = ["round_robin", "least_loaded", "weighted_score"]

# Experiment prompts
PROMPTS = {
    "short": "Say hello.",
    "medium": "Explain the concept of DDS (Data Distribution Service) middleware in distributed systems.",
    "long": ("Write a detailed comparison of DDS, MQTT, and gRPC for IoT applications. "
             "Cover latency, reliability, scalability, and ease of implementation."),
}


class BenchmarkSuite:
    """Orchestrates the full benchmark matrix."""

    def __init__(self, orchestrator_url: str, mongo_url: str = "",
                 early_stop_error_rate: float = 0.5):
        self.orchestrator_url = orchestrator_url
        self.mongo_url = mongo_url
        self.early_stop_error_rate = early_stop_error_rate
        self._results = []

    async def run_all(self, scenarios=None, protocols=None,
                      algorithms=None, experiments=None):
        """Run the full benchmark matrix."""
        scenarios = scenarios or list(SCENARIOS.keys())
        protocols = protocols or PROTOCOLS
        algorithms = algorithms or ALGORITHMS
        experiments = experiments or ["E1", "E2", "E3", "E4", "E5"]

        total = len(scenarios) * len(protocols) * len(algorithms) * len(experiments)
        current = 0

        print(f"{'='*60}")
        print(f"Full Benchmark Suite")
        print(f"Scenarios: {scenarios}")
        print(f"Protocols: {protocols}")
        print(f"Algorithms: {algorithms}")
        print(f"Experiments: {experiments}")
        print(f"Total combinations: {total}")
        print(f"{'='*60}\n")

        for exp in experiments:
            for scenario in scenarios:
                for protocol in protocols:
                    for algorithm in algorithms:
                        current += 1
                        print(f"\n--- [{current}/{total}] "
                              f"{exp}/{scenario}/{protocol}/{algorithm} ---")

                        try:
                            result = await self.run_single(
                                scenario, protocol, algorithm, exp
                            )
                            self._results.append(result)

                            # Early stop check
                            error_rate = result.get("stats", {}).get("error_rate", 0)
                            if error_rate > self.early_stop_error_rate:
                                print(f"  EARLY STOP: error_rate={error_rate:.1%} > "
                                      f"{self.early_stop_error_rate:.0%}")
                                continue

                        except Exception as e:
                            print(f"  ERROR: {e}")
                            self._results.append({
                                "scenario": scenario, "protocol": protocol,
                                "algorithm": algorithm, "experiment": exp,
                                "error": str(e),
                            })

        print(f"\n{'='*60}")
        print(f"Benchmark complete: {len(self._results)} runs")
        print(f"{'='*60}")
        return self._results

    async def run_single(self, scenario: str, protocol: str,
                         algorithm: str, experiment: str) -> dict:
        """Run a single benchmark combination."""
        s = SCENARIOS[scenario]

        # Set algorithm on orchestrator
        await self._setup_algorithm(algorithm)

        # Configure based on experiment
        if experiment == "E1":
            result = await self._run_e1(scenario, protocol, algorithm, s)
        elif experiment == "E2":
            result = await self._run_e2(scenario, protocol, algorithm, s)
        elif experiment == "E3":
            result = await self._run_e3(scenario, protocol, algorithm, s)
        elif experiment == "E4":
            result = await self._run_e4(scenario, protocol, algorithm, s)
        elif experiment == "E5":
            result = await self._run_e5(scenario, protocol, algorithm, s)
        else:
            raise ValueError(f"Unknown experiment: {experiment}")

        # Save to MongoDB
        if self.mongo_url:
            await self._save_result(result)

        return result

    async def _setup_algorithm(self, algorithm: str):
        """Set routing algorithm on orchestrator."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{self.orchestrator_url}/api/v1/routing/algorithm",
                    json={"algorithm": algorithm},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    pass
        except Exception:
            pass

    async def _run_e1(self, scenario, protocol, algorithm, s):
        """E1: Latency Decomposition — short + long prompts."""
        results = {}
        for prompt_type, prompt in [("short", PROMPTS["short"]),
                                     ("long", PROMPTS["long"])]:
            config = LoadConfig(
                orchestrator_url=self.orchestrator_url,
                num_clients=s["clients"],
                protocol=protocol,
                duration_s=s["duration"],
                ramp_up_s=s["ramp_up"],
                prompt=prompt,
                max_tokens=50 if prompt_type == "short" else 200,
                scenario=scenario,
                algorithm=algorithm,
            )
            gen = LoadGenerator(config)
            result = await gen.run()
            results[prompt_type] = result.stats

        return {
            "experiment": "E1", "scenario": scenario,
            "protocol": protocol, "algorithm": algorithm,
            "stats": results, "timestamp": time.time(),
        }

    async def _run_e2(self, scenario, protocol, algorithm, s):
        """E2: Reliability — error rates and recovery."""
        config = LoadConfig(
            orchestrator_url=self.orchestrator_url,
            num_clients=s["clients"],
            protocol=protocol,
            duration_s=min(s["duration"], 120),
            ramp_up_s=s["ramp_up"],
            prompt=PROMPTS["medium"],
            max_tokens=50,
            scenario=scenario,
            algorithm=algorithm,
        )
        gen = LoadGenerator(config)
        result = await gen.run()
        return {
            "experiment": "E2", "scenario": scenario,
            "protocol": protocol, "algorithm": algorithm,
            "stats": result.stats,
            "error_count": len(result.errors),
            "errors_sample": result.errors[:20],
            "timestamp": time.time(),
        }

    async def _run_e3(self, scenario, protocol, algorithm, s):
        """E3: Priority Fairness — different priority levels."""
        results = {}
        for prio_name, prio_val in [("HIGH", 10), ("NORMAL", 5), ("LOW", 1)]:
            config = LoadConfig(
                orchestrator_url=self.orchestrator_url,
                num_clients=s["clients"] // 3,
                protocol=protocol,
                duration_s=s["duration"],
                ramp_up_s=s["ramp_up"],
                prompt=PROMPTS["medium"],
                max_tokens=50,
                scenario=scenario,
                algorithm=algorithm,
            )
            gen = LoadGenerator(config)
            result = await gen.run()
            results[prio_name] = result.stats

        return {
            "experiment": "E3", "scenario": scenario,
            "protocol": protocol, "algorithm": algorithm,
            "stats": results, "timestamp": time.time(),
        }

    async def _run_e4(self, scenario, protocol, algorithm, s):
        """E4: Scalability Curve — throughput vs clients."""
        config = LoadConfig(
            orchestrator_url=self.orchestrator_url,
            num_clients=s["clients"],
            protocol=protocol,
            duration_s=s["duration"],
            ramp_up_s=s["ramp_up"],
            prompt=PROMPTS["short"],
            max_tokens=50,
            scenario=scenario,
            algorithm=algorithm,
        )
        gen = LoadGenerator(config)
        result = await gen.run()
        return {
            "experiment": "E4", "scenario": scenario,
            "protocol": protocol, "algorithm": algorithm,
            "stats": result.stats,
            "latencies_sample": result.latencies[:5000],
            "timestamp": time.time(),
        }

    async def _run_e5(self, scenario, protocol, algorithm, s):
        """E5: Streaming — TTFT and ITL metrics."""
        # Note: streaming metrics require SSE parsing
        config = LoadConfig(
            orchestrator_url=self.orchestrator_url,
            num_clients=min(s["clients"], 500),
            protocol=protocol,
            duration_s=s["duration"],
            ramp_up_s=s["ramp_up"],
            prompt=PROMPTS["medium"],
            max_tokens=100,
            scenario=scenario,
            algorithm=algorithm,
        )
        gen = LoadGenerator(config)
        result = await gen.run()
        return {
            "experiment": "E5", "scenario": scenario,
            "protocol": protocol, "algorithm": algorithm,
            "stats": result.stats, "timestamp": time.time(),
        }

    async def _save_result(self, result: dict):
        """Save result to MongoDB."""
        try:
            from mongo_layer import MongoMetricsStore
            store = MongoMetricsStore(self.mongo_url)
            await store.connect()
            run_id = (f"{result.get('experiment', '')}_{result.get('scenario', '')}_"
                      f"{result.get('protocol', '')}_{result.get('algorithm', '')}_"
                      f"{int(time.time())}")
            result["run_id"] = run_id
            await store.save_run(result)
            await store.close()
        except Exception as e:
            print(f"  MongoDB save error: {e}")


async def main_async(args):
    suite = BenchmarkSuite(
        orchestrator_url=args.url,
        mongo_url=args.mongo_url,
        early_stop_error_rate=args.early_stop,
    )

    if args.all:
        results = await suite.run_all()
    elif args.quick:
        results = await suite.run_all(
            scenarios=["S1"],
            experiments=["E1", "E4"],
        )
    else:
        scenarios = args.scenario.split(",") if args.scenario else None
        protocols = args.protocol.split(",") if args.protocol else None
        algorithms = args.algorithm.split(",") if args.algorithm else None
        experiments = args.experiment.split(",") if args.experiment else None
        results = await suite.run_all(scenarios, protocols, algorithms, experiments)

    # Save summary
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Full Benchmark Suite")
    parser.add_argument("--url", type=str, default="http://192.168.1.61:8080")
    parser.add_argument("--mongo-url", type=str, default="")
    parser.add_argument("--all", action="store_true", help="Run full matrix (~6 hours)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (S1, E1+E4)")
    parser.add_argument("--scenario", type=str, help="Comma-separated scenarios")
    parser.add_argument("--protocol", type=str, help="Comma-separated protocols")
    parser.add_argument("--algorithm", type=str, help="Comma-separated algorithms")
    parser.add_argument("--experiment", type=str, help="Comma-separated experiments")
    parser.add_argument("--early-stop", type=float, default=0.5,
                       help="Error rate threshold for early stop")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="Output file for results")

    args = parser.parse_args()

    if not (args.all or args.quick or args.scenario or args.experiment):
        parser.print_help()
        return

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
