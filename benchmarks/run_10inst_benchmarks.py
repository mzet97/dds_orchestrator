#!/usr/bin/env python3
"""
Full benchmark suite runner for the 10-instance topology.
Runs the complete matrix: 5 scenarios x 3 protocols x 3 algorithms = 45 runs.

Topology:
  .61 (RTX 3080 10GB): 6 instances, ports 8082-8087, parallel=15
  .60 (RX 6600M 8GB):  4 instances, ports 8088-8091, parallel=10
  .62: Orchestrator on port 8080
  .63: Client machine (run this script here)
  .51: Redis + MongoDB (k8s)

Scenario Matrix:
  S1:  100 clients,   60s duration,  5s ramp-up,   50 warmup
  S2:  500 clients,   60s duration, 10s ramp-up,  100 warmup
  S3: 1000 clients,  120s duration, 15s ramp-up,  200 warmup
  S4: 5000 clients,  120s duration, 20s ramp-up,  500 warmup
  S5: 10000 clients, 180s duration, 30s ramp-up, 1000 warmup

For each scenario: 3 protocols (http, dds, grpc) x 3 algorithms = 9 runs.
Total: 45 runs. Early-stop if error_rate > 50%.
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from load_generator import LoadConfig, LoadGenerator

ORCHESTRATOR = "http://192.168.1.62:8080"
MONGO_URL = "mongodb://admin:Admin%40123@mongodb.home.arpa:27017/?authSource=admin"

SCENARIOS = {
    "S1": {"clients": 100,   "duration": 60,  "ramp_up": 5,  "warmup": 50},
    "S2": {"clients": 500,   "duration": 60,  "ramp_up": 10, "warmup": 100},
    "S3": {"clients": 1000,  "duration": 120, "ramp_up": 15, "warmup": 200},
    "S4": {"clients": 5000,  "duration": 120, "ramp_up": 20, "warmup": 500},
    "S5": {"clients": 10000, "duration": 180, "ramp_up": 30, "warmup": 1000},
}

PROTOCOLS = ["http", "dds", "grpc"]
ALGORITHMS = ["least_loaded", "round_robin", "weighted_score"]

EARLY_STOP_ERROR_RATE = 0.50


async def run_suite(args):
    """Run the full benchmark matrix."""
    scenarios = args.scenarios or list(SCENARIOS.keys())
    protocols = args.protocols or PROTOCOLS
    algorithms = args.algorithms or ALGORITHMS

    total_runs = len(scenarios) * len(protocols) * len(algorithms)
    current = 0
    results = []
    skipped = 0
    early_stopped = 0

    # Connect to MongoDB once for the entire suite
    mongo_store = None
    if args.save_mongo:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from mongo_layer import MongoMetricsStore

            mongo_store = MongoMetricsStore(args.save_mongo)
            await mongo_store.connect()
            print(f"MongoDB connected: {args.save_mongo}")
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            mongo_store = None

    suite_start = time.time()

    print(f"{'='*70}")
    print(f"  10-INSTANCE BENCHMARK SUITE")
    print(f"{'='*70}")
    print(f"  Target:     {args.url}")
    print(f"  Scenarios:  {scenarios}")
    print(f"  Protocols:  {protocols}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Total runs: {total_runs}")
    print(f"  Early-stop: error_rate > {EARLY_STOP_ERROR_RATE*100:.0f}%")
    print(f"  Topology:   6x RTX 3080 (.61) + 4x RX 6600M (.60)")
    print(f"{'='*70}\n")

    # Estimate total time
    est_seconds = sum(
        SCENARIOS[s]["duration"] + SCENARIOS[s]["ramp_up"] + 10  # +10s overhead
        for s in scenarios
        for _ in protocols
        for _ in algorithms
    )
    est_minutes = est_seconds / 60
    print(f"  Estimated time: ~{est_minutes:.0f} minutes ({est_seconds/3600:.1f} hours)\n")

    for scenario in scenarios:
        s = SCENARIOS[scenario]
        for protocol in protocols:
            for algorithm in algorithms:
                current += 1
                run_label = f"{scenario}/{protocol}/{algorithm}"

                print(f"\n{'='*70}")
                print(f"  [{current}/{total_runs}] {run_label}")
                print(f"  Clients={s['clients']}  Duration={s['duration']}s  "
                      f"Ramp-up={s['ramp_up']}s  Warmup={s['warmup']}")
                print(f"{'='*70}")

                try:
                    config = LoadConfig(
                        orchestrator_url=args.url,
                        num_clients=s["clients"],
                        protocol=protocol,
                        duration_s=s["duration"],
                        ramp_up_s=s["ramp_up"],
                        prompt="Explain DDS middleware in one sentence.",
                        max_tokens=50,
                        scenario=scenario,
                        warmup_requests=s["warmup"],
                        algorithm=algorithm,
                    )
                    gen = LoadGenerator(config)
                    result = await gen.run()

                    run_data = {
                        "run_id": f"10inst_{scenario}_{protocol}_{algorithm}_{time.strftime('%Y%m%d_%H%M%S')}",
                        "experiment": "10inst_full_suite",
                        "scenario": scenario,
                        "protocol": protocol,
                        "algorithm": algorithm,
                        "num_clients": s["clients"],
                        "num_instances": 10,
                        "topology": {"rtx3080": 6, "rx6600m": 4},
                        "config": {
                            "duration_s": s["duration"],
                            "ramp_up_s": s["ramp_up"],
                            "warmup": s["warmup"],
                        },
                        "results": result.stats,
                        "timestamp": time.time(),
                    }
                    results.append(run_data)

                    # Save to MongoDB
                    if mongo_store:
                        try:
                            await mongo_store.save_run(run_data)
                        except Exception as e:
                            print(f"  MongoDB save error: {e}")

                    # Early-stop check
                    error_rate = result.stats.get("error_rate", 0)
                    if error_rate > EARLY_STOP_ERROR_RATE:
                        early_stopped += 1
                        print(f"\n  ** EARLY STOP: error_rate={error_rate*100:.1f}% "
                              f"> {EARLY_STOP_ERROR_RATE*100:.0f}% **")
                        print(f"  Skipping remaining algorithms for "
                              f"{scenario}/{protocol}")
                        # Skip to next protocol (remaining algorithms are likely
                        # to fail too at this load level)
                        break

                except Exception as e:
                    print(f"\n  ERROR running {run_label}: {e}")
                    results.append({
                        "scenario": scenario,
                        "protocol": protocol,
                        "algorithm": algorithm,
                        "error": str(e),
                        "timestamp": time.time(),
                    })
                    skipped += 1

                # Brief pause between runs to let the system stabilize
                if current < total_runs:
                    await asyncio.sleep(3)

    suite_duration = time.time() - suite_start

    # Close MongoDB
    if mongo_store:
        try:
            await mongo_store.close()
        except Exception:
            pass

    # Save all results to file
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(
        results_dir, f"10inst_suite_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_file, "w") as f:
        json.dump(
            {
                "experiment": "10inst_full_suite",
                "topology": {
                    "instances": 10,
                    "rtx3080": 6,
                    "rx6600m": 4,
                    "orchestrator": "192.168.1.62:8080",
                },
                "runs": results,
                "summary": {
                    "total_runs": len(results),
                    "early_stopped": early_stopped,
                    "skipped": skipped,
                    "duration_s": suite_duration,
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to {out_file}")

    # Print summary table
    _print_summary(results, suite_duration, early_stopped, skipped)


def _print_summary(results: list, suite_duration: float,
                   early_stopped: int, skipped: int):
    """Print a formatted summary table of all results."""
    print(f"\n{'='*90}")
    print(f"  BENCHMARK SUITE SUMMARY")
    print(f"  Duration: {suite_duration/60:.1f} minutes | "
          f"Runs: {len(results)} | "
          f"Early-stopped: {early_stopped} | "
          f"Errors: {skipped}")
    print(f"{'='*90}")

    # Header
    print(f"\n  {'Scenario':<10} {'Protocol':<10} {'Algorithm':<16} "
          f"{'Clients':>8} {'Reqs':>8} {'Err%':>7} "
          f"{'p50ms':>8} {'p95ms':>8} {'p99ms':>8} {'RPS':>8}")
    print(f"  {'-'*86}")

    for r in results:
        if "error" in r:
            print(f"  {r.get('scenario','?'):<10} {r.get('protocol','?'):<10} "
                  f"{r.get('algorithm','?'):<16} {'ERROR':>8}")
            continue

        stats = r.get("results", {})
        scenario = r.get("scenario", "?")
        protocol = r.get("protocol", "?")
        algorithm = r.get("algorithm", "?")
        clients = r.get("num_clients", 0)
        total_reqs = stats.get("total_requests", 0)
        error_rate = stats.get("error_rate", 0) * 100
        p50 = stats.get("p50", 0)
        p95 = stats.get("p95", 0)
        p99 = stats.get("p99", 0)
        rps = stats.get("throughput_rps", 0)

        # Highlight high error rates
        err_str = f"{error_rate:>6.1f}%"
        if error_rate > 50:
            err_str = f"*{error_rate:>5.1f}%"

        print(f"  {scenario:<10} {protocol:<10} {algorithm:<16} "
              f"{clients:>8} {total_reqs:>8} {err_str:>7} "
              f"{p50:>8.1f} {p95:>8.1f} {p99:>8.1f} {rps:>8.1f}")

    print(f"  {'-'*86}")

    # Per-protocol aggregation
    print(f"\n  Per-Protocol Averages (excluding errors):")
    print(f"  {'Protocol':<10} {'Runs':>6} {'Avg p50':>10} {'Avg p95':>10} "
          f"{'Avg RPS':>10} {'Avg Err%':>10}")
    print(f"  {'-'*58}")

    for protocol in PROTOCOLS:
        proto_runs = [
            r for r in results
            if r.get("protocol") == protocol and "error" not in r
        ]
        if not proto_runs:
            continue
        avg_p50 = sum(r["results"].get("p50", 0) for r in proto_runs) / len(proto_runs)
        avg_p95 = sum(r["results"].get("p95", 0) for r in proto_runs) / len(proto_runs)
        avg_rps = sum(r["results"].get("throughput_rps", 0) for r in proto_runs) / len(proto_runs)
        avg_err = sum(r["results"].get("error_rate", 0) for r in proto_runs) / len(proto_runs) * 100
        print(f"  {protocol:<10} {len(proto_runs):>6} {avg_p50:>10.1f} "
              f"{avg_p95:>10.1f} {avg_rps:>10.1f} {avg_err:>9.1f}%")

    print(f"\n{'='*90}")


def main():
    parser = argparse.ArgumentParser(
        description="10-instance full benchmark suite runner"
    )
    parser.add_argument("--url", default=ORCHESTRATOR,
                       help="Orchestrator URL (default: %(default)s)")
    parser.add_argument("--scenarios", nargs="+", default=None,
                       choices=list(SCENARIOS.keys()),
                       help="Scenarios to run (default: all S1-S5)")
    parser.add_argument("--protocols", nargs="+", default=None,
                       choices=PROTOCOLS,
                       help="Protocols to test (default: all)")
    parser.add_argument("--algorithms", nargs="+", default=None,
                       choices=ALGORITHMS,
                       help="Algorithms to test (default: all)")
    parser.add_argument("--save-mongo", default=MONGO_URL,
                       help="MongoDB URL for saving results")
    parser.add_argument("--no-mongo", action="store_true",
                       help="Disable MongoDB persistence")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: S1+S2 only, http+dds, least_loaded")

    args = parser.parse_args()

    if args.no_mongo:
        args.save_mongo = ""

    if args.quick:
        args.scenarios = args.scenarios or ["S1", "S2"]
        args.protocols = args.protocols or ["http", "dds"]
        args.algorithms = args.algorithms or ["least_loaded"]
        print("Quick mode: S1+S2, http+dds, least_loaded")

    asyncio.run(run_suite(args))


if __name__ == "__main__":
    main()
