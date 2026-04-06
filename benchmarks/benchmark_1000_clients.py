#!/usr/bin/env python3
"""
1000-client load test against 10-instance deployment.
Run from .63 (client machine) targeting orchestrator on .62.

Topology:
  .61 (RTX 3080 10GB): 6 instances, ports 8082-8087, parallel=15
  .60 (RX 6600M 8GB):  4 instances, ports 8088-8091, parallel=10
  .62: Orchestrator on port 8080
  .51: Redis + MongoDB (k8s)
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


async def run(args):
    for protocol in args.protocols:
        for algorithm in args.algorithms:
            print(f"\n{'='*60}")
            print(f"  S3_1000 | {protocol} | {algorithm}")
            print(f"{'='*60}")

            config = LoadConfig(
                orchestrator_url=args.url,
                num_clients=1000,
                protocol=protocol,
                duration_s=120,
                ramp_up_s=15,
                prompt="Explain DDS middleware in one sentence.",
                max_tokens=50,
                scenario="S3_1000",
                warmup_requests=200,
                algorithm=algorithm,
            )
            gen = LoadGenerator(config)
            result = await gen.run()

            # Save to MongoDB if available
            if args.save_mongo:
                try:
                    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
                    from mongo_layer import MongoMetricsStore

                    store = MongoMetricsStore(args.save_mongo)
                    await store.connect()
                    run_data = {
                        "run_id": f"S3_1000_{protocol}_{algorithm}_{time.strftime('%Y%m%d_%H%M%S')}",
                        "experiment": "10inst_1000agents",
                        "scenario": "S3_1000",
                        "protocol": protocol,
                        "algorithm": algorithm,
                        "num_clients": 1000,
                        "num_instances": 10,
                        "num_agents": 1000,
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
            fname = f"results_1000_{protocol}_{algorithm}.json"
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
    parser = argparse.ArgumentParser(description="1000-client benchmark")
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
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
