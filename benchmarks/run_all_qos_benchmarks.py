#!/usr/bin/env python3
"""
Run All QoS Benchmarks (B5-B7)
==============================
Executes all benchmark scenarios described in the thesis.

Usage:
    python run_all_qos_benchmarks.py
    python run_all_qos_benchmarks.py --skip B5.1 B5.2
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd: list, description: str) -> int:
    """Run a command and return exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run All QoS Benchmarks B5-B7")
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["B5.1", "B5.2", "B5.3", "B6", "B7"],
        default=[],
        help="Skip specific benchmarks"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results",
        help="Output directory"
    )

    args = parser.parse_args()

    benchmark_dir = Path(__file__).parent
    os.makedirs(args.output, exist_ok=True)

    benchmarks = [
        ("B5.1", ["python", "-m", "benchmarks.qos.benchmark_b51_failure_detection", "--mode", "all", "--output", args.output],
         "Failure Detection (DDS vs HTTP vs gRPC)"),
        ("B5.2", ["python", "-m", "benchmarks.qos.benchmark_b52_message_priority", "--mode", "all", "--output", args.output],
         "Message Prioritization (DDS vs Python vs Redis)"),
        ("B5.3", ["python", "-m", "benchmarks.qos.benchmark_b53_load_balancing", "--mode", "all", "--output", args.output],
         "Load Balancing (DDS vs Python vs Nginx vs K8s)"),
        ("B6", ["python", "-m", "benchmarks.qos.benchmark_b61_autogen_comparison", "--mode", "all", "--output", args.output],
         "AutoGen vs DDS Comparison"),
        ("B7", ["python", "-m", "benchmarks.qos.benchmark_b71_dds_grpc", "--mode", "all", "--output", args.output],
         "DDS vs gRPC Comparison"),
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(args.output, f"benchmark_summary_{timestamp}.txt")

    results = []

    for name, cmd, description in benchmarks:
        if name in args.skip:
            print(f"\nSkipping {name}")
            continue

        print(f"\n{'#'*60}")
        print(f"# {name}: {description}")
        print(f"{'#'*60}")

        returncode = run_command(cmd, description)

        results.append({
            "benchmark": name,
            "description": description,
            "status": "SUCCESS" if returncode == 0 else "FAILED",
            "returncode": returncode,
        })

        if returncode != 0:
            print(f"WARNING: {name} returned non-zero exit code: {returncode}")

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    for r in results:
        status_symbol = "✓" if r["status"] == "SUCCESS" else "✗"
        print(f"{status_symbol} {r['benchmark']}: {r['status']}")

    print(f"\nResults saved to: {args.output}/")

    # Save summary
    with open(summary_file, "w") as f:
        f.write(f"QoS Benchmarks Run - {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"{r['benchmark']}: {r['status']}\n")

    print(f"Summary saved to: {summary_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
