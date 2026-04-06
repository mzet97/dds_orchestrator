#!/usr/bin/env python3
"""
Generate plots for 38-instance benchmarks from MongoDB data.
Produces 12 figures for the dissertation.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "38inst"

# Academic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.titlesize': 16, 'font.family': 'sans-serif',
})

COLORS = {
    'http': '#E69F00', 'grpc': '#009E73', 'dds': '#0072B2',
    'round_robin': '#CC79A7', 'least_loaded': '#56B4E9', 'weighted_score': '#D55E00',
    'gpu': '#0072B2', 'cpu': '#E69F00',
}

PROTOCOL_LABELS = {'http': 'HTTP', 'grpc': 'gRPC', 'dds': 'DDS'}
SCENARIO_CLIENTS = {'S1': 100, 'S2': 500, 'S3': 1000, 'S4': 5000, 'S5': 10000}


async def load_runs(mongo_url, **filters):
    """Load benchmark runs from MongoDB."""
    from mongo_layer import MongoMetricsStore
    store = MongoMetricsStore(mongo_url)
    await store.connect()
    runs = await store.get_runs(**filters)
    await store.close()
    return runs


def plot_e1_cdf(runs, output_dir, fmt="png"):
    """Plot 1: E1 Latency CDF — HTTP vs gRPC vs DDS."""
    e1_runs = [r for r in runs if r.get("experiment") == "E1"]
    if not e1_runs:
        print("  No E1 data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scenario in zip(axes, ["S1", "S3"]):
        for protocol in ["http", "grpc", "dds"]:
            matching = [r for r in e1_runs
                        if r.get("scenario") == scenario
                        and r.get("protocol") == protocol]
            if not matching:
                continue
            r = matching[0]
            latencies = sorted(r.get("latencies_sample", r.get("stats", {}).get("short", {}).get("latencies", [])))
            if not latencies:
                continue
            cdf = np.arange(1, len(latencies) + 1) / len(latencies)
            ax.plot(latencies, cdf, label=PROTOCOL_LABELS.get(protocol, protocol),
                    color=COLORS.get(protocol, '#333'), linewidth=2)

        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('CDF')
        ax.set_title(f'Latency CDF — {scenario} ({SCENARIO_CLIENTS.get(scenario, "?")} clients)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('E1: Latency Distribution (CDF)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"01_e1_latency_cdf.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 1: E1 Latency CDF")


def plot_e1_boxplot(runs, output_dir, fmt="png"):
    """Plot 2: E1 Latency Box Plot — p50/p95/p99 per protocol."""
    e1_runs = [r for r in runs if r.get("experiment") == "E1"]
    if not e1_runs:
        print("  No E1 data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    colors_list = []
    for protocol in ["http", "grpc", "dds"]:
        for scenario in ["S1", "S3", "S5"]:
            matching = [r for r in e1_runs
                        if r.get("scenario") == scenario
                        and r.get("protocol") == protocol]
            if matching:
                stats = matching[0].get("stats", {})
                short_stats = stats.get("short", stats)
                vals = [short_stats.get("p50", 0), short_stats.get("p95", 0),
                        short_stats.get("p99", 0)]
                data.append(vals)
                labels.append(f"{PROTOCOL_LABELS.get(protocol, protocol)}\n{scenario}")
                colors_list.append(COLORS.get(protocol, '#333'))

    if data:
        x = np.arange(len(data))
        width = 0.25
        for i, pct_label in enumerate(["p50", "p95", "p99"]):
            vals = [d[i] for d in data]
            ax.bar(x + i * width, vals, width, label=pct_label, alpha=0.8)

        ax.set_xlabel('Protocol / Scenario')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('E1: Latency Percentiles')
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f"02_e1_latency_boxplot.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 2: E1 Latency Box Plot")


def plot_e4_throughput(runs, output_dir, fmt="png"):
    """Plot 3: E4 Throughput Bars."""
    e4_runs = [r for r in runs if r.get("experiment") == "E4"]
    if not e4_runs:
        print("  No E4 data")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = ["S1", "S2", "S3", "S4", "S5"]
    protocols = ["http", "grpc", "dds"]
    x = np.arange(len(scenarios))
    width = 0.25

    for i, protocol in enumerate(protocols):
        throughputs = []
        for scenario in scenarios:
            matching = [r for r in e4_runs
                        if r.get("scenario") == scenario
                        and r.get("protocol") == protocol]
            if matching:
                throughputs.append(matching[0].get("stats", {}).get("throughput_rps", 0))
            else:
                throughputs.append(0)
        ax.bar(x + i * width, throughputs, width,
               label=PROTOCOL_LABELS.get(protocol, protocol),
               color=COLORS.get(protocol, '#333'), alpha=0.8)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Throughput (req/s)')
    ax.set_title('E4: Throughput by Scenario and Protocol')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{s}\n({SCENARIO_CLIENTS[s]})" for s in scenarios])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f"03_e4_throughput.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 3: E4 Throughput Bars")


def plot_e4_scaling_curve(runs, output_dir, fmt="png"):
    """Plot 4: E4 Scaling Curve — latency p95 vs num_clients."""
    e4_runs = [r for r in runs if r.get("experiment") == "E4"]
    if not e4_runs:
        print("  No E4 data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for protocol in ["http", "grpc", "dds"]:
        clients = []
        p95s = []
        for scenario in ["S1", "S2", "S3", "S4", "S5"]:
            matching = [r for r in e4_runs
                        if r.get("scenario") == scenario
                        and r.get("protocol") == protocol]
            if matching:
                clients.append(SCENARIO_CLIENTS[scenario])
                p95s.append(matching[0].get("stats", {}).get("p95", 0))
        if clients:
            ax.plot(clients, p95s, 'o-', label=PROTOCOL_LABELS.get(protocol, protocol),
                    color=COLORS.get(protocol, '#333'), linewidth=2, markersize=8)

    ax.set_xlabel('Concurrent Clients')
    ax.set_ylabel('Latency p95 (ms)')
    ax.set_title('E4: Scaling Curve — Latency vs Clients')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"04_e4_scaling_curve.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 4: E4 Scaling Curve")


def plot_algorithm_comparison(runs, output_dir, fmt="png"):
    """Plot 5: Algorithm Comparison — box plots."""
    fig, ax = plt.subplots(figsize=(10, 6))

    algorithms = ["round_robin", "least_loaded", "weighted_score"]
    data = []
    labels = []

    for algo in algorithms:
        algo_runs = [r for r in runs
                     if r.get("algorithm") == algo and r.get("experiment") == "E4"]
        if algo_runs:
            all_p50 = [r.get("stats", {}).get("p50", 0) for r in algo_runs]
            data.append(all_p50)
            labels.append(algo.replace("_", "\n"))

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, algo in zip(bp['boxes'], algorithms):
            patch.set_facecolor(COLORS.get(algo, '#999'))
            patch.set_alpha(0.7)

    ax.set_ylabel('Latency p50 (ms)')
    ax.set_title('Routing Algorithm Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f"05_algorithm_comparison.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 5: Algorithm Comparison")


def plot_e2_reliability(runs, output_dir, fmt="png"):
    """Plot 6: E2 Reliability — error rate bar chart."""
    e2_runs = [r for r in runs if r.get("experiment") == "E2"]
    if not e2_runs:
        print("  No E2 data")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = ["S1", "S2", "S3", "S4", "S5"]
    protocols = ["http", "grpc", "dds"]
    x = np.arange(len(scenarios))
    width = 0.25

    for i, protocol in enumerate(protocols):
        error_rates = []
        for scenario in scenarios:
            matching = [r for r in e2_runs
                        if r.get("scenario") == scenario
                        and r.get("protocol") == protocol]
            if matching:
                error_rates.append(matching[0].get("stats", {}).get("error_rate", 0) * 100)
            else:
                error_rates.append(0)
        ax.bar(x + i * width, error_rates, width,
               label=PROTOCOL_LABELS.get(protocol, protocol),
               color=COLORS.get(protocol, '#333'), alpha=0.8)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('E2: Reliability — Error Rate by Protocol')
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f"06_e2_reliability.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 6: E2 Reliability")


def plot_e3_priority(runs, output_dir, fmt="png"):
    """Plot 7: E3 Priority Fairness."""
    e3_runs = [r for r in runs if r.get("experiment") == "E3"]
    if not e3_runs:
        print("  No E3 data")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    priorities = ["HIGH", "NORMAL", "LOW"]
    colors_prio = ['#0072B2', '#E69F00', '#CC79A7']

    for r in e3_runs[:1]:  # Use first run
        stats = r.get("stats", {})
        vals = [stats.get(p, {}).get("p50", 0) for p in priorities]
        if any(v > 0 for v in vals):
            ax.bar(priorities, vals, color=colors_prio, alpha=0.8)

    ax.set_xlabel('Priority Level')
    ax.set_ylabel('Latency p50 (ms)')
    ax.set_title('E3: Priority Fairness — Latency by Priority')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f"07_e3_priority.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 7: E3 Priority Fairness")


def plot_e5_streaming(runs, output_dir, fmt="png"):
    """Plot 8: E5 Streaming — TTFT + ITL."""
    e5_runs = [r for r in runs if r.get("experiment") == "E5"]
    if not e5_runs:
        print("  No E5 data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    protocols = ["http", "grpc", "dds"]
    x = np.arange(len(protocols))

    p50s = []
    for protocol in protocols:
        matching = [r for r in e5_runs if r.get("protocol") == protocol]
        if matching:
            p50s.append(matching[0].get("stats", {}).get("p50", 0))
        else:
            p50s.append(0)

    ax.bar(x, p50s, color=[COLORS.get(p, '#333') for p in protocols], alpha=0.8)
    ax.set_xlabel('Protocol')
    ax.set_ylabel('Latency p50 (ms)')
    ax.set_title('E5: Streaming Latency')
    ax.set_xticks(x)
    ax.set_xticklabels([PROTOCOL_LABELS.get(p, p) for p in protocols])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f"08_e5_streaming.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 8: E5 Streaming")


def plot_gpu_vs_cpu(runs, output_dir, fmt="png"):
    """Plot 10: GPU vs CPU instance latency comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    # This requires per-instance metrics from MongoDB
    ax.text(0.5, 0.5, 'Requires per-instance metrics\nfrom MongoDB metrics collection',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('GPU vs CPU Instance Latency')
    plt.tight_layout()
    plt.savefig(output_dir / f"10_gpu_vs_cpu.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot 10: GPU vs CPU (placeholder)")


def generate_latex_table(runs, output_dir):
    """Plot 12: Generate LaTeX summary table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Benchmark Results Summary — 38 Instances, 1000 Agents}",
        r"\label{tab:benchmark_summary}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Scenario & Protocol & p50 (ms) & p95 (ms) & p99 (ms) & RPS & Error\% \\",
        r"\midrule",
    ]

    for scenario in ["S1", "S2", "S3", "S4", "S5"]:
        for protocol in ["http", "grpc", "dds"]:
            matching = [r for r in runs
                        if r.get("scenario") == scenario
                        and r.get("protocol") == protocol
                        and r.get("experiment") == "E4"]
            if matching:
                s = matching[0].get("stats", {})
                lines.append(
                    f"{scenario} & {PROTOCOL_LABELS.get(protocol, protocol)} & "
                    f"{s.get('p50', 0):.1f} & {s.get('p95', 0):.1f} & "
                    f"{s.get('p99', 0):.1f} & {s.get('throughput_rps', 0):.0f} & "
                    f"{s.get('error_rate', 0)*100:.1f} \\\\"
                )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)
    table_path = output_dir / "benchmark_table.tex"
    with open(table_path, "w") as f:
        f.write(latex)
    print(f"  LaTeX table: {table_path}")
    return latex


async def main_async(args):
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    print("=" * 60)
    print("Generating 38-Instance Benchmark Plots")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load data
    if args.mongo_url:
        runs = await load_runs(args.mongo_url)
        print(f"Loaded {len(runs)} runs from MongoDB")
    elif args.input_file:
        with open(args.input_file) as f:
            runs = json.load(f)
        print(f"Loaded {len(runs)} runs from {args.input_file}")
    else:
        print("ERROR: Specify --mongo-url or --input-file")
        return

    if args.all or args.plot == "all":
        plot_e1_cdf(runs, output_dir, fmt)
        plot_e1_boxplot(runs, output_dir, fmt)
        plot_e4_throughput(runs, output_dir, fmt)
        plot_e4_scaling_curve(runs, output_dir, fmt)
        plot_algorithm_comparison(runs, output_dir, fmt)
        plot_e2_reliability(runs, output_dir, fmt)
        plot_e3_priority(runs, output_dir, fmt)
        plot_e5_streaming(runs, output_dir, fmt)
        plot_gpu_vs_cpu(runs, output_dir, fmt)
        generate_latex_table(runs, output_dir)
    elif args.plot:
        plot_fn = {
            "e1_cdf": plot_e1_cdf,
            "e1_boxplot": plot_e1_boxplot,
            "e4_throughput": plot_e4_throughput,
            "e4_scaling_curve": plot_e4_scaling_curve,
            "algorithm_comparison": plot_algorithm_comparison,
            "e2_reliability": plot_e2_reliability,
            "e3_priority": plot_e3_priority,
            "e5_streaming": plot_e5_streaming,
            "gpu_vs_cpu": plot_gpu_vs_cpu,
        }.get(args.plot)
        if plot_fn:
            plot_fn(runs, output_dir, fmt)
        else:
            print(f"Unknown plot: {args.plot}")
    elif args.latex_table:
        generate_latex_table(runs, output_dir)

    print(f"\nDone! Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate 38-instance benchmark plots")
    parser.add_argument("--mongo-url", type=str, default="",
                       help="MongoDB URL to load results")
    parser.add_argument("--input-file", type=str, default="",
                       help="JSON file with benchmark results")
    parser.add_argument("--output-dir", type=str, default="",
                       help="Output directory for plots")
    parser.add_argument("--format", type=str, default="png",
                       choices=["png", "pdf", "svg"])
    parser.add_argument("--all", action="store_true",
                       help="Generate all plots")
    parser.add_argument("--plot", type=str, default="",
                       help="Generate specific plot")
    parser.add_argument("--latex-table", action="store_true",
                       help="Generate LaTeX summary table")

    args = parser.parse_args()

    if not (args.all or args.plot or args.latex_table):
        args.all = True

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
