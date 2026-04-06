#!/usr/bin/env python3
"""
Generate plots for 10-instance benchmarks from MongoDB data.
Produces 8 figures for the dissertation.

Topology:
  .61 (RTX 3080 10GB): 6 instances, ports 8082-8087, parallel=15
  .60 (RX 6600M 8GB):  4 instances, ports 8088-8091, parallel=10
  Orchestrator on .62:8080

Usage:
  python generate_10inst_plots.py --mongo-url "mongodb://admin:Admin%40123@mongodb.home.arpa:27017/?authSource=admin"
  python generate_10inst_plots.py --mongo-url "..." --experiment 10inst_1000agents --output-dir ./plots/10inst
  python generate_10inst_plots.py --mongo-url "..." --format pdf
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ===== Output Directory =====

OUTPUT_DIR = Path(__file__).resolve().parent / "plots" / "10inst"

# ===== Academic Style =====

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ===== Color Palettes =====

PROTOCOL_COLORS = {
    "http": "#E69F00",
    "grpc": "#009E73",
    "dds": "#0072B2",
}

ALGORITHM_COLORS = {
    "round_robin": "#CC79A7",
    "least_loaded": "#56B4E9",
    "weighted_score": "#D55E00",
}

GPU_COLORS = {
    "rtx3080": "#0072B2",
    "rx6600m": "#E69F00",
}

PROTOCOL_LABELS = {"http": "HTTP", "grpc": "gRPC", "dds": "DDS"}
ALGORITHM_LABELS = {
    "round_robin": "Round Robin",
    "least_loaded": "Least Loaded",
    "weighted_score": "Weighted Score",
}

SCENARIO_CLIENTS = {"S1": 100, "S2": 500, "S3": 1000, "S4": 5000, "S5": 10000}
PROTOCOLS = ["http", "grpc", "dds"]
ALGORITHMS = ["round_robin", "least_loaded", "weighted_score"]

# ===== Data Loading =====


async def load_runs_from_mongo(mongo_url: str, experiment: str = None):
    """Load benchmark runs from MongoDB."""
    from mongo_layer import MongoMetricsStore

    store = MongoMetricsStore(mongo_url)
    await store.connect()

    filters = {}
    if experiment:
        filters["experiment"] = experiment

    runs = await store.get_runs(**filters)
    await store.close()
    return runs


async def load_metrics_from_mongo(mongo_url: str, experiment: str = None):
    """Load raw request metrics from MongoDB for per-instance analysis."""
    from mongo_layer import MongoMetricsStore

    store = MongoMetricsStore(mongo_url)
    await store.connect()

    query = {}
    if experiment:
        query["scenario"] = {"$regex": experiment}

    cursor = store._db["metrics"].find(query, {"_id": 0}).limit(50000)
    metrics = await cursor.to_list(length=50000)
    await store.close()
    return metrics


def _get_runs(runs, experiment=None, scenario=None, protocol=None, algorithm=None):
    """Filter runs by criteria."""
    result = runs
    if experiment:
        result = [r for r in result if r.get("experiment") == experiment]
    if scenario:
        result = [r for r in result if r.get("scenario") == scenario]
    if protocol:
        result = [r for r in result if r.get("protocol") == protocol]
    if algorithm:
        result = [r for r in result if r.get("algorithm") == algorithm]
    return result


def _safe_stats(run, key, default=0):
    """Safely extract a stats key from a run document."""
    stats = run.get("stats", run.get("results", {}))
    return stats.get(key, default)


# ===== Figure 1: Latency CDF =====


def plot_latency_cdf(runs, output_dir, fmt="png"):
    """Figure 1: Latency CDF -- DDS vs gRPC vs HTTP for scenarios S1, S3, S5.

    Three subplots side by side, one per scenario.
    """
    scenarios = ["S1", "S3", "S5"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, scenario in zip(axes, scenarios):
        has_data = False
        for protocol in PROTOCOLS:
            matching = _get_runs(runs, scenario=scenario, protocol=protocol)
            if not matching:
                continue

            r = matching[0]
            # Try latencies_sample first, then fall back to stats.latencies
            latencies = sorted(
                r.get("latencies_sample", [])
                or r.get("stats", {}).get("latencies", [])
                or r.get("results", {}).get("latencies", [])
            )
            if not latencies:
                continue

            has_data = True
            cdf = np.arange(1, len(latencies) + 1) / len(latencies)
            ax.plot(
                latencies, cdf,
                label=PROTOCOL_LABELS.get(protocol, protocol),
                color=PROTOCOL_COLORS.get(protocol, "#333"),
                linewidth=2,
            )

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("CDF")
        n_clients = SCENARIO_CLIENTS.get(scenario, "?")
        ax.set_title(f"{scenario} ({n_clients} clients)")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        if not has_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="gray")

    fig.suptitle("Latency CDF by Protocol", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"01_latency_cdf.{fmt}", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"01_latency_cdf.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 1: Latency CDF")


# ===== Figure 2: Throughput vs Clients =====


def plot_throughput(runs, output_dir, fmt="png"):
    """Figure 2: Throughput vs Clients -- line chart 100 to 10K clients, 3 protocol lines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for protocol in PROTOCOLS:
        clients_list = []
        throughput_list = []

        for scenario in ["S1", "S2", "S3", "S4", "S5"]:
            matching = _get_runs(runs, scenario=scenario, protocol=protocol)
            if not matching:
                continue
            r = matching[0]
            rps = _safe_stats(r, "throughput_rps", 0)
            if rps > 0:
                clients_list.append(SCENARIO_CLIENTS[scenario])
                throughput_list.append(rps)

        if clients_list:
            ax.plot(
                clients_list, throughput_list, "o-",
                label=PROTOCOL_LABELS.get(protocol, protocol),
                color=PROTOCOL_COLORS.get(protocol, "#333"),
                linewidth=2, markersize=8,
            )

    ax.set_xlabel("Concurrent Clients")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput vs Number of Clients")
    ax.set_xscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"02_throughput_vs_clients.{fmt}", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"02_throughput_vs_clients.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 2: Throughput vs Clients")


# ===== Figure 3: Algorithm Comparison =====


def plot_algorithm_comparison(runs, output_dir, fmt="png"):
    """Figure 3: Algorithm Comparison -- bar chart, 3 algorithms x 3 protocols, p50 latency."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(ALGORITHMS))
    width = 0.22
    offsets = [-width, 0, width]

    for i, protocol in enumerate(PROTOCOLS):
        p50_values = []
        for algo in ALGORITHMS:
            matching = _get_runs(runs, protocol=protocol, algorithm=algo)
            if matching:
                # Average p50 across all scenarios for this protocol+algo
                p50s = [_safe_stats(r, "p50", 0) for r in matching]
                p50_values.append(np.mean(p50s) if p50s else 0)
            else:
                p50_values.append(0)

        bars = ax.bar(
            x + offsets[i], p50_values, width,
            label=PROTOCOL_LABELS.get(protocol, protocol),
            color=PROTOCOL_COLORS.get(protocol, "#333"),
            alpha=0.85,
        )
        # Value labels on bars
        for bar, val in zip(bars, p50_values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xlabel("Routing Algorithm")
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_title("Routing Algorithm Comparison (p50 Latency)")
    ax.set_xticks(x)
    ax.set_xticklabels([ALGORITHM_LABELS.get(a, a) for a in ALGORITHMS])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / f"03_algorithm_comparison.{fmt}", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"03_algorithm_comparison.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 3: Algorithm Comparison")


# ===== Figure 4: Latency Box Plots =====


def plot_latency_boxplots(runs, output_dir, fmt="png"):
    """Figure 4: Latency Box Plots -- p50/p95/p99 per protocol per scenario."""
    scenarios = ["S1", "S3", "S5"]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(16, 6), sharey=True)

    percentile_labels = ["p50", "p95", "p99"]

    for ax, scenario in zip(axes, scenarios):
        x = np.arange(len(PROTOCOLS))
        width = 0.22
        offsets = [-width, 0, width]

        for j, pct in enumerate(percentile_labels):
            values = []
            for protocol in PROTOCOLS:
                matching = _get_runs(runs, scenario=scenario, protocol=protocol)
                if matching:
                    values.append(_safe_stats(matching[0], pct, 0))
                else:
                    values.append(0)

            ax.bar(
                x + offsets[j], values, width,
                label=pct if scenario == scenarios[0] else None,
                alpha=0.8,
            )

        n_clients = SCENARIO_CLIENTS.get(scenario, "?")
        ax.set_title(f"{scenario} ({n_clients} clients)")
        ax.set_xlabel("Protocol")
        ax.set_xticks(x)
        ax.set_xticklabels([PROTOCOL_LABELS.get(p, p) for p in PROTOCOLS])
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Latency (ms)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Latency Percentiles by Protocol and Scenario", fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / f"04_latency_boxplots.{fmt}", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"04_latency_boxplots.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 4: Latency Box Plots")


# ===== Figure 5: Error Rate vs Clients =====


def plot_error_rate(runs, output_dir, fmt="png"):
    """Figure 5: Error Rate vs Clients -- line chart, error_rate vs client count per protocol."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for protocol in PROTOCOLS:
        clients_list = []
        error_rates = []

        for scenario in ["S1", "S2", "S3", "S4", "S5"]:
            matching = _get_runs(runs, scenario=scenario, protocol=protocol)
            if not matching:
                continue
            r = matching[0]
            er = _safe_stats(r, "error_rate", 0)
            clients_list.append(SCENARIO_CLIENTS[scenario])
            error_rates.append(er * 100)  # Convert to percentage

        if clients_list:
            ax.plot(
                clients_list, error_rates, "s-",
                label=PROTOCOL_LABELS.get(protocol, protocol),
                color=PROTOCOL_COLORS.get(protocol, "#333"),
                linewidth=2, markersize=8,
            )

    ax.set_xlabel("Concurrent Clients")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Error Rate vs Number of Clients")
    ax.set_xscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / f"05_error_rate_vs_clients.{fmt}", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"05_error_rate_vs_clients.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 5: Error Rate vs Clients")


# ===== Figure 6: Slot Saturation Curve =====


def plot_slot_saturation(runs, output_dir, fmt="png"):
    """Figure 6: Slot Saturation Curve -- how slot usage grows with client count.

    Uses Redis snapshot data stored in MongoDB benchmark_runs collection
    (fields: slots_used_snapshot, slots_total=130).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect saturation data from runs that include slot snapshots
    clients_list = []
    utilization_list = []
    peak_used_list = []

    for scenario in ["S1", "S2", "S3", "S4", "S5"]:
        matching = _get_runs(runs, scenario=scenario)
        if not matching:
            continue

        r = matching[0]
        stats = r.get("stats", r.get("results", {}))

        # Try to find slot usage data
        slots_used = stats.get("peak_slots_used",
                               stats.get("slots_used_snapshot",
                               stats.get("avg_slots_used", 0)))
        slots_total = stats.get("slots_total", 130)

        if slots_used > 0:
            n_clients = SCENARIO_CLIENTS.get(scenario, 0)
            clients_list.append(n_clients)
            utilization_list.append(slots_used / max(slots_total, 1) * 100)
            peak_used_list.append(slots_used)

    if clients_list:
        ax.plot(clients_list, utilization_list, "o-",
                color="#0072B2", linewidth=2, markersize=8,
                label="Slot Utilization")

        # Add capacity line
        ax.axhline(y=100, color="#CC0000", linestyle="--", linewidth=1.5,
                    label="Full Capacity (130 slots)")

        # Annotate peak values
        for x, y, used in zip(clients_list, utilization_list, peak_used_list):
            ax.annotate(
                f"{int(used)}/130", (x, y),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=9,
            )

    ax.set_xlabel("Concurrent Clients")
    ax.set_ylabel("Slot Utilization (%)")
    ax.set_title("Slot Saturation Curve (130 Total Slots)")
    ax.set_xscale("log")
    ax.set_ylim(0, 120)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if not clients_list:
        ax.text(0.5, 0.5,
                "No slot saturation data available.\n"
                "Requires benchmark runs with peak_slots_used field.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray")

    plt.tight_layout()
    plt.savefig(output_dir / f"06_slot_saturation.{fmt}", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"06_slot_saturation.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 6: Slot Saturation Curve")


# ===== Figure 7: GPU Comparison =====


def plot_gpu_comparison(runs, output_dir, fmt="png"):
    """Figure 7: GPU Comparison -- RTX 3080 vs RX 6600M latency distribution.

    Uses per-instance metrics to separate RTX (.61) from RX (.60) performance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Ports by GPU type
    rtx_ports = set(range(8082, 8088))  # .61: 8082-8087
    rx_ports = set(range(8088, 8092))    # .60: 8088-8091

    # Subplot 1: p50 latency per GPU type across scenarios
    ax = axes[0]
    scenarios = ["S1", "S3", "S5"]
    x = np.arange(len(scenarios))
    width = 0.35

    rtx_p50s = []
    rx_p50s = []

    for scenario in scenarios:
        matching = _get_runs(runs, scenario=scenario)
        if not matching:
            rtx_p50s.append(0)
            rx_p50s.append(0)
            continue

        r = matching[0]
        # Try to get per-GPU-type stats
        per_gpu = r.get("stats", {}).get("per_gpu_type", {})
        if per_gpu:
            rtx_p50s.append(per_gpu.get("rtx3080", {}).get("p50", 0))
            rx_p50s.append(per_gpu.get("rx6600m", {}).get("p50", 0))
        else:
            # Fall back to per-instance stats
            per_inst = r.get("stats", {}).get("per_instance", {})
            rtx_vals = [
                v.get("p50", 0) for k, v in per_inst.items()
                if int(k) in rtx_ports and v.get("p50", 0) > 0
            ]
            rx_vals = [
                v.get("p50", 0) for k, v in per_inst.items()
                if int(k) in rx_ports and v.get("p50", 0) > 0
            ]
            rtx_p50s.append(np.mean(rtx_vals) if rtx_vals else 0)
            rx_p50s.append(np.mean(rx_vals) if rx_vals else 0)

    bars1 = ax.bar(x - width / 2, rtx_p50s, width,
                   label="RTX 3080 (10GB)", color=GPU_COLORS["rtx3080"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, rx_p50s, width,
                   label="RX 6600M (8GB)", color=GPU_COLORS["rx6600m"], alpha=0.85)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_title("p50 Latency by GPU Type")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({SCENARIO_CLIENTS[s]})" for s in scenarios])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Subplot 2: Throughput per GPU
    ax2 = axes[1]

    rtx_rps = []
    rx_rps = []

    for scenario in scenarios:
        matching = _get_runs(runs, scenario=scenario)
        if not matching:
            rtx_rps.append(0)
            rx_rps.append(0)
            continue

        r = matching[0]
        per_gpu = r.get("stats", {}).get("per_gpu_type", {})
        if per_gpu:
            rtx_rps.append(per_gpu.get("rtx3080", {}).get("throughput_rps", 0))
            rx_rps.append(per_gpu.get("rx6600m", {}).get("throughput_rps", 0))
        else:
            # Estimate from total throughput proportional to instance count
            total_rps = _safe_stats(r, "throughput_rps", 0)
            rtx_rps.append(total_rps * 6 / 10)  # 6 out of 10 instances
            rx_rps.append(total_rps * 4 / 10)    # 4 out of 10 instances

    ax2.bar(x - width / 2, rtx_rps, width,
            label="RTX 3080 (6 inst)", color=GPU_COLORS["rtx3080"], alpha=0.85)
    ax2.bar(x + width / 2, rx_rps, width,
            label="RX 6600M (4 inst)", color=GPU_COLORS["rx6600m"], alpha=0.85)

    ax2.set_xlabel("Scenario")
    ax2.set_ylabel("Throughput (req/s)")
    ax2.set_title("Throughput by GPU Type")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s}\n({SCENARIO_CLIENTS[s]})" for s in scenarios])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    has_data = any(v > 0 for v in rtx_p50s + rx_p50s + rtx_rps + rx_rps)
    if not has_data:
        for ax_item in axes:
            ax_item.text(0.5, 0.5,
                         "Requires per-GPU-type metrics\nin benchmark runs",
                         ha="center", va="center", transform=ax_item.transAxes,
                         fontsize=12, color="gray")

    fig.suptitle("GPU Comparison: RTX 3080 vs RX 6600M", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"07_gpu_comparison.{fmt}", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"07_gpu_comparison.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 7: GPU Comparison")


# ===== Figure 8: Summary Table (LaTeX) =====


def generate_summary_table(runs, output_dir):
    """Figure 8: Generate a LaTeX-formatted summary table of all results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Benchmark Results Summary --- 10 Instances, 1000 Agents}",
        r"\label{tab:10inst_benchmark_summary}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Scenario & Protocol & p50 & p95 & p99 & Throughput & Error",
        r"         &          & (ms) & (ms) & (ms) & (req/s)   & (\%) \\",
        r"\midrule",
    ]

    for scenario in ["S1", "S2", "S3", "S4", "S5"]:
        first_in_scenario = True
        for protocol in PROTOCOLS:
            matching = _get_runs(runs, scenario=scenario, protocol=protocol)
            if not matching:
                continue

            r = matching[0]
            p50 = _safe_stats(r, "p50", 0)
            p95 = _safe_stats(r, "p95", 0)
            p99 = _safe_stats(r, "p99", 0)
            rps = _safe_stats(r, "throughput_rps", 0)
            err = _safe_stats(r, "error_rate", 0) * 100

            scenario_col = f"{scenario} ({SCENARIO_CLIENTS.get(scenario, '?')})" if first_in_scenario else ""
            first_in_scenario = False

            lines.append(
                f"  {scenario_col} & {PROTOCOL_LABELS.get(protocol, protocol)} & "
                f"{p50:.1f} & {p95:.1f} & {p99:.1f} & {rps:.0f} & {err:.2f} \\\\"
            )

        if scenario != "S5":
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)

    # Write .tex file
    table_path = output_dir / "benchmark_summary_10inst.tex"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(latex)

    # Also render as a matplotlib figure for PNG output
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    # Build table data
    col_labels = ["Scenario", "Protocol", "p50 (ms)", "p95 (ms)", "p99 (ms)", "RPS", "Error %"]
    table_data = []

    for scenario in ["S1", "S2", "S3", "S4", "S5"]:
        for protocol in PROTOCOLS:
            matching = _get_runs(runs, scenario=scenario, protocol=protocol)
            if not matching:
                continue
            r = matching[0]
            table_data.append([
                f"{scenario} ({SCENARIO_CLIENTS.get(scenario, '?')})",
                PROTOCOL_LABELS.get(protocol, protocol),
                f"{_safe_stats(r, 'p50', 0):.1f}",
                f"{_safe_stats(r, 'p95', 0):.1f}",
                f"{_safe_stats(r, 'p99', 0):.1f}",
                f"{_safe_stats(r, 'throughput_rps', 0):.0f}",
                f"{_safe_stats(r, 'error_rate', 0) * 100:.2f}",
            ])

    if table_data:
        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.5)

        # Style header
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#4472C4")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No benchmark data available",
                ha="center", va="center", fontsize=14, color="gray")

    ax.set_title("Benchmark Results Summary -- 10 Instances, 1000 Agents",
                 fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "08_summary_table.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "08_summary_table.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Figure 8: Summary Table (LaTeX: {table_path})")
    return latex


# ===== Main =====


async def main_async(args):
    """Load data and generate all plots."""
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format
    experiment = args.experiment

    print("=" * 60)
    print("Generating 10-Instance Benchmark Plots")
    print(f"  Experiment: {experiment}")
    print(f"  Output:     {output_dir}")
    print(f"  Format:     {fmt}")
    print("=" * 60)

    # Load data from MongoDB
    if not args.mongo_url:
        print("ERROR: --mongo-url is required")
        print("Usage: python generate_10inst_plots.py --mongo-url 'mongodb://...'")
        return

    print(f"\nLoading data from MongoDB...")
    runs = await load_runs_from_mongo(args.mongo_url, experiment)
    print(f"  Loaded {len(runs)} benchmark runs")

    if not runs:
        print("\nWARNING: No benchmark runs found.")
        print(f"  Filter: experiment={experiment}")
        print("  Generating empty placeholder plots...")

    # Generate all figures
    print(f"\nGenerating figures...")
    plot_latency_cdf(runs, output_dir, fmt)
    plot_throughput(runs, output_dir, fmt)
    plot_algorithm_comparison(runs, output_dir, fmt)
    plot_latency_boxplots(runs, output_dir, fmt)
    plot_error_rate(runs, output_dir, fmt)
    plot_slot_saturation(runs, output_dir, fmt)
    plot_gpu_comparison(runs, output_dir, fmt)
    generate_summary_table(runs, output_dir)

    print(f"\nDone! {8} figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 10-instance benchmark plots for dissertation"
    )
    parser.add_argument(
        "--mongo-url", type=str, required=True,
        help="MongoDB connection URL",
    )
    parser.add_argument(
        "--experiment", type=str, default="10inst_1000agents",
        help="Experiment name to filter runs (default: 10inst_1000agents)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="",
        help="Output directory for plots (default: benchmarks/plots/10inst)",
    )
    parser.add_argument(
        "--format", type=str, default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format (default: png; PDF always also generated)",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
