#!/usr/bin/env python3
"""
Análise e Visualização dos Resultados do Benchmark DDS

Lê CSVs gerados por benchmark_orchestrator_dds.py, agrega estatísticas,
remove outliers (IQR), gera summary.csv e 6 gráficos.

Uso:
  python analyze_benchmark.py --input results/
  python analyze_benchmark.py --input results/ --remove-outliers --format pdf
"""

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ===========================================================================
# Data Loading
# ===========================================================================

def load_raw_csvs(raw_dir: str) -> List[Dict]:
    """Load all raw CSV files from the results/raw/ directory."""
    rows = []
    csv_files = sorted(Path(raw_dir).glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {raw_dir}")
        sys.exit(1)

    for f in csv_files:
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row["rtt_ms"] = float(row["rtt_ms"])
                row["success"] = row["success"] == "True"
                row["num_clients"] = int(row["num_clients"])
                row["request_num"] = int(row["request_num"])
                row["prompt_tokens"] = int(row.get("prompt_tokens", 0))
                row["completion_tokens"] = int(row.get("completion_tokens", 0))
                row["processing_time_ms"] = int(row.get("processing_time_ms", 0))
                row["timestamp"] = float(row.get("timestamp", 0))
                rows.append(row)

    print(f"Loaded {len(rows)} rows from {len(csv_files)} CSV files")
    return rows


def group_by_scenario(rows: List[Dict]) -> Dict[str, List[Dict]]:
    """Group rows by scenario name."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario"]].append(row)
    return dict(groups)


# ===========================================================================
# Outlier Removal (IQR)
# ===========================================================================

def remove_outliers_iqr(latencies: List[float], factor: float = 1.5) -> Tuple[List[float], int]:
    """
    Remove outliers using IQR method.
    Returns (cleaned latencies, num_removed).
    """
    if len(latencies) < 4:
        return latencies, 0

    s = sorted(latencies)
    n = len(s)
    q1 = s[int(n * 0.25)]
    q3 = s[int(n * 0.75)]
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    cleaned = [x for x in latencies if lower <= x <= upper]
    return cleaned, len(latencies) - len(cleaned)


# ===========================================================================
# Statistics
# ===========================================================================

def compute_stats(latencies: List[float]) -> Dict[str, float]:
    """Compute full statistical summary."""
    if not latencies:
        return {k: 0.0 for k in [
            "count", "mean_ms", "stddev_ms", "min_ms", "max_ms",
            "p50_ms", "p90_ms", "p95_ms", "p99_ms", "cv"
        ]}

    s = sorted(latencies)
    n = len(s)
    mean = statistics.mean(s)
    stddev = statistics.stdev(s) if n > 1 else 0.0

    return {
        "count": n,
        "mean_ms": round(mean, 3),
        "stddev_ms": round(stddev, 3),
        "min_ms": round(s[0], 3),
        "max_ms": round(s[-1], 3),
        "p50_ms": round(s[int(n * 0.50)], 3),
        "p90_ms": round(s[min(int(n * 0.90), n - 1)], 3),
        "p95_ms": round(s[min(int(n * 0.95), n - 1)], 3),
        "p99_ms": round(s[min(int(n * 0.99), n - 1)], 3),
        "cv": round(stddev / mean, 4) if mean > 0 else 0.0,
    }


def compute_per_client_stats(rows: List[Dict]) -> Dict[str, Dict]:
    """Compute stats per client within a scenario."""
    by_client = defaultdict(list)
    for row in rows:
        if row["success"] and row["rtt_ms"] > 0:
            by_client[row["client_id"]].append(row["rtt_ms"])

    result = {}
    for cid, lats in sorted(by_client.items()):
        result[cid] = compute_stats(lats)
    return result


# ===========================================================================
# Aggregation & Summary
# ===========================================================================

def aggregate_and_summarize(
    groups: Dict[str, List[Dict]],
    output_dir: str,
    do_remove_outliers: bool,
) -> List[Dict]:
    """Aggregate per-scenario, write agg CSVs and summary."""

    agg_dir = os.path.join(output_dir, "agg")
    os.makedirs(agg_dir, exist_ok=True)

    summary_rows = []

    for scenario, rows in sorted(groups.items()):
        # Write aggregated CSV
        agg_path = os.path.join(agg_dir, f"{scenario}.csv")
        fieldnames = list(rows[0].keys())
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(rows, key=lambda r: r["timestamp"]):
                writer.writerow(row)

        # Extract successful latencies
        latencies = [r["rtt_ms"] for r in rows if r["success"] and r["rtt_ms"] > 0]
        total = len(rows)
        successful = len(latencies)
        failed = total - successful

        outliers_removed = 0
        if do_remove_outliers and latencies:
            latencies, outliers_removed = remove_outliers_iqr(latencies)
            if outliers_removed > 0:
                print(f"  {scenario}: removed {outliers_removed} outliers via IQR")

        # Wall time
        timestamps = [r["timestamp"] for r in rows if r["timestamp"] > 0]
        wall_time = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0

        stats = compute_stats(latencies)

        # Parse mode and num_clients from scenario name
        parts = scenario.split("_c")
        mode = parts[0] if parts else scenario
        num_clients = int(parts[1]) if len(parts) > 1 else 0

        summary = {
            "scenario": scenario,
            "mode": mode,
            "num_clients": num_clients,
            "total_requests": total,
            "successful": successful,
            "failed": failed,
            "outliers_removed": outliers_removed,
            **stats,
            "throughput_rps": round(successful / wall_time, 4) if wall_time > 0 else 0,
            "wall_time_s": round(wall_time, 2),
        }
        summary_rows.append(summary)

    # Write summary CSV
    summary_path = os.path.join(output_dir, "summary.csv")
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"\nSummary written to {summary_path}")

    return summary_rows


# ===========================================================================
# Plotting
# ===========================================================================

COLORS = {
    "sync": "#2196F3",
    "async": "#4CAF50",
    "parallel": "#FF9800",
}
MARKERS = {"sync": "o", "async": "s", "parallel": "D"}


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "figure.dpi": 150,
    })


def plot_latency_vs_clients(summary: List[Dict], plot_dir: str, fmt: str):
    """Plot 1: Mean + P95 latency vs number of clients, one line per mode."""
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Latência vs Número de Clientes (100% DDS)", fontsize=14, fontweight="bold")

    by_mode = defaultdict(list)
    for s in summary:
        by_mode[s["mode"]].append(s)

    for mode, data in sorted(by_mode.items()):
        data.sort(key=lambda x: x["num_clients"])
        clients = [d["num_clients"] for d in data]
        means = [d["mean_ms"] for d in data]
        p95s = [d["p95_ms"] for d in data]

        color = COLORS.get(mode, "gray")
        marker = MARKERS.get(mode, "o")

        ax1.plot(clients, means, f"-{marker}", color=color, label=mode, linewidth=2, markersize=8)
        ax2.plot(clients, p95s, f"-{marker}", color=color, label=mode, linewidth=2, markersize=8)

    ax1.set_xlabel("Número de Clientes")
    ax1.set_ylabel("Latência Média (ms)")
    ax1.set_title("Mean RTT")
    ax1.legend()

    ax2.set_xlabel("Número de Clientes")
    ax2.set_ylabel("Latência P95 (ms)")
    ax2.set_title("P95 RTT")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(plot_dir, f"latency_vs_clients.{fmt}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_throughput_vs_clients(summary: List[Dict], plot_dir: str, fmt: str):
    """Plot 2: Throughput (req/s) vs number of clients."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Throughput vs Número de Clientes (100% DDS)", fontsize=14, fontweight="bold")

    by_mode = defaultdict(list)
    for s in summary:
        by_mode[s["mode"]].append(s)

    for mode, data in sorted(by_mode.items()):
        data.sort(key=lambda x: x["num_clients"])
        clients = [d["num_clients"] for d in data]
        tput = [d["throughput_rps"] for d in data]

        color = COLORS.get(mode, "gray")
        marker = MARKERS.get(mode, "o")
        ax.plot(clients, tput, f"-{marker}", color=color, label=mode, linewidth=2, markersize=8)

    ax.set_xlabel("Número de Clientes")
    ax.set_ylabel("Throughput (req/s)")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(plot_dir, f"throughput_vs_clients.{fmt}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_boxplot_by_mode(groups: Dict[str, List[Dict]], plot_dir: str, fmt: str):
    """Plot 3: Boxplot of RTT distribution grouped by (mode, client_count)."""
    _setup_style()

    # Organize data
    scenarios = sorted(groups.keys())
    if not scenarios:
        return

    data = []
    labels = []
    colors_list = []
    for sc in scenarios:
        lats = [r["rtt_ms"] for r in groups[sc] if r["success"] and r["rtt_ms"] > 0]
        if lats:
            data.append(lats)
            labels.append(sc.replace("_c", "\nc"))
            mode = sc.split("_c")[0]
            colors_list.append(COLORS.get(mode, "gray"))

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(data) * 1.2), 7))
    fig.suptitle("Distribuição RTT por Cenário (100% DDS)", fontsize=14, fontweight="bold")

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("RTT (ms)")
    ax.set_xlabel("Cenário")
    plt.xticks(rotation=45, ha="right", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.6, label=m) for m, c in COLORS.items()]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    path = os.path.join(plot_dir, f"boxplot_by_mode.{fmt}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cdf_comparison(groups: Dict[str, List[Dict]], plot_dir: str, fmt: str):
    """Plot 4: CDF of latencies for selected scenarios."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("CDF de Latência — Cenários Selecionados (100% DDS)", fontsize=14, fontweight="bold")

    # Select interesting scenarios (parallel at various client counts)
    interesting = []
    for sc in sorted(groups.keys()):
        if "parallel" in sc or "sync_c1" in sc:
            interesting.append(sc)
    if not interesting:
        interesting = list(groups.keys())[:6]

    for sc in interesting:
        lats = sorted([r["rtt_ms"] for r in groups[sc] if r["success"] and r["rtt_ms"] > 0])
        if not lats:
            continue
        n = len(lats)
        cdf = np.arange(1, n + 1) / n
        mode = sc.split("_c")[0]
        color = COLORS.get(mode, "gray")
        ax.plot(lats, cdf, label=sc, linewidth=1.5)

    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(plot_dir, f"cdf_comparison.{fmt}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_heatmap_p95(summary: List[Dict], plot_dir: str, fmt: str):
    """Plot 5: Heatmap of P95 latency — mode (rows) x client_count (cols)."""
    _setup_style()

    modes = sorted(set(s["mode"] for s in summary))
    clients = sorted(set(s["num_clients"] for s in summary))

    if not modes or not clients:
        return

    # Build matrix
    matrix = np.full((len(modes), len(clients)), np.nan)
    for s in summary:
        mi = modes.index(s["mode"])
        ci = clients.index(s["num_clients"])
        matrix[mi, ci] = s["p95_ms"]

    fig, ax = plt.subplots(figsize=(max(8, len(clients) * 1.5), 4))
    fig.suptitle("Heatmap P95 Latência (ms) — 100% DDS", fontsize=14, fontweight="bold")

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(clients)))
    ax.set_xticklabels([str(c) for c in clients])
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(modes)
    ax.set_xlabel("Número de Clientes")
    ax.set_ylabel("Modo")

    # Annotate cells
    for mi in range(len(modes)):
        for ci in range(len(clients)):
            val = matrix[mi, ci]
            if not np.isnan(val):
                ax.text(ci, mi, f"{val:.0f}", ha="center", va="center",
                        color="white" if val > np.nanmax(matrix) * 0.6 else "black",
                        fontweight="bold")

    plt.colorbar(im, ax=ax, label="P95 (ms)")
    plt.tight_layout()
    path = os.path.join(plot_dir, f"heatmap_p95.{fmt}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_per_client_fairness(groups: Dict[str, List[Dict]], plot_dir: str, fmt: str):
    """Plot 6: Per-client mean latency for parallel scenarios (fairness)."""
    _setup_style()

    # Select parallel scenarios with multiple clients
    parallel_scenarios = [
        sc for sc in sorted(groups.keys())
        if "parallel" in sc and int(sc.split("_c")[1]) > 1
    ]

    if not parallel_scenarios:
        print("  Skipping per_client_fairness (no multi-client parallel scenarios)")
        return

    n_plots = len(parallel_scenarios)
    cols = min(3, n_plots)
    rows_count = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows_count, cols, figsize=(5 * cols, 4 * rows_count), squeeze=False)
    fig.suptitle("Fairness per Client — Parallel Mode (100% DDS)", fontsize=14, fontweight="bold")

    for idx, sc in enumerate(parallel_scenarios):
        ax = axes[idx // cols][idx % cols]

        per_client = compute_per_client_stats(groups[sc])
        if not per_client:
            continue

        client_ids = list(per_client.keys())
        means = [per_client[c]["mean_ms"] for c in client_ids]
        stddevs = [per_client[c]["stddev_ms"] for c in client_ids]

        x = range(len(client_ids))
        ax.bar(x, means, yerr=stddevs, color=COLORS["parallel"], alpha=0.7,
               capsize=3, edgecolor="white", linewidth=0.5)
        ax.set_title(sc, fontsize=10)
        ax.set_ylabel("Mean RTT (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("client_", "C") for c in client_ids],
                           rotation=45, fontsize=7)

    # Hide unused axes
    for idx in range(n_plots, rows_count * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    path = os.path.join(plot_dir, f"per_client_fairness.{fmt}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Análise e visualização dos resultados do benchmark DDS",
    )
    parser.add_argument("--input", type=str, default="results", help="Input directory (default: results/)")
    parser.add_argument("--output", type=str, default=None, help="Output plot directory (default: <input>/plots/)")
    parser.add_argument("--remove-outliers", action="store_true", help="Remove outliers via IQR method")
    parser.add_argument("--format", choices=["png", "pdf"], default="png", help="Plot format (default: png)")
    args = parser.parse_args()

    raw_dir = os.path.join(args.input, "raw")
    plot_dir = args.output or os.path.join(args.input, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print()
    print("=" * 65)
    print(" DDS Benchmark Analysis")
    print("=" * 65)
    print()

    # Load data
    rows = load_raw_csvs(raw_dir)
    groups = group_by_scenario(rows)
    print(f"Scenarios found: {len(groups)}")
    for sc, rows_sc in sorted(groups.items()):
        success = sum(1 for r in rows_sc if r["success"])
        print(f"  {sc}: {len(rows_sc)} rows ({success} successful)")
    print()

    # Aggregate and compute summary
    print("Computing statistics...")
    summary = aggregate_and_summarize(groups, args.input, args.remove_outliers)

    # Print summary table
    print()
    print(f"{'Scenario':<20} {'N':>5} {'Mean':>8} {'Std':>8} "
          f"{'P50':>8} {'P90':>8} {'P95':>8} {'P99':>8} {'CV':>6} {'Tput':>8}")
    print("-" * 105)
    for s in summary:
        print(
            f"{s['scenario']:<20} {s['count']:>5} {s['mean_ms']:>7.0f}ms {s['stddev_ms']:>7.0f}ms "
            f"{s['p50_ms']:>7.0f}ms {s['p90_ms']:>7.0f}ms {s['p95_ms']:>7.0f}ms "
            f"{s['p99_ms']:>7.0f}ms {s['cv']:>5.3f} {s['throughput_rps']:>7.2f}"
        )

    # Generate plots
    print()
    print(f"Generating plots in {plot_dir}/...")

    plot_latency_vs_clients(summary, plot_dir, args.format)
    plot_throughput_vs_clients(summary, plot_dir, args.format)
    plot_boxplot_by_mode(groups, plot_dir, args.format)
    plot_cdf_comparison(groups, plot_dir, args.format)
    plot_heatmap_p95(summary, plot_dir, args.format)
    plot_per_client_fairness(groups, plot_dir, args.format)

    print()
    print(f"Analysis complete!")
    print(f"  Summary:  {os.path.join(args.input, 'summary.csv')}")
    print(f"  Plots:    {plot_dir}/")


if __name__ == "__main__":
    main()
