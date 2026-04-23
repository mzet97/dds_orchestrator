#!/usr/bin/env python3
"""
Generate comparison plots for HTTP vs gRPC vs DDS benchmarks.
Uses the final n=100 results from client VM (.63).
"""

import json
import statistics
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# RX 6600M + Phi-4-mini (n=1000) — use for single-GPU comparison
FILES_RX6600M = {
    "HTTP": "e2e_http_20260315_133259.json",
    "gRPC": "e2e_grpc_20260315_142734.json",
    "DDS":  "e2e_dds_20260315_152325.json",
}
# RTX 3080 + Phi-4-mini (n=1000) — use for GPU comparison
FILES_RTX3080 = {
    "HTTP": "e2e_http_20260315_210700.json",
    "gRPC": "e2e_grpc_20260315_212821.json",
    "DDS":  "e2e_dds_20260315_215109.json",
}
# RTX 3080 final results (n=1000, Phi-4-mini, all fixes)
FILES_RTX3080_FINAL = {
    "HTTP": "e2e_http_20260316_141112.json",
    "gRPC": "e2e_grpc_20260316_143311.json",
    "DDS":  "e2e_dds_20260316_145502.json",
}
FILES = FILES_RTX3080_FINAL

COLORS = {"HTTP": "#2196F3", "gRPC": "#FF9800", "DDS": "#4CAF50"}
HATCHES = {"HTTP": "", "gRPC": "//", "DDS": ".."}


def load_data():
    data = {}
    for label, fname in FILES.items():
        path = RESULTS_DIR / fname
        if path.exists():
            data[label] = json.load(open(path))
            print(f"  Loaded {label}: {fname}")
        else:
            print(f"  WARNING: {fname} not found!")
    return data


def get_latencies(protocol_data, scenario, prompt_type):
    """Extract successful roundtrip latencies."""
    items = protocol_data.get(scenario, {}).get(prompt_type, [])
    return [r["roundtrip_ms"] for r in items if r.get("success")]


def percentile(data, p):
    if not data:
        return 0
    s = sorted(data)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


# ─── Plot 1: E1 Latency Bar Chart ───────────────────────────────────────────

def plot_e1_latency_bars(data):
    """Bar chart comparing p50 latency across protocols for short and long prompts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, prompt_type, title in [(axes[0], "short", "Short Prompt (~20 tokens)"),
                                     (axes[1], "long", "Long Prompt (~500 tokens)")]:
        protocols = []
        p50s = []
        means = []
        stds = []

        for label in ["HTTP", "gRPC", "DDS"]:
            if label not in data:
                continue
            lats = get_latencies(data[label], "E1", prompt_type)
            if lats:
                protocols.append(label)
                p50s.append(percentile(lats, 50))
                means.append(statistics.mean(lats))
                stds.append(statistics.stdev(lats) if len(lats) > 1 else 0)

        x = np.arange(len(protocols))
        bars = ax.bar(x, p50s, 0.5, color=[COLORS[p] for p in protocols],
                      edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, p50s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    f'{val:.0f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(protocols, fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle('E1: End-to-End Latency (p50) — HTTP vs gRPC vs DDS',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'E1_latency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: E1_latency_comparison.png")


# ─── Plot 2: E1 Latency Box Plot ────────────────────────────────────────────

def plot_e1_boxplot(data):
    """Box plot showing latency distribution for each protocol."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, prompt_type, title in [(axes[0], "short", "Short Prompt"),
                                     (axes[1], "long", "Long Prompt")]:
        all_lats = []
        labels = []
        colors = []
        for label in ["HTTP", "gRPC", "DDS"]:
            if label not in data:
                continue
            lats = get_latencies(data[label], "E1", prompt_type)
            if lats:
                all_lats.append(lats)
                labels.append(label)
                colors.append(COLORS[label])

        bp = ax.boxplot(all_lats, labels=labels, patch_artist=True,
                        widths=0.5, showfliers=True,
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle('E1: Latency Distribution — HTTP vs gRPC vs DDS',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'E1_latency_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: E1_latency_boxplot.png")


# ─── Plot 3: E1 CDF ─────────────────────────────────────────────────────────

def plot_e1_cdf(data):
    """CDF of latency for short and long prompts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, prompt_type, title in [(axes[0], "short", "Short Prompt"),
                                     (axes[1], "long", "Long Prompt")]:
        for label in ["HTTP", "gRPC", "DDS"]:
            if label not in data:
                continue
            lats = sorted(get_latencies(data[label], "E1", prompt_type))
            if lats:
                cdf = np.arange(1, len(lats) + 1) / len(lats)
                ax.plot(lats, cdf, label=label, color=COLORS[label], linewidth=2)

        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('CDF', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle('E1: Cumulative Distribution of Latency',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'E1_latency_cdf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: E1_latency_cdf.png")


# ─── Plot 4: E4 Scalability ─────────────────────────────────────────────────

def plot_e4_scalability(data):
    """Throughput and p50 latency vs number of clients."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for label in ["HTTP", "gRPC", "DDS"]:
        if label not in data:
            continue
        e4 = data[label].get("E4", {})
        clients = []
        throughputs = []
        p50s = []

        for key in sorted(e4.keys()):
            if key.startswith("clients_"):
                n = int(key.split("_")[1])
                clients.append(n)
                throughputs.append(e4[key].get("throughput_rps", 0))
                p50s.append(e4[key].get("p50", 0))

        if clients:
            axes[0].plot(clients, throughputs, 'o-', label=label,
                        color=COLORS[label], linewidth=2, markersize=8)
            axes[1].plot(clients, p50s, 's-', label=label,
                        color=COLORS[label], linewidth=2, markersize=8)

    axes[0].set_xlabel('Number of Clients', fontsize=12)
    axes[0].set_ylabel('Throughput (req/s)', fontsize=12)
    axes[0].set_title('Throughput vs Clients', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    axes[0].set_axisbelow(True)

    axes[1].set_xlabel('Number of Clients', fontsize=12)
    axes[1].set_ylabel('Latency p50 (ms)', fontsize=12)
    axes[1].set_title('Latency p50 vs Clients', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].set_axisbelow(True)

    fig.suptitle('E4: Scalability — Throughput and Latency',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'E4_scalability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: E4_scalability.png")


# ─── Plot 5: E5 Streaming TTFT ──────────────────────────────────────────────

def plot_e5_streaming(data):
    """TTFT comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    protocols = []
    ttfts = []
    for label in ["HTTP", "gRPC", "DDS"]:
        if label not in data:
            continue
        e5 = data[label].get("E5", [])
        if e5:
            vals = [r.get("ttft_ms", 0) for r in e5 if r.get("ttft_ms", 0) > 0]
            if vals:
                protocols.append(label)
                ttfts.append(statistics.mean(vals))

    if protocols:
        x = np.arange(len(protocols))
        bars = ax.bar(x, ttfts, 0.5, color=[COLORS[p] for p in protocols],
                      edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, ttfts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f'{val:.0f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(protocols, fontsize=12)
        ax.set_ylabel('TTFT (ms)', fontsize=12)
        ax.set_title('E5: Time-to-First-Token (TTFT)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'E5_ttft_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: E5_ttft_comparison.png")


# ─── Plot 6: E2 Failure Detection ────────────────────────────────────────────

def plot_e2_failure_detection():
    """Bar chart of failure detection times for DDS DEADLINE, gRPC Health, HTTP Heartbeat."""
    import json

    fig, ax = plt.subplots(figsize=(8, 5))

    protocols = []
    means = []
    colors_list = []

    for fname, label, color in [
        ("E2_DDS_kill9_1000ms_lease200ms_summary.json", "DDS\nDEADLINE", COLORS["DDS"]),
        ("E2_gRPC_HEALTH_kill9_1000ms_summary.json", "gRPC\nHealth Check", COLORS["gRPC"]),
        ("E2_HTTP_HEARTBEAT_kill9_1000ms_summary.json", "HTTP\nHeartbeat", COLORS["HTTP"]),
    ]:
        path = RESULTS_DIR / fname
        if path.exists():
            data = json.load(open(path))
            mean = data.get("detection_mean_ms", -1)
            if mean > 0:
                protocols.append(label)
                means.append(mean)
                colors_list.append(color)

    if protocols:
        x = np.arange(len(protocols))
        bars = ax.bar(x, means, 0.5, color=colors_list, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f'{val:.1f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(protocols, fontsize=11)
        ax.set_ylabel('Detection Time (ms)', fontsize=12)
        ax.set_title('E2: Failure Detection Time (kill -9)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'E2_failure_detection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: E2_failure_detection.png")


# ─── Plot 7: E3 Priority ────────────────────────────────────────────────────

def plot_e3_priority(data):
    """Bar chart comparing normal vs high priority latency across protocols."""
    fig, ax = plt.subplots(figsize=(10, 5))

    protocols = []
    normal_means = []
    high_means = []

    for label in ["HTTP", "gRPC", "DDS"]:
        if label not in data:
            continue
        e3 = data[label].get("E3", {})
        normal = [r["roundtrip_ms"] for r in e3.get("normal", []) if r.get("success")]
        high = [r["roundtrip_ms"] for r in e3.get("high", []) if r.get("success")]
        if normal and high:
            protocols.append(label)
            normal_means.append(statistics.mean(normal))
            high_means.append(statistics.mean(high))

    if protocols:
        x = np.arange(len(protocols))
        w = 0.3
        bars1 = ax.bar(x - w/2, normal_means, w, label='Normal Priority',
                        color=[COLORS[p] for p in protocols], alpha=0.6, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + w/2, high_means, w, label='High Priority',
                        color=[COLORS[p] for p in protocols], edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars1, normal_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, high_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(protocols, fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('E3: Normal vs High Priority Latency', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'E3_priority_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: E3_priority_comparison.png")


# ─── Plot 8: Summary Table ──────────────────────────────────────────────────

def plot_summary_table(data):
    """Summary table as image."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    headers = ['Metric', 'HTTP', 'gRPC', 'DDS']
    rows = []

    # E1 short p50
    row = ['E1 Short p50']
    for label in ["HTTP", "gRPC", "DDS"]:
        lats = get_latencies(data.get(label, {}), "E1", "short")
        row.append(f'{percentile(lats, 50):.0f}ms' if lats else 'N/A')
    rows.append(row)

    # E1 long p50
    row = ['E1 Long p50']
    for label in ["HTTP", "gRPC", "DDS"]:
        lats = get_latencies(data.get(label, {}), "E1", "long")
        row.append(f'{percentile(lats, 50):.0f}ms' if lats else 'N/A')
    rows.append(row)

    # E1 success rate
    row = ['E1 Success Rate']
    for label in ["HTTP", "gRPC", "DDS"]:
        s = data.get(label, {}).get("E1", {}).get("short", [])
        ok = sum(1 for r in s if r.get("success"))
        row.append(f'{ok}/{len(s)}' if s else 'N/A')
    rows.append(row)

    # E4 throughput (1 client)
    row = ['E4 Throughput (1 client)']
    for label in ["HTTP", "gRPC", "DDS"]:
        e4 = data.get(label, {}).get("E4", {})
        v = e4.get("clients_1", {}).get("throughput_rps", 0)
        row.append(f'{v:.1f} req/s' if v else 'N/A')
    rows.append(row)

    # E5 TTFT
    row = ['E5 TTFT (mean)']
    for label in ["HTTP", "gRPC", "DDS"]:
        e5 = data.get(label, {}).get("E5", [])
        vals = [r.get("ttft_ms", 0) for r in e5 if r.get("ttft_ms", 0) > 0]
        row.append(f'{statistics.mean(vals):.0f}ms' if vals else 'N/A')
    rows.append(row)

    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#E0E0E0'] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Color protocol columns
    for i in range(len(rows)):
        for j, label in enumerate(["HTTP", "gRPC", "DDS"]):
            table[i + 1, j + 1].set_facecolor(COLORS[label] + '30')  # 30 = alpha hex

    ax.set_title('Summary: HTTP vs gRPC vs DDS (n=100, Phi-4-mini, AMD RX 6600M)',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: summary_table.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    data = load_data()

    if not data:
        print("No data loaded!")
        return

    print("\nGenerating plots...")
    plot_e1_latency_bars(data)
    plot_e1_boxplot(data)
    plot_e1_cdf(data)
    plot_e2_failure_detection()
    plot_e3_priority(data)
    plot_e4_scalability(data)
    plot_e5_streaming(data)
    plot_summary_table(data)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
