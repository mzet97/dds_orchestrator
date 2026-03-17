#!/usr/bin/env python3
"""Generate comparison plots for E1-E5 benchmarks across HTTP, gRPC, DDS."""

import json
import statistics
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ─── Load Data ───────────────────────────────────────────────────────────────

RESULTS_FILE = "results/e2e_combined_20260317_101052.json"
PLOTS_DIR = Path("plots/e2e_comparison")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_FILE) as f:
    data = json.load(f)

PROTOCOLS = ["http", "grpc", "dds"]
LABELS = {"http": "HTTP", "grpc": "gRPC", "dds": "DDS"}
COLORS = {"http": "#2196F3", "grpc": "#FF9800", "dds": "#4CAF50"}

def get_latencies(proto, scenario, subkey=None):
    """Extract latencies from results."""
    r = data.get(proto, {}).get(scenario, {})
    if subkey:
        r = r.get(subkey, [])
    if isinstance(r, list):
        return sorted([x["roundtrip_ms"] for x in r if x.get("success")])
    return []


# ─── E1: Latency Bar Chart ──────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, ptype in enumerate(["short", "long"]):
    ax = axes[idx]
    p50s, p95s, means = [], [], []
    proto_labels = []

    for proto in PROTOCOLS:
        lats = get_latencies(proto, "E1", ptype)
        if lats:
            p50s.append(lats[len(lats)//2])
            p95s.append(lats[int(len(lats)*0.95)])
            means.append(statistics.mean(lats))
            proto_labels.append(LABELS[proto])

    x = np.arange(len(proto_labels))
    width = 0.25

    bars1 = ax.bar(x - width, p50s, width, label='p50', color=[COLORS[p] for p in PROTOCOLS], alpha=0.9)
    bars2 = ax.bar(x, p95s, width, label='p95', color=[COLORS[p] for p in PROTOCOLS], alpha=0.5)
    bars3 = ax.bar(x + width, means, width, label='mean', color=[COLORS[p] for p in PROTOCOLS], alpha=0.3, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'E1: {ptype.capitalize()} Prompt Latency')
    ax.set_xticks(x)
    ax.set_xticklabels(proto_labels)
    ax.legend()

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "E1_latency_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  E1 latency comparison: OK")


# ─── E1: Boxplot ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, ptype in enumerate(["short", "long"]):
    ax = axes[idx]
    all_data = []
    labels = []

    for proto in PROTOCOLS:
        lats = get_latencies(proto, "E1", ptype)
        if lats:
            all_data.append(lats)
            labels.append(LABELS[proto])

    bp = ax.boxplot(all_data, labels=labels, patch_artist=True, showfliers=True)
    for patch, proto in zip(bp['boxes'], PROTOCOLS):
        patch.set_facecolor(COLORS[proto])
        patch.set_alpha(0.7)

    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'E1: {ptype.capitalize()} Prompt Distribution')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "E1_latency_boxplot.png", dpi=150, bbox_inches='tight')
plt.close()
print("  E1 boxplot: OK")


# ─── E1: CDF ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, ptype in enumerate(["short", "long"]):
    ax = axes[idx]

    for proto in PROTOCOLS:
        lats = get_latencies(proto, "E1", ptype)
        if lats:
            y = np.arange(1, len(lats)+1) / len(lats)
            ax.plot(lats, y, label=LABELS[proto], color=COLORS[proto], linewidth=2)

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_title(f'E1: {ptype.capitalize()} Prompt CDF')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "E1_latency_cdf.png", dpi=150, bbox_inches='tight')
plt.close()
print("  E1 CDF: OK")


# ─── E2: Reliability ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

success_rates = []
proto_labels = []
for proto in PROTOCOLS:
    e1_short = data.get(proto, {}).get("E1", {}).get("short", [])
    e1_long = data.get(proto, {}).get("E1", {}).get("long", [])
    all_reqs = e1_short + e1_long
    if all_reqs:
        success = sum(1 for x in all_reqs if x.get("success"))
        rate = success / len(all_reqs) * 100
        success_rates.append(rate)
        proto_labels.append(LABELS[proto])

bars = ax.bar(proto_labels, success_rates, color=[COLORS[p] for p in PROTOCOLS], alpha=0.8)
ax.set_ylabel('Success Rate (%)')
ax.set_title('E2: Reliability (Success Rate)')
ax.set_ylim(0, 105)
ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)

for bar, rate in zip(bars, success_rates):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "E2_reliability.png", dpi=150, bbox_inches='tight')
plt.close()
print("  E2 reliability: OK")


# ─── E3: Priority ────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(PROTOCOLS))
width = 0.35

normal_p50s = []
high_p50s = []

for proto in PROTOCOLS:
    e3 = data.get(proto, {}).get("E3", {})
    normal = sorted([x["roundtrip_ms"] for x in e3.get("normal", []) if x.get("success")])
    high = sorted([x["roundtrip_ms"] for x in e3.get("high", []) if x.get("success")])
    normal_p50s.append(normal[len(normal)//2] if normal else 0)
    high_p50s.append(high[len(high)//2] if high else 0)

bars1 = ax.bar(x - width/2, normal_p50s, width, label='Normal Priority', alpha=0.8)
bars2 = ax.bar(x + width/2, high_p50s, width, label='High Priority', alpha=0.8)

ax.set_ylabel('Latency p50 (ms)')
ax.set_title('E3: Priority Comparison')
ax.set_xticks(x)
ax.set_xticklabels([LABELS[p] for p in PROTOCOLS])
ax.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "E3_priority.png", dpi=150, bbox_inches='tight')
plt.close()
print("  E3 priority: OK")


# ─── E4: Scalability ─────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

client_counts = [1, 2, 4]

# Throughput
for proto in PROTOCOLS:
    e4 = data.get(proto, {}).get("E4", {})
    throughputs = []
    for nc in client_counts:
        key = f"clients_{nc}"
        throughputs.append(e4.get(key, {}).get("throughput_rps", 0))
    ax1.plot(client_counts, throughputs, 'o-', label=LABELS[proto],
             color=COLORS[proto], linewidth=2, markersize=8)

ax1.set_xlabel('Concurrent Clients')
ax1.set_ylabel('Throughput (req/s)')
ax1.set_title('E4: Throughput Scalability')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xticks(client_counts)

# p50 latency
for proto in PROTOCOLS:
    e4 = data.get(proto, {}).get("E4", {})
    p50s = []
    for nc in client_counts:
        key = f"clients_{nc}"
        p50s.append(e4.get(key, {}).get("p50", 0))
    ax2.plot(client_counts, p50s, 'o-', label=LABELS[proto],
             color=COLORS[proto], linewidth=2, markersize=8)

ax2.set_xlabel('Concurrent Clients')
ax2.set_ylabel('Latency p50 (ms)')
ax2.set_title('E4: Latency vs Concurrency')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xticks(client_counts)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "E4_scalability.png", dpi=150, bbox_inches='tight')
plt.close()
print("  E4 scalability: OK")


# ─── E5: Streaming ───────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

ttft_p50s = []
proto_labels = []

for proto in PROTOCOLS:
    e5 = data.get(proto, {}).get("E5", [])
    ttfts = sorted([x["ttft_ms"] for x in e5 if x.get("ttft_ms", 0) > 0])
    if ttfts:
        ttft_p50s.append(ttfts[len(ttfts)//2])
        proto_labels.append(LABELS[proto])

bars = ax.bar(proto_labels, ttft_p50s, color=[COLORS[p] for p in PROTOCOLS], alpha=0.8)
ax.set_ylabel('TTFT p50 (ms)')
ax.set_title('E5: Time to First Token (Streaming)')

for bar, val in zip(bars, ttft_p50s):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            f'{val:.0f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "E5_streaming_ttft.png", dpi=150, bbox_inches='tight')
plt.close()
print("  E5 streaming TTFT: OK")


# ─── Summary Table ───────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

headers = ['Metric', 'HTTP', 'gRPC', 'DDS']
rows = []

# E1
for ptype in ['short', 'long']:
    row = [f'E1 {ptype} p50']
    for proto in PROTOCOLS:
        lats = get_latencies(proto, "E1", ptype)
        row.append(f'{lats[len(lats)//2]:.1f}ms' if lats else 'N/A')
    rows.append(row)

# E2 success rate
row = ['E2 Success Rate']
for proto in PROTOCOLS:
    e1 = data.get(proto, {}).get("E1", {})
    all_reqs = e1.get("short", []) + e1.get("long", [])
    success = sum(1 for x in all_reqs if x.get("success"))
    row.append(f'{success}/{len(all_reqs)} ({success/max(1,len(all_reqs))*100:.0f}%)')
rows.append(row)

# E3
row = ['E3 Normal p50']
for proto in PROTOCOLS:
    e3 = data.get(proto, {}).get("E3", {})
    lats = sorted([x["roundtrip_ms"] for x in e3.get("normal", []) if x.get("success")])
    row.append(f'{lats[len(lats)//2]:.1f}ms' if lats else 'N/A')
rows.append(row)

# E4
row = ['E4 Throughput (1 client)']
for proto in PROTOCOLS:
    e4 = data.get(proto, {}).get("E4", {})
    rps = e4.get("clients_1", {}).get("throughput_rps", 0)
    row.append(f'{rps:.1f} rps')
rows.append(row)

# E5
row = ['E5 TTFT p50']
for proto in PROTOCOLS:
    e5 = data.get(proto, {}).get("E5", [])
    ttfts = sorted([x["ttft_ms"] for x in e5 if x.get("ttft_ms", 0) > 0])
    row.append(f'{ttfts[len(ttfts)//2]:.0f}ms' if ttfts else 'N/A')
rows.append(row)

table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style header
for j in range(len(headers)):
    table[0, j].set_facecolor('#333333')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Alternate row colors
for i in range(len(rows)):
    color = '#f0f0f0' if i % 2 == 0 else 'white'
    for j in range(len(headers)):
        table[i+1, j].set_facecolor(color)

ax.set_title('E1-E5 Benchmark Summary: HTTP vs gRPC vs DDS', fontsize=14, fontweight='bold', pad=20)

plt.savefig(PLOTS_DIR / "summary_table.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Summary table: OK")


print(f"\nAll plots saved to {PLOTS_DIR}/")
