import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data for client 1 (client_000) and client 2 (client_001)
df1 = pd.read_csv('benchmark_results/gpu_20260224_165051/raw/sync_c5_client_000.csv')
df5 = pd.read_csv('benchmark_results/gpu_20260224_165051/raw/sync_c5_client_001.csv')

# Filter successful requests
df1_ok = df1[df1['success'] == True]
df5_ok = df5[df5['success'] == True]

print(f"Client 1 (client_000): {len(df1_ok)} successful requests")
print(f"Client 2 (client_001): {len(df5_ok)} successful requests")
print()

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. CDF comparison
ax1 = axes[0, 0]
for df, label, color in [(df1_ok, 'Client 1', 'blue'), (df5_ok, 'Client 5', 'red')]:
    latencies = np.sort(df['rtt_ms'].values)
    cdf = np.arange(1, len(latencies)+1) / len(latencies)
    ax1.plot(latencies, cdf, label=label, linewidth=2, color=color)
ax1.set_xlabel('Latency (ms)')
ax1.set_ylabel('CDF')
ax1.set_title('CDF Comparison: Client 1 vs Client 5')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1000)

# 2. Latency over time
ax2 = axes[0, 1]
ax2.plot(df1_ok['request_num'].values, df1_ok['rtt_ms'].values, 'b-', label='Client 1', alpha=0.7)
ax2.plot(df5_ok['request_num'].values, df5_ok['rtt_ms'].values, 'r-', label='Client 5', alpha=0.7)
ax2.set_xlabel('Request ID')
ax2.set_ylabel('Latency (ms)')
ax2.set_title('Latency Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Boxplot
ax3 = axes[1, 0]
data = [df1_ok['rtt_ms'].values, df5_ok['rtt_ms'].values]
bp = ax3.boxplot(data, labels=['Client 1', 'Client 5'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax3.set_ylabel('Latency (ms)')
ax3.set_title('Latency Distribution')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Statistics
ax4 = axes[1, 1]
stats = {
    'Client 1': [df1_ok['rtt_ms'].mean(), df1_ok['rtt_ms'].median(),
                 df1_ok['rtt_ms'].quantile(0.95), df1_ok['rtt_ms'].std()],
    'Client 5': [df5_ok['rtt_ms'].mean(), df5_ok['rtt_ms'].median(),
                 df5_ok['rtt_ms'].quantile(0.95), df5_ok['rtt_ms'].std()]
}
x = np.arange(4)
width = 0.35
bars1 = ax4.bar(x - width/2, stats['Client 1'], width, label='Client 1', color='blue', alpha=0.7)
bars2 = ax4.bar(x + width/2, stats['Client 5'], width, label='Client 5', color='red', alpha=0.7)
ax4.set_ylabel('Latency (ms)')
ax4.set_title('Statistics Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(['Mean', 'P50', 'P95', 'Std'])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('benchmark_results/gpu_20260224_165051/plots/client_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: client_comparison.png")

# Print statistics
print()
print("Client 1 Statistics:")
print(f"  Mean: {df1_ok['rtt_ms'].mean():.1f}ms")
print(f"  P50:  {df1_ok['rtt_ms'].median():.1f}ms")
print(f"  P95:  {df1_ok['rtt_ms'].quantile(0.95):.1f}ms")
print(f"  P99:  {df1_ok['rtt_ms'].quantile(0.99):.1f}ms")
print(f"  Std:  {df1_ok['rtt_ms'].std():.1f}ms")
print()
print("Client 5 Statistics:")
print(f"  Mean: {df5_ok['rtt_ms'].mean():.1f}ms")
print(f"  P50:  {df5_ok['rtt_ms'].median():.1f}ms")
print(f"  P95:  {df5_ok['rtt_ms'].quantile(0.95):.1f}ms")
print(f"  P99:  {df5_ok['rtt_ms'].quantile(0.99):.1f}ms")
print(f"  Std:  {df5_ok['rtt_ms'].std():.1f}ms")
