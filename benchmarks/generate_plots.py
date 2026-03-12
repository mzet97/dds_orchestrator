#!/usr/bin/env python3
"""
Gera plots dos resultados dos benchmarks E1-E5
=============================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configuração
RESULTS_DIR = Path("E:/TI/git/tese/dds_orchestrator/benchmarks/results")
OUTPUT_DIR = Path("E:/TI/git/tese/dds_orchestrator/benchmarks/plots")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def plot_e1_latency():
    """E1: Decomposição de Latência - DDS vs HTTP"""
    print("Gerando E1: Decomposição de Latência...")

    # Ler dados
    dds_short = pd.read_csv(RESULTS_DIR / "E1_DDS_phi4-mini_short.csv")
    dds_long = pd.read_csv(RESULTS_DIR / "E1_DDS_phi4-mini_long.csv")
    http_short = pd.read_csv(RESULTS_DIR / "E1_HTTP_phi4-mini_short.csv")
    http_long = pd.read_csv(RESULTS_DIR / "E1_HTTP_phi4-mini_long.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = ['DDS', 'HTTP']
    short_means = [dds_short['T_total_ms'].mean(), http_short['T_total_ms'].mean()]
    long_means = [dds_long['T_total_ms'].mean(), http_long['T_total_ms'].mean()]

    x = range(len(labels))
    width = 0.35

    axes[0].bar([i - width/2 for i in x], short_means, width, label='Prompt Curto', color=['#2ecc71', '#e74c3c'])
    axes[0].set_ylabel('Latência Total (ms)')
    axes[0].set_title('Latência - Prompt Curto (12 chars)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()

    for i, v in enumerate(short_means):
        axes[0].text(i, v + 0.1, f'{v:.2f}ms', ha='center', fontsize=10)

    axes[1].bar([i - width/2 for i in x], long_means, width, label='Prompt Longo', color=['#2ecc71', '#e74c3c'])
    axes[1].set_ylabel('Latência Total (ms)')
    axes[1].set_title('Latência - Prompt Longo (137 chars)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()

    for i, v in enumerate(long_means):
        axes[1].text(i, v + 0.1, f'{v:.2f}ms', ha='center', fontsize=10)

    plt.suptitle('E1: Decomposição de Latência - DDS vs HTTP', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "E1_latency_dds_vs_http.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Imprimir comparação
    print(f"  DDS short: {short_means[0]:.2f}ms | HTTP short: {short_means[1]:.2f}ms | Speedup: {short_means[1]/short_means[0]:.1f}x")
    print(f"  DDS long: {long_means[0]:.2f}ms | HTTP long: {long_means[1]:.2f}ms | Speedup: {long_means[1]/long_means[0]:.1f}x")


def plot_e2_failure():
    """E2: Detecção de Falha"""
    print("Gerando E2: Detecção de Falha...")

    df_1000 = pd.read_csv(RESULTS_DIR / "E2_DDS_DEADLINE_kill9_1000ms.csv")
    df_5000 = pd.read_csv(RESULTS_DIR / "E2_DDS_DEADLINE_kill9_5000ms.csv")
    df_10000 = pd.read_csv(RESULTS_DIR / "E2_DDS_DEADLINE_kill9_10000ms.csv")

    fig, ax = plt.subplots(figsize=(10, 6))

    periods = ['1000ms', '5000ms', '10000ms']
    detection_times = [
        df_1000['detection_time_ms'].mean(),
        df_5000['detection_time_ms'].mean(),
        df_10000['detection_time_ms'].mean()
    ]

    colors = ['#3498db', '#e74c3c', '#f39c12']
    bars = ax.bar(periods, [abs(x) for x in detection_times], color=colors)

    ax.set_xlabel('Período DEADLINE')
    ax.set_ylabel('Tempo de Detecção (ms)')
    ax.set_title('E2: Tempo de Detecção de Falha (kill -9)')

    for bar, val in zip(bars, detection_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{abs(val):.0f}ms', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "E2_failure_detection.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_e3_priority():
    """E3: Priorização"""
    print("Gerando E3: Priorização...")

    df = pd.read_csv(RESULTS_DIR / "E3_PRIORITY_carga10.csv")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot de latência por prioridade (priority é NORMAL ou HIGH)
    colors = ['#e74c3c' if p == 'HIGH' else '#3498db' for p in df['priority']]
    ax.scatter(df['send_time_s'], df['latency_ms'], c=colors, alpha=0.6, s=30)

    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Latência (ms)')
    ax.set_title('E3: Latência por Prioridade de Mensagem')

    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Alta Prioridade'),
        Patch(facecolor='#3498db', label='Baixa Prioridade')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "E3_priority.png", dpi=150, bbox_inches='tight')
    plt.close()

    normal_latency = df[df['priority'] == 'NORMAL']['latency_ms'].mean()
    high_latency = df[df['priority'] == 'HIGH']['latency_ms'].mean()
    print(f"  Latência normal: {normal_latency:.2f}ms")
    print(f"  Latência alta prioridade: {high_latency:.2f}ms")


def plot_e4_scalability():
    """E4: Escalabilidade"""
    print("Gerando E4: Escalabilidade...")

    df_1 = pd.read_csv(RESULTS_DIR / "E4_scalability_1ag_1cl.csv")
    df_2 = pd.read_csv(RESULTS_DIR / "E4_scalability_1ag_2cl.csv")
    df_4 = pd.read_csv(RESULTS_DIR / "E4_scalability_1ag_4cl.csv")
    df_8 = pd.read_csv(RESULTS_DIR / "E4_scalability_1ag_8cl.csv")

    clients = [1, 2, 4, 8]
    throughputs = [
        df_1['latency_ms'].count() / (df_1['latency_ms'].max() / 1000) if len(df_1) > 0 else 0,
        df_2['latency_ms'].count() / (df_2['latency_ms'].max() / 1000) if len(df_2) > 0 else 0,
        df_4['latency_ms'].count() / (df_4['latency_ms'].max() / 1000) if len(df_4) > 0 else 0,
        df_8['latency_ms'].count() / (df_8['latency_ms'].max() / 1000) if len(df_8) > 0 else 0,
    ]

    latencies_p50 = [
        df_1['latency_ms'].median(),
        df_2['latency_ms'].median(),
        df_4['latency_ms'].median(),
        df_8['latency_ms'].median(),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Throughput
    axes[0].plot(clients, throughputs, 'o-', color='#2ecc71', linewidth=2, markersize=10)
    axes[0].set_xlabel('Clientes Simultâneos')
    axes[0].set_ylabel('Throughput (req/s)')
    axes[0].set_title('E4: Throughput vs Clientes')
    axes[0].set_xticks(clients)

    for i, v in enumerate(throughputs):
        axes[0].text(clients[i], v + 50, f'{v:.0f}', ha='center', fontsize=10)

    # Latência
    axes[1].plot(clients, latencies_p50, 'o-', color='#e74c3c', linewidth=2, markersize=10)
    axes[1].set_xlabel('Clientes Simultâneos')
    axes[1].set_ylabel('Latência p50 (ms)')
    axes[1].set_title('E4: Latência p50 vs Clientes')
    axes[1].set_xticks(clients)

    for i, v in enumerate(latencies_p50):
        axes[1].text(clients[i], v + 1, f'{v:.1f}ms', ha='center', fontsize=10)

    plt.suptitle('E4: Escalabilidade Multi-Cliente', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "E4_scalability.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_e5_streaming():
    """E5: Streaming"""
    print("Gerando E5: Streaming...")

    df_phi = pd.read_csv(RESULTS_DIR / "E5_streaming_phi4-mini.csv")
    df_qwen = pd.read_csv(RESULTS_DIR / "E5_streaming_qwen3.5-9b.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Phi-4-mini
    axes[0].bar(['TTFT', 'ITL (mean)'], [df_phi['ttft_ms'].mean(), df_phi['itl_mean_ms'].mean()],
                color=['#3498db', '#2ecc71'])
    axes[0].set_ylabel('Tempo (ms)')
    axes[0].set_title('Phi-4-mini Streaming')
    max_val = max(df_phi['ttft_ms'].mean(), df_phi['itl_mean_ms'].mean())
    if max_val > 0:
        axes[0].set_ylim(0, max_val * 1.3)

    # Qwen3.5-9B
    axes[1].bar(['TTFT', 'ITL (mean)'], [df_qwen['ttft_ms'].mean(), df_qwen['itl_mean_ms'].mean()],
                color=['#3498db', '#2ecc71'])
    axes[1].set_ylabel('Tempo (ms)')
    axes[1].set_title('Qwen3.5-9B Streaming')
    max_val = max(df_qwen['ttft_ms'].mean(), df_qwen['itl_mean_ms'].mean())
    if max_val > 0:
        axes[1].set_ylim(0, max_val * 1.3)

    plt.suptitle('E5: Streaming Token-a-Token', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "E5_streaming.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Phi-4-mini: TTFT={df_phi['ttft_ms'].mean():.2f}ms, ITL={df_phi['itl_mean_ms'].mean():.2f}ms")
    print(f"  Qwen3.5-9B: TTFT={df_qwen['ttft_ms'].mean():.2f}ms, ITL={df_qwen['itl_mean_ms'].mean():.2f}ms")


def main():
    print("=" * 60)
    print("GERANDO PLOTS DOS BENCHMARKS E1-E5")
    print("=" * 60)
    print(f"\nResultados de: {RESULTS_DIR}")
    print(f"Saída em: {OUTPUT_DIR}\n")

    # Gerar plots
    plot_e1_latency()
    plot_e2_failure()
    plot_e3_priority()
    plot_e4_scalability()
    plot_e5_streaming()

    print(f"\n{'-'*60}")
    print(f"Plots salvos em: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
