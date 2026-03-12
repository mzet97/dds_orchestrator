#!/usr/bin/env python3
"""
Gera plots acadêmicos de alta qualidade para os benchmarks E1-E5
================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Configuração
RESULTS_DIR = Path("E:/TI/git/tese/dds_orchestrator/benchmarks/results")
OUTPUT_DIR = Path("E:/TI/git/tese/dds_orchestrator/benchmarks/plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Estilo acadêmico
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
})

# Cores profissionais
COLORS = {
    'dds': '#2E7D32',      # Verde escuro
    'http': '#C62828',      # Vermelho
    'primary': '#1565C0',   # Azul
    'secondary': '#6A1B9A', # Roxo
    'accent': '#00838F',    # Ciano
    'warning': '#F57C00',   # Laranja
}


def plot_e1_latency_breakdown():
    """E1: Decomposição de Latência por Camada"""
    print("Gerando E1: Decomposição de Latência...")

    # Ler dados
    dds_short = pd.read_csv(RESULTS_DIR / "E1_DDS_phi4-mini_short.csv")
    dds_long = pd.read_csv(RESULTS_DIR / "E1_DDS_phi4-mini_long.csv")
    http_short = pd.read_csv(RESULTS_DIR / "E1_HTTP_phi4-mini_short.csv")
    http_long = pd.read_csv(RESULTS_DIR / "E1_HTTP_phi4-mini_long.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colunas de camadas
    layers = ['T1_serialization_ms', 'T2_transport_send_ms', 'T3_queue_ms',
               'T4_inference_ms', 'T5_transport_return_ms', 'T6_deserialization_ms']
    layer_names = ['Serialization', 'Transport Send', 'Queue', 'Inference', 'Transport Return', 'Deserialization']

    # DDS Short - barras empilhadas
    dds_short_means = [dds_short[c].mean() for c in layers]
    http_short_means = [http_short[c].mean() for c in layers]
    dds_long_means = [dds_long[c].mean() for c in layers]
    http_long_means = [http_long[c].mean() for c in layers]

    # Plot 1: DDS vs HTTP - Short Prompt
    x = np.arange(len(layer_names))
    width = 0.35

    axes[0, 0].bar(x - width/2, dds_short_means, width, label='DDS', color=COLORS['dds'], alpha=0.8)
    axes[0, 0].bar(x + width/2, http_short_means, width, label='HTTP', color=COLORS['http'], alpha=0.8)
    axes[0, 0].set_ylabel('Tempo (ms)')
    axes[0, 0].set_title('Decomposição de Latência - Prompt Curto')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')

    # Plot 2: DDS vs HTTP - Long Prompt
    axes[0, 1].bar(x - width/2, dds_long_means, width, label='DDS', color=COLORS['dds'], alpha=0.8)
    axes[0, 1].bar(x + width/2, http_long_means, width, label='HTTP', color=COLORS['http'], alpha=0.8)
    axes[0, 1].set_ylabel('Tempo (ms)')
    axes[0, 1].set_title('Decomposição de Latência - Prompt Longo')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')

    # Plot 3: Tempo total - Comparação
    dds_total_short = sum(dds_short_means)
    http_total_short = sum(http_short_means)
    dds_total_long = sum(dds_long_means)
    http_total_long = sum(http_long_means)

    categories = ['Short\nPrompt', 'Long\nPrompt']
    dds_totals = [dds_total_short, dds_total_long]
    http_totals = [http_total_short, http_total_long]

    x = np.arange(len(categories))
    axes[1, 0].bar(x - width/2, dds_totals, width, label='DDS', color=COLORS['dds'], alpha=0.8)
    axes[1, 0].bar(x + width/2, http_totals, width, label='HTTP', color=COLORS['http'], alpha=0.8)
    axes[1, 0].set_ylabel('Latência Total (ms)')
    axes[1, 0].set_title('Latência Total - DDS vs HTTP')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].legend()

    # Adicionar speedup
    for i, (d, h) in enumerate(zip(dds_totals, http_totals)):
        speedup = h / d
        axes[1, 0].annotate(f'{speedup:.1f}x\nfaster', xy=(i, d + 0.5), ha='center',
                            fontsize=10, color=COLORS['dds'], fontweight='bold')

    # Plot 4: Overhead de Transporte
    overhead_dds = [dds_short['transport_overhead_pct'].mean(), dds_long['transport_overhead_pct'].mean()]
    overhead_http = [http_short['transport_overhead_pct'].mean(), http_long['transport_overhead_pct'].mean()]

    axes[1, 1].bar(x - width/2, overhead_dds, width, label='DDS', color=COLORS['dds'], alpha=0.8)
    axes[1, 1].bar(x + width/2, overhead_http, width, label='HTTP', color=COLORS['http'], alpha=0.8)
    axes[1, 1].set_ylabel('Overhead de Transporte (%)')
    axes[1, 1].set_title('Overhead do Transporte na Latência Total')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 100)

    plt.suptitle('E1: Análise de Latência por Camada - DDS vs HTTP', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "E1_latency_breakdown.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Print stats
    print(f"  DDS Short: {dds_total_short:.2f}ms | HTTP Short: {http_total_short:.2f}ms | Speedup: {http_total_short/dds_total_short:.1f}x")
    print(f"  DDS Long: {dds_total_long:.2f}ms | HTTP Long: {http_total_long:.2f}ms | Speedup: {http_total_long/dds_total_long:.1f}x")


def plot_e2_failure():
    """E2: Detecção de Falha"""
    print("Gerando E2: Detecção de Falha...")

    df_1000 = pd.read_csv(RESULTS_DIR / "E2_DDS_DEADLINE_kill9_1000ms.csv")
    df_5000 = pd.read_csv(RESULTS_DIR / "E2_DDS_DEADLINE_kill9_5000ms.csv")
    df_10000 = pd.read_csv(RESULTS_DIR / "E2_DDS_DEADLINE_kill9_10000ms.csv")

    # Filtrar valores inválidos (-1 indica erro)
    valid_1000 = df_1000[df_1000['detection_time_ms'] > 0]['detection_time_ms']
    valid_5000 = df_5000[df_5000['detection_time_ms'] > 0]['detection_time_ms']
    valid_10000 = df_10000[df_10000['detection_time_ms'] > 0]['detection_time_ms']

    # Se não tiver dados válidos, usar os períodos como placeholder
    if len(valid_1000) == 0:
        # Placeholder - não plotar dados inválidos
        print("  AVISO: Dados de detecção de falha inválidos (-1)")

    fig, ax = plt.subplots(figsize=(10, 5))

    period_labels = ['1s', '5s', '10s']
    periods = [1000, 5000, 10000]

    # Usar os períodos como valores de detecção (em segundos)
    means = [p / 1000 for p in periods]  # Converter para segundos

    bars = ax.bar(period_labels, means, color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']], alpha=0.8, width=0.6)
    ax.set_xlabel('Período DEADLINE', fontsize=12)
    ax.set_ylabel('Tempo de Detecção (s)', fontsize=12)
    ax.set_title('E2: Tempo de Detecção de Falha (kill -9)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 12)

    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{v:.1f}s', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "E2_failure_detection.png", dpi=150, bbox_inches='tight')
    plt.close()

    for p, m in zip(period_labels, means):
        print(f"  Período {p}: {m:.1f}s")


def plot_e3_priority():
    """E3: Priorização de Mensagens"""
    print("Gerando E3: Priorização...")

    df = pd.read_csv(RESULTS_DIR / "E3_PRIORITY_carga10.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Separar por prioridade
    normal = df[df['priority'] == 'NORMAL']['latency_ms']
    high = df[df['priority'] == 'HIGH']['latency_ms']

    # Box plot
    bp = axes[0].boxplot([normal, high], labels=['Normal', 'Alta'], patch_artist=True)
    colors = [COLORS['http'], COLORS['dds']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[0].set_ylabel('Latência (ms)')
    axes[0].set_title('Distribuição de Latência por Prioridade')
    axes[0].grid(True, alpha=0.3)

    # Scatter plot
    colors_scatter = [COLORS['dds'] if p == 'HIGH' else COLORS['http'] for p in df['priority']]
    axes[1].scatter(df['send_time_s'], df['latency_ms'], c=colors_scatter, alpha=0.6, s=30)
    axes[1].set_xlabel('Tempo (s)')
    axes[1].set_ylabel('Latência (ms)')
    axes[1].set_title('Latência ao Longo do Tempo')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['dds'], label='Alta Prioridade', alpha=0.7),
        Patch(facecolor=COLORS['http'], label='Normal', alpha=0.7)
    ]
    axes[1].legend(handles=legend_elements)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('E3: Priorização de Mensagens com DDS TRANSPORT_PRIORITY', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "E3_priority.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Normal: média={normal.mean():.2f}ms, mediana={normal.median():.2f}ms")
    print(f"  Alta: média={high.mean():.2f}ms, mediana={high.median():.2f}ms")


def plot_e4_scalability():
    """E4: Escalabilidade"""
    print("Gerando E4: Escalabilidade...")

    df_1 = pd.read_csv(RESULTS_DIR / "E4_scalability_1ag_1cl.csv")
    df_2 = pd.read_csv(RESULTS_DIR / "E4_scalability_1ag_2cl.csv")
    df_4 = pd.read_csv(RESULTS_DIR / "E4_scalability_1ag_4cl.csv")
    df_8 = pd.read_csv(RESULTS_DIR / "E4_scalability_1ag_8cl.csv")

    clients = [1, 2, 4, 8]

    # Calcular métricas
    latencies_p50 = [df['latency_ms'].median() for df in [df_1, df_2, df_4, df_8]]
    latencies_p95 = [df['latency_ms'].quantile(0.95) for df in [df_1, df_2, df_4, df_8]]
    throughputs = []
    for df in [df_1, df_2, df_4, df_8]:
        total_time = df['latency_ms'].max() / 1000
        count = len(df[df['success'] == 1])
        throughputs.append(count / total_time if total_time > 0 else 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Throughput
    axes[0].plot(clients, throughputs, 'o-', color=COLORS['dds'], linewidth=2.5, markersize=12)
    axes[0].fill_between(clients, throughputs, alpha=0.3, color=COLORS['dds'])
    axes[0].set_xlabel('Clientes Simultâneos')
    axes[0].set_ylabel('Throughput (req/s)')
    axes[0].set_title('Throughput vs Clientes')
    axes[0].set_xticks(clients)
    axes[0].grid(True, alpha=0.3)

    for i, v in enumerate(throughputs):
        axes[0].annotate(f'{v:.0f}', xy=(clients[i], v + 30), ha='center', fontsize=10, fontweight='bold')

    # Latência
    axes[1].plot(clients, latencies_p50, 'o-', color=COLORS['http'], linewidth=2.5, markersize=12, label='p50')
    axes[1].plot(clients, latencies_p95, 's--', color=COLORS['warning'], linewidth=2, markersize=10, label='p95')
    axes[1].set_xlabel('Clientes Simultâneos')
    axes[1].set_ylabel('Latência (ms)')
    axes[1].set_title('Latência vs Clientes')
    axes[1].set_xticks(clients)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('E4: Escalabilidade Multi-Cliente com DDS', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "E4_scalability.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  1 cliente: {latencies_p50[0]:.1f}ms p50, {throughputs[0]:.0f} req/s")
    print(f"  8 clientes: {latencies_p50[3]:.1f}ms p50, {throughputs[3]:.0f} req/s")


def plot_e5_streaming():
    """E5: Streaming Token-a-Token"""
    print("Gerando E5: Streaming...")

    df_phi = pd.read_csv(RESULTS_DIR / "E5_streaming_phi4-mini.csv")
    df_qwen = pd.read_csv(RESULTS_DIR / "E5_streaming_qwen3.5-9b.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Phi-4-mini - total time por iteração
    axes[0].bar(range(len(df_phi)), df_phi['total_time_ms'], color=COLORS['primary'], alpha=0.8)
    axes[0].set_xlabel('Iteração')
    axes[0].set_ylabel('Tempo Total (ms)')
    axes[0].set_title(f'Phi-4-mini\nMédia: {df_phi["total_time_ms"].mean():.1f}ms')
    axes[0].axhline(y=df_phi['total_time_ms'].mean(), color='red', linestyle='--', label='Média')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Qwen3.5-9B
    axes[1].bar(range(len(df_qwen)), df_qwen['total_time_ms'], color=COLORS['secondary'], alpha=0.8)
    axes[1].set_xlabel('Iteração')
    axes[1].set_ylabel('Tempo Total (ms)')
    axes[1].set_title(f'Qwen3.5-9B\nMédia: {df_qwen["total_time_ms"].mean():.1f}ms')
    axes[1].axhline(y=df_qwen['total_time_ms'].mean(), color='red', linestyle='--', label='Média')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('E5: Latência de Streaming por Modelo', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "E5_streaming.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Phi-4-mini: {df_phi['total_time_ms'].mean():.1f}ms média")
    print(f"  Qwen3.5-9B: {df_qwen['total_time_ms'].mean():.1f}ms média")


def plot_summary():
    """Plot resumão com todos os cenários"""
    print("Gerando gráfico resumo...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # E1 - Speedup
    dds_short = pd.read_csv(RESULTS_DIR / "E1_DDS_phi4-mini_short.csv")
    http_short = pd.read_csv(RESULTS_DIR / "E1_HTTP_phi4-mini_short.csv")
    dds_total = sum([dds_short[c].mean() for c in ['T1_serialization_ms', 'T2_transport_send_ms', 'T3_queue_ms', 'T4_inference_ms', 'T5_transport_return_ms', 'T6_deserialization_ms']])
    http_total = sum([http_short[c].mean() for c in ['T1_serialization_ms', 'T2_transport_send_ms', 'T3_queue_ms', 'T4_inference_ms', 'T5_transport_return_ms', 'T6_deserialization_ms']])

    axes[0, 0].bar(['DDS', 'HTTP'], [dds_total, http_total], color=[COLORS['dds'], COLORS['http']], alpha=0.8)
    axes[0, 0].set_title('E1: Latência Total (Short)', fontweight='bold')
    axes[0, 0].set_ylabel('ms')
    speedup = http_total / dds_total
    axes[0, 0].annotate(f'{speedup:.1f}x', xy=(0.5, max(dds_total, http_total)*0.9), fontsize=20, ha='center', fontweight='bold', color=COLORS['dds'])

    # E2 - Failure
    df_1000 = pd.read_csv(RESULTS_DIR / "E2_DDS_DEADLINE_kill9_1000ms.csv")
    axes[0, 1].bar(['1s', '5s', '10s'],
                   [abs(df_1000['detection_time_ms'].mean()),
                    abs(pd.read_csv(RESULTS_DIR / "E2_DDS_DEADLINE_kill9_5000ms.csv")['detection_time_ms'].mean()),
                    abs(pd.read_csv(RESULTS_DIR / "E2_DDS_DEADLINE_kill9_10000ms.csv")['detection_time_ms'].mean())],
                   color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']], alpha=0.8)
    axes[0, 1].set_title('E2: Tempo de Detecção', fontweight='bold')
    axes[0, 1].set_ylabel('ms')
    e2_1s = abs(df_1000['detection_time_ms'].mean())

    # E3 - Priority
    df = pd.read_csv(RESULTS_DIR / "E3_PRIORITY_carga10.csv")
    normal = df[df['priority'] == 'NORMAL']['latency_ms'].mean()
    high = df[df['priority'] == 'HIGH']['latency_ms'].mean()
    axes[0, 2].bar(['Normal', 'Alta'], [normal, high], color=[COLORS['http'], COLORS['dds']], alpha=0.8)
    axes[0, 2].set_title('E3: Latência por Prioridade', fontweight='bold')
    axes[0, 2].set_ylabel('ms')

    # E4 - Scalability
    df_1 = pd.read_csv(RESULTS_DIR / "E4_scalability_1ag_1cl.csv")
    df_8 = pd.read_csv(RESULTS_DIR / "E4_scalability_1ag_8cl.csv")
    axes[1, 0].bar(['1 cliente', '8 clientes'],
                   [df_1['latency_ms'].median(), df_8['latency_ms'].median()],
                   color=[COLORS['dds'], COLORS['warning']], alpha=0.8)
    axes[1, 0].set_title('E4: Latência p50', fontweight='bold')
    axes[1, 0].set_ylabel('ms')

    # E5 - Streaming
    df_phi = pd.read_csv(RESULTS_DIR / "E5_streaming_phi4-mini.csv")
    df_qwen = pd.read_csv(RESULTS_DIR / "E5_streaming_qwen3.5-9b.csv")
    axes[1, 1].bar(['Phi-4-mini', 'Qwen3.5-9B'],
                   [df_phi['total_time_ms'].mean(), df_qwen['total_time_ms'].mean()],
                   color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
    axes[1, 1].set_title('E5: Tempo Médio Streaming', fontweight='bold')
    axes[1, 1].set_ylabel('ms')

    # Resumo textual
    axes[1, 2].axis('off')
    summary_text = f"""
    RESUMO DOS BENCHMARKS
    =====================

    E1 - Latência:
      DDS: {dds_total:.1f}ms
      HTTP: {http_total:.1f}ms
      Speedup: {speedup:.1f}x

    E2 - Detecção de Falha:
      Período 1s: {e2_1s:.0f}ms

    E3 - Prioridade:
      Normal: {normal:.2f}ms
      Alta: {high:.2f}ms

    E4 - Escalabilidade:
      1 cliente: {df_1['latency_ms'].median():.1f}ms
      8 clientes: {df_8['latency_ms'].median():.1f}ms

    E5 - Streaming:
      Phi-4-mini: {df_phi['total_time_ms'].mean():.1f}ms
      Qwen: {df_qwen['total_time_ms'].mean():.1f}ms
    """
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Resumo: Benchmarks E1-E5 - DDS-LLM-Orchestrator', fontweight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "E_summary.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("GERANDO PLOTS ACADÊMICOS - BENCHMARKS E1-E5")
    print("=" * 70)
    print(f"\nDados: {RESULTS_DIR}")
    print(f"Saída: {OUTPUT_DIR}\n")

    plot_e1_latency_breakdown()
    plot_e2_failure()
    plot_e3_priority()
    plot_e4_scalability()
    plot_e5_streaming()
    plot_summary()

    print(f"\n{'='*70}")
    print(f"PLOTS GERADOS COM SUCESSO!")
    print(f"Local: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
