#!/usr/bin/env python3
"""
Gera plots academicos de alta qualidade para os benchmarks E1-E5
================================================================
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Paths relativos ao script
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots"

# Estilo academico
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

# Cores profissionais (colorblind-friendly - sem red/green juntos)
COLORS = {
    'dds': '#0072B2',      # Azul
    'http': '#E69F00',     # Laranja
    'primary': '#1565C0',  # Azul escuro
    'secondary': '#6A1B9A', # Roxo
    'accent': '#009E73',   # Verde-azulado
    'warning': '#F57C00',  # Laranja
}


def _try_read_csv(filepath):
    """Tenta ler um CSV, retorna None se nao existir."""
    if filepath.exists():
        return pd.read_csv(filepath)
    return None


def plot_e1_latency_breakdown(results_dir, output_dir):
    """E1: Decomposicao de Latencia por Camada"""
    print("Gerando E1: Decomposicao de Latencia...")

    # Ler dados
    dds_short = _try_read_csv(results_dir / "E1_DDS_phi4-mini_short.csv")
    dds_long = _try_read_csv(results_dir / "E1_DDS_phi4-mini_long.csv")
    http_short = _try_read_csv(results_dir / "E1_HTTP_phi4-mini_short.csv")
    http_long = _try_read_csv(results_dir / "E1_HTTP_phi4-mini_long.csv")

    if any(df is None for df in [dds_short, dds_long, http_short, http_long]):
        print("  AVISO: Faltam CSVs de E1. Execute os benchmarks E1 antes de gerar plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Colunas de camadas - DDS usa nomes originais, HTTP usa nomes corrigidos
    # Tentar ambos os esquemas de nomes para compatibilidade
    dds_layers = ['T1_serialization_ms', 'T2_transport_send_ms', 'T3_queue_ms',
                  'T4_inference_ms', 'T5_transport_return_ms', 'T6_deserialization_ms']

    # HTTP pode ter nomes antigos ou novos
    http_layers_new = ['T1_serialization_ms', 'T2_http_overhead_ms', 'T3_queue_est_ms',
                       'T4_inference_ms', 'T5_transport_return_est_ms', 'T6_deserialization_ms']
    http_layers_old = ['T1_serialization_ms', 'T2_transport_send_ms', 'T3_queue_ms',
                       'T4_inference_ms', 'T5_transport_return_ms', 'T6_deserialization_ms']

    # Detectar qual esquema o CSV HTTP usa
    if 'T2_http_overhead_ms' in http_short.columns:
        http_layers = http_layers_new
    else:
        http_layers = http_layers_old

    layer_names = ['Serialization', 'Transp Send', 'Queue', 'Inference', 'Transp Return', 'Deserial']

    # Calcular totais
    dds_total_short = sum([dds_short[c].mean() for c in dds_layers])
    http_total_short = sum([http_short[c].mean() for c in http_layers])
    dds_total_long = sum([dds_long[c].mean() for c in dds_layers])
    http_total_long = sum([http_long[c].mean() for c in http_layers])

    # Plot 1: Comparacao total
    categories = ['Short Prompt', 'Long Prompt']
    dds_totals = [dds_total_short, dds_total_long]
    http_totals = [http_total_short, http_total_long]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, dds_totals, width, label='DDS', color=COLORS['dds'], alpha=0.8)
    bars2 = axes[0].bar(x + width/2, http_totals, width, label='HTTP', color=COLORS['http'], alpha=0.8)
    axes[0].set_ylabel('Latencia Total (ms)', fontsize=12)
    axes[0].set_title('Latencia Total: DDS vs HTTP', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=11)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Adicionar speedup
    for i, (d, h) in enumerate(zip(dds_totals, http_totals)):
        if d > 0:
            speedup = h / d
            axes[0].annotate(f'{speedup:.1f}x faster', xy=(i, max(d, h) * 0.9),
                            ha='center', fontsize=12, fontweight='bold', color=COLORS['dds'])

    # Plot 2: Breakdown por camada (apenas short prompt)
    dds_short_means = [dds_short[c].mean() for c in dds_layers]
    http_short_means = [http_short[c].mean() for c in http_layers]

    x2 = np.arange(len(layer_names))
    axes[1].bar(x2 - width/2, dds_short_means, width, label='DDS', color=COLORS['dds'], alpha=0.8)
    axes[1].bar(x2 + width/2, http_short_means, width, label='HTTP', color=COLORS['http'], alpha=0.8)
    axes[1].set_ylabel('Tempo (ms)', fontsize=12)
    axes[1].set_title('Decomposicao por Camada (Short Prompt)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(layer_names, rotation=30, ha='right', fontsize=9)
    axes[1].legend(fontsize=11)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "E1_latency_breakdown.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Print stats
    print(f"  DDS Short: {dds_total_short:.2f}ms | HTTP Short: {http_total_short:.2f}ms | Speedup: {http_total_short/dds_total_short:.1f}x" if dds_total_short > 0 else "  DDS Short: 0ms")
    print(f"  DDS Long: {dds_total_long:.2f}ms | HTTP Long: {http_total_long:.2f}ms | Speedup: {http_total_long/dds_total_long:.1f}x" if dds_total_long > 0 else "  DDS Long: 0ms")


def plot_e2_failure(results_dir, output_dir):
    """E2: Detecao de Falha - usa dados reais dos CSVs."""
    print("Gerando E2: Detecao de Falha...")

    protocols = ["DDS_DEADLINE", "GRPC_HEALTH", "HTTP_HEARTBEAT"]
    periodos = [1000, 5000, 10000]
    periodo_labels = ['1s', '5s', '10s']

    # Coletar dados reais de todos os protocolos e periodos
    all_data = {}
    has_any_data = False

    for protocol in protocols:
        means = []
        valid_periods = []
        for periodo, label in zip(periodos, periodo_labels):
            csv_path = results_dir / f"E2_{protocol}_kill9_{periodo}ms.csv"
            df = _try_read_csv(csv_path)
            if df is not None and 'detection_time_ms' in df.columns:
                # Filtrar valores invalidos (negativos = erro)
                valid = df[df['detection_time_ms'] > 0]['detection_time_ms']
                if len(valid) > 0:
                    means.append(valid.mean())
                    valid_periods.append(label)
                    has_any_data = True

        if means:
            all_data[protocol] = (valid_periods, means)

    if not has_any_data:
        print("  AVISO: Sem dados E2 validos. Execute os benchmarks E2 antes de gerar plots.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plotar cada protocolo que tem dados
    protocol_colors = {
        'DDS_DEADLINE': COLORS['primary'],
        'GRPC_HEALTH': COLORS['secondary'],
        'HTTP_HEARTBEAT': COLORS['accent']
    }

    if len(all_data) == 1:
        # Apenas um protocolo: barras simples
        protocol = list(all_data.keys())[0]
        labels, means = all_data[protocol]
        bars = ax.bar(labels, means, color=protocol_colors.get(protocol, COLORS['primary']),
                      alpha=0.8, width=0.6, label=protocol)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.02,
                    f'{v:.0f}ms', ha='center', fontsize=11, fontweight='bold')
    else:
        # Multiplos protocolos: barras agrupadas
        n_protocols = len(all_data)
        width = 0.8 / n_protocols
        x_base = np.arange(len(periodo_labels))

        for idx, (protocol, (labels, means)) in enumerate(all_data.items()):
            # Mapear labels para posicoes
            positions = [periodo_labels.index(l) for l in labels]
            x_pos = [x_base[p] + (idx - n_protocols/2 + 0.5) * width for p in positions]
            bars = ax.bar(x_pos, means, width, label=protocol,
                         color=protocol_colors.get(protocol, COLORS['primary']), alpha=0.8)

        ax.set_xticks(x_base)
        ax.set_xticklabels(periodo_labels)
        ax.legend()

    ax.set_xlabel('Periodo de Heartbeat/Deadline', fontsize=12)
    ax.set_ylabel('Tempo de Detecao (ms)', fontsize=12)
    ax.set_title('E2: Tempo de Detecao de Falha (kill -9)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "E2_failure_detection.png", dpi=150, bbox_inches='tight')
    plt.close()

    for protocol, (labels, means) in all_data.items():
        for label, mean in zip(labels, means):
            print(f"  {protocol} periodo {label}: {mean:.0f}ms")


def plot_e3_priority(results_dir, output_dir):
    """E3: Priorizacao de Mensagens"""
    print("Gerando E3: Priorizacao...")

    df = _try_read_csv(results_dir / "E3_PRIORITY_carga10.csv")
    if df is None:
        print("  AVISO: Sem dados E3. Execute os benchmarks E3 antes de gerar plots.")
        return

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

    axes[0].set_ylabel('Latencia (ms)')
    axes[0].set_title('Distribuicao de Latencia por Prioridade')
    axes[0].grid(True, alpha=0.3)

    # Scatter plot
    colors_scatter = [COLORS['dds'] if p == 'HIGH' else COLORS['http'] for p in df['priority']]
    axes[1].scatter(df['send_time_s'], df['latency_ms'], c=colors_scatter, alpha=0.6, s=30)
    axes[1].set_xlabel('Tempo (s)')
    axes[1].set_ylabel('Latencia (ms)')
    axes[1].set_title('Latencia ao Longo do Tempo')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['dds'], label='Alta Prioridade', alpha=0.7),
        Patch(facecolor=COLORS['http'], label='Normal', alpha=0.7)
    ]
    axes[1].legend(handles=legend_elements)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('E3: Priorizacao de Mensagens com DDS TRANSPORT_PRIORITY', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "E3_priority.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Normal: media={normal.mean():.2f}ms, mediana={normal.median():.2f}ms")
    print(f"  Alta: media={high.mean():.2f}ms, mediana={high.median():.2f}ms")


def plot_e4_scalability(results_dir, output_dir):
    """E4: Escalabilidade"""
    print("Gerando E4: Escalabilidade...")

    dfs = []
    clients = [1, 2, 4, 8]
    for c in clients:
        df = _try_read_csv(results_dir / f"E4_scalability_1ag_{c}cl.csv")
        if df is None:
            print(f"  AVISO: Sem dados E4 para {c} clientes.")
            return
        dfs.append(df)

    # Calcular metricas
    latencies_p50 = [df['latency_ms'].median() for df in dfs]
    latencies_p95 = [df['latency_ms'].quantile(0.95) for df in dfs]

    # Throughput: usar wall_clock_time se disponivel, senao calcular com nota
    throughputs = []
    throughput_note = ""
    for df in dfs:
        count = len(df[df['success'] == 1])
        if 'wall_clock_time_s' in df.columns:
            total_time = df['wall_clock_time_s'].max()
            throughputs.append(count / total_time if total_time > 0 else 0)
        elif 'timestamp_s' in df.columns:
            # Calcular a partir de timestamps
            total_time = df['timestamp_s'].max() - df['timestamp_s'].min()
            throughputs.append(count / total_time if total_time > 0 else 0)
        else:
            # Fallback: usar max latency (limitacao conhecida)
            total_time = df['latency_ms'].max() / 1000
            throughputs.append(count / total_time if total_time > 0 else 0)
            throughput_note = " (NOTA: throughput calculado via max_latency - pode ser impreciso)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Throughput
    axes[0].plot(clients, throughputs, 'o-', color=COLORS['dds'], linewidth=2.5, markersize=12)
    axes[0].fill_between(clients, throughputs, alpha=0.3, color=COLORS['dds'])
    axes[0].set_xlabel('Clientes Simultaneos')
    axes[0].set_ylabel('Throughput (req/s)')
    title_suffix = '*' if throughput_note else ''
    axes[0].set_title(f'Throughput vs Clientes{title_suffix}')
    axes[0].set_xticks(clients)
    axes[0].grid(True, alpha=0.3)

    for i, v in enumerate(throughputs):
        axes[0].annotate(f'{v:.0f}', xy=(clients[i], v + max(throughputs)*0.05), ha='center', fontsize=10, fontweight='bold')

    # Latencia
    axes[1].plot(clients, latencies_p50, 'o-', color=COLORS['http'], linewidth=2.5, markersize=12, label='p50')
    axes[1].plot(clients, latencies_p95, 's--', color=COLORS['warning'], linewidth=2, markersize=10, label='p95')
    axes[1].set_xlabel('Clientes Simultaneos')
    axes[1].set_ylabel('Latencia (ms)')
    axes[1].set_title('Latencia vs Clientes')
    axes[1].set_xticks(clients)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('E4: Escalabilidade Multi-Cliente com DDS', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "E4_scalability.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  1 cliente: {latencies_p50[0]:.1f}ms p50, {throughputs[0]:.0f} req/s")
    print(f"  8 clientes: {latencies_p50[3]:.1f}ms p50, {throughputs[3]:.0f} req/s")
    if throughput_note:
        print(f"  {throughput_note}")


def plot_e5_streaming(results_dir, output_dir):
    """E5: Streaming Token-a-Token"""
    print("Gerando E5: Streaming...")

    # Tentar nomes novos e antigos dos CSVs
    df_phi = _try_read_csv(results_dir / "E5_DDS_VIA_ORCH_streaming_phi4-mini.csv")
    if df_phi is None:
        df_phi = _try_read_csv(results_dir / "E5_DDS_streaming_phi4-mini.csv")
    if df_phi is None:
        df_phi = _try_read_csv(results_dir / "E5_streaming_phi4-mini.csv")

    df_qwen = _try_read_csv(results_dir / "E5_DDS_VIA_ORCH_streaming_qwen3.5-9b.csv")
    if df_qwen is None:
        df_qwen = _try_read_csv(results_dir / "E5_DDS_streaming_qwen3.5-9b.csv")
    if df_qwen is None:
        df_qwen = _try_read_csv(results_dir / "E5_streaming_qwen3.5-9b.csv")

    if df_phi is None and df_qwen is None:
        print("  AVISO: Sem dados E5. Execute os benchmarks E5 antes de gerar plots.")
        return

    available = [(name, df) for name, df in [("Phi-4-mini", df_phi), ("Qwen3.5-9B", df_qwen)] if df is not None]

    fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    plot_colors = [COLORS['primary'], COLORS['secondary']]
    for idx, (name, df) in enumerate(available):
        axes[idx].bar(range(len(df)), df['total_time_ms'], color=plot_colors[idx], alpha=0.8)
        axes[idx].set_xlabel('Iteracao')
        axes[idx].set_ylabel('Tempo Total (ms)')
        axes[idx].set_title(f'{name}\nMedia: {df["total_time_ms"].mean():.1f}ms')
        axes[idx].axhline(y=df['total_time_ms'].mean(), color='red', linestyle='--', label='Media')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.suptitle('E5: Latencia de Streaming por Modelo', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "E5_streaming.png", dpi=150, bbox_inches='tight')
    plt.close()

    for name, df in available:
        print(f"  {name}: {df['total_time_ms'].mean():.1f}ms media")


def plot_summary(results_dir, output_dir):
    """Plot resumao com todos os cenarios"""
    print("Gerando grafico resumo...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # E1 - Speedup
    dds_short = _try_read_csv(results_dir / "E1_DDS_phi4-mini_short.csv")
    http_short = _try_read_csv(results_dir / "E1_HTTP_phi4-mini_short.csv")

    dds_layers = ['T1_serialization_ms', 'T2_transport_send_ms', 'T3_queue_ms',
                  'T4_inference_ms', 'T5_transport_return_ms', 'T6_deserialization_ms']

    if dds_short is not None and http_short is not None:
        # Detectar esquema de colunas HTTP
        if 'T2_http_overhead_ms' in http_short.columns:
            http_layers = ['T1_serialization_ms', 'T2_http_overhead_ms', 'T3_queue_est_ms',
                          'T4_inference_ms', 'T5_transport_return_est_ms', 'T6_deserialization_ms']
        else:
            http_layers = dds_layers

        dds_total = sum([dds_short[c].mean() for c in dds_layers])
        http_total = sum([http_short[c].mean() for c in http_layers])

        axes[0, 0].bar(['DDS', 'HTTP'], [dds_total, http_total], color=[COLORS['dds'], COLORS['http']], alpha=0.8)
        axes[0, 0].set_title('E1: Latencia Total (Short)', fontweight='bold')
        axes[0, 0].set_ylabel('ms')
        if dds_total > 0:
            speedup = http_total / dds_total
            axes[0, 0].annotate(f'{speedup:.1f}x', xy=(0.5, max(dds_total, http_total)*0.9), fontsize=20, ha='center', fontweight='bold', color=COLORS['dds'])
    else:
        axes[0, 0].text(0.5, 0.5, 'Sem dados E1', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('E1: Latencia Total (Short)', fontweight='bold')
        dds_total = 0
        http_total = 0
        speedup = 0

    # E2 - Failure (usar dados reais, filtrar negativos)
    e2_means = []
    e2_labels = ['1s', '5s', '10s']
    e2_periodos = [1000, 5000, 10000]
    e2_has_data = False

    for periodo in e2_periodos:
        df_e2 = _try_read_csv(results_dir / f"E2_DDS_DEADLINE_kill9_{periodo}ms.csv")
        if df_e2 is not None and 'detection_time_ms' in df_e2.columns:
            valid = df_e2[df_e2['detection_time_ms'] > 0]['detection_time_ms']
            if len(valid) > 0:
                e2_means.append(valid.mean())
                e2_has_data = True
            else:
                e2_means.append(0)
        else:
            e2_means.append(0)

    if e2_has_data:
        axes[0, 1].bar(e2_labels, e2_means,
                       color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']], alpha=0.8)
        axes[0, 1].set_title('E2: Tempo de Detecao', fontweight='bold')
        axes[0, 1].set_ylabel('ms')
    else:
        axes[0, 1].text(0.5, 0.5, 'Sem dados E2', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('E2: Tempo de Detecao', fontweight='bold')

    e2_1s = e2_means[0] if e2_means else 0

    # E3 - Priority
    df_e3 = _try_read_csv(results_dir / "E3_PRIORITY_carga10.csv")
    if df_e3 is not None:
        normal = df_e3[df_e3['priority'] == 'NORMAL']['latency_ms'].mean()
        high = df_e3[df_e3['priority'] == 'HIGH']['latency_ms'].mean()
        axes[0, 2].bar(['Normal', 'Alta'], [normal, high], color=[COLORS['http'], COLORS['dds']], alpha=0.8)
    else:
        axes[0, 2].text(0.5, 0.5, 'Sem dados E3', ha='center', va='center', transform=axes[0, 2].transAxes)
        normal = 0
        high = 0
    axes[0, 2].set_title('E3: Latencia por Prioridade', fontweight='bold')
    axes[0, 2].set_ylabel('ms')

    # E4 - Scalability
    df_1 = _try_read_csv(results_dir / "E4_scalability_1ag_1cl.csv")
    df_8 = _try_read_csv(results_dir / "E4_scalability_1ag_8cl.csv")
    if df_1 is not None and df_8 is not None:
        axes[1, 0].bar(['1 cliente', '8 clientes'],
                       [df_1['latency_ms'].median(), df_8['latency_ms'].median()],
                       color=[COLORS['dds'], COLORS['warning']], alpha=0.8)
    else:
        axes[1, 0].text(0.5, 0.5, 'Sem dados E4', ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('E4: Latencia p50', fontweight='bold')
    axes[1, 0].set_ylabel('ms')

    # E5 - Streaming (tentar nomes novos e antigos)
    df_phi = _try_read_csv(results_dir / "E5_DDS_VIA_ORCH_streaming_phi4-mini.csv")
    if df_phi is None:
        df_phi = _try_read_csv(results_dir / "E5_DDS_streaming_phi4-mini.csv")
    if df_phi is None:
        df_phi = _try_read_csv(results_dir / "E5_streaming_phi4-mini.csv")

    df_qwen = _try_read_csv(results_dir / "E5_DDS_VIA_ORCH_streaming_qwen3.5-9b.csv")
    if df_qwen is None:
        df_qwen = _try_read_csv(results_dir / "E5_DDS_streaming_qwen3.5-9b.csv")
    if df_qwen is None:
        df_qwen = _try_read_csv(results_dir / "E5_streaming_qwen3.5-9b.csv")

    if df_phi is not None and df_qwen is not None:
        axes[1, 1].bar(['Phi-4-mini', 'Qwen3.5-9B'],
                       [df_phi['total_time_ms'].mean(), df_qwen['total_time_ms'].mean()],
                       color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
    elif df_phi is not None:
        axes[1, 1].bar(['Phi-4-mini'], [df_phi['total_time_ms'].mean()],
                       color=[COLORS['primary']], alpha=0.8)
    else:
        axes[1, 1].text(0.5, 0.5, 'Sem dados E5', ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('E5: Tempo Medio Streaming', fontweight='bold')
    axes[1, 1].set_ylabel('ms')

    # Resumo textual
    axes[1, 2].axis('off')
    summary_lines = [
        "RESUMO DOS BENCHMARKS",
        "=====================",
        "",
        "E1 - Latencia:",
        f"  DDS: {dds_total:.1f}ms" if dds_total > 0 else "  DDS: sem dados",
        f"  HTTP: {http_total:.1f}ms" if http_total > 0 else "  HTTP: sem dados",
        f"  Speedup: {speedup:.1f}x" if speedup > 0 else "",
        "",
        "E2 - Detecao de Falha:",
        f"  Periodo 1s: {e2_1s:.0f}ms" if e2_1s > 0 else "  sem dados",
        "",
        "E3 - Prioridade:",
        f"  Normal: {normal:.2f}ms" if normal > 0 else "  sem dados",
        f"  Alta: {high:.2f}ms" if high > 0 else "",
        "",
        "E4 - Escalabilidade:",
        f"  1 cliente: {df_1['latency_ms'].median():.1f}ms" if df_1 is not None else "  sem dados",
        f"  8 clientes: {df_8['latency_ms'].median():.1f}ms" if df_8 is not None else "",
        "",
        "E5 - Streaming:",
        f"  Phi-4-mini: {df_phi['total_time_ms'].mean():.1f}ms" if df_phi is not None else "  sem dados",
        f"  Qwen: {df_qwen['total_time_ms'].mean():.1f}ms" if df_qwen is not None else "",
    ]
    summary_text = "\n".join(summary_lines)

    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Resumo: Benchmarks E1-E5 - DDS-LLM-Orchestrator', fontweight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(output_dir / "E_summary.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Gera plots academicos para benchmarks E1-E5")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Diretorio com resultados CSV (default: ../results relativo ao script)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Diretorio de saida para plots (default: ../plots relativo ao script)")
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir else RESULTS_DIR
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("GERANDO PLOTS ACADEMICOS - BENCHMARKS E1-E5")
    print("=" * 70)
    print(f"\nDados: {results_dir}")
    print(f"Saida: {output_dir}\n")

    plot_e1_latency_breakdown(results_dir, output_dir)
    plot_e2_failure(results_dir, output_dir)
    plot_e3_priority(results_dir, output_dir)
    plot_e4_scalability(results_dir, output_dir)
    plot_e5_streaming(results_dir, output_dir)
    plot_summary(results_dir, output_dir)

    print(f"\n{'='*70}")
    print(f"PLOTS GERADOS COM SUCESSO!")
    print(f"Local: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
