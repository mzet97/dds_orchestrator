#!/usr/bin/env python3
"""
E1: Decomposição de Latência por Camada - DDS
==============================================
Mede T1-T6 através do orquestrador com transporte DDS interno.

Arquitetura medida:
    Cliente → HTTP → Orquestrador → DDS (UDP) → Agente → llama-server

T1: Serialização do request (JSON)
T2: Transporte de envio (HTTP cliente→orquestrador + DDS orquestrador→agente)
T3: Fila no agente (estimada: ~0.1ms localhost)
T4: Inferência LLM (T_total - T1 - T2 - T3 - T5 - T6)
T5: Transporte de retorno (DDS agente→orquestrador + HTTP orquestrador→cliente; embutido em T3+T4)
T6: Deserialização da response

Nota: T5 não é mensurável separadamente no cliente sem instrumentação do agente.
Idêntico ao E1_HTTP, mas o orquestrador usa DDS internamente para roteamento.

Usage:
    python E1_decompose_latency_dds.py --url http://localhost:8080 --model qwen3.5-0.8b --prompt short --n 100
"""

import argparse
import asyncio
import json
import sys
import time
import statistics
from pathlib import Path
from typing import Dict, List
import aiohttp


class LatencyDecomposerDDS:
    """Instrumenta cada camada da requisição via orquestrador DDS."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    async def measure_full_request(self, model: str, prompt: str) -> Dict[str, float]:
        """
        Mede latência em 6 pontos via orquestrador com transporte DDS:
        T1: Serialização do request
        T2: Envio (HTTP cliente→orquestrador; DDS orquestrador→agente)
        T3: Fila no agente (estimada)
        T4: Inferência LLM
        T5: Transporte de retorno (embutido em T3+T4; DDS usa UDP, menor overhead que TCP)
        T6: Deserialização da response
        """

        # T1: Serialização
        t1_start = time.perf_counter_ns()
        request_data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50
        }
        serialized = json.dumps(request_data)
        t1_end = time.perf_counter_ns()
        T1 = (t1_end - t1_start) / 1e6  # ms

        # T_rtt: round-trip completo (serialização + envio + fila + inferência + retorno + deserialização)
        t_rtt_start = time.perf_counter_ns()
        response_text = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                data=serialized,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                response_text = await resp.text()
        t_rtt_end = time.perf_counter_ns()
        T_rtt = (t_rtt_end - t_rtt_start) / 1e6  # ms

        # T6: Deserialização
        t6_start = time.perf_counter_ns()
        if response_text:
            try:
                response_json = json.loads(response_text)
            except Exception:
                response_json = {}
        else:
            response_json = {}
        t6_end = time.perf_counter_ns()
        T6 = (t6_end - t6_start) / 1e6  # ms

        # T4: Inferência real (reportada pelo servidor no campo processing_time_ms)
        T4 = response_json.get("processing_time_ms", 0)

        # T_transport: overhead de transporte = round-trip - inferência
        T_transport = max(T_rtt - T4, 0)

        T_total = T1 + T_rtt + T6  # Total wall-clock incluindo serialização

        return {
            "T1_serialization_ms": T1,
            "T2_transport_send_ms": T_transport / 2,  # estimativa: metade ida, metade volta
            "T3_queue_ms": 0,  # incluído em T_transport
            "T4_inference_ms": T4,
            "T5_transport_return_ms": T_transport / 2,
            "T6_deserialization_ms": T6,
            "T_total_ms": T_rtt,  # round-trip é o total real
            "transport_overhead_pct": (T_transport / T_rtt * 100) if T_rtt > 0 else 0
        }


async def run_benchmark(args):
    """Executa o benchmark completo."""

    decomposer = LatencyDecomposerDDS(base_url=args.url)

    # Prompts conforme especificação da dissertação
    prompts = {
        "short": "O que e 2+2?",
        "long": "Explique detalhadamente a teoria da relatividade geral de Albert Einstein, incluindo suas implicacoes para a fisica moderna e cosmologia."
    }

    prompt = prompts.get(args.prompt_type, prompts["short"])

    results = []

    print(f"E1: Decomposição de Latência - DDS (via orquestrador)")
    print(f"URL: {args.url}")
    print(f"Modelo: {args.model}")
    print(f"Prompt: {args.prompt_type} ({len(prompt)} chars)")
    print(f"Iterações: {args.n}")
    print("-" * 50)

    for i in range(args.n):
        try:
            result = await decomposer.measure_full_request(args.model, prompt)
            result["iteration"] = i + 1
            results.append(result)
        except Exception as e:
            print(f"Iteração {i+1}/{args.n}: ERRO - {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"Iteração {i+1}/{args.n}: T_total={result['T_total_ms']:.1f}ms, T4={result['T4_inference_ms']:.1f}ms")

    if not results:
        print("Nenhum resultado coletado.")
        return None

    # Estatísticas
    metrics = ["T1_serialization_ms", "T2_transport_send_ms", "T3_queue_ms",
               "T4_inference_ms", "T5_transport_return_ms", "T6_deserialization_ms",
               "T_total_ms", "transport_overhead_pct"]

    protocol_label = args.protocol_label
    summary = {"protocol": protocol_label, "model": args.model, "prompt_type": args.prompt_type, "n": len(results)}

    for m in metrics:
        values = [r[m] for r in results if m in r]
        if values:
            summary[f"{m}_mean"] = round(statistics.mean(values), 4)
            summary[f"{m}_median"] = round(statistics.median(values), 4)
            summary[f"{m}_stdev"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0
            summary[f"{m}_p95"] = round(sorted(values)[int(len(values) * 0.95)], 4)
            summary[f"{m}_p99"] = round(sorted(values)[int(len(values) * 0.99)], 4)
            summary[f"{m}_min"] = round(min(values), 4)
            summary[f"{m}_max"] = round(max(values), 4)

    # Salvar CSV
    csv_file = f"results/E1_{protocol_label}_{args.model}_{args.prompt_type}.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write(",".join(["iteration"] + metrics) + "\n")
        for r in results:
            f.write(",".join([str(r.get(m, 0)) for m in ["iteration"] + metrics]) + "\n")

    # Salvar JSON summary
    json_file = f"results/E1_{protocol_label}_{args.model}_{args.prompt_type}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"T_total médio:          {summary['T_total_ms_mean']:.2f} ms")
    print(f"T4 (inferência) médio:  {summary['T4_inference_ms_mean']:.2f} ms")
    print(f"Overhead transporte:    {summary['transport_overhead_pct_mean']:.2f}%")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E1: Decomposição de Latência - DDS")
    parser.add_argument("--url", default="http://localhost:8080", help="URL do orquestrador com DDS habilitado")
    parser.add_argument("--model", default="qwen3.5-0.8b", help="Modelo a usar")
    parser.add_argument("--prompt", dest="prompt_type", choices=["short", "long"], default="short",
                        help="Tipo de prompt")
    parser.add_argument("--n", type=int, default=1000, help="Número de iterações")
    parser.add_argument("--protocol-label", default="DDS", help="Label do protocolo para nomes de arquivos (DDS, gRPC, HTTP)")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
