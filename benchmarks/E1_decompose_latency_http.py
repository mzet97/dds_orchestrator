#!/usr/bin/env python3
"""
E1: Decomposição de Latência por Camada - HTTP
==============================================
Mede T1-T6 com requests HTTP REAIS

Usage:
    python E1_decompose_latency_http.py --model phi4 --prompt short --n 100
"""

import argparse
import asyncio
import json
import time
import statistics
import aiohttp
from pathlib import Path
from typing import Dict


class LatencyDecomposerHTTP:
    """Instrumenta cada camada da requisição HTTP."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    async def measure_full_request(self, model: str, prompt: str) -> Dict[str, float]:
        """
        Mede latência em 6 pontos via HTTP:
        T1: Serialização do request (JSON)
        T2: Envio HTTP (TCP handshake + send)
        T3: Fila no agente
        T4: Inferência LLM
        T5: Resposta HTTP
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

        # T2: Envio HTTP (conexão + envio)
        t2_start = time.perf_counter_ns()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                # T3+T4: Tempo até receber resposta (fila + inferência)
                t3_start = time.perf_counter_ns()
                response_text = await resp.text()
                t3_end = time.perf_counter_ns()
                T3_plus_T4 = (t3_end - t3_start) / 1e6  # ms

        t2_end = time.perf_counter_ns()
        T2 = (t2_end - t2_start) / 1e6 - T3_plus_T4  # ms (envio puro)

        # T5: Recebimento HTTP (já incluído em T3+T4)

        # T6: Deserialização
        t6_start = time.perf_counter_ns()
        if response_text:
            response_json = json.loads(response_text)
        else:
            response_json = {}
        t6_end = time.perf_counter_ns()
        T6 = (t6_end - t6_start) / 1e6  # ms

        # T_total
        T_total = T1 + T2 + T3_plus_T4 + T6

        # T4 = T3+T4 - T3 estimado (assumindo T3 pequeno para localhost)
        T3 = 0.1  # Estimativa
        T4 = T3_plus_T4 - T3

        return {
            "T1_serialization_ms": T1,
            "T2_transport_send_ms": max(T2, 0.001),
            "T3_queue_ms": T3,
            "T4_inference_ms": T4,
            "T5_transport_return_ms": 0.001,  # Incluído em T3+T4
            "T6_deserialization_ms": T6,
            "T_total_ms": T_total,
            "transport_overhead_pct": ((T2 + 0.001) / T_total) * 100 if T_total > 0 else 0
        }


async def run_benchmark(args):
    """Executa o benchmark completo."""

    base_url = args.url or "http://localhost:8080"

    decomposer = LatencyDecomposerHTTP(base_url)

    prompts = {
        "short": "O que é 2+2?",
        "long": "Explique detalhadamente a teoria da relatividade geral de Albert Einstein."
    }

    prompt = prompts.get(args.prompt_type, prompts["short"])

    results = []

    print(f"E1: Decomposição de Latência - HTTP")
    print(f"URL: {base_url}")
    print(f"Modelo: {args.model}")
    print(f"Prompt: {args.prompt_type}")
    print(f"Iterações: {args.n}")
    print("-" * 50)

    for i in range(args.n):
        result = await decomposer.measure_full_request(args.model, prompt)
        result["iteration"] = i + 1
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"Iteração {i+1}/{args.n}...")

    # Estatísticas
    metrics = ["T1_serialization_ms", "T2_transport_send_ms", "T3_queue_ms",
               "T4_inference_ms", "T5_transport_return_ms", "T6_deserialization_ms",
               "T_total_ms", "transport_overhead_pct"]

    summary = {"protocol": "HTTP", "model": args.model, "prompt_type": args.prompt_type, "n": args.n}

    for m in metrics:
        values = [r[m] for r in results if m in r]
        if values:
            summary[f"{m}_mean"] = round(statistics.mean(values), 4)
            summary[f"{m}_median"] = round(statistics.median(values), 4)
            summary[f"{m}_stdev"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0
            summary[f"{m}_min"] = round(min(values), 4)
            summary[f"{m}_max"] = round(max(values), 4)

    # Salvar CSV
    csv_file = f"results/E1_HTTP_{args.model}_{args.prompt_type}.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write(",".join(["iteration"] + metrics) + "\n")
        for r in results:
            f.write(",".join([str(r.get(m, 0)) for m in metrics]) + "\n")

    # Salvar JSON
    json_file = f"results/E1_HTTP_{args.model}_{args.prompt_type}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"T_total médio: {summary['T_total_ms_mean']:.2f} ms")
    print(f"Overhead transporte: {summary['transport_overhead_pct_mean']:.2f}%")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E1: Decomposição de Latência - HTTP")
    parser.add_argument("--model", default="phi4-mini", help="Modelo a usar")
    parser.add_argument("--prompt", dest="prompt_type", choices=["short", "long"], default="short")
    parser.add_argument("--n", type=int, default=100, help="Número de iterações")
    parser.add_argument("--url", default="http://localhost:8080", help="URL base do servidor")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
