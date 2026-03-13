#!/usr/bin/env python3
"""
E1: Decomposicao de Latencia por Camada - HTTP
==============================================
Mede T1-T6 com requests HTTP REAIS

NOTA: T3 (fila no agente) e T5 (transporte de retorno) nao sao mensuráveis
pelo cliente HTTP. Os valores exportados sao ESTIMATIVAS, marcados com sufixo
"_est" nos headers CSV e no JSON summary.

T2 representa o overhead de conexao TCP + envio de headers HTTP (antes do
servidor processar), estimado a partir do tempo ate receber os headers de
resposta.

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
    """Instrumenta cada camada da requisicao HTTP."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    async def measure_full_request(self, model: str, prompt: str) -> Dict[str, float]:
        """
        Mede latencia em 6 pontos via HTTP:
        T1: Serializacao do request (JSON)
        T2: Overhead HTTP (TCP handshake + envio headers) - estimado
        T3: Fila no agente - ESTIMATIVA (nao mensuravel pelo cliente)
        T4: Inferencia LLM - inferido (round_trip - T2_est - T3_est - T5_est)
        T5: Transporte de retorno - ESTIMATIVA (nao mensuravel pelo cliente)
        T6: Deserializacao da response
        """

        # T1: Serializacao
        t1_start = time.perf_counter_ns()
        request_data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50
        }
        payload_bytes = json.dumps(request_data).encode()
        t1_end = time.perf_counter_ns()
        T1 = (t1_end - t1_start) / 1e6  # ms

        # T2+T3+T4+T5: round-trip HTTP completo (nao totalmente separavel pelo cliente)
        t_send = time.perf_counter_ns()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                data=payload_bytes,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                # t_headers: tempo ate receber os headers (inicio do processamento do servidor)
                t_headers = time.perf_counter_ns()
                body = await resp.read()
                t_recv = time.perf_counter_ns()

        T_round_trip = (t_recv - t_send) / 1e6  # ms total
        T_to_headers = (t_headers - t_send) / 1e6  # ms ate headers

        # T2: overhead HTTP estimado (~10% do tempo ate receber headers)
        T2_estimated = max(T_to_headers * 0.1, 0.001)

        # T3: fila no servidor (nao mensuravel pelo cliente)
        T3_estimated = 0.1  # ms

        # T5: transporte de retorno (nao mensuravel pelo cliente)
        T5_estimated = 0.001  # ms

        # T4: inferencia inferida (o resto do round-trip)
        T4_inferred = max(T_round_trip - T2_estimated - T3_estimated - T5_estimated, 0.0)

        # T6: Deserializacao
        t6_start = time.perf_counter_ns()
        if body:
            response_json = json.loads(body)
        else:
            response_json = {}
        t6_end = time.perf_counter_ns()
        T6 = (t6_end - t6_start) / 1e6  # ms

        # T_total: soma de todas as componentes
        T_total = T1 + T2_estimated + T3_estimated + T4_inferred + T5_estimated + T6

        return {
            "T1_serialization_ms": T1,
            "T2_http_overhead_ms": T2_estimated,
            "T3_queue_est_ms": T3_estimated,
            "T4_inference_ms": T4_inferred,
            "T5_transport_return_est_ms": T5_estimated,
            "T6_deserialization_ms": T6,
            "T_total_ms": T_total,
            "T_round_trip_ms": T_round_trip,
            "transport_overhead_pct": ((T2_estimated + T5_estimated) / T_total) * 100 if T_total > 0 else 0
        }


async def run_benchmark(args):
    """Executa o benchmark completo."""

    base_url = args.url or "http://localhost:8080"

    decomposer = LatencyDecomposerHTTP(base_url)

    prompts = {
        "short": "O que e 2+2?",
        "long": "Explique detalhadamente a teoria da relatividade geral de Albert Einstein."
    }

    prompt = prompts.get(args.prompt_type, prompts["short"])

    results = []

    print(f"E1: Decomposicao de Latencia - HTTP")
    print(f"URL: {base_url}")
    print(f"Modelo: {args.model}")
    print(f"Prompt: {args.prompt_type}")
    print(f"Iteracoes: {args.n}")
    print("-" * 50)

    for i in range(args.n):
        result = await decomposer.measure_full_request(args.model, prompt)
        result["iteration"] = i + 1
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"Iteracao {i+1}/{args.n}...")

    # Estatisticas
    metrics = ["T1_serialization_ms", "T2_http_overhead_ms", "T3_queue_est_ms",
               "T4_inference_ms", "T5_transport_return_est_ms", "T6_deserialization_ms",
               "T_total_ms", "T_round_trip_ms", "transport_overhead_pct"]

    summary = {"protocol": "HTTP", "model": args.model, "prompt_type": args.prompt_type, "n": args.n}
    summary["estimated_fields"] = ["T3_queue_est_ms", "T5_transport_return_est_ms", "T2_http_overhead_ms"]

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

    csv_columns = ["iteration"] + metrics
    with open(csv_file, "w") as f:
        f.write(",".join(csv_columns) + "\n")
        for r in results:
            f.write(",".join([str(r.get(m, 0)) for m in csv_columns]) + "\n")

    # Salvar JSON
    json_file = f"results/E1_HTTP_{args.model}_{args.prompt_type}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"T1 (serializacao): {summary['T1_serialization_ms_mean']:.4f} ms")
    print(f"T2 (overhead HTTP, estimado): {summary['T2_http_overhead_ms_mean']:.4f} ms")
    print(f"T3 (estimado): {summary['T3_queue_est_ms_mean']:.4f} ms")
    print(f"T4 (inferencia, inferido): {summary['T4_inference_ms_mean']:.2f} ms")
    print(f"T5 (estimado): {summary['T5_transport_return_est_ms_mean']:.4f} ms")
    print(f"T6 (deserializacao): {summary['T6_deserialization_ms_mean']:.4f} ms")
    print(f"T_total medio: {summary['T_total_ms_mean']:.2f} ms")
    print(f"T_round_trip medio: {summary['T_round_trip_ms_mean']:.2f} ms")
    print(f"Overhead transporte: {summary['transport_overhead_pct_mean']:.2f}%")
    print(f"NOTA: T2, T3 e T5 sao estimativas do cliente, nao medicoes diretas.")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E1: Decomposicao de Latencia - HTTP")
    parser.add_argument("--model", default="phi4-mini", help="Modelo a usar")
    parser.add_argument("--prompt", dest="prompt_type", choices=["short", "long"], default="short")
    parser.add_argument("--n", type=int, default=100, help="Numero de iteracoes")
    parser.add_argument("--url", default="http://localhost:8080", help="URL base do servidor")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
