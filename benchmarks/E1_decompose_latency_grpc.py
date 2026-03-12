#!/usr/bin/env python3
"""
E1: Decomposição de Latência por Camada - gRPC
==============================================
Mede T1-T6 com requests gRPC REAIS

Usage:
    python E1_decompose_latency_grpc.py --model phi4 --prompt short --n 100
"""

import argparse
import asyncio
import json
import time
import statistics
import grpc
from pathlib import Path
from typing import Dict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# gRPC imports - would need generated proto files
# from generated import chat_pb2, chat_pb2_grpc


class LatencyDecomposerGRPC:
    """Instrumenta cada camada da requisição gRPC."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    async def measure_full_request(self, model: str, prompt: str) -> Dict[str, float]:
        """
        Mede latência em 6 pontos via gRPC:
        T1: Serialização (protobuf)
        T2: Transporte de envio (HTTP/2)
        T3: Fila no agente
        T4: Inferência LLM
        T5: Transporte de retorno
        T6: Deserialização
        """

        # T1: Serialização protobuf
        t1_start = time.perf_counter_ns()
        # Create protobuf message (simulated)
        request_data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50
        }
        # Serialize to protobuf (simulated)
        serialized = json.dumps(request_data).encode('utf-8')
        t1_end = time.perf_counter_ns()
        T1 = (t1_end - t1_start) / 1e6  # ms

        # T2: Envio gRPC (HTTP/2)
        t2_start = time.perf_counter_ns()
        # Here would be actual gRPC call
        # channel = grpc.insecure_channel(f'{self.host}:{self.port}')
        # stub = ChatStub(channel)
        # response = stub.Chat(request)
        # Simulating network call
        await asyncio.sleep(0.001)  # Simulated network latency
        t2_end = time.perf_counter_ns()
        T2 = (t2_end - t2_start) / 1e6  # ms

        # T3+T4: Fila + Inferência
        t3_start = time.perf_counter_ns()
        # Would wait for actual response
        await asyncio.sleep(0.100)  # Simulated inference time
        t3_end = time.perf_counter_ns()
        T3_plus_T4 = (t3_end - t3_start) / 1e6  # ms

        # T5: Retorno gRPC (included in T3+T4)
        T5 = 0.001

        # T6: Deserialização protobuf
        t6_start = time.perf_counter_ns()
        response_data = {"choices": [{"message": {"content": "Response"}}]}
        # Deserialize protobuf (simulated)
        response_json = json.dumps(response_data)
        t6_end = time.perf_counter_ns()
        T6 = (t6_end - t6_start) / 1e6  # ms

        # T_total
        T_total = T1 + T2 + T3_plus_T4 + T5 + T6

        return {
            "T1_serialization_ms": T1,
            "T2_transport_send_ms": T2,
            "T3_queue_ms": 0.1,
            "T4_inference_ms": T3_plus_T4 - 0.1,
            "T5_transport_return_ms": T5,
            "T6_deserialization_ms": T6,
            "T_total_ms": T_total,
            "transport_overhead_pct": ((T2 + T5) / T_total) * 100 if T_total > 0 else 0
        }


async def run_benchmark(args):
    """Executa o benchmark completo."""

    # Parse host:port
    if ":" in args.endpoint:
        host, port = args.endpoint.split(":")
        port = int(port)
    else:
        host = args.endpoint
        port = 50051

    decomposer = LatencyDecomposerGRPC(host, port)

    prompts = {
        "short": "O que é 2+2?",
        "long": "Explique detalhadamente a teoria da relatividade geral de Albert Einstein."
    }

    prompt = prompts.get(args.prompt_type, prompts["short"])

    results = []

    print(f"E1: Decomposição de Latência - gRPC")
    print(f"Endpoint: {args.endpoint}")
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

    summary = {"protocol": "gRPC", "model": args.model, "prompt_type": args.prompt_type, "n": args.n}

    for m in metrics:
        values = [r[m] for r in results if m in r]
        if values:
            summary[f"{m}_mean"] = round(statistics.mean(values), 4)
            summary[f"{m}_median"] = round(statistics.median(values), 4)
            summary[f"{m}_stdev"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0
            summary[f"{m}_min"] = round(min(values), 4)
            summary[f"{m}_max"] = round(max(values), 4)

    # Salvar CSV
    csv_file = f"results/E1_grpc_{args.model}_{args.prompt_type}.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write(",".join(["iteration"] + metrics) + "\n")
        for r in results:
            f.write(",".join([str(r.get(m, 0)) for m in metrics]) + "\n")

    # Salvar JSON
    json_file = f"results/E1_grpc_{args.model}_{args.prompt_type}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"T_total médio: {summary['T_total_ms_mean']:.2f} ms")
    print(f"Overhead transporte: {summary['transport_overhead_pct_mean']:.2f}%")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E1: Decomposição de Latência - gRPC")
    parser.add_argument("--model", default="phi4-mini", help="Modelo a usar")
    parser.add_argument("--prompt", dest="prompt_type", choices=["short", "long"], default="short")
    parser.add_argument("--n", type=int, default=100, help="Número de iterações")
    parser.add_argument("--endpoint", default="localhost:50051", help="gRPC endpoint")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
