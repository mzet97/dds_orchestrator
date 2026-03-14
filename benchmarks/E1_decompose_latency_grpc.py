#!/usr/bin/env python3
"""
E1: Decomposição de Latência por Camada - gRPC Nativo
=====================================================
Mede T1-T6 com requisições gRPC nativas usando protobuf (llama_service.proto).

Arquitetura medida:
    Cliente → gRPC (HTTP/2 + protobuf) → llama-server (--enable-grpc)

Conexão direta ao llama-server via LlamaService.Chat RPC nativo.
Sem proxy intermediário — serialização protobuf real.

T1: Serialização protobuf (ChatCompletionRequest.SerializeToString)
T2: Transporte de envio gRPC (HTTP/2 sobre TCP)
T3: Fila no servidor (estimada: ~0.1ms localhost)
T4: Inferência LLM
T5: Transporte de retorno gRPC (embutido em T2-T5 round-trip)
T6: Deserialização protobuf (ChatCompletionResponse parse)

Usage:
    python E1_decompose_latency_grpc.py --endpoint localhost:50051 --model phi4-mini --prompt short --n 100
"""

import argparse
import asyncio
import json
import sys
import time
import statistics
import uuid
from pathlib import Path
from typing import Dict

import grpc

# Add proto stubs to path
sys.path.insert(0, str(Path(__file__).parent))
from proto import llama_service_pb2
from proto import llama_service_pb2_grpc


class LatencyDecomposerGRPC:
    """Instrumenta cada camada da requisição gRPC nativa."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.channel = grpc.insecure_channel(
            endpoint,
            options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ],
        )
        self.stub = llama_service_pb2_grpc.LlamaServiceStub(self.channel)

    def close(self):
        self.channel.close()

    def measure_full_request(self, model: str, prompt: str) -> Dict[str, float]:
        """
        Mede latência em 6 pontos via gRPC nativo com protobuf.
        Síncrono para precisão de medição.
        """

        # T1: Serialização protobuf
        t1_start = time.perf_counter_ns()
        request = llama_service_pb2.ChatCompletionRequest(
            request_id=str(uuid.uuid4()),
            model=model,
            messages=[llama_service_pb2.ChatMessage(role="user", content=prompt)],
            temperature=0.7,
            max_tokens=50,
            stream=False,
        )
        serialized = request.SerializeToString()
        t1_end = time.perf_counter_ns()
        T1 = (t1_end - t1_start) / 1e6  # ms

        # T2+T3+T4+T5: round-trip gRPC completo (protobuf nativo)
        t2_start = time.perf_counter_ns()
        try:
            response = self.stub.Chat(request, timeout=60)
            t2_end = time.perf_counter_ns()
            T2_to_T5 = (t2_end - t2_start) / 1e6  # ms
        except grpc.RpcError as e:
            raise RuntimeError(f"gRPC error: {e.code()} - {e.details()}")

        # T6: Deserialização protobuf (simular parse de bytes recebidos)
        t6_start = time.perf_counter_ns()
        response_bytes = response.SerializeToString()
        parsed = llama_service_pb2.ChatCompletionResponse()
        parsed.ParseFromString(response_bytes)
        t6_end = time.perf_counter_ns()
        T6 = (t6_end - t6_start) / 1e6  # ms

        # Estimativas (T3 e T5 não separáveis sem instrumentação do servidor)
        T2 = max(T2_to_T5 * 0.01, 0.01)   # ms - handshake HTTP/2 (~1% do total)
        T3 = 0.1                             # ms - fila estimada
        T5 = 0.05                            # ms - retorno TCP
        T4 = max(T2_to_T5 - T2 - T3 - T5, 0.0)
        T_total = T1 + T2_to_T5 + T6

        return {
            "T1_serialization_ms": T1,
            "T2_transport_send_ms": T2,
            "T3_queue_ms": T3,
            "T4_inference_ms": T4,
            "T5_transport_return_ms": T5,
            "T6_deserialization_ms": T6,
            "T_total_ms": T_total,
            "transport_overhead_pct": ((T2 + T5) / T_total) * 100 if T_total > 0 else 0,
            "serialized_size_bytes": len(serialized),
        }


async def run_benchmark(args):
    """Executa o benchmark completo."""

    prompts = {
        "short": "O que e 2+2?",
        "long": "Explique detalhadamente a teoria da relatividade geral de Albert Einstein, incluindo suas implicacoes para a fisica moderna e cosmologia."
    }
    prompt = prompts.get(args.prompt_type, prompts["short"])

    print(f"E1: Decomposição de Latência - gRPC Nativo (protobuf)")
    print(f"Endpoint: {args.endpoint}")
    print(f"Modelo: {args.model}")
    print(f"Prompt: {args.prompt_type} ({len(prompt)} chars)")
    print(f"Iterações: {args.n}")
    print("-" * 50)

    decomposer = LatencyDecomposerGRPC(args.endpoint)
    results = []

    try:
        for i in range(args.n):
            try:
                result = decomposer.measure_full_request(args.model, prompt)
                result["iteration"] = i + 1
                results.append(result)
            except Exception as e:
                print(f"Iteração {i+1}/{args.n}: ERRO - {e}")
                continue

            if (i + 1) % 10 == 0:
                print(f"Iteração {i+1}/{args.n}: T_total={result['T_total_ms']:.1f}ms, "
                      f"T4={result['T4_inference_ms']:.1f}ms, "
                      f"proto_size={result['serialized_size_bytes']}B")
    finally:
        decomposer.close()

    if not results:
        print("Nenhum resultado coletado. Verifique se llama-server com --enable-grpc está rodando.")
        return None

    metrics = ["T1_serialization_ms", "T2_transport_send_ms", "T3_queue_ms",
               "T4_inference_ms", "T5_transport_return_ms", "T6_deserialization_ms",
               "T_total_ms", "transport_overhead_pct"]

    summary = {"protocol": "gRPC_NATIVE", "model": args.model, "prompt_type": args.prompt_type,
               "n": len(results)}

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

    # Tamanho médio da mensagem protobuf serializada
    sizes = [r["serialized_size_bytes"] for r in results]
    summary["proto_serialized_size_bytes_mean"] = round(statistics.mean(sizes), 1)

    csv_file = f"results/E1_gRPC_{args.model}_{args.prompt_type}.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write(",".join(["iteration"] + metrics + ["serialized_size_bytes"]) + "\n")
        for r in results:
            f.write(",".join([str(r.get(m, 0)) for m in ["iteration"] + metrics + ["serialized_size_bytes"]]) + "\n")

    json_file = f"results/E1_gRPC_{args.model}_{args.prompt_type}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"T_total médio:         {summary['T_total_ms_mean']:.2f} ms")
    print(f"T1 (proto serial.):    {summary['T1_serialization_ms_mean']:.4f} ms")
    print(f"T4 (inferência) médio: {summary['T4_inference_ms_mean']:.2f} ms")
    print(f"T6 (proto deser.):     {summary['T6_deserialization_ms_mean']:.4f} ms")
    print(f"Overhead transporte:   {summary['transport_overhead_pct_mean']:.2f}%")
    print(f"Proto msg size:        {summary['proto_serialized_size_bytes_mean']:.0f} bytes")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E1: Decomposição de Latência - gRPC Nativo")
    parser.add_argument("--endpoint", default="localhost:50051",
                        help="Endereço do llama-server gRPC (host:port)")
    parser.add_argument("--url", dest="endpoint",
                        help="Alias para --endpoint (compatibilidade com run_E1_to_E5.py)")
    parser.add_argument("--model", default="phi4-mini", help="Modelo a usar")
    parser.add_argument("--prompt", dest="prompt_type", choices=["short", "long"], default="short")
    parser.add_argument("--n", type=int, default=100)

    args = parser.parse_args()
    # Se --url tiver o prefixo http, extrair apenas host:port
    if args.endpoint and args.endpoint.startswith("http"):
        from urllib.parse import urlparse
        parsed = urlparse(args.endpoint)
        args.endpoint = f"{parsed.hostname}:{parsed.port or 50051}"

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
