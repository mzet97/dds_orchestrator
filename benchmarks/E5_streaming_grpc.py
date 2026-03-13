#!/usr/bin/env python3
"""
E5: Streaming Token-a-Token - gRPC Server Streaming
====================================================
Mede TTFT (Time-to-First-Token) e ITL (Inter-Token Latency) via gRPC streaming.

Requer: servidor gRPC rodando (_grpc_server.py --backend http://... --port 50051)

Arquitetura medida:
    Cliente → gRPC server streaming → _grpc_server.py → HTTP SSE → llama-server

Cada token do llama-server é encaminhado como um StreamChunk gRPC separado,
permitindo medir ITL genuíno no nível do protocolo gRPC.

Métricas:
  TTFT = tempo até receber o primeiro StreamChunk com token (ms)
  ITL  = tempo entre StreamChunks consecutivos (ms)

Usage:
    python _grpc_server.py --backend http://192.168.1.60:8082 --port 50051 &
    python E5_streaming_grpc.py --endpoint 192.168.1.60:50051 --model phi4-mini --n 50
"""

import argparse
import asyncio
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List
import grpc


def _json_serialize(obj: dict) -> bytes:
    return json.dumps(obj).encode("utf-8")


def _json_deserialize(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


class StreamingBenchmarkGRPC:
    """Benchmark de streaming token-a-token via gRPC server streaming."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._channel = grpc.insecure_channel(endpoint)
        # gRPC unary-to-server-streaming stub
        self._stream_stub = self._channel.unary_stream(
            "/LLMService/StreamChat",
            request_serializer=_json_serialize,
            response_deserializer=_json_deserialize,
        )

    def close(self):
        self._channel.close()

    def measure_streaming(self, model: str, prompt: str, max_tokens: int = 200) -> Dict:
        """
        Mede TTFT e ITL com gRPC server streaming.
        Síncrono para precisão de medição de timestamps.
        """
        ttft = None
        itl_list = []
        total_tokens = 0
        previous_time = None

        payload = {
            "model": model,
            "content": prompt,
            "max_tokens": max_tokens
        }

        start = time.perf_counter()

        try:
            stream = self._stream_stub(payload, timeout=120)

            for chunk in stream:
                if chunk.get("error"):
                    return {"error": chunk["error"]}

                current_time = time.perf_counter()

                if chunk.get("done", False):
                    break

                token = chunk.get("token", "")
                if not token:
                    continue

                # Primeiro token = TTFT
                if ttft is None:
                    ttft = (current_time - start) * 1000  # ms
                    previous_time = current_time
                else:
                    # Tokens subsequentes = ITL
                    itl = (current_time - previous_time) * 1000  # ms
                    itl_list.append(itl)
                    previous_time = current_time

                total_tokens += 1

        except grpc.RpcError as e:
            return {"error": f"gRPC {e.code()}: {e.details()}"}
        except Exception as e:
            return {"error": str(e)}

        end = time.perf_counter()
        total_time = (end - start) * 1000  # ms

        return {
            "ttft_ms": ttft if ttft else 0,
            "tokens": total_tokens,
            "total_time_ms": total_time,
            "itl_mean_ms": statistics.mean(itl_list) if itl_list else 0,
            "itl_median_ms": statistics.median(itl_list) if itl_list else 0,
            "itl_p99_ms": (
                sorted(itl_list)[int(len(itl_list) * 0.99)]
                if len(itl_list) >= 100
                else max(itl_list)
                if itl_list
                else 0
            )
        }


async def run_benchmark(args):
    """Executa benchmark de streaming gRPC."""

    endpoint = args.endpoint
    if endpoint.startswith("http"):
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        endpoint = f"{parsed.hostname}:{parsed.port or 50051}"

    # Prompt padronizado conforme dissertação (~200 tokens)
    prompt = "Conte uma historia sobre um robo que aprende a sentir emocoes. Com pelo menos 200 palavras."

    print(f"E5: Streaming Token-a-Token - gRPC")
    print(f"Endpoint: {endpoint}")
    print(f"Modelo: {args.model}")
    print(f"Iterações: {args.n}")
    print("-" * 50)

    benchmark = StreamingBenchmarkGRPC(endpoint)
    results = []
    loop = asyncio.get_event_loop()

    try:
        for i in range(args.n):
            try:
                result = await loop.run_in_executor(
                    None, benchmark.measure_streaming, args.model, prompt
                )

                if "error" not in result:
                    results.append({
                        "iteration": i + 1,
                        "ttft_ms": result["ttft_ms"],
                        "itl_mean_ms": result["itl_mean_ms"],
                        "itl_median_ms": result["itl_median_ms"],
                        "itl_p99_ms": result["itl_p99_ms"],
                        "tokens": result["tokens"],
                        "total_time_ms": result["total_time_ms"]
                    })
                    print(f"Iteração {i+1}/{args.n}: TTFT={result['ttft_ms']:.1f}ms, "
                          f"ITL={result['itl_mean_ms']:.2f}ms, Tokens={result['tokens']}")
                else:
                    print(f"Iteração {i+1}/{args.n}: ERRO - {result['error']}")

            except Exception as e:
                print(f"Iteração {i+1}/{args.n}: ERRO - {e}")
    finally:
        benchmark.close()

    if not results:
        print("Nenhum resultado coletado. Verifique se _grpc_server.py está rodando.")
        return None

    ttft_values = [r["ttft_ms"] for r in results]
    itl_mean_values = [r["itl_mean_ms"] for r in results]
    itl_p99_values = [r["itl_p99_ms"] for r in results]
    tokens_values = [r["tokens"] for r in results]

    summary = {
        "protocol": "gRPC_STREAMING",
        "model": args.model,
        "endpoint": endpoint,
        "n": len(results),
        "ttft": {
            "mean_ms": round(statistics.mean(ttft_values), 2),
            "median_ms": round(statistics.median(ttft_values), 2),
            "stdev_ms": round(statistics.stdev(ttft_values), 2) if len(ttft_values) > 1 else 0,
            "p95_ms": round(sorted(ttft_values)[int(len(ttft_values) * 0.95)], 2),
        },
        "itl_mean": {
            "mean_ms": round(statistics.mean(itl_mean_values), 4),
            "median_ms": round(statistics.median(itl_mean_values), 4),
            "stdev_ms": round(statistics.stdev(itl_mean_values), 4) if len(itl_mean_values) > 1 else 0,
        },
        "itl_p99": {
            "mean_ms": round(statistics.mean(itl_p99_values), 2) if itl_p99_values else 0
        },
        "tokens": {
            "mean": round(statistics.mean(tokens_values), 1)
        }
    }

    csv_file = f"results/E5_gRPC_streaming_{args.model}.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write("iteration,ttft_ms,itl_mean_ms,itl_median_ms,itl_p99_ms,tokens,total_time_ms\n")
        for r in results:
            f.write(f"{r['iteration']},{r['ttft_ms']},{r['itl_mean_ms']},"
                    f"{r['itl_median_ms']},{r['itl_p99_ms']},{r['tokens']},{r['total_time_ms']}\n")

    json_file = f"results/E5_gRPC_streaming_{args.model}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"TTFT médio: {summary['ttft']['mean_ms']:.2f}ms")
    print(f"ITL médio:  {summary['itl_mean']['mean_ms']:.2f}ms")
    print(f"ITL p99:    {summary['itl_p99']['mean_ms']:.2f}ms")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E5: Streaming Token-a-Token - gRPC")
    parser.add_argument("--endpoint", default="localhost:50051",
                        help="Endereço do servidor gRPC")
    parser.add_argument("--url", dest="endpoint",
                        help="Alias para --endpoint (compatibilidade)")
    parser.add_argument("--model", default="phi4-mini")
    parser.add_argument("--n", type=int, default=50)

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
