#!/usr/bin/env python3
"""
E5: Streaming Token-a-Token - HTTP
==================================
Mede TTFT (Time-to-First-Token) e ITL (Inter-Token Latency)
100% REAL - streaming real via HTTP

Usage:
    python E5_streaming_http.py --model phi4-mini --n 50
"""

import argparse
import asyncio
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List
import aiohttp


class StreamingBenchmarkHTTP:
    """Benchmark de streaming HTTP token-a-token."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    async def measure_streaming(self, model: str, prompt: str, max_tokens: int = 200) -> Dict:
        """
        Mede TTFT e ITL com streaming HTTP REAL.
        """
        ttft = None  # Time-to-First-Token
        itl_list = []  # Inter-Token Latency
        total_tokens = 0

        start = time.perf_counter()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True,
                        "max_tokens": max_tokens
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    previous_time = time.perf_counter()

                    async for line in resp.content:
                        if not line:
                            continue

                        # Parse SSE line
                        if line.startswith(b"data: "):
                            data = line[6:]
                            if data == b"[DONE]":
                                break

                            try:
                                parsed = json.loads(data)
                                current_time = time.perf_counter()

                                # Primeira token = TTFT
                                if ttft is None:
                                    ttft = (current_time - start) * 1000  # ms

                                # Tokens subsequentes = ITL
                                else:
                                    itl = (current_time - previous_time) * 1000  # ms
                                    itl_list.append(itl)

                                previous_time = current_time
                                total_tokens += 1

                            except:
                                pass

        except Exception as e:
            return {"error": str(e)}

        end = time.perf_counter()
        total_time = (end - start) * 1000  # ms

        return {
            "ttft_ms": ttft if ttft else 0,
            "itl_list_ms": itl_list,
            "tokens": total_tokens,
            "total_time_ms": total_time,
            "itl_mean_ms": statistics.mean(itl_list) if itl_list else 0,
            "itl_median_ms": statistics.median(itl_list) if itl_list else 0,
            "itl_p99_ms": sorted(itl_list)[int(len(itl_list) * 0.99)] if len(itl_list) >= 100 else max(itl_list) if itl_list else 0
        }


async def run_benchmark(args):
    """Executa benchmark de streaming HTTP."""

    benchmark = StreamingBenchmarkHTTP(base_url=args.url)

    # Prompt padronizado
    prompt = "Conte uma história sobre um robô que aprende a sentir emoções. Com pelo menos 200 palavras."

    results = []

    print(f"E5: Streaming Token-a-Token - HTTP")
    print(f"URL: {args.url}")
    print(f"Modelo: {args.model}")
    print(f"Iterações: {args.n}")
    print("-" * 50)

    for i in range(args.n):
        result = await benchmark.measure_streaming(args.model, prompt)

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

            print(f"Iteração {i+1}/{args.n}: TTFT={result['ttft_ms']:.1f}ms, ITL={result['itl_mean_ms']:.2f}ms, Tokens={result['tokens']}")
        else:
            print(f"Iteração {i+1}/{args.n}: ERRO - {result['error']}")

    # Estatísticas
    ttft_values = [r["ttft_ms"] for r in results]
    itl_mean_values = [r["itl_mean_ms"] for r in results]
    itl_p99_values = [r["itl_p99_ms"] for r in results]
    tokens_values = [r["tokens"] for r in results]

    summary = {
        "protocol": "HTTP_STREAMING",
        "model": args.model,
        "n": len(results),
        "ttft": {
            "mean_ms": round(statistics.mean(ttft_values), 2) if ttft_values else 0,
            "median_ms": round(statistics.median(ttft_values), 2) if ttft_values else 0,
            "stdev_ms": round(statistics.stdev(ttft_values), 2) if len(ttft_values) > 1 else 0
        },
        "itl_mean": {
            "mean_ms": round(statistics.mean(itl_mean_values), 4) if itl_mean_values else 0,
            "median_ms": round(statistics.median(itl_mean_values), 4) if itl_mean_values else 0,
            "stdev_ms": round(statistics.stdev(itl_mean_values), 4) if len(itl_mean_values) > 1 else 0
        },
        "itl_p99": {
            "mean_ms": round(statistics.mean(itl_p99_values), 2) if itl_p99_values else 0
        },
        "tokens": {
            "mean": round(statistics.mean(tokens_values), 1) if tokens_values else 0
        }
    }

    # Salvar CSV
    csv_file = f"results/E5_HTTP_streaming_{args.model}.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write("iteration,ttft_ms,itl_mean_ms,itl_median_ms,itl_p99_ms,tokens,total_time_ms\n")
        for r in results:
            f.write(f"{r['iteration']},{r['ttft_ms']},{r['itl_mean_ms']},{r['itl_median_ms']},{r['itl_p99_ms']},{r['tokens']},{r['total_time_ms']}\n")

    # Salvar JSON
    json_file = f"results/E5_HTTP_streaming_{args.model}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"TTFT médio: {summary['ttft']['mean_ms']:.2f}ms")
    print(f"ITL médio: {summary['itl_mean']['mean_ms']:.2f}ms")
    print(f"ITL p99: {summary['itl_p99']['mean_ms']:.2f}ms")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E5: Streaming Token-a-Token - HTTP")
    parser.add_argument("--model", default="phi4-mini", help="Modelo a usar")
    parser.add_argument("--url", default="http://localhost:8080", help="URL do servidor")
    parser.add_argument("--n", type=int, default=50, help="Número de iterações")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
