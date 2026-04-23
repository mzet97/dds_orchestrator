#!/usr/bin/env python3
"""
E5: Streaming Token-a-Token - DDS (via Orquestrador)
=====================================================
Mede TTFT (Time-to-First-Token) e ITL (Inter-Token Latency).

Rota: Cliente -> Orquestrador HTTP -> DDS -> Agente -> DDS -> Orquestrador -> SSE -> Cliente

O cliente envia a requisicao ao Orquestrador via HTTP. O Orquestrador roteia
internamente via DDS pub/sub para o Agente, que executa a inferencia e retorna
tokens via DDS. O Orquestrador entrega os tokens ao cliente via SSE (Server-Sent Events).

Comparar com E5_streaming_http.py (HTTP direto ao agente, sem orquestrador DDS).

Metricas:
  TTFT = tempo ate receber o primeiro token (ms)
  ITL  = tempo entre tokens consecutivos (ms); mede jitter de entrega

Usage:
    python E5_streaming_dds.py --url http://localhost:8080 --model qwen3.5-0.8b --n 50
"""

import argparse
import asyncio
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List
import aiohttp


class StreamingBenchmarkDDS:
    """Benchmark de streaming token-a-token via DDS (orquestrador)."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    async def measure_streaming(self, model: str, prompt: str, max_tokens: int = 200) -> Dict:
        """
        Mede TTFT e ITL com streaming REAL via orquestrador DDS.
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

                    # Buffer para parse correto de SSE (chunks podem conter múltiplas linhas)
                    sse_buffer = b""
                    _done = False
                    async for chunk in resp.content.iter_any():
                        sse_buffer += chunk
                        while b"\n" in sse_buffer:
                            raw_line, sse_buffer = sse_buffer.split(b"\n", 1)
                            line = raw_line.rstrip(b"\r")
                            if not line:
                                continue

                            if line.startswith(b"data: "):
                                data = line[6:]
                                if data.strip() == b"[DONE]":
                                    _done = True
                                    break

                                try:
                                    parsed = json.loads(data)
                                    delta = parsed.get("choices", [{}])[0].get("delta", {})
                                    token = delta.get("content", "")
                                    if not token:
                                        continue
                                    current_time = time.perf_counter()

                                    if ttft is None:
                                        ttft = (current_time - start) * 1000

                                    else:
                                        itl = (current_time - previous_time) * 1000
                                        itl_list.append(itl)

                                    previous_time = current_time
                                    total_tokens += 1

                                except Exception:
                                    pass

                        if _done:
                            break

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
    """Executa benchmark de streaming via orquestrador DDS."""

    benchmark = StreamingBenchmarkDDS(base_url=args.url)

    # Prompt padronizado
    prompt = "Conte uma historia sobre um robo que aprende a sentir emocoes. Com pelo menos 200 palavras."

    results = []

    print(f"E5: Streaming Token-a-Token - DDS (via orquestrador)")
    print(f"Rota: Cliente -> Orquestrador HTTP -> DDS -> Agente -> DDS -> Orquestrador -> SSE -> Cliente")
    print(f"URL: {args.url}")
    print(f"Modelo: {args.model}")
    print(f"Iteracoes: {args.n}")
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

            print(f"Iteracao {i+1}/{args.n}: TTFT={result['ttft_ms']:.1f}ms, ITL={result['itl_mean_ms']:.2f}ms, Tokens={result['tokens']}")
        else:
            print(f"Iteracao {i+1}/{args.n}: ERRO - {result['error']}")

    # Estatisticas
    ttft_values = [r["ttft_ms"] for r in results]
    itl_mean_values = [r["itl_mean_ms"] for r in results]
    itl_p99_values = [r["itl_p99_ms"] for r in results]
    tokens_values = [r["tokens"] for r in results]

    summary = {
        "protocol": f"{args.protocol_label}_VIA_ORCHESTRADOR",
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

    # Salvar CSV - nome distinto para DDS
    csv_file = f"results/E5_{args.protocol_label}_VIA_ORCH_streaming_{args.model}.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write("iteration,ttft_ms,itl_mean_ms,itl_median_ms,itl_p99_ms,tokens,total_time_ms\n")
        for r in results:
            f.write(f"{r['iteration']},{r['ttft_ms']},{r['itl_mean_ms']},{r['itl_median_ms']},{r['itl_p99_ms']},{r['tokens']},{r['total_time_ms']}\n")

    # Salvar JSON
    json_file = f"results/E5_{args.protocol_label}_VIA_ORCH_streaming_{args.model}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"TTFT medio: {summary['ttft']['mean_ms']:.2f}ms")
    print(f"ITL medio: {summary['itl_mean']['mean_ms']:.2f}ms")
    print(f"ITL p99: {summary['itl_p99']['mean_ms']:.2f}ms")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E5: Streaming Token-a-Token - DDS (via orquestrador)")
    parser.add_argument("--model", default="qwen3.5-0.8b", help="Modelo a usar")
    parser.add_argument("--url", default="http://localhost:8080", help="URL do orquestrador DDS")
    parser.add_argument("--n", type=int, default=1000, help="Numero de iteracoes")
    parser.add_argument("--protocol-label", default="DDS", help="Label do protocolo para nomes de arquivos")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
