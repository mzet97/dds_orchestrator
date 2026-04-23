#!/usr/bin/env python3
"""
E5: Streaming Token-a-Token - DDS Nativo (Pub/Sub Direto)
==========================================================
Mede TTFT (Time-to-First-Token) e ITL (Inter-Token Latency) via DDS nativo
usando pub/sub direto ao llama-server (sem orchestrador).

Arquitetura medida:
    Cliente -> DDS pub/sub -> llama-server (--enable-dds)

Conexao direta ao llama-server. Cada token e um ChatCompletionResponse DDS
com is_final=False, ate o ultimo chunk com is_final=True.

Metricas:
  TTFT = tempo ate receber o primeiro ChatCompletionResponse com content nao-vazio (ms)
  ITL  = tempo entre ChatCompletionResponses consecutivos (ms)

Usage:
    python E5_streaming_dds_direct.py --domain 0 --model Qwen3.5-9B --n 5
"""

import argparse
import asyncio
import json
import os
import sys
import time
import statistics
import uuid
from pathlib import Path
from typing import Dict, List

# Add parent paths for IDL imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_orch_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _orch_dir)


def init_dds(domain_id: int):
    """Initialize DDS participant, writer (request), reader (response)."""
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.topic import Topic
    from cyclonedds.pub import Publisher, DataWriter
    from cyclonedds.sub import Subscriber, DataReader
    from cyclonedds.core import Policy
    from cyclonedds.qos import Qos
    from cyclonedds.util import duration

    from llama import ChatMessage, ChatCompletionRequest, ChatCompletionResponse

    participant = DomainParticipant(domain_id)

    # QoS must match C++ llama-server: Reliable, TransientLocal, KeepLast(8)
    qos_llm = Qos(
        Policy.Reliability.Reliable(duration(seconds=10)),
        Policy.Durability.TransientLocal,
        Policy.History.KeepLast(8),
    )

    topic_req = Topic(participant, "llama_chat_completion_request", ChatCompletionRequest)
    topic_resp = Topic(participant, "llama_chat_completion_response", ChatCompletionResponse)

    publisher = Publisher(participant)
    subscriber = Subscriber(participant)

    writer = DataWriter(publisher, topic_req, qos_llm)
    reader = DataReader(subscriber, topic_resp, qos_llm)

    return {
        "participant": participant,
        "writer": writer,
        "reader": reader,
        "ChatMessage": ChatMessage,
        "ChatCompletionRequest": ChatCompletionRequest,
    }


def measure_streaming(dds_ctx, model: str, prompt: str, max_tokens: int = 200) -> Dict:
    """
    Mede TTFT e ITL com DDS nativo pub/sub direto.
    Sincrono para precisao de timestamps.
    """
    ChatMessage = dds_ctx["ChatMessage"]
    ChatCompletionRequest = dds_ctx["ChatCompletionRequest"]
    writer = dds_ctx["writer"]
    reader = dds_ctx["reader"]

    request_id = str(uuid.uuid4())

    req = ChatCompletionRequest(
        request_id=request_id,
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.7,
        max_tokens=max_tokens,
        stream=True,
    )

    ttft = None
    itl_list = []
    total_tokens = 0
    previous_time = None

    start = time.perf_counter()
    writer.write(req)

    timeout_s = 120
    deadline = time.perf_counter() + timeout_s

    while time.perf_counter() < deadline:
        try:
            samples = reader.take()
            for sample in samples:
                if not sample:
                    continue
                if getattr(sample, "request_id", None) != request_id:
                    continue

                current_time = time.perf_counter()

                # Final marker
                if getattr(sample, "is_final", False):
                    # Final chunk may have content too
                    content = getattr(sample, "content", "")
                    if content and ttft is not None:
                        itl = (current_time - previous_time) * 1000
                        itl_list.append(itl)
                        total_tokens += 1
                    elif content and ttft is None:
                        ttft = (current_time - start) * 1000
                        total_tokens += 1
                    # Done
                    end = time.perf_counter()
                    total_time = (end - start) * 1000
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
                        ),
                    }

                content = getattr(sample, "content", "")
                if not content:
                    continue

                if ttft is None:
                    ttft = (current_time - start) * 1000
                    previous_time = current_time
                else:
                    itl = (current_time - previous_time) * 1000
                    itl_list.append(itl)
                    previous_time = current_time

                total_tokens += 1

        except Exception as e:
            pass

        time.sleep(0.001)  # 1ms poll

    return {"error": "Timeout waiting for DDS response"}


async def run_benchmark(args):
    """Executa benchmark de streaming DDS nativo."""

    prompt = "Conte uma historia sobre um robo que aprende a sentir emocoes. Com pelo menos 200 palavras."

    print(f"E5: Streaming Token-a-Token - DDS Nativo (Pub/Sub Direto)")
    print(f"Domain: {args.domain}")
    print(f"Modelo: {args.model}")
    print(f"Iteracoes: {args.n}")
    print("-" * 50)

    dds_ctx = init_dds(args.domain)

    # Wait a moment for DDS discovery
    await asyncio.sleep(2)

    results = []
    loop = asyncio.get_event_loop()

    for i in range(args.n):
        try:
            result = await loop.run_in_executor(
                None, measure_streaming, dds_ctx, args.model, prompt
            )

            if "error" not in result:
                results.append({
                    "iteration": i + 1,
                    "ttft_ms": result["ttft_ms"],
                    "itl_mean_ms": result["itl_mean_ms"],
                    "itl_median_ms": result["itl_median_ms"],
                    "itl_p99_ms": result["itl_p99_ms"],
                    "tokens": result["tokens"],
                    "total_time_ms": result["total_time_ms"],
                })
                print(f"Iteracao {i+1}/{args.n}: TTFT={result['ttft_ms']:.1f}ms, "
                      f"ITL={result['itl_mean_ms']:.2f}ms, Tokens={result['tokens']}")
            else:
                print(f"Iteracao {i+1}/{args.n}: ERRO - {result['error']}")

        except Exception as e:
            print(f"Iteracao {i+1}/{args.n}: ERRO - {e}")

    if not results:
        print("Nenhum resultado coletado. Verifique se llama-server com --enable-dds esta rodando.")
        return None

    ttft_values = [r["ttft_ms"] for r in results]
    itl_mean_values = [r["itl_mean_ms"] for r in results]
    itl_p99_values = [r["itl_p99_ms"] for r in results]
    tokens_values = [r["tokens"] for r in results]

    summary = {
        "protocol": "DDS_NATIVE_STREAMING",
        "model": args.model,
        "domain": args.domain,
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
        },
    }

    csv_file = f"results/E5_DDS_NATIVE_streaming_{args.model}.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write("iteration,ttft_ms,itl_mean_ms,itl_median_ms,itl_p99_ms,tokens,total_time_ms\n")
        for r in results:
            f.write(f"{r['iteration']},{r['ttft_ms']},{r['itl_mean_ms']},"
                    f"{r['itl_median_ms']},{r['itl_p99_ms']},{r['tokens']},{r['total_time_ms']}\n")

    json_file = f"results/E5_DDS_NATIVE_streaming_{args.model}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"TTFT medio: {summary['ttft']['mean_ms']:.2f}ms")
    print(f"ITL medio:  {summary['itl_mean']['mean_ms']:.2f}ms")
    print(f"ITL p99:    {summary['itl_p99']['mean_ms']:.2f}ms")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E5: Streaming Token-a-Token - DDS Nativo")
    parser.add_argument("--domain", type=int, default=0,
                        help="DDS Domain ID (must match llama-server)")
    parser.add_argument("--model", default="qwen3.5-0.8b")
    parser.add_argument("--n", type=int, default=1000)

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
