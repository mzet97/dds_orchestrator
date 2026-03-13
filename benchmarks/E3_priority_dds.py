#!/usr/bin/env python3
"""
E3: Priorização sob Carga - DDS TRANSPORT_PRIORITY
====================================================
Mede latência de mensagens HIGH vs NORMAL sob carga sustentada.

O orquestrador usa DDS TRANSPORT_PRIORITY internamente para rotear requisições
de alta prioridade antes das normais na camada de transporte DDS.

Metodologia:
  - Carga sustentada: 10 req/s com prioridade NORMAL
  - A cada inject_interval segundos: injeção de 1 requisição HIGH
  - Duração: 5 minutos, 30 injeções HIGH
  - max_tokens=5: minimiza contribuição da inferência, isola latência de transporte
  - Métrica: latência de transporte = tempo de envio até início de processamento
    (aproximado por latência end-to-end com inferência mínima)

Usage:
    python E3_priority_dds.py --url http://localhost:8080 --carga 10 --n 30 --duracao 300
"""

import argparse
import asyncio
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List
import aiohttp


class PriorityBenchmarkDDS:
    """Benchmark de priorização via orquestrador com DDS TRANSPORT_PRIORITY."""

    def __init__(self, base_url: str, carga_req_s: int = 10):
        self.base_url = base_url
        self.carga_req_s = carga_req_s
        self.normal_results: List[Dict] = []
        self.stop_load = False

    async def _send_request(self, session: aiohttp.ClientSession, priority: int) -> Dict:
        """
        Envia requisição com prioridade explícita.
        priority=10 → HIGH (DDS TRANSPORT_PRIORITY=10)
        priority=1  → NORMAL (DDS TRANSPORT_PRIORITY=1)
        max_tokens=5 para minimizar inferência e isolar transporte.
        """
        payload = {
            "model": "phi4-mini",
            "messages": [{"role": "user", "content": "Responda apenas: ok"}],
            "max_tokens": 5,
            "priority": priority
        }

        send_time = time.perf_counter()
        try:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                await resp.text()
                recv_time = time.perf_counter()
                return {
                    "priority": "HIGH" if priority >= 10 else "NORMAL",
                    "send_time": send_time,
                    "latency_ms": (recv_time - send_time) * 1000,
                    "status": resp.status
                }
        except Exception as e:
            recv_time = time.perf_counter()
            return {
                "priority": "HIGH" if priority >= 10 else "NORMAL",
                "send_time": send_time,
                "latency_ms": (recv_time - send_time) * 1000,
                "error": str(e)
            }

    async def run_background_load(self, duration_s: int):
        """Gera carga sustentada de requisições NORMAL."""
        interval = 1.0 / self.carga_req_s
        start = time.perf_counter()
        count = 0

        async with aiohttp.ClientSession() as session:
            while (time.perf_counter() - start) < duration_s and not self.stop_load:
                result = await self._send_request(session, priority=1)
                result["elapsed_s"] = time.perf_counter() - start
                self.normal_results.append(result)
                count += 1
                await asyncio.sleep(interval)

        return count

    async def inject_priority_message(self, session: aiohttp.ClientSession) -> Dict:
        """Injeta mensagem HIGH e retorna latência."""
        result = await self._send_request(session, priority=10)
        return result


async def run_benchmark(args):
    """Executa benchmark de priorização DDS."""

    benchmark = PriorityBenchmarkDDS(base_url=args.url, carga_req_s=args.carga)

    print(f"E3: Priorização - DDS TRANSPORT_PRIORITY (via orquestrador)")
    print(f"URL: {args.url}")
    print(f"Carga NORMAL: {args.carga} req/s")
    print(f"Duração: {args.duracao}s")
    print(f"Injeções HIGH: {args.n}")
    print(f"Intervalo entre injeções: {args.duracao / args.n:.1f}s")
    print("-" * 50)

    inject_interval = args.duracao / args.n  # intervalo entre injeções HIGH

    # Iniciar carga NORMAL em background
    load_task = asyncio.create_task(benchmark.run_background_load(args.duracao))

    # Aguardar carga estabilizar
    await asyncio.sleep(1.0)

    # Injetar mensagens HIGH a cada inject_interval segundos
    priority_results = []
    async with aiohttp.ClientSession() as session:
        for i in range(args.n):
            await asyncio.sleep(inject_interval)

            if load_task.done():
                print(f"Aviso: carga background terminou antes das {args.n} injeções")
                break

            result = await benchmark.inject_priority_message(session)
            priority_results.append(result)
            print(f"Injeção {i+1}/{args.n}: latência={result['latency_ms']:.2f}ms "
                  f"({'OK' if 'error' not in result else 'ERRO'})")

    # Parar carga normal
    benchmark.stop_load = True
    try:
        await asyncio.wait_for(load_task, timeout=5.0)
    except asyncio.TimeoutError:
        pass

    # Análise
    normal_latencies = [r["latency_ms"] for r in benchmark.normal_results
                        if "error" not in r]
    priority_latencies = [r["latency_ms"] for r in priority_results
                          if "error" not in r]

    if not normal_latencies or not priority_latencies:
        print("ERRO: dados insuficientes para análise")
        return None

    summary = {
        "protocol": "DDS_TRANSPORT_PRIORITY",
        "url": args.url,
        "carga_req_s": args.carga,
        "duracao_s": args.duracao,
        "n_injections": len(priority_latencies),
        "normal": {
            "n": len(normal_latencies),
            "mean_ms": round(statistics.mean(normal_latencies), 4),
            "median_ms": round(statistics.median(normal_latencies), 4),
            "stdev_ms": round(statistics.stdev(normal_latencies), 4) if len(normal_latencies) > 1 else 0,
            "p95_ms": round(sorted(normal_latencies)[int(len(normal_latencies) * 0.95)], 4),
            "p99_ms": round(sorted(normal_latencies)[int(len(normal_latencies) * 0.99)], 4),
        },
        "priority_high": {
            "n": len(priority_latencies),
            "mean_ms": round(statistics.mean(priority_latencies), 4),
            "median_ms": round(statistics.median(priority_latencies), 4),
            "stdev_ms": round(statistics.stdev(priority_latencies), 4) if len(priority_latencies) > 1 else 0,
            "p95_ms": round(sorted(priority_latencies)[int(len(priority_latencies) * 0.95)] if len(priority_latencies) >= 20 else max(priority_latencies), 4),
        }
    }

    # Diferença de latência (hipótese: HIGH < NORMAL sob carga)
    diff = summary["normal"]["median_ms"] - summary["priority_high"]["median_ms"]
    summary["priority_advantage_ms"] = round(diff, 4)

    # Salvar CSV
    csv_file = f"results/E3_DDS_PRIORITY_carga{args.carga}.csv"
    Path("results").mkdir(exist_ok=True)

    all_results = [(r, "NORMAL") for r in benchmark.normal_results] + \
                  [(r, "HIGH") for r in priority_results]

    with open(csv_file, "w") as f:
        f.write("priority,latency_ms,elapsed_s,error\n")
        for r, prio in all_results:
            elapsed = r.get("elapsed_s", r.get("send_time", 0))
            error = r.get("error", "")
            f.write(f"{prio},{r['latency_ms']},{elapsed},{error}\n")

    json_file = f"results/E3_DDS_PRIORITY_carga{args.carga}_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"NORMAL  (n={summary['normal']['n']}): "
          f"mediana={summary['normal']['median_ms']:.2f}ms, "
          f"p95={summary['normal']['p95_ms']:.2f}ms")
    print(f"HIGH    (n={summary['priority_high']['n']}): "
          f"mediana={summary['priority_high']['median_ms']:.2f}ms")
    print(f"Vantagem HIGH sobre NORMAL: {diff:.2f}ms")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E3: Priorização - DDS TRANSPORT_PRIORITY")
    parser.add_argument("--url", default="http://localhost:8080", help="URL do orquestrador")
    parser.add_argument("--carga", type=int, default=10, help="Carga NORMAL em req/s")
    parser.add_argument("--duracao", type=int, default=300, help="Duração em segundos (padrão: 300=5min)")
    parser.add_argument("--n", type=int, default=30, help="Número de injeções HIGH priority")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
