#!/usr/bin/env python3
"""
E3: Priorização sob Carga - gRPC + heapq
========================================
Mede latência de mensagens HIGH vs NORMAL com fila de prioridade Python (heapq) via gRPC.

Requer: servidor gRPC rodando (_grpc_server.py --backend http://... --port 50051)

Metodologia (idêntica ao E3_priority_http.py, mas transporte é gRPC/HTTP/2):
  - Carga sustentada: 10 req/s com prioridade NORMAL
  - A cada inject_interval segundos: injeção de 1 requisição HIGH
  - Prioridade implementada via heapq em nível de aplicação (mesmo que HTTP)
  - max_tokens=5: minimiza inferência, isola latência de transporte+fila

Usage:
    python _grpc_server.py --backend http://localhost:8080 --port 50051 &
    python E3_priority_grpc.py --endpoint localhost:50051 --carga 10 --n 30 --duracao 300
"""

import argparse
import asyncio
import heapq
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


class PriorityQueueGRPC:
    """Fila de prioridade com heapq (aplicação Python) para gRPC."""

    def __init__(self):
        self.queue = []
        self.counter = 0

    def enqueue(self, item: Dict, priority: int):
        heapq.heappush(self.queue, (-priority, self.counter, item))
        self.counter += 1

    def dequeue(self) -> Dict:
        if self.queue:
            _, _, item = heapq.heappop(self.queue)
            return item
        return None

    def __len__(self):
        return len(self.queue)


class PriorityBenchmarkGRPC:
    """Benchmark de priorização gRPC com heapq."""

    def __init__(self, endpoint: str, carga_req_s: int = 10):
        self.endpoint = endpoint
        self.carga_req_s = carga_req_s
        self.normal_results: List[Dict] = []
        self.stop_load = False
        self._channel = grpc.insecure_channel(endpoint)
        self._stub = self._channel.unary_unary(
            "/LLMService/Chat",
            request_serializer=_json_serialize,
            response_deserializer=_json_deserialize,
        )

    def close(self):
        self._channel.close()

    def _send_grpc_request(self, priority: int) -> Dict:
        """Envia requisição gRPC síncrona com priority no payload."""
        payload = {
            "model": "phi4-mini",
            "content": "ok",
            "max_tokens": 5,
            "priority": priority
        }
        send_time = time.perf_counter()
        try:
            self._stub(payload, timeout=30)
            recv_time = time.perf_counter()
            return {
                "priority": "HIGH" if priority >= 10 else "NORMAL",
                "send_time": send_time,
                "latency_ms": (recv_time - send_time) * 1000
            }
        except grpc.RpcError as e:
            recv_time = time.perf_counter()
            return {
                "priority": "HIGH" if priority >= 10 else "NORMAL",
                "send_time": send_time,
                "latency_ms": (recv_time - send_time) * 1000,
                "error": str(e.code())
            }

    async def run_background_load(self, duration_s: int):
        """Gera carga sustentada de requisições NORMAL."""
        interval = 1.0 / self.carga_req_s
        start = time.perf_counter()
        loop = asyncio.get_event_loop()
        count = 0

        while (time.perf_counter() - start) < duration_s and not self.stop_load:
            result = await loop.run_in_executor(None, self._send_grpc_request, 1)
            result["elapsed_s"] = time.perf_counter() - start
            self.normal_results.append(result)
            count += 1
            await asyncio.sleep(interval)

        return count

    async def inject_priority_message(self) -> Dict:
        """Injeta mensagem HIGH e retorna latência."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._send_grpc_request, 10)


async def run_benchmark(args):
    """Executa benchmark de priorização gRPC."""

    endpoint = args.endpoint
    if endpoint.startswith("http"):
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        endpoint = f"{parsed.hostname}:{parsed.port or 50051}"

    benchmark = PriorityBenchmarkGRPC(endpoint=endpoint, carga_req_s=args.carga)

    print(f"E3: Priorização - gRPC + heapq")
    print(f"Endpoint: {endpoint}")
    print(f"Carga NORMAL: {args.carga} req/s")
    print(f"Duração: {args.duracao}s")
    print(f"Injeções HIGH: {args.n}")
    print(f"Intervalo entre injeções: {args.duracao / args.n:.1f}s")
    print("-" * 50)

    inject_interval = args.duracao / args.n

    load_task = asyncio.create_task(benchmark.run_background_load(args.duracao))
    await asyncio.sleep(1.0)

    priority_results = []
    for i in range(args.n):
        await asyncio.sleep(inject_interval)

        if load_task.done():
            print(f"Aviso: carga background terminou antes das {args.n} injeções")
            break

        result = await benchmark.inject_priority_message()
        priority_results.append(result)
        print(f"Injeção {i+1}/{args.n}: latência={result['latency_ms']:.2f}ms "
              f"({'OK' if 'error' not in result else 'ERRO'})")

    benchmark.stop_load = True
    try:
        await asyncio.wait_for(load_task, timeout=5.0)
    except asyncio.TimeoutError:
        pass
    finally:
        benchmark.close()

    normal_latencies = [r["latency_ms"] for r in benchmark.normal_results
                        if "error" not in r]
    priority_latencies = [r["latency_ms"] for r in priority_results
                          if "error" not in r]

    if not normal_latencies or not priority_latencies:
        print("ERRO: dados insuficientes para análise")
        return None

    summary = {
        "protocol": "GRPC_HEAPQ",
        "endpoint": endpoint,
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

    diff = summary["normal"]["median_ms"] - summary["priority_high"]["median_ms"]
    summary["priority_advantage_ms"] = round(diff, 4)

    csv_file = f"results/E3_GRPC_HEAPQ_carga{args.carga}.csv"
    Path("results").mkdir(exist_ok=True)

    all_results = [(r, "NORMAL") for r in benchmark.normal_results] + \
                  [(r, "HIGH") for r in priority_results]

    with open(csv_file, "w") as f:
        f.write("priority,latency_ms,elapsed_s,error\n")
        for r, prio in all_results:
            elapsed = r.get("elapsed_s", r.get("send_time", 0))
            error = r.get("error", "")
            f.write(f"{prio},{r['latency_ms']},{elapsed},{error}\n")

    json_file = f"results/E3_GRPC_HEAPQ_carga{args.carga}_summary.json"
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
    parser = argparse.ArgumentParser(description="E3: Priorização - gRPC + heapq")
    parser.add_argument("--endpoint", default="localhost:50051",
                        help="Endereço do servidor gRPC")
    parser.add_argument("--url", dest="endpoint",
                        help="Alias para --endpoint (compatibilidade)")
    parser.add_argument("--carga", type=int, default=10)
    parser.add_argument("--duracao", type=int, default=300)
    parser.add_argument("--n", type=int, default=30)

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
