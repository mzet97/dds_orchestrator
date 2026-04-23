#!/usr/bin/env python3
"""
E4: Escalabilidade Multi-Agente - gRPC Nativo
==============================================
Avalia throughput e latência com múltiplos clientes via gRPC nativo (protobuf)
conectando diretamente ao llama-server.

Arquitetura medida:
    Clientes → gRPC (protobuf) → llama-server(s) (--enable-grpc)

Canais gRPC persistentes (HTTP/2 multiplexing) por agente.

Design experimental (conforme dissertação):
  Fase A: 1 agente, clientes = 1, 2, 4, 8
  Fase B: 2 agentes, clientes = 1, 2, 4, 8
  N = 50 requisições por cliente por configuração

Usage:
    # Fase A (1 agente):
    python E4_scalability_grpc.py --agentes 192.168.1.60:50051 --n 50

    # Fase B (2 agentes):
    python E4_scalability_grpc.py --agentes 192.168.1.60:50051,192.168.1.61:50051 --n 50
"""

import argparse
import asyncio
import json
import random
import sys
import time
import statistics
import uuid
from pathlib import Path
from typing import Dict, List

import grpc
import psutil

sys.path.insert(0, str(Path(__file__).parent))
from proto import llama_service_pb2
from proto import llama_service_pb2_grpc


CLIENT_COUNTS = [1, 2, 4, 8]


class ScalabilityBenchmarkGRPC:
    """Benchmark de escalabilidade com gRPC nativo (protobuf)."""

    def __init__(self, agentes: List[str]):
        self.agentes = agentes
        # Canal persistente por agente (reutilizado — HTTP/2 multiplexing)
        self._channels = {
            addr: grpc.insecure_channel(
                addr,
                options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)],
            )
            for addr in agentes
        }
        self._stubs = {
            addr: llama_service_pb2_grpc.LlamaServiceStub(ch)
            for addr, ch in self._channels.items()
        }

    def close(self):
        for ch in self._channels.values():
            ch.close()

    def _single_request(self, addr: str) -> Dict:
        """Requisição síncrona ao llama-server via gRPC nativo."""
        stub = self._stubs[addr]
        request = llama_service_pb2.ChatCompletionRequest(
            request_id=str(uuid.uuid4()),
            model="phi4-mini",
            messages=[llama_service_pb2.ChatMessage(role="user", content="O que e 2+2?")],
            max_tokens=20,
            stream=False,
        )
        start = time.perf_counter()
        try:
            stub.Chat(request, timeout=120)
            end = time.perf_counter()
            return {"success": True, "latency_ms": (end - start) * 1000}
        except grpc.RpcError as e:
            end = time.perf_counter()
            return {"success": False, "latency_ms": (end - start) * 1000,
                    "error": str(e.code())}
        except Exception as e:
            end = time.perf_counter()
            return {"success": False, "latency_ms": (end - start) * 1000, "error": str(e)}

    async def run_concurrent(self, num_clientes: int, n_per_client: int) -> List[Dict]:
        """Executa num_clientes × n_per_client requisições concorrentes."""
        total = num_clientes * n_per_client
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self._single_request, random.choice(self.agentes))
            for _ in range(total)
        ]
        results = await asyncio.gather(*tasks)
        return list(results)


async def run_benchmark(args):
    """Executa benchmark E4 completo (Fase A e Fase B)."""

    raw_agentes = [a.strip() for a in args.agentes.split(",") if a.strip()]
    agentes = []
    for a in raw_agentes:
        if a.startswith("http"):
            from urllib.parse import urlparse
            parsed = urlparse(a)
            agentes.append(f"{parsed.hostname}:{parsed.port or 50051}")
        else:
            agentes.append(a)

    num_agentes = len(agentes)
    phase = "A" if num_agentes == 1 else "B" if num_agentes == 2 else f"{num_agentes}ag"

    benchmark = ScalabilityBenchmarkGRPC(agentes)

    print(f"E4: Escalabilidade Multi-Agente - gRPC Nativo (protobuf)")
    print(f"Agentes ({num_agentes}): {agentes}")
    print(f"Fase: {phase}")
    print(f"Configurações de clientes: {CLIENT_COUNTS}")
    print(f"Requisições por cliente: {args.n}")
    print("=" * 60)

    all_summaries = []
    Path("results").mkdir(exist_ok=True)

    try:
        for num_clientes in CLIENT_COUNTS:
            print(f"\n--- Fase {phase}: {num_agentes} agente(s), {num_clientes} cliente(s) ---")

            results = await benchmark.run_concurrent(num_clientes, args.n)

            latencies = sorted([r["latency_ms"] for r in results if r.get("success")])
            successes = len(latencies)
            total = len(results)

            if not latencies:
                print("  ERRO: nenhuma requisição bem-sucedida")
                all_summaries.append({"phase": phase, "num_clientes": num_clientes, "error": "no data"})
                continue

            throughput = successes / (latencies[-1] / 1000.0) if latencies[-1] > 0 else 0

            summary = {
                "protocol": "gRPC_NATIVE",
                "phase": phase,
                "num_agentes": num_agentes,
                "num_clientes": num_clientes,
                "total_requests": total,
                "successful_requests": successes,
                "throughput_req_s": round(throughput, 3),
                "latency_p50_ms": round(statistics.median(latencies), 2),
                "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
                "latency_p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
                "latency_mean_ms": round(statistics.mean(latencies), 2),
                "latency_stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
                "cpu_pct": round(psutil.cpu_percent(interval=0.1), 1),
                "mem_mb": round(psutil.virtual_memory().used / 1024 / 1024, 1)
            }
            all_summaries.append(summary)

            print(f"  Throughput: {summary['throughput_req_s']:.2f} req/s")
            print(f"  Latência p50: {summary['latency_p50_ms']:.2f}ms  "
                  f"p95: {summary['latency_p95_ms']:.2f}ms  "
                  f"p99: {summary['latency_p99_ms']:.2f}ms")
            print(f"  Sucesso: {successes}/{total}")

            csv_file = f"results/E4_gRPC_NATIVE_fase{phase}_{num_agentes}ag_{num_clientes}cl.csv"
            with open(csv_file, "w") as f:
                f.write("latency_ms,success\n")
                for r in results:
                    f.write(f"{r['latency_ms']},{1 if r.get('success') else 0}\n")

            await asyncio.sleep(2.0)

    finally:
        benchmark.close()

    json_file = f"results/E4_gRPC_NATIVE_fase{phase}_{num_agentes}ag_summary.json"
    with open(json_file, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Resumo Fase {phase} ({num_agentes} agente(s)):")
    print(f"{'Clientes':>8} {'p50(ms)':>10} {'p95(ms)':>10} {'req/s':>8}")
    for s in all_summaries:
        if "error" not in s:
            print(f"{s['num_clientes']:>8} {s['latency_p50_ms']:>10.1f} "
                  f"{s['latency_p95_ms']:>10.1f} {s['throughput_req_s']:>8.2f}")

    print(f"\nJSON: {json_file}")
    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="E4: Escalabilidade - gRPC Nativo")
    parser.add_argument("--agentes", default="localhost:50051",
                        help="Endereços gRPC dos llama-servers separados por vírgula. "
                             "1 agente = Fase A, 2 agentes = Fase B")
    parser.add_argument("--n", type=int, default=50,
                        help="Requisições por cliente por configuração")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
