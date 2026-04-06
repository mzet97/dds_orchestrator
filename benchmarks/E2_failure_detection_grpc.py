#!/usr/bin/env python3
"""
E2: Detecção de Falha - gRPC Health Check Polling (GetStatus RPC)
==================================================================
Mede tempo de detecção de falha usando polling gRPC GetStatus.

Metodologia:
  1. Subprocesso _e2_grpc_health_server.py responde a GetStatus RPC.
  2. Thread de polling chama GetStatus a cada periodo_ms/10 ms.
  3. Servidor é terminado (kill -9, SIGTERM ou SIGSTOP).
  4. T_detect = instante em que GetStatus falha (gRPC UNAVAILABLE).
  5. Tempo de detecção = T_detect - T_fail (ms).

Comparação com DDS DEADLINE:
  - gRPC: polling periódico Python → detecção em até intervalo_poll ms
  - DDS:  DEADLINE gerenciado em C++ → detecção determinística em ≤ D ms

Tipos de falha:
  - kill9:    kill -9 (terminação abrupta)
  - sigterm:  kill -15 (shutdown gracioso)
  - deadlock: kill -STOP (processo travado)

Usage:
    python E2_failure_detection_grpc.py --periodo 1000 --tipo kill9 --n 10
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import statistics
from pathlib import Path
from typing import Optional

import grpc

sys.path.insert(0, str(Path(__file__).parent))
from proto import llama_service_pb2
from proto import llama_service_pb2_grpc


class GRPCHealthDetector:
    """Detecta falha de servidor via polling gRPC GetStatus."""

    def __init__(self, periodo_ms: int, port: int = 50099):
        self.periodo_ms = periodo_ms
        self.port = port
        self.poll_interval_s = (periodo_ms / 10) / 1000.0

    async def run_iteration(self, tipo: str) -> float:
        """
        Executa uma iteração: inicia servidor, polling, simula falha, mede detecção.
        Retorna tempo de detecção em ms, ou -1 em caso de falha.
        """
        # Iniciar servidor gRPC como subprocesso
        server_script = Path(__file__).parent / "_e2_grpc_health_server.py"
        proc = subprocess.Popen(
            [sys.executable, str(server_script), str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Aguardar servidor inicializar
        await asyncio.sleep(1.0)

        if proc.poll() is not None:
            stderr = proc.stderr.read().decode()
            print(f"  Servidor não iniciou: {stderr[:200]}")
            return -1

        # Confirmar que servidor responde
        endpoint = f"localhost:{self.port}"
        channel = grpc.insecure_channel(endpoint)
        stub = llama_service_pb2_grpc.LlamaServiceStub(channel)

        try:
            status = stub.GetStatus(llama_service_pb2.Empty(), timeout=5.0)
            if not status.ready:
                print("  Servidor não está ready")
                channel.close()
                proc.kill()
                proc.wait(timeout=2)
                return -1
        except Exception as e:
            print(f"  Servidor não responde: {e}")
            channel.close()
            proc.kill()
            proc.wait(timeout=2)
            return -1

        # Executar falha e registrar T_fail
        t_fail = time.perf_counter()

        if tipo == "kill9":
            proc.kill()
        elif tipo == "sigterm":
            proc.terminate()
        elif tipo == "deadlock":
            if sys.platform != "win32":
                proc.send_signal(signal.SIGSTOP)
            else:
                proc.kill()

        # Polling até detectar falha (máximo: 3x o período)
        timeout_s = (self.periodo_ms * 3) / 1000.0
        t_detect = None
        poll_start = time.perf_counter()

        while (time.perf_counter() - poll_start) < timeout_s:
            try:
                stub.GetStatus(llama_service_pb2.Empty(), timeout=self.poll_interval_s)
                # Servidor ainda respondendo
            except grpc.RpcError:
                t_detect = time.perf_counter()
                break
            except Exception:
                t_detect = time.perf_counter()
                break
            time.sleep(self.poll_interval_s)

        channel.close()

        # Cleanup
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass

        if t_detect is None:
            return -1

        detection_ms = (t_detect - t_fail) * 1000.0

        # Pequena pausa para porta liberar
        await asyncio.sleep(0.5)

        return detection_ms


async def run_benchmark(args):
    """Executa benchmark de detecção de falha com gRPC GetStatus polling."""

    poll_interval_ms = args.periodo / 10

    print(f"E2: Detecção de Falha - gRPC Health Check (GetStatus RPC)")
    print(f"Período equivalente: {args.periodo}ms (poll a cada {poll_interval_ms:.0f}ms)")
    print(f"Tipo de falha: {args.tipo}")
    print(f"Iterações: {args.n}")
    print("-" * 50)

    detector = GRPCHealthDetector(periodo_ms=args.periodo, port=args.port)
    results = []

    for i in range(args.n):
        detection_time = await detector.run_iteration(args.tipo)

        results.append({
            "iteration": i + 1,
            "detection_time_ms": detection_time,
            "tipo": args.tipo,
            "periodo_ms": args.periodo,
        })

        status = f"{detection_time:.2f}ms" if detection_time > 0 else "TIMEOUT"
        print(f"Iteração {i+1}/{args.n}: {status}")

    # Estatísticas
    detection_times = [r["detection_time_ms"] for r in results if r["detection_time_ms"] > 0]
    timeout_count = sum(1 for r in results if r["detection_time_ms"] < 0)

    if not detection_times:
        print("ERRO: nenhuma detecção bem-sucedida.")
        return None

    summary = {
        "protocol": "gRPC_HEALTH_CHECK",
        "periodo_ms": args.periodo,
        "poll_interval_ms": poll_interval_ms,
        "tipo_falha": args.tipo,
        "n_total": args.n,
        "n_successful": len(detection_times),
        "n_timeout": timeout_count,
        "detection_mean_ms": round(statistics.mean(detection_times), 2),
        "detection_median_ms": round(statistics.median(detection_times), 2),
        "detection_stdev_ms": round(statistics.stdev(detection_times), 2) if len(detection_times) > 1 else 0,
        "detection_p95_ms": round(sorted(detection_times)[int(len(detection_times) * 0.95)], 2),
        "detection_min_ms": round(min(detection_times), 2),
        "detection_max_ms": round(max(detection_times), 2),
    }

    # Salvar CSV
    csv_file = f"results/E2_gRPC_HEALTH_{args.tipo}_{args.periodo}ms.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write("iteration,detection_time_ms,tipo,periodo_ms\n")
        for r in results:
            f.write(f"{r['iteration']},{r['detection_time_ms']},{r['tipo']},{r['periodo_ms']}\n")

    json_file = f"results/E2_gRPC_HEALTH_{args.tipo}_{args.periodo}ms_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResultados:")
    print(f"Tempo médio de detecção: {summary['detection_mean_ms']:.2f}ms")
    print(f"Mediana:                 {summary['detection_median_ms']:.2f}ms")
    print(f"p95:                     {summary['detection_p95_ms']:.2f}ms")
    print(f"Intervalo de poll:       {poll_interval_ms:.0f}ms")
    print(f"Timeouts:                {timeout_count}/{args.n}")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E2: Detecção de Falha - gRPC Health Check")
    parser.add_argument("--periodo", type=int, default=1000,
                        help="Período equivalente em ms (poll a cada periodo/10)")
    parser.add_argument("--tipo", choices=["kill9", "sigterm", "deadlock"], default="kill9",
                        help="Tipo de falha")
    parser.add_argument("--n", type=int, default=10, help="Número de iterações")
    parser.add_argument("--port", type=int, default=50099,
                        help="Porta do servidor gRPC de health check")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
