#!/usr/bin/env python3
"""
E2: Detecção de Falha - gRPC Health Check
==========================================
Mede tempo de detecção de falha usando polling do gRPC Health Check padrão.

Metodologia:
  1. Servidor gRPC (_grpc_server.py) é iniciado como subprocesso.
  2. Thread de polling chama /LLMService/HealthCheck a cada `intervalo_ms`.
  3. Falha é simulada (kill -9, SIGTERM ou SIGSTOP) no processo do servidor.
  4. T_fail = instante do kill.
  5. T_detect = instante em que o próximo health check falha.
  6. Tempo de detecção = T_detect - T_fail (ms).

Comparação com DDS DEADLINE:
  - gRPC: polling periódico em Python → detecção em até (intervalo_ms + RTT)
  - DDS:  deadline gerenciado em C++ → detecção determinística em ≤ D ms

Tipos de falha:
  - kill9:    kill -9 (terminação abrupta)
  - sigterm:  kill -15 (shutdown gracioso)
  - deadlock: kill -STOP (processo travado)

Usage:
    python E2_failure_detection_grpc.py --agent-url localhost:50051 --intervalo 1000 --tipo kill9 --n 50
"""

import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional
import grpc


def _json_serialize(obj: dict) -> bytes:
    return json.dumps(obj).encode("utf-8")


def _json_deserialize(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


class GRPCHealthChecker:
    """Detecta falha de servidor gRPC via polling de Health Check."""

    def __init__(self, endpoint: str, intervalo_ms: int):
        self.endpoint = endpoint
        self.intervalo_s = intervalo_ms / 1000.0
        self.server_pid: Optional[int] = None
        self._detection_time: Optional[float] = None
        self._running = False

    def _create_health_stub(self):
        channel = grpc.insecure_channel(self.endpoint)
        return channel.unary_unary(
            "/LLMService/HealthCheck",
            request_serializer=_json_serialize,
            response_deserializer=_json_deserialize,
        ), channel

    def start_grpc_server(self, backend_url: str) -> subprocess.Popen:
        """Inicia o servidor gRPC proxy como subprocesso."""
        server_script = Path(__file__).parent / "_grpc_server.py"
        host, port = self.endpoint.split(":")
        proc = subprocess.Popen(
            [sys.executable, str(server_script),
             "--backend", backend_url,
             "--port", port],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.server_pid = proc.pid
        time.sleep(1.5)  # Aguardar inicialização
        return proc

    def start_health_monitor(self, t_fail_ref: list):
        """
        Inicia thread de polling health check.
        t_fail_ref: lista de 1 elemento; quando preenchida, indica que o kill foi executado.
        """
        self._running = True
        self._detection_time = None

        def monitor():
            stub, channel = self._create_health_stub()
            try:
                while self._running:
                    # Aguardar intervalo antes de checar
                    time.sleep(self.intervalo_s)

                    # Se kill ainda não aconteceu, não detectar
                    if not t_fail_ref[0]:
                        continue

                    try:
                        response = stub({}, timeout=1.0)
                        if not response.get("serving", False):
                            self._detection_time = time.perf_counter()
                            self._running = False
                            break
                    except grpc.RpcError:
                        # Falha detectada: servidor não responde
                        self._detection_time = time.perf_counter()
                        self._running = False
                        break
                    except Exception:
                        self._detection_time = time.perf_counter()
                        self._running = False
                        break
            finally:
                channel.close()

        self._thread = threading.Thread(target=monitor, daemon=True)
        self._thread.start()

    def stop_health_monitor(self):
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2.0)


async def run_benchmark(args):
    """Executa benchmark de detecção de falha gRPC Health Check."""

    # Extrair host:port do endpoint
    endpoint = args.agent_url
    if endpoint.startswith("http"):
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        endpoint = f"{parsed.hostname}:{parsed.port or 50051}"

    checker = GRPCHealthChecker(endpoint=endpoint, intervalo_ms=args.intervalo)

    print(f"E2: Detecção de Falha - gRPC Health Check")
    print(f"Endpoint: {endpoint}")
    print(f"Intervalo: {args.intervalo}ms")
    print(f"Tipo de falha: {args.tipo}")
    print(f"Iterações: {args.n}")
    print("-" * 50)

    results = []

    for i in range(args.n):
        # Iniciar servidor gRPC
        server_proc = checker.start_grpc_server(args.backend)

        if server_proc.poll() is not None:
            print(f"Iteração {i+1}/{args.n}: ERRO - servidor gRPC não iniciou")
            continue

        # t_fail_ref: sinaliza para thread quando o kill aconteceu
        t_fail_ref = [None]

        # Iniciar polling health check
        checker.start_health_monitor(t_fail_ref)

        # Aguardar estabilização do monitor
        await asyncio.sleep(0.3)

        # Executar falha e registrar T_fail
        t_fail = time.perf_counter()
        t_fail_ref[0] = t_fail  # Sinalizar para thread de monitoramento

        if args.tipo == "kill9":
            server_proc.kill()
        elif args.tipo == "sigterm":
            server_proc.terminate()
        elif args.tipo == "deadlock":
            if sys.platform != "win32":
                server_proc.send_signal(signal.SIGSTOP)
            else:
                server_proc.kill()

        # Aguardar detecção (máximo: 3x o intervalo)
        timeout_s = (args.intervalo * 3) / 1000.0
        elapsed = 0.0
        while checker._detection_time is None and elapsed < timeout_s:
            await asyncio.sleep(0.01)
            elapsed += 0.01

        checker.stop_health_monitor()

        if checker._detection_time is not None:
            detection_ms = (checker._detection_time - t_fail) * 1000.0
        else:
            detection_ms = -1  # timeout

        results.append({
            "iteration": i + 1,
            "detection_time_ms": detection_ms,
            "tipo": args.tipo,
            "intervalo_ms": args.intervalo
        })

        status = f"{detection_ms:.2f}ms" if detection_ms > 0 else "TIMEOUT"
        print(f"Iteração {i+1}/{args.n}: {status}")

        # Limpar
        try:
            server_proc.kill()
        except Exception:
            pass
        server_proc.wait(timeout=2)

        await asyncio.sleep(0.5)

    # Estatísticas
    detection_times = [r["detection_time_ms"] for r in results if r["detection_time_ms"] > 0]
    timeout_count = sum(1 for r in results if r["detection_time_ms"] < 0)

    if not detection_times:
        print("ERRO: nenhuma detecção bem-sucedida.")
        return None

    summary = {
        "protocol": "GRPC_HEALTH_CHECK",
        "intervalo_ms": args.intervalo,
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

    csv_file = f"results/E2_GRPC_HEALTH_{args.tipo}_{args.intervalo}ms.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write("iteration,detection_time_ms,tipo,intervalo_ms\n")
        for r in results:
            f.write(f"{r['iteration']},{r['detection_time_ms']},{r['tipo']},{r['intervalo_ms']}\n")

    json_file = f"results/E2_GRPC_HEALTH_{args.tipo}_{args.intervalo}ms_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"Tempo médio de detecção: {summary['detection_mean_ms']:.2f}ms")
    print(f"Mediana:                 {summary['detection_median_ms']:.2f}ms")
    print(f"p95:                     {summary['detection_p95_ms']:.2f}ms")
    print(f"Timeouts:                {timeout_count}/{args.n}")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E2: Detecção de Falha - gRPC Health Check")
    parser.add_argument("--agent-url", default="localhost:50051",
                        help="Endereço do servidor gRPC (host:port)")
    parser.add_argument("--backend", default="http://localhost:8080",
                        help="URL do backend HTTP para o servidor gRPC proxy")
    parser.add_argument("--intervalo", type=int, default=1000,
                        help="Intervalo de health check em ms (1000, 5000, 10000)")
    parser.add_argument("--tipo", choices=["kill9", "sigterm", "deadlock"], default="kill9")
    parser.add_argument("--n", type=int, default=50)

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
