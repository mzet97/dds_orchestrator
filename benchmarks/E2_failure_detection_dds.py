#!/usr/bin/env python3
"""
E2: Detecção de Falha - DDS DEADLINE
=====================================
Mede o tempo de detecção de falha usando a política DEADLINE do CycloneDDS.

Metodologia:
  1. Subprocesso publicador envia heartbeats a cada periodo_ms/10 ms.
  2. DataReader configurado com DEADLINE = periodo_ms.
  3. Publicador é terminado (kill -9, SIGTERM ou SIGSTOP).
  4. A camada C++ do CycloneDDS detecta a ausência de mensagens e dispara
     o callback on_requested_deadline_missed via Listener.
  5. Tempo de detecção = T_detect - T_fail (ms).

Diferença em relação ao heartbeat HTTP/gRPC:
  - DDS: detecção determinística em exatamente D ms após a falha (camada C++).
  - HTTP/gRPC: detecção em até D ms após o próximo ciclo de polling Python.

Tipos de falha:
  - kill9:    kill -9 (terminação abrupta, sem shutdown gracioso)
  - sigterm:  kill -15 (shutdown gracioso)
  - deadlock: kill -STOP (processo travado, não envia mais mensagens)

Usage:
    python E2_failure_detection_dds.py --periodo 1000 --tipo kill9 --n 50 --domain 0
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
from typing import Dict, List, Optional


class DDSDeadlineDetector:
    """Detecta falha de agente via política DEADLINE do CycloneDDS."""

    def __init__(self, periodo_ms: int, domain_id: int = 0):
        self.periodo_ms = periodo_ms
        self.domain_id = domain_id
        self._detection_event: Optional[asyncio.Event] = None
        self._t_detect: Optional[float] = None
        self._dds_available = False

        # Inicializar CycloneDDS
        try:
            from cyclonedds.domain import DomainParticipant
            from cyclonedds.topic import Topic
            from cyclonedds.sub import DataReader, Subscriber
            from cyclonedds.qos import Qos, Policy
            from cyclonedds.util import duration
            from cyclonedds.core import Listener
            from cyclonedds.idl import IdlStruct
            from dataclasses import dataclass

            @dataclass
            class BenchmarkHeartbeat(IdlStruct, typename="BenchmarkHeartbeat"):
                agent_id: str = ""
                seq: int = 0
                timestamp: float = 0.0

            self.BenchmarkHeartbeat = BenchmarkHeartbeat

            self.participant = DomainParticipant(domain_id)
            self.topic = Topic(self.participant, "benchmark/heartbeat", BenchmarkHeartbeat)
            self._dds_available = True

        except ImportError as e:
            print(f"ERRO: cyclonedds não disponível: {e}", file=sys.stderr)
            print("Instale: pip install cyclonedds", file=sys.stderr)
            raise

    def _create_reader_with_deadline(self, event: asyncio.Event, loop: asyncio.AbstractEventLoop):
        """Cria DataReader com DEADLINE QoS e Listener."""
        from cyclonedds.sub import DataReader, Subscriber
        from cyclonedds.qos import Qos, Policy
        from cyclonedds.util import duration
        from cyclonedds.core import Listener

        detector_self = self

        class DeadlineListener(Listener):
            def on_requested_deadline_missed(self, reader, status):
                """Chamado pelo C++ do CycloneDDS quando DEADLINE é violado."""
                detector_self._t_detect = time.perf_counter()
                # Sinalizar asyncio event de forma thread-safe
                loop.call_soon_threadsafe(event.set)

        qos = Qos(Policy.Deadline(duration(milliseconds=self.periodo_ms)))
        self._listener = DeadlineListener()
        self._reader = DataReader(
            Subscriber(self.participant),
            self.topic,
            qos=qos,
            listener=self._listener
        )

    async def run_iteration(self, tipo: str) -> float:
        """
        Executa uma iteração: inicia publicador, espera conexão,
        simula falha e mede tempo de detecção.
        Retorna tempo de detecção em ms, ou -1 em caso de falha.
        """
        loop = asyncio.get_running_loop()
        self._detection_event = asyncio.Event()
        self._t_detect = None

        # Criar reader com DEADLINE e listener
        self._create_reader_with_deadline(self._detection_event, loop)

        # Iniciar publicador como subprocesso
        publisher_script = Path(__file__).parent / "_e2_heartbeat_publisher.py"
        pub_proc = subprocess.Popen(
            [sys.executable, str(publisher_script), str(self.periodo_ms), str(self.domain_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Aguardar publicador inicializar e conectar ao DDS
        await asyncio.sleep(1.0)

        # Confirmar que publicador está rodando
        if pub_proc.poll() is not None:
            stderr = pub_proc.stderr.read().decode()
            return -1  # publicador falhou ao iniciar

        # Executar falha e registrar T_fail
        t_fail = time.perf_counter()

        if tipo == "kill9":
            pub_proc.kill()           # SIGKILL - terminação abrupta
        elif tipo == "sigterm":
            pub_proc.terminate()      # SIGTERM - shutdown gracioso
        elif tipo == "deadlock":
            if sys.platform != "win32":
                pub_proc.send_signal(signal.SIGSTOP)  # processo travado
            else:
                pub_proc.kill()       # Windows: sem SIGSTOP, usar SIGKILL

        # Aguardar DEADLINE disparar (máximo: 3x o período)
        timeout_s = (self.periodo_ms * 3) / 1000.0
        try:
            await asyncio.wait_for(self._detection_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            # DEADLINE não disparou dentro do timeout esperado
            pub_proc.kill()
            return -1

        t_detect = self._t_detect
        detection_ms = (t_detect - t_fail) * 1000.0

        # Limpar
        try:
            pub_proc.kill()
        except Exception:
            pass
        pub_proc.wait(timeout=2)

        # Destruir reader para próxima iteração ter estado limpo
        del self._reader

        # Pequena pausa para DDS estabilizar entre iterações
        await asyncio.sleep(0.5)

        return detection_ms


async def run_benchmark(args):
    """Executa benchmark de detecção de falha com DDS DEADLINE."""

    print(f"E2: Detecção de Falha - DDS DEADLINE")
    print(f"Período DEADLINE: {args.periodo}ms")
    print(f"Tipo de falha: {args.tipo}")
    print(f"Iterações: {args.n}")
    print(f"DDS Domain: {args.domain}")
    print("-" * 50)

    try:
        detector = DDSDeadlineDetector(periodo_ms=args.periodo, domain_id=args.domain)
    except ImportError:
        sys.exit(1)

    results = []

    for i in range(args.n):
        detection_time = await detector.run_iteration(args.tipo)

        results.append({
            "iteration": i + 1,
            "detection_time_ms": detection_time,
            "tipo": args.tipo,
            "periodo_ms": args.periodo
        })

        status = f"{detection_time:.2f}ms" if detection_time > 0 else "TIMEOUT"
        print(f"Iteração {i+1}/{args.n}: {status}")

    # Estatísticas
    detection_times = [r["detection_time_ms"] for r in results if r["detection_time_ms"] > 0]
    timeout_count = sum(1 for r in results if r["detection_time_ms"] < 0)

    if not detection_times:
        print("ERRO: nenhuma detecção bem-sucedida. Verifique se cyclonedds está instalado.")
        return None

    summary = {
        "protocol": "DDS_DEADLINE",
        "periodo_ms": args.periodo,
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
    csv_file = f"results/E2_DDS_DEADLINE_{args.tipo}_{args.periodo}ms.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write("iteration,detection_time_ms,tipo,periodo_ms\n")
        for r in results:
            f.write(f"{r['iteration']},{r['detection_time_ms']},{r['tipo']},{r['periodo_ms']}\n")

    json_file = f"results/E2_DDS_DEADLINE_{args.tipo}_{args.periodo}ms_summary.json"
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
    parser = argparse.ArgumentParser(description="E2: Detecção de Falha - DDS DEADLINE")
    parser.add_argument("--periodo", type=int, default=1000,
                        help="Período DEADLINE em ms (1000, 5000, 10000)")
    parser.add_argument("--tipo", choices=["kill9", "sigterm", "deadlock"], default="kill9",
                        help="Tipo de falha")
    parser.add_argument("--n", type=int, default=50, help="Número de iterações")
    parser.add_argument("--domain", type=int, default=0, help="DDS Domain ID")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
