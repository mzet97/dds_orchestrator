#!/usr/bin/env python3
"""
E2: Detecção de Falha - DDS DEADLINE + LIVELINESS
===================================================
Mede o tempo de detecção de falha usando políticas QoS do CycloneDDS.

Mecanismos testados:
  - DEADLINE: detecta ausência de dados (publicações não chegam no prazo).
    Tempo de detecção ≈ periodo_ms (by design).
  - LIVELINESS AUTOMATIC: detecta falha de processo (lease expira sem heartbeat).
    Tempo de detecção ≈ lease_duration (configurável).

Metodologia:
  1. Subprocesso publicador envia heartbeats a cada periodo_ms/10 ms.
  2. DataReader configurado com DEADLINE = periodo_ms e LIVELINESS = lease_ms.
  3. Publicador é terminado (kill -9, SIGTERM ou SIGSTOP).
  4. T_detect = primeiro callback disparado (DEADLINE ou LIVELINESS).
  5. Tempo de detecção = T_detect - T_fail (ms).

Comparação com gRPC:
  - gRPC: TCP socket fecha imediatamente → detecção em ~5ms.
  - DDS DEADLINE: espera período completo → detecção em ~D ms.
  - DDS LIVELINESS: espera lease expirar → detecção em ~lease_ms.

Tipos de falha:
  - kill9:    kill -9 (terminação abrupta)
  - sigterm:  kill -15 (shutdown gracioso)
  - deadlock: kill -STOP (processo travado)

Usage:
    python E2_failure_detection_dds.py --periodo 1000 --lease 200 --n 10 --domain 0
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


class DDSFailureDetector:
    """Detecta falha via DEADLINE e/ou LIVELINESS do CycloneDDS."""

    def __init__(self, periodo_ms: int, lease_ms: int, domain_id: int = 0):
        self.periodo_ms = periodo_ms
        self.lease_ms = lease_ms
        self.domain_id = domain_id
        self._detection_event: Optional[asyncio.Event] = None
        self._t_detect: Optional[float] = None
        self._detected_by: Optional[str] = None
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

    def _create_reader(self, event: asyncio.Event, loop: asyncio.AbstractEventLoop):
        """Cria DataReader com DEADLINE + LIVELINESS QoS e Listener."""
        from cyclonedds.sub import DataReader, Subscriber
        from cyclonedds.qos import Qos, Policy
        from cyclonedds.util import duration
        from cyclonedds.core import Listener

        detector_self = self

        class FailureListener(Listener):
            def on_requested_deadline_missed(self, reader, status):
                """Chamado pelo C++ do CycloneDDS quando DEADLINE é violado."""
                if detector_self._t_detect is None:
                    detector_self._t_detect = time.perf_counter()
                    detector_self._detected_by = "DEADLINE"
                    loop.call_soon_threadsafe(event.set)

            def on_liveliness_changed(self, reader, status):
                """Chamado pelo C++ do CycloneDDS quando LIVELINESS muda."""
                # alive_count diminui = writer morreu
                if status.alive_count == 0 and status.alive_count_change < 0:
                    if detector_self._t_detect is None:
                        detector_self._t_detect = time.perf_counter()
                        detector_self._detected_by = "LIVELINESS"
                        loop.call_soon_threadsafe(event.set)

        qos_policies = [
            Policy.Reliability.Reliable(duration(seconds=10)),
            Policy.Deadline(duration(milliseconds=self.periodo_ms)),
        ]

        # Add LIVELINESS if lease_ms > 0
        if self.lease_ms > 0:
            qos_policies.append(
                Policy.Liveliness.Automatic(lease_duration=duration(milliseconds=self.lease_ms))
            )

        qos = Qos(*qos_policies)
        self._listener = FailureListener()
        self._reader = DataReader(
            Subscriber(self.participant),
            self.topic,
            qos=qos,
            listener=self._listener
        )

    async def run_iteration(self, tipo: str) -> dict:
        """
        Executa uma iteração: inicia publicador, espera conexão,
        simula falha e mede tempo de detecção.
        Retorna dict com detection_time_ms e detected_by, ou detection_time_ms=-1.
        """
        loop = asyncio.get_running_loop()
        self._detection_event = asyncio.Event()
        self._t_detect = None
        self._detected_by = None

        # Criar reader com DEADLINE + LIVELINESS e listener
        self._create_reader(self._detection_event, loop)

        # Iniciar publicador como subprocesso
        publisher_script = Path(__file__).parent / "_e2_heartbeat_publisher.py"
        pub_proc = subprocess.Popen(
            [sys.executable, str(publisher_script), str(self.periodo_ms), str(self.domain_id), str(self.lease_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Aguardar publicador inicializar e conectar ao DDS
        await asyncio.sleep(1.0)

        # Confirmar que publicador está rodando
        if pub_proc.poll() is not None:
            stderr = pub_proc.stderr.read().decode()
            return {"detection_time_ms": -1, "detected_by": "NONE"}

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

        # Aguardar detecção (máximo: 3x o período ou 3x o lease, o que for maior)
        max_wait = max(self.periodo_ms, self.lease_ms) * 3
        timeout_s = max_wait / 1000.0
        try:
            await asyncio.wait_for(self._detection_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            self._cleanup_iteration(pub_proc)
            return {"detection_time_ms": -1, "detected_by": "TIMEOUT"}

        t_detect = self._t_detect
        detection_ms = (t_detect - t_fail) * 1000.0

        self._cleanup_iteration(pub_proc)
        return {"detection_time_ms": detection_ms, "detected_by": self._detected_by}

    def _cleanup_iteration(self, pub_proc):
        """Limpeza determinística entre iterações.

        Bug 1 fix: o cleanup anterior usava ``del self._reader`` + sleep 0.5s,
        que não destrói o reader DDS imediatamente (depende de GC) e deixa o
        listener ainda registrado. Para sigterm/deadlock o publisher demora a
        morrer e os dados continuam chegando na entidade DDS antiga, fazendo
        a próxima iteração começar com estado contaminado e o callback
        DEADLINE/LIVELINESS nunca dispara (pattern "TIMEOUT (NONE)").

        Aqui:
          - SIGKILL forçado e ``wait`` para garantir o publisher morre.
          - ``set_listener(None)`` desregistra antes de fechar.
          - ``close()`` libera as entidades CycloneDDS imediatamente.
          - ``gc.collect()`` força liberação dos handles Python.
        """
        import gc

        # garantir que o publisher subprocess realmente morreu, mesmo com
        # SIGTERM/SIGSTOP que dão chance de cleanup ao filho
        try:
            pub_proc.kill()
        except Exception:
            pass
        try:
            pub_proc.wait(timeout=3)
        except Exception:
            pass

        # tear down DDS reader determinístico
        try:
            self._reader.set_listener(None)
        except Exception:
            pass
        try:
            self._reader.close()
        except Exception:
            pass
        self._reader = None
        self._listener = None
        gc.collect()

        # Sleep maior para CycloneDDS reciclar entidades + libera-thread do
        # listener que está em mid-callback no SIGSTOP/SIGTERM cases.
        # Não usamos await aqui (síncrono) — o caller ainda fará outro sleep.
        time.sleep(2.0)


async def run_benchmark(args):
    """Executa benchmark de detecção de falha com DDS DEADLINE + LIVELINESS."""

    print(f"E2: Detecção de Falha - DDS (DEADLINE + LIVELINESS)")
    print(f"Período DEADLINE: {args.periodo}ms")
    print(f"Lease LIVELINESS: {args.lease}ms (0=disabled)")
    print(f"Tipo de falha: {args.tipo}")
    print(f"Iterações: {args.n}")
    print(f"DDS Domain: {args.domain}")
    print("-" * 60)

    try:
        detector = DDSFailureDetector(
            periodo_ms=args.periodo,
            lease_ms=args.lease,
            domain_id=args.domain,
        )
    except ImportError:
        sys.exit(1)

    results = []

    for i in range(args.n):
        result = await detector.run_iteration(args.tipo)

        results.append({
            "iteration": i + 1,
            "detection_time_ms": result["detection_time_ms"],
            "detected_by": result["detected_by"],
            "tipo": args.tipo,
            "periodo_ms": args.periodo,
            "lease_ms": args.lease,
        })

        dt = result["detection_time_ms"]
        by = result["detected_by"]
        status = f"{dt:.2f}ms ({by})" if dt > 0 else f"TIMEOUT ({by})"
        print(f"Iteração {i+1}/{args.n}: {status}")

    # Estatísticas
    detection_times = [r["detection_time_ms"] for r in results if r["detection_time_ms"] > 0]
    timeout_count = sum(1 for r in results if r["detection_time_ms"] < 0)
    detected_by_counts = {}
    for r in results:
        by = r["detected_by"]
        detected_by_counts[by] = detected_by_counts.get(by, 0) + 1

    if not detection_times:
        print("ERRO: nenhuma detecção bem-sucedida. Verifique se cyclonedds está instalado.")
        return None

    summary = {
        "protocol": "DDS_DEADLINE_LIVELINESS",
        "periodo_ms": args.periodo,
        "lease_ms": args.lease,
        "tipo_falha": args.tipo,
        "n_total": args.n,
        "n_successful": len(detection_times),
        "n_timeout": timeout_count,
        "detected_by_counts": detected_by_counts,
        "detection_mean_ms": round(statistics.mean(detection_times), 2),
        "detection_median_ms": round(statistics.median(detection_times), 2),
        "detection_stdev_ms": round(statistics.stdev(detection_times), 2) if len(detection_times) > 1 else 0,
        "detection_p95_ms": round(sorted(detection_times)[int(len(detection_times) * 0.95)], 2),
        "detection_min_ms": round(min(detection_times), 2),
        "detection_max_ms": round(max(detection_times), 2),
    }

    # Salvar CSV
    lease_label = f"_lease{args.lease}ms" if args.lease > 0 else ""
    csv_file = f"results/E2_DDS_{args.tipo}_{args.periodo}ms{lease_label}.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write("iteration,detection_time_ms,detected_by,tipo,periodo_ms,lease_ms\n")
        for r in results:
            f.write(f"{r['iteration']},{r['detection_time_ms']},{r['detected_by']},"
                    f"{r['tipo']},{r['periodo_ms']},{r['lease_ms']}\n")

    json_file = csv_file.replace(".csv", "_summary.json")
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResultados:")
    print(f"Tempo médio de detecção: {summary['detection_mean_ms']:.2f}ms")
    print(f"Mediana:                 {summary['detection_median_ms']:.2f}ms")
    print(f"p95:                     {summary['detection_p95_ms']:.2f}ms")
    print(f"Detecção por mecanismo:  {detected_by_counts}")
    print(f"Timeouts:                {timeout_count}/{args.n}")
    print(f"\nCSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E2: Detecção de Falha - DDS DEADLINE + LIVELINESS")
    parser.add_argument("--periodo", type=int, default=1000,
                        help="Período DEADLINE em ms (default: 1000)")
    parser.add_argument("--lease", type=int, default=200,
                        help="Lease LIVELINESS em ms (0=disabled, default: 200)")
    parser.add_argument("--tipo", choices=["kill9", "sigterm", "deadlock"], default="kill9",
                        help="Tipo de falha")
    parser.add_argument("--n", type=int, default=1000, help="Número de iterações (v3: N=1000)")
    parser.add_argument("--domain", type=int, default=0, help="DDS Domain ID")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
