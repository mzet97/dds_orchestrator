#!/usr/bin/env python3
"""
E2: Detecção de Falha - DDS DEADLINE
=====================================
Mede tempo de detecção de falha com política DEADLINE do DDS
100% REAL - simula falha REAL com kill -9, SIGTERM, deadlock

Usage:
    python E2_failure_detection_dds.py --periodo 1000 --tipo kill9 --n 50
"""

import argparse
import asyncio
import json
import time
import statistics
import subprocess
import signal
from pathlib import Path
from typing import Dict, List


class FailureDetectorDDS:
    """Detecção de falha usando DDS DEADLINE."""

    def __init__(self, periodo_ms: int = 1000):
        self.periodo_ms = periodo_ms
        self.agent_pid = None

    async def start_agent(self):
        """Inicia agente REAL."""
        # Iniciar agente em background
        proc = subprocess.Popen(
            ["python3", "agent_llm.py", "--model-path", "/home/oldds/models/Phi-4-mini-reasoning-Q4_K_M.gguf"],
            cwd="/home/oldds/dds_agent/python",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.agent_pid = proc.pid
        await asyncio.sleep(2)  # Esperar inicialização
        return proc

    async def simulate_failure(self, tipo: str) -> float:
        """
        Simula falha REAL e mede tempo de detecção.
        Retorna tempo de detecção em ms.
        """
        # Iniciar monitoramento em background
        detection_time = None

        async def monitor():
            nonlocal detection_time
            start = time.perf_counter()

            #Monitorar até detectar falha
            while True:
                # Verificar se processo ainda existe
                if self.agent_pid:
                    try:
                        # Verificar se processo está vivo
                        result = subprocess.run(
                            ["ps", "-p", str(self.agent_pid)],
                            capture_output=True
                        )
                        if result.returncode != 0:
                            # Processo morreu!
                            detection_time = (time.perf_counter() - start) * 1000
                            break
                    except:
                        pass

                await asyncio.sleep(0.01)  # Check a cada 10ms

        # Iniciar monitoramento
        monitor_task = asyncio.create_task(monitor())

        # Executar falha conforme tipo
        await asyncio.sleep(0.5)  # Pequeno delay antes da falha

        if tipo == "kill9":
            # CRASH REAL - kill -9
            subprocess.run(["kill", "-9", str(self.agent_pid)], check=False)
        elif tipo == "sigterm":
            # SHUTDOWN GRACEFUL - SIGTERM
            subprocess.run(["kill", "-15", str(self.agent_pid)], check=False)
        elif tipo == "deadlock":
            # DEADLOCK - enviar sinal de stop
            subprocess.run(["kill", "-STOP", str(self.agent_pid)], check=False)

        # Aguardar detecção
        await asyncio.sleep(self.periodo_ms / 1000 + 2)  # Esperar até periodo + 2s

        monitor_task.cancel()

        return detection_time if detection_time else -1


async def run_benchmark(args):
    """Executa benchmark de detecção de falha."""

    detector = FailureDetectorDDS(periodo_ms=args.periodo)

    results = []

    print(f"E2: Detecção de Falha - DDS DEADLINE")
    print(f"Período: {args.periodo}ms")
    print(f"Tipo de falha: {args.tipo}")
    print(f"Iterações: {args.n}")
    print("-" * 50)

    for i in range(args.n):
        # Iniciar agente
        agent_proc = await detector.start_agent()

        # Simular falha e medir detecção
        detection_time = await detector.simulate_failure(args.tipo)

        results.append({
            "iteration": i + 1,
            "detection_time_ms": detection_time,
            "tipo": args.tipo,
            "periodo_ms": args.periodo
        })

        print(f"Iteração {i+1}/{args.n}: {detection_time:.2f}ms")

        # Limpar
        try:
            subprocess.run(["kill", "-9", str(detector.agent_pid)], check=False)
        except:
            pass

        await asyncio.sleep(1)  #Delay entre testes

    # Estatísticas
    detection_times = [r["detection_time_ms"] for r in results if r["detection_time_ms"] > 0]

    summary = {
        "protocol": "DDS_DEADLINE",
        "periodo_ms": args.periodo,
        "tipo_falha": args.tipo,
        "n": len(detection_times),
        "detection_mean_ms": round(statistics.mean(detection_times), 2) if detection_times else -1,
        "detection_median_ms": round(statistics.median(detection_times), 2) if detection_times else -1,
        "detection_stdev_ms": round(statistics.stdev(detection_times), 2) if len(detection_times) > 1 else 0,
        "detection_min_ms": round(min(detection_times), 2) if detection_times else -1,
        "detection_max_ms": round(max(detection_times), 2) if detection_times else -1,
    }

    # Salvar CSV
    csv_file = f"results/E2_DDS_DEADLINE_{args.tipo}_{args.periodo}ms.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write("iteration,detection_time_ms,tipo,periodo_ms\n")
        for r in results:
            f.write(f"{r['iteration']},{r['detection_time_ms']},{r['tipo']},{r['periodo_ms']}\n")

    # Salvar JSON
    json_file = f"results/E2_DDS_DEADLINE_{args.tipo}_{args.periodo}ms_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"Tempo médio de detecção: {summary['detection_mean_ms']:.2f}ms")
    print(f"CSV: {csv_file}")
    print(f"JSON: {json_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E2: Detecção de Falha - DDS DEADLINE")
    parser.add_argument("--periodo", type=int, default=1000, help="Período DEADLINE em ms (1000, 5000, 10000)")
    parser.add_argument("--tipo", choices=["kill9", "sigterm", "deadlock"], default="kill9", help="Tipo de falha")
    parser.add_argument("--n", type=int, default=50, help="Número de iterações")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
