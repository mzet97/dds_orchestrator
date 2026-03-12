#!/usr/bin/env python3
"""
E2: Detecção de Falha - HTTP Heartbeat
======================================
Mede tempo de detecção com heartbeat HTTP
100% REAL - usa threads para polling

Usage:
    python E2_failure_detection_http.py --intervalo 1000 --tipo kill9 --n 50
"""

import argparse
import asyncio
import json
import time
import statistics
import subprocess
import threading
from pathlib import Path
from typing import Dict, List
import requests


class HeartbeatDetector:
    """Detecção de falha via HTTP Heartbeat."""

    def __init__(self, agent_url: str, intervalo_ms: int = 1000):
        self.agent_url = agent_url
        self.intervalo = intervalo_ms / 1000.0  # seconds
        self.agent_pid = None
        self.detection_time = None
        self.running = False

    def start_agent(self):
        """Inicia agente REAL."""
        proc = subprocess.Popen(
            ["python3", "agent_llm.py", "--model-path", "../models/phi4-mini.gguf"],
            cwd="/mnt/e/TI/git/tese/dds_agent/python",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.agent_pid = proc.pid
        time.sleep(2)  # Esperar inicialização
        return proc

    def start_heartbeat_monitor(self):
        """Inicia thread de heartbeat polling."""
        self.running = True
        self.detection_time = None

        def monitor():
            start = time.perf_counter()
            while self.running:
                try:
                    # HTTP Health check
                    resp = requests.get(f"{self.agent_url}/health", timeout=1)
                    if resp.status_code != 200:
                        # Falha detectada!
                        self.detection_time = (time.perf_counter() - start) * 1000  # ms
                        self.running = False
                        break
                except:
                    # Falha detectada!
                    self.detection_time = (time.perf_counter() - start) * 1000  # ms
                    self.running = False
                    break

                time.sleep(self.intervalo)

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()

    def stop_heartbeat_monitor(self):
        """Para o monitor."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1)

    async def simulate_failure(self, tipo: str) -> float:
        """Simula falha e retorna tempo de detecção."""
        # Iniciar monitoramento
        self.start_heartbeat_monitor()

        # Pequeno delay
        await asyncio.sleep(0.5)

        # Executar falha
        if tipo == "kill9":
            subprocess.run(["kill", "-9", str(self.agent_pid)], check=False)
        elif tipo == "sigterm":
            subprocess.run(["kill", "-15", str(self.agent_pid)], check=False)
        elif tipo == "deadlock":
            subprocess.run(["kill", "-STOP", str(self.agent_pid)], check=False)

        # Aguardar detecção
        await asyncio.sleep(self.intervalo + 1)

        # Parar monitor
        self.stop_heartbeat_monitor()

        return self.detection_time if self.detection_time else -1


async def run_benchmark(args):
    """Executa benchmark."""

    detector = HeartbeatDetector(
        agent_url=args.agent_url,
        intervalo_ms=args.intervalo
    )

    results = []

    print(f"E2: Detecção de Falha - HTTP Heartbeat")
    print(f"Agente: {args.agent_url}")
    print(f"Intervalo: {args.intervalo}ms")
    print(f"Tipo de falha: {args.tipo}")
    print(f"Iterações: {args.n}")
    print("-" * 50)

    for i in range(args.n):
        # Iniciar agente
        agent_proc = detector.start_agent()

        # Simular falha e medir
        detection_time = await detector.simulate_failure(args.tipo)

        results.append({
            "iteration": i + 1,
            "detection_time_ms": detection_time,
            "tipo": args.tipo,
            "intervalo_ms": args.intervalo
        })

        print(f"Iteração {i+1}/{args.n}: {detection_time:.2f}ms")

        # Limpar
        try:
            subprocess.run(["kill", "-9", str(detector.agent_pid)], check=False)
        except:
            pass

        await asyncio.sleep(1)

    # Estatísticas
    detection_times = [r["detection_time_ms"] for r in results if r["detection_time_ms"] > 0]

    summary = {
        "protocol": "HTTP_HEARTBEAT",
        "intervalo_ms": args.intervalo,
        "tipo_falha": args.tipo,
        "n": len(detection_times),
        "detection_mean_ms": round(statistics.mean(detection_times), 2) if detection_times else -1,
        "detection_median_ms": round(statistics.median(detection_times), 2) if detection_times else -1,
        "detection_stdev_ms": round(statistics.stdev(detection_times), 2) if len(detection_times) > 1 else 0,
    }

    # Salvar CSV
    csv_file = f"results/E2_HTTP_HEARTBEAT_{args.tipo}_{args.intervalo}ms.csv"
    Path("results").mkdir(exist_ok=True)

    with open(csv_file, "w") as f:
        f.write("iteration,detection_time_ms,tipo,intervalo_ms\n")
        for r in results:
            f.write(f"{r['iteration']},{r['detection_time_ms']},{r['tipo']},{r['intervalo_ms']}\n")

    # Salvar JSON
    json_file = f"results/E2_HTTP_HEARTBEAT_{args.tipo}_{args.intervalo}ms_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResultados:")
    print(f"Tempo médio de detecção: {summary['detection_mean_ms']:.2f}ms")
    print(f"CSV: {csv_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E2: Detecção de Falha - HTTP Heartbeat")
    parser.add_argument("--agent-url", default="http://localhost:8082", help="URL do agente")
    parser.add_argument("--intervalo", type=int, default=1000, help="Intervalo de heartbeat em ms")
    parser.add_argument("--tipo", choices=["kill9", "sigterm", "deadlock"], default="kill9", help="Tipo de falha")
    parser.add_argument("--n", type=int, default=50, help="Número de iterações")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
