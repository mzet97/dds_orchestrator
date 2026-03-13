#!/usr/bin/env python3
"""
E2: Detecao de Falha - HTTP Heartbeat
======================================
Mede tempo de detecao com heartbeat HTTP
100% REAL - usa threads para polling

Nota sobre threading:
- O loop principal usa asyncio.sleep (correto para o event loop async)
- As threads internas de heartbeat usam time.sleep (correto para threads)

Usage:
    python E2_failure_detection_http.py --intervalo 1000 --tipo kill9 --n 50
"""

import argparse
import asyncio
import json
import time
import statistics
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional
import requests


class HeartbeatDetector:
    """Detecao de falha via HTTP Heartbeat."""

    def __init__(self, agent_url: str, intervalo_ms: int = 1000):
        self.agent_url = agent_url
        self.intervalo = intervalo_ms / 1000.0  # seconds
        self.agent_pid = None
        self.detection_time: Optional[float] = None
        self.running = False
        self._t_fail_ref: List[Optional[float]] = [None]

    def start_agent(self) -> Optional[subprocess.Popen]:
        """Inicia agente REAL."""
        # Calcular path relativo ao script (raiz do repo)
        script_dir = Path(__file__).resolve().parent.parent.parent
        agent_script = script_dir / "dds_agent" / "python" / "agent_llm.py"
        agent_cwd = script_dir / "dds_agent" / "python"

        if not agent_script.exists():
            print(f"ERRO: Script do agente nao encontrado: {agent_script}")
            print(f"      Verifique se o diretorio dds_agent/python/ existe na raiz do repositorio.")
            return None

        # Usar sys.executable para garantir o mesmo interpretador Python
        proc = subprocess.Popen(
            [sys.executable, str(agent_script), "--model-path", "../models/phi4-mini.gguf"],
            cwd=str(agent_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.agent_pid = proc.pid
        time.sleep(2)  # Esperar inicializacao
        return proc

    def start_heartbeat_monitor(self, t_fail_ref: List[Optional[float]]):
        """
        Inicia thread de heartbeat polling.
        t_fail_ref: lista de 1 elemento [None ou float]; o tempo do kill sera
        escrito nela pelo caller, e esta thread usa esse valor como referencia.

        Nota: esta thread usa time.sleep (correto para threads, nao asyncio).
        Sinaliza _first_success_event apos o primeiro heartbeat bem-sucedido.
        """
        self.running = True
        self.detection_time = None
        self._t_fail_ref = t_fail_ref
        self._first_success_event = threading.Event()

        def monitor():
            while self.running:
                try:
                    # HTTP Health check
                    resp = requests.get(f"{self.agent_url}/health", timeout=1)
                    if resp.status_code != 200:
                        # Falha detectada!
                        t_detected = time.perf_counter()
                        t_fail = self._t_fail_ref[0]
                        if t_fail is not None:
                            self.detection_time = t_detected
                        self.running = False
                        break
                    else:
                        # Heartbeat bem-sucedido
                        self._first_success_event.set()
                except Exception:
                    # Falha detectada (conexao recusada, timeout, etc.)
                    t_detected = time.perf_counter()
                    t_fail = self._t_fail_ref[0]
                    if t_fail is not None:
                        self.detection_time = t_detected
                    self.running = False
                    break

                # Polling com intervalo configurado (time.sleep em thread)
                time.sleep(self.intervalo)

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()

    def stop_heartbeat_monitor(self):
        """Para o monitor."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)

    async def simulate_failure(self, tipo: str) -> float:
        """Simula falha e retorna tempo de detecao em ms."""
        # Resetar detection_time entre iteracoes
        self.detection_time = None
        t_fail_ref: List[Optional[float]] = [None]

        # Iniciar monitoramento ANTES do kill
        self.start_heartbeat_monitor(t_fail_ref)

        # Aguardar confirmacao do primeiro heartbeat bem-sucedido antes de matar.
        # Isso garante que o kill ocorre logo apos um heartbeat confirmado,
        # tornando o tempo de deteccao proximo a um intervalo completo (distribuicao
        # uniforme em [0, H] conforme metodologia da dissertacao).
        deadline = time.perf_counter() + 5.0
        while not self._first_success_event.is_set():
            await asyncio.sleep(0.02)
            if time.perf_counter() > deadline:
                self.stop_heartbeat_monitor()
                return -1  # Agente nao respondeu no tempo limite

        # Registrar T_fail no MOMENTO do kill
        t_fail = time.perf_counter()
        t_fail_ref[0] = t_fail

        # Executar falha
        if tipo == "kill9":
            subprocess.run(["kill", "-9", str(self.agent_pid)], check=False)
        elif tipo == "sigterm":
            subprocess.run(["kill", "-15", str(self.agent_pid)], check=False)
        elif tipo == "deadlock":
            subprocess.run(["kill", "-STOP", str(self.agent_pid)], check=False)

        # Aguardar detecao com timeout
        timeout_s = self.intervalo * 3 + 2  # 3x intervalo + margem
        elapsed = 0.0
        poll_interval = 0.01
        while self.detection_time is None and elapsed < timeout_s:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Parar monitor
        self.stop_heartbeat_monitor()

        if self.detection_time is not None and t_fail_ref[0] is not None:
            detection_ms = (self.detection_time - t_fail_ref[0]) * 1000
            return detection_ms
        else:
            return -1


async def run_benchmark(args):
    """Executa benchmark."""

    detector = HeartbeatDetector(
        agent_url=args.agent_url,
        intervalo_ms=args.intervalo
    )

    results = []

    print(f"E2: Detecao de Falha - HTTP Heartbeat")
    print(f"Agente: {args.agent_url}")
    print(f"Intervalo: {args.intervalo}ms")
    print(f"Tipo de falha: {args.tipo}")
    print(f"Iteracoes: {args.n}")
    print("-" * 50)

    for i in range(args.n):
        # Iniciar agente
        agent_proc = detector.start_agent()

        if agent_proc is None:
            print(f"Iteracao {i+1}/{args.n}: PULADA - agente nao pode ser iniciado")
            results.append({
                "iteration": i + 1,
                "detection_time_ms": -1,
                "tipo": args.tipo,
                "intervalo_ms": args.intervalo
            })
            continue

        # Simular falha e medir
        detection_time = await detector.simulate_failure(args.tipo)

        results.append({
            "iteration": i + 1,
            "detection_time_ms": detection_time,
            "tipo": args.tipo,
            "intervalo_ms": args.intervalo
        })

        if detection_time > 0:
            print(f"Iteracao {i+1}/{args.n}: {detection_time:.2f}ms")
        else:
            print(f"Iteracao {i+1}/{args.n}: FALHA (detecao nao ocorreu)")

        # Limpar - garantir que o processo foi terminado
        try:
            subprocess.run(["kill", "-9", str(detector.agent_pid)], check=False)
        except Exception:
            pass

        # Esperar limpeza (asyncio.sleep no event loop async)
        await asyncio.sleep(1)

    # Estatisticas (filtrar valores invalidos)
    detection_times = [r["detection_time_ms"] for r in results if r["detection_time_ms"] > 0]

    summary = {
        "protocol": "HTTP_HEARTBEAT",
        "intervalo_ms": args.intervalo,
        "tipo_falha": args.tipo,
        "n_total": args.n,
        "n_valid": len(detection_times),
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
    if detection_times:
        print(f"Tempo medio de detecao: {summary['detection_mean_ms']:.2f}ms")
        print(f"Iteracoes validas: {len(detection_times)}/{args.n}")
    else:
        print("Nenhuma detecao valida registrada.")
    print(f"CSV: {csv_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="E2: Detecao de Falha - HTTP Heartbeat")
    parser.add_argument("--agent-url", default="http://localhost:8082", help="URL do agente")
    parser.add_argument("--intervalo", type=int, default=1000, help="Intervalo de heartbeat em ms")
    parser.add_argument("--tipo", choices=["kill9", "sigterm", "deadlock"], default="kill9", help="Tipo de falha")
    parser.add_argument("--n", type=int, default=50, help="Numero de iteracoes")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
