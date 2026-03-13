#!/usr/bin/env python3
"""
Executa TODOS os benchmarks E1-E5 (DDS, HTTP, gRPC)
=====================================================
Gera resultados em CSV e JSON.

Configurar URLs antes de executar:
  ORCHESTRADOR_URL: orquestrador com DDS habilitado (VM3)
  AGENTE1_URL:      agente AMD RX6600M - Phi-4-mini (VM1)
  AGENTE2_URL:      agente NVIDIA RTX 3080 - Qwen3.5-9B (VM2)

Usage:
    python run_E1_to_E5.py --cenario all --n 50 \\
        --orchestrador http://192.168.1.62:8080 \\
        --agentes http://192.168.1.60:8082,http://192.168.1.61:8082
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Criar diretório results
Path("results").mkdir(exist_ok=True)


def build_cenarios(orchestrador: str, agentes: str, agente1: str) -> dict:
    """Constrói os comandos com as URLs fornecidas."""
    return {
        "E1": [
            # E1: Decomposição de latência - DDS (via orquestrador)
            f"python3 benchmarks/E1_decompose_latency_dds.py --url {orchestrador} --model phi4-mini --prompt short --n {{n}}",
            f"python3 benchmarks/E1_decompose_latency_dds.py --url {orchestrador} --model phi4-mini --prompt long --n {{n}}",
            # E1: HTTP (direto ao agente)
            f"python3 benchmarks/E1_decompose_latency_http.py --url {agente1} --model phi4-mini --prompt short --n {{n}}",
            f"python3 benchmarks/E1_decompose_latency_http.py --url {agente1} --model phi4-mini --prompt long --n {{n}}",
            # E1: gRPC
            f"python3 benchmarks/E1_decompose_latency_grpc.py --url {agente1} --model phi4-mini --prompt short --n {{n}}",
            f"python3 benchmarks/E1_decompose_latency_grpc.py --url {agente1} --model phi4-mini --prompt long --n {{n}}",
        ],
        "E2": [
            # E2: Detecção de falha - DDS DEADLINE
            f"python3 benchmarks/E2_failure_detection_dds.py --periodo 1000 --tipo kill9 --n {{n}}",
            f"python3 benchmarks/E2_failure_detection_dds.py --periodo 5000 --tipo kill9 --n {{n}}",
            f"python3 benchmarks/E2_failure_detection_dds.py --periodo 10000 --tipo kill9 --n {{n}}",
            f"python3 benchmarks/E2_failure_detection_dds.py --periodo 1000 --tipo sigterm --n {{n}}",
            f"python3 benchmarks/E2_failure_detection_dds.py --periodo 1000 --tipo deadlock --n {{n}}",
            # E2: HTTP Heartbeat
            f"python3 benchmarks/E2_failure_detection_http.py --agent-url {agente1} --intervalo 1000 --tipo kill9 --n {{n}}",
            f"python3 benchmarks/E2_failure_detection_http.py --agent-url {agente1} --intervalo 5000 --tipo kill9 --n {{n}}",
            f"python3 benchmarks/E2_failure_detection_http.py --agent-url {agente1} --intervalo 10000 --tipo kill9 --n {{n}}",
            # E2: gRPC Health Check
            f"python3 benchmarks/E2_failure_detection_grpc.py --agent-url {agente1} --intervalo 1000 --tipo kill9 --n {{n}}",
            f"python3 benchmarks/E2_failure_detection_grpc.py --agent-url {agente1} --intervalo 5000 --tipo kill9 --n {{n}}",
        ],
        "E3": [
            # E3: Priorização - 5 minutos, 30 injeções HIGH
            f"python3 benchmarks/E3_priority_dds.py --url {orchestrador} --carga 10 --n 30 --duracao 300",
            f"python3 benchmarks/E3_priority_http.py --url {orchestrador} --carga 10 --n 30 --duracao 300",
            f"python3 benchmarks/E3_priority_grpc.py --url {agente1} --carga 10 --n 30 --duracao 300",
        ],
        "E4": [
            # E4: Escalabilidade - Fase A (1 agente) e Fase B (2 agentes)
            # As fases A/B são controladas pelo número de --agentes fornecidos
            f"python3 benchmarks/E4_scalability_dds.py --orchestrador {orchestrador} --agentes {agente1} --n {{n}}",
            f"python3 benchmarks/E4_scalability_dds.py --orchestrador {orchestrador} --agentes {agentes} --n {{n}}",
            f"python3 benchmarks/E4_scalability_http.py --agentes {agente1} --n {{n}}",
            f"python3 benchmarks/E4_scalability_http.py --agentes {agentes} --n {{n}}",
            f"python3 benchmarks/E4_scalability_grpc.py --agentes {agente1} --n {{n}}",
            f"python3 benchmarks/E4_scalability_grpc.py --agentes {agentes} --n {{n}}",
        ],
        "E5": [
            # E5: Streaming - DDS via orquestrador, HTTP direto ao agente
            f"python3 benchmarks/E5_streaming_dds.py --url {orchestrador} --model phi4-mini --n {{n}}",
            f"python3 benchmarks/E5_streaming_http.py --url {agente1} --model phi4-mini --n {{n}}",
            f"python3 benchmarks/E5_streaming_grpc.py --url {agente1} --model phi4-mini --n {{n}}",
        ]
    }


def run_command(cmd: str, cwd: str) -> int:
    """Executa comando e retorna código de saída."""
    print(f"\n{'='*60}")
    print(f"Executando: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Executa todos os benchmarks E1-E5")
    parser.add_argument("--cenario", choices=["all", "E1", "E2", "E3", "E4", "E5"], default="all")
    parser.add_argument("--n", type=int, default=50, help="Número de iterações (default: 50)")
    parser.add_argument("--skip-errors", action="store_true", help="Continuar se algum teste falhar")
    parser.add_argument("--orchestrador", default="http://localhost:8080",
                        help="URL do orquestrador DDS (VM3)")
    parser.add_argument("--agentes", default="http://localhost:8082",
                        help="URLs dos agentes separados por vírgula (VM1,VM2)")
    parser.add_argument("--cwd", default=str(Path(__file__).parent.parent),
                        help="Diretório de trabalho (padrão: dds_orchestrator/)")

    args = parser.parse_args()

    # Agente 1 (primeiro da lista)
    agente1 = args.agentes.split(",")[0].strip()

    CENARIOS = build_cenarios(args.orchestrador, args.agentes, agente1)

    print(f"===========================================")
    print(f"EXECUTANDO BENCHMARKS E1-E5")
    print(f"===========================================")
    print(f"Cenário: {args.cenario}")
    print(f"N iterações: {args.n}")
    print(f"Orquestrador: {args.orchestrador}")
    print(f"Agentes: {args.agentes}")
    print()

    # Determinar cenários a executar
    if args.cenario == "all":
        scenarios_to_run = list(CENARIOS.keys())
    else:
        scenarios_to_run = [args.cenario]

    total = 0
    passed = 0
    failed = 0

    for scenario in scenarios_to_run:
        print(f"\n{'#'*60}")
        print(f"# CENÁRIO: {scenario}")
        print(f"{'#'*60}")

        commands = CENARIOS[scenario]

        for cmd_template in commands:
            cmd = cmd_template.format(n=args.n)
            total += 1

            return_code = run_command(cmd, args.cwd)

            if return_code == 0:
                passed += 1
                print(f"✓ SUCESSO")
            else:
                failed += 1
                print(f"✗ FALHOU (código: {return_code})")

                if not args.skip_errors:
                    print("\nInterrompendo devido a erro.")
                    break

    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Passou: {passed}")
    print(f"Falhou: {failed}")
    print(f"\nResultados salvos em: results/")

    # Listar arquivos
    result_files = list(Path("results").glob("*.csv")) + list(Path("results").glob("*.json"))
    for f in sorted(result_files):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
