#!/usr/bin/env python3
"""
Executa TODOS os benchmarks E1-E5 (100% REAIS)
=============================================
Gera resultados em CSV

Usage:
    python run_E1_to_E5.py --cenario all --n 100
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Criar diretório results
Path("results").mkdir(exist_ok=True)

# Cenários
CENARIOS = {
    "E1": [
        # E1: Decomposição de latência
        # DDS
        "python3 benchmarks/E1_decompose_latency_dds.py --model phi4-mini --prompt short --n {n}",
        "python3 benchmarks/E1_decompose_latency_dds.py --model phi4-mini --prompt long --n {n}",
        # HTTP
        "python3 benchmarks/E1_decompose_latency_http.py --model phi4-mini --prompt short --n {n}",
        "python3 benchmarks/E1_decompose_latency_http.py --model phi4-mini --prompt long --n {n}",
    ],
    "E2": [
        # E2: Detecção de falha
        "python3 benchmarks/E2_failure_detection_dds.py --periodo 1000 --tipo kill9 --n {n}",
        "python3 benchmarks/E2_failure_detection_dds.py --periodo 5000 --tipo kill9 --n {n}",
        "python3 benchmarks/E2_failure_detection_dds.py --periodo 10000 --tipo kill9 --n {n}",
    ],
    "E3": [
        # E3: Priorização
        "python3 benchmarks/E3_priority_dds.py --carga 10 --n 30",
    ],
    "E4": [
        # E4: Escalabilidade
        "python3 benchmarks/E4_scalability_dds.py --clientes 1 --n {n}",
        "python3 benchmarks/E4_scalability_dds.py --clientes 2 --n {n}",
        "python3 benchmarks/E4_scalability_dds.py --clientes 4 --n {n}",
        "python3 benchmarks/E4_scalability_dds.py --clientes 8 --n {n}",
    ],
    "E5": [
        # E5: Streaming
        "python3 benchmarks/E5_streaming_dds.py --model phi4-mini --n {n}",
        "python3 benchmarks/E5_streaming_dds.py --model qwen3.5-9b --n {n}",
    ]
}


def run_command(cmd: str) -> int:
    """Executa comando e retorna código de saída."""
    print(f"\n{'='*60}")
    print(f"Executando: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd="/mnt/e/TI/git/tese/dds_orchestrator")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Executa todos os benchmarks E1-E5")
    parser.add_argument("--cenario", choices=["all", "E1", "E2", "E3", "E4", "E5"], default="all")
    parser.add_argument("--n", type=int, default=50, help="Número de iterações (default: 50)")
    parser.add_argument("--skip-errors", action="store_true", help="Continuar se algum teste falhar")

    args = parser.parse_args()

    print(f"===========================================")
    print(f"EXECUTANDO BENCHMARKS E1-E5 (100% REAIS)")
    print(f"===========================================")
    print(f"Cenário: {args.cenario}")
    print(f"N iterações: {args.n}")
    print()

    # Determinar cenários a executar
    if args.cenario == "all":
        scenarios_to_run = CENARIOS.keys()
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

            return_code = run_command(cmd)

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
