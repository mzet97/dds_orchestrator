#!/usr/bin/env python3
"""
Executa TODOS os benchmarks E1-E5 com cadeia nativa ponta-a-ponta
==================================================================
Gera resultados em CSV e JSON.

Regra de ouro (dissertação v3, Seção sec:protocolo_medicao):
    "Se iniciou em DDS, finaliza em DDS." — o cliente é nativo por protocolo,
    e os quatro hops (cliente → orquestrador → agente → llama-server)
    usam o MESMO transporte. Zero atalhos.

N=1000 iterações por configuração em TODOS os cenários (v3, parágrafo
"Número de repetições" consolidado).

Configurar URLs antes de executar:
  ORCHESTRADOR_URL: endpoint do orquestrador (HTTP para o client HTTP; gRPC
                    e DDS usam seus próprios transportes configurados nos
                    scripts *_native.py).
  AGENTE1_URL:      agente primário (RX6600M).
  AGENTE2_URL:      agente secundário (RTX 3080).

Usage:
    python run_E1_to_E5.py --cenario all --n 1000 \\
        --orchestrador http://192.168.1.62:18080 \\
        --orch-grpc   192.168.1.62:50052 \\
        --agentes     http://192.168.1.60:8082,http://192.168.1.61:8082
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Criar diretório results
Path("results").mkdir(exist_ok=True)


def build_cenarios(orchestrador: str, orch_grpc: str, agentes: str, agente1: str) -> dict:
    """Constrói os comandos para os três protocolos com cadeia nativa.

    - DDS:  cliente CycloneDDS em client/request → orquestrador DDS → agente DDS → llama-server DDS.
    - gRPC: cliente ClientOrchestratorService.Chat → orquestrador gRPC → agente gRPC → llama-server gRPC.
    - HTTP: cliente HTTP → orquestrador HTTP → agente HTTP → llama-server HTTP.
    """
    # Saco de flags específicas por script (cada um expõe as opções do seu
    # transporte; o runner passa só o essencial comum).
    N = "{n}"  # placeholder expandido depois

    return {
        "E1": [
            # DDS native (full chain, 4 hops)
            f"python3 benchmarks/E1_decompose_latency_dds_native.py  --model qwen3.5-0.8b --prompt short --n {N}",
            f"python3 benchmarks/E1_decompose_latency_dds_native.py  --model qwen3.5-0.8b --prompt long  --n {N}",
            # gRPC native (full chain, via ClientOrchestratorService)
            f"python3 benchmarks/E1_decompose_latency_grpc_native.py --orch {orch_grpc} --model qwen3.5-0.8b --prompt short --n {N}",
            f"python3 benchmarks/E1_decompose_latency_grpc_native.py --orch {orch_grpc} --model qwen3.5-0.8b --prompt long  --n {N}",
            # HTTP (client → orchestrator HTTP → agent HTTP → llama HTTP)
            f"python3 benchmarks/E1_decompose_latency_http.py --url {orchestrador} --model qwen3.5-0.8b --prompt short --n {N}",
            f"python3 benchmarks/E1_decompose_latency_http.py --url {orchestrador} --model qwen3.5-0.8b --prompt long  --n {N}",
        ],
        "E2": [
            # DDS DEADLINE + LIVELINESS
            f"python3 benchmarks/E2_failure_detection_dds.py  --periodo 1000  --tipo kill9    --n {N}",
            f"python3 benchmarks/E2_failure_detection_dds.py  --periodo 5000  --tipo kill9    --n {N}",
            f"python3 benchmarks/E2_failure_detection_dds.py  --periodo 10000 --tipo kill9    --n {N}",
            f"python3 benchmarks/E2_failure_detection_dds.py  --periodo 1000  --tipo sigterm  --n {N}",
            f"python3 benchmarks/E2_failure_detection_dds.py  --periodo 1000  --tipo deadlock --n {N}",
            # gRPC Health Check
            f"python3 benchmarks/E2_failure_detection_grpc.py --periodo 1000  --tipo kill9 --n {N}",
            f"python3 benchmarks/E2_failure_detection_grpc.py --periodo 5000  --tipo kill9 --n {N}",
            f"python3 benchmarks/E2_failure_detection_grpc.py --periodo 10000 --tipo kill9 --n {N}",
            # HTTP Heartbeat
            f"python3 benchmarks/E2_failure_detection_http.py --agent-url {agente1} --intervalo 1000  --tipo kill9 --n {N}",
            f"python3 benchmarks/E2_failure_detection_http.py --agent-url {agente1} --intervalo 5000  --tipo kill9 --n {N}",
            f"python3 benchmarks/E2_failure_detection_http.py --agent-url {agente1} --intervalo 10000 --tipo kill9 --n {N}",
        ],
        "E3": [
            # Cada execução: N injeções HIGH; carga NORMAL sustentada em req/s.
            # Duração dos scripts já escala com N (default 10000s p/ N=1000 a 10 req/s).
            f"python3 benchmarks/E3_priority_dds_native.py  --carga 10 --n {N}",
            f"python3 benchmarks/E3_priority_grpc_native.py --orch {orch_grpc} --carga 10 --n {N}",
            f"python3 benchmarks/E3_priority_http.py        --url {orchestrador} --carga 10 --n {N}",
        ],
        "E4": [
            # Fase A: 1 agente. Fase B: 2 agentes. A fase é controlada pelo
            # número de URLs em --agentes (HTTP). Nos _native, os clientes
            # falam com o orquestrador (que conhece todos os agentes).
            f"python3 benchmarks/E4_scalability_dds_native.py  --n {N}",
            f"python3 benchmarks/E4_scalability_grpc_native.py --orch {orch_grpc} --n {N}",
            f"python3 benchmarks/E4_scalability_http.py        --agentes {agente1} --n {N}",
            f"python3 benchmarks/E4_scalability_http.py        --agentes {agentes} --n {N}",
        ],
        "E5": [
            f"python3 benchmarks/E5_streaming_dds_native.py  --model qwen3.5-0.8b --n {N}",
            f"python3 benchmarks/E5_streaming_grpc_native.py --orch {orch_grpc} --model qwen3.5-0.8b --n {N}",
            f"python3 benchmarks/E5_streaming_http.py        --url {orchestrador} --model qwen3.5-0.8b --n {N}",
        ],
    }


def run_command(cmd: str, cwd: str) -> int:
    """Executa comando e retorna código de saída."""
    print(f"\n{'='*60}")
    print(f"Executando: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Executa todos os benchmarks E1-E5 (cadeia nativa)")
    parser.add_argument("--cenario", choices=["all", "E1", "E2", "E3", "E4", "E5"], default="all")
    parser.add_argument("--n", type=int, default=1000,
                        help="Número de iterações por configuração (v3: N=1000)")
    parser.add_argument("--skip-errors", action="store_true",
                        help="Continuar se algum teste falhar")
    parser.add_argument("--orchestrador", default="http://localhost:18080",
                        help="URL HTTP do orquestrador (para cliente HTTP)")
    parser.add_argument("--orch-grpc", default="localhost:50052",
                        help="Endpoint gRPC do orquestrador (ClientOrchestratorService)")
    parser.add_argument("--agentes", default="http://localhost:8082",
                        help="URLs HTTP dos agentes (vírgula-separadas) para E2 HTTP e E4 HTTP")
    parser.add_argument("--cwd", default=str(Path(__file__).parent.parent),
                        help="Diretório de trabalho (padrão: dds_orchestrator/)")

    args = parser.parse_args()

    agente1 = args.agentes.split(",")[0].strip()

    CENARIOS = build_cenarios(args.orchestrador, args.orch_grpc, args.agentes, agente1)

    print("=" * 60)
    print("EXECUTANDO BENCHMARKS E1-E5 (cadeia nativa ponta-a-ponta)")
    print("=" * 60)
    print(f"Cenário      : {args.cenario}")
    print(f"N iterações  : {args.n}")
    print(f"Orquestrador : {args.orchestrador}  |  gRPC: {args.orch_grpc}")
    print(f"Agentes      : {args.agentes}")
    print()

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

        for cmd_template in CENARIOS[scenario]:
            cmd = cmd_template.format(n=args.n)
            total += 1
            return_code = run_command(cmd, args.cwd)

            if return_code == 0:
                passed += 1
                print("✓ SUCESSO")
            else:
                failed += 1
                print(f"✗ FALHOU (código: {return_code})")
                if not args.skip_errors:
                    print("\nInterrompendo devido a erro.")
                    sys.exit(return_code)

    print(f"\n{'='*60}")
    print("RESULTADO FINAL")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Passou: {passed}")
    print(f"Falhou: {failed}")
    print("\nResultados salvos em: results/")

    result_files = list(Path("results").glob("*.csv")) + list(Path("results").glob("*.json"))
    for f in sorted(result_files):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
