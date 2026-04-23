# deprecated/ — benchmarks aposentados

Os scripts neste diretório foram movidos de `benchmarks/` em 2026-04-23
por violarem a regra de cadeia nativa ponta-a-ponta imposta pela
dissertação v3: **"se iniciou em DDS, finaliza em DDS"**.

## Por que foram aposentados

| Script | Violação |
|---|---|
| `E1_decompose_latency_dds.py` | Cliente HTTP → orquestrador → DDS interno. Cliente não é nativo DDS. |
| `E1_decompose_latency_grpc.py` | Cliente gRPC → **llama-server direto**. Pula orquestrador e agente (3 hops em vez de 4). |
| `E3_priority_dds.py` | Cliente HTTP → orquestrador → DDS interno. |
| `E3_priority_grpc.py` | Cliente gRPC → llama-server direto. |
| `E4_scalability_dds.py` | Cliente HTTP → orquestrador → DDS interno. |
| `E4_scalability_grpc.py` | Cliente gRPC → llama-server direto. |
| `E5_streaming_dds.py` | Cliente HTTP + SSE nas pontas, DDS só no meio. Mistura protocolos. |
| `E5_streaming_dds_direct.py` | Cliente DDS → llama-server direto. Pula orquestrador e agente. |
| `E5_streaming_grpc.py` | Cliente gRPC → llama-server direto. |

## O que usar no lugar

Em `benchmarks/`:

| Cenário | DDS | gRPC | HTTP |
|---|---|---|---|
| E1 | `E1_decompose_latency_dds_native.py` | `E1_decompose_latency_grpc_native.py` | `E1_decompose_latency_http.py` |
| E2 | `E2_failure_detection_dds.py` | `E2_failure_detection_grpc.py` | `E2_failure_detection_http.py` |
| E3 | `E3_priority_dds_native.py` | `E3_priority_grpc_native.py` | `E3_priority_http.py` |
| E4 | `E4_scalability_dds_native.py` | `E4_scalability_grpc_native.py` | `E4_scalability_http.py` |
| E5 | `E5_streaming_dds_native.py` | `E5_streaming_grpc_native.py` | `E5_streaming_http.py` |

Todas as 15 variantes têm `--n 1000` como default, conforme a dissertação v3
(parágrafo "Número de repetições" da Seção `sec:protocolo_medicao`).

O runner consolidado é `run_E1_to_E5.py`.
