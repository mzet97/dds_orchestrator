#!/usr/bin/env python3
"""
Helper: Servidor gRPC de health check para benchmark E2.
Lançado como subprocesso pelo E2_failure_detection_grpc.py.

Implementa LlamaService.GetStatus RPC usando os mesmos proto stubs
que o llama-server --enable-grpc, para comparação justa com DDS DEADLINE.

Usage (interno):
    python _e2_grpc_health_server.py <port> <intervalo_ms>
"""
import sys
import time
from concurrent import futures
from pathlib import Path

# Import proto stubs
sys.path.insert(0, str(Path(__file__).parent))
from proto import llama_service_pb2
from proto import llama_service_pb2_grpc

import grpc


class HealthServicer(llama_service_pb2_grpc.LlamaServiceServicer):
    """Minimal LlamaService: only implements GetStatus for health checking."""

    def GetStatus(self, request, context):
        return llama_service_pb2.ServerStatus(
            server_id="e2-health-server",
            slots_idle=1,
            slots_processing=0,
            model_loaded="benchmark",
            ready=True,
        )

    def Chat(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        return llama_service_pb2.ChatCompletionResponse()

    def StreamChat(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        return


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 50051

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    llama_service_pb2_grpc.add_LlamaServiceServicer_to_server(
        HealthServicer(), server
    )
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()

    print(f"gRPC health server started on port {port}", flush=True)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=0)


if __name__ == "__main__":
    main()
