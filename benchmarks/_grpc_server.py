#!/usr/bin/env python3
"""
Servidor gRPC proxy para benchmarks E1-E5.
==========================================
Implementa um servidor gRPC que encaminha requisições ao backend HTTP
(llama-server ou orquestrador HTTP), permitindo comparação real de
overhead gRPC (HTTP/2 + serialização) vs HTTP/1.1 vs DDS.

Serialização: JSON-over-gRPC (sem necessidade de arquivos .proto ou protoc).
Protocolo: gRPC padrão sobre HTTP/2.

Serviços implementados:
  /LLMService/Chat             - Requisição unária (não-streaming)
  /LLMService/StreamChat       - Streaming server-side token-a-token
  /grpc.health.v1.Health/Check - Health check padrão gRPC

Usage:
    python _grpc_server.py --backend http://localhost:8080 --port 50051

Dependências:
    pip install grpcio grpcio-health-checking requests
"""

import argparse
import json
import logging
import sys
import time
from concurrent import futures

import grpc
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Serialização JSON-over-gRPC ──────────────────────────────────────────────

def _json_serialize(obj: dict) -> bytes:
    return json.dumps(obj).encode("utf-8")


def _json_deserialize(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


# ── Handlers de serviço ──────────────────────────────────────────────────────

class LLMServiceHandlers:
    """Handlers do serviço gRPC LLM (proxy para HTTP backend)."""

    def __init__(self, backend_url: str):
        self.backend_url = backend_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"

    def chat(self, request: dict, context: grpc.ServicerContext) -> dict:
        """
        Requisição unária: encaminha para /v1/chat/completions (não-streaming).
        """
        payload = {
            "model": request.get("model", "qwen3.5-0.8b"),
            "messages": [{"role": "user", "content": request.get("content", "")}],
            "max_tokens": request.get("max_tokens", 50),
            "stream": False,
        }
        if "priority" in request:
            payload["priority"] = request["priority"]

        t_start = time.perf_counter()
        try:
            resp = self.session.post(
                f"{self.backend_url}/v1/chat/completions",
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            processing_ms = int((time.perf_counter() - t_start) * 1000)
            return {"content": content, "success": True, "processing_ms": processing_ms}
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {"content": "", "success": False, "processing_ms": 0, "error": str(e)}

    def stream_chat(self, request: dict, context: grpc.ServicerContext):
        """
        Streaming server-side: encaminha para /v1/chat/completions (stream=True).
        Produz um chunk por token recebido via SSE.
        """
        payload = {
            "model": request.get("model", "qwen3.5-0.8b"),
            "messages": [{"role": "user", "content": request.get("content", "")}],
            "max_tokens": request.get("max_tokens", 200),
            "stream": True,
        }

        try:
            with self.session.post(
                f"{self.backend_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=120
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        data = line[6:]
                        if data == b"[DONE]":
                            yield {"token": "", "done": True, "timestamp_ns": time.perf_counter_ns()}
                            return
                        try:
                            parsed = json.loads(data)
                            delta = parsed.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield {"token": token, "done": False,
                                       "timestamp_ns": time.perf_counter_ns()}
                        except Exception:
                            pass
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            yield {"token": "", "done": True, "timestamp_ns": time.perf_counter_ns(),
                   "error": str(e)}

    def health_check(self, request: dict, context: grpc.ServicerContext) -> dict:
        """Health check: tenta um HEAD no backend."""
        try:
            resp = self.session.get(f"{self.backend_url}/health", timeout=5)
            serving = resp.status_code == 200
        except Exception:
            serving = False
        return {"serving": serving}


# ── Registro do servidor ──────────────────────────────────────────────────────

def _make_handler(fn, streaming_output=False):
    """Cria um RPC method handler com serialização JSON."""
    if streaming_output:
        def wrapper(request_bytes, context):
            request = _json_deserialize(request_bytes)
            for response in fn(request, context):
                yield _json_serialize(response)
        return grpc.server_streaming_rpc_method_handler(
            wrapper,
            request_deserializer=None,
            response_serializer=None,
        )
    else:
        def wrapper(request_bytes, context):
            request = _json_deserialize(request_bytes)
            response = fn(request, context)
            return _json_serialize(response)
        return grpc.unary_unary_rpc_method_handler(
            wrapper,
            request_deserializer=None,
            response_serializer=None,
        )


def build_server(backend_url: str, port: int, max_workers: int = 10) -> grpc.Server:
    """Constrói e retorna o servidor gRPC configurado."""
    handlers = LLMServiceHandlers(backend_url)

    method_handlers = {
        "Chat": _make_handler(handlers.chat, streaming_output=False),
        "StreamChat": _make_handler(handlers.stream_chat, streaming_output=True),
        "HealthCheck": _make_handler(handlers.health_check, streaming_output=False),
    }

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    server.add_generic_rpc_handlers([
        _build_generic_handler("LLMService", method_handlers)
    ])

    server.add_insecure_port(f"[::]:{port}")
    return server


def _build_generic_handler(service_name: str, method_handlers: dict):
    """Cria um GenericRpcHandler compatível com grpcio."""

    class Handler(grpc.GenericRpcHandler):
        def service_name(self):
            return service_name

        def service(self, handler_call_details):
            method = handler_call_details.method.split("/")[-1]
            return method_handlers.get(method)

    return Handler()


def main():
    parser = argparse.ArgumentParser(description="gRPC proxy server para benchmarks LLM")
    parser.add_argument("--backend", default="http://localhost:8080",
                        help="URL do backend HTTP (llama-server ou orquestrador)")
    parser.add_argument("--port", type=int, default=50051, help="Porta gRPC")
    parser.add_argument("--workers", type=int, default=10, help="Thread pool workers")

    args = parser.parse_args()

    server = build_server(args.backend, args.port, args.workers)
    server.start()

    logger.info(f"gRPC server listening on port {args.port}")
    logger.info(f"Backend HTTP: {args.backend}")
    logger.info("Serviços: /LLMService/Chat, /LLMService/StreamChat, /LLMService/HealthCheck")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5)
        logger.info("Servidor encerrado.")


if __name__ == "__main__":
    main()
