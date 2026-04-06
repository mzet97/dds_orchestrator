#!/usr/bin/env python3
"""
gRPC Server para Orchestrator
Implementa serviço gRPC paralelo ao HTTP
"""

import grpc
import asyncio
import time
import logging
from concurrent import futures
from typing import AsyncIterator

from dds_orchestrator.proto import orchestrator_pb2, orchestrator_pb2_grpc

logger = logging.getLogger(__name__)


class OrchestratorGRPCService(orchestrator_pb2_grpc.OrchestratorServicer):
    """Implementa serviço gRPC do Orchestrator"""

    def __init__(self, orchestrator_server):
        self.orch = orchestrator_server
        logger.info("gRPC Service initialized")

    async def ChatCompletion(self, request, context):
        """Completa conversação via gRPC"""
        try:
            # Converter request gRPC para dict
            data = {
                "model": request.model,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature if request.temperature > 0 else 0.7,
            }

            # Chamar handler HTTP existente
            result = await self.orch.handle_chat(data)

            # Retornar response gRPC
            return orchestrator_pb2.ChatCompletionResponse(
                task_id=result.get("task_id", ""),
                content=result.get("response", ""),
                tokens=result.get("tokens", 0),
                latency_ms=int(result.get("processing_time_ms", 0))
            )

        except Exception as e:
            logger.error(f"ChatCompletion error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def RegisterAgent(self, request, context):
        """Registra agente via gRPC"""
        try:
            data = {
                "agent_id": request.agent_id,
                "model_name": request.model_name,
                "capabilities": list(request.capabilities),
                "port": request.port,
            }

            # Registrar no registry existente
            agent_info = self.orch.registry.register(
                agent_id=data["agent_id"],
                model_name=data["model_name"],
                capabilities=data["capabilities"],
                port=data["port"]
            )

            return orchestrator_pb2.RegisterAgentResponse(
                success=True,
                agent_id=agent_info.agent_id,
                registered_at=int(time.time() * 1000)
            )

        except Exception as e:
            logger.error(f"RegisterAgent error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def Heartbeat(self, request, context):
        """Heartbeat de agente via gRPC"""
        try:
            # Atualizar status no registry
            self.orch.registry.heartbeat(request.agent_id)

            return orchestrator_pb2.HeartbeatResponse(
                success=True,
                timestamp=int(time.time() * 1000)
            )

        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


async def start_grpc_server(orchestrator_server, host: str = "0.0.0.0", port: int = 50051):
    """Inicia servidor gRPC"""
    logger.info(f"Starting gRPC server on {host}:{port}")

    # Criar servidor gRPC
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            (grpc.ChannelArguments.max_send_message_length, -1),
            (grpc.ChannelArguments.max_receive_message_length, -1),
        ]
    )

    # Adicionar service
    service = OrchestratorGRPCService(orchestrator_server)
    orchestrator_pb2_grpc.add_OrchestratorServicer_to_server(service, server)

    # Adicionar port
    server.add_insecure_port(f"{host}:{port}")

    # Iniciar
    await server.start()
    logger.info(f"gRPC server started on {host}:{port}")

    return server


async def shutdown_grpc_server(server):
    """Para servidor gRPC"""
    logger.info("Shutting down gRPC server")
    await server.stop(grace=5)
    logger.info("gRPC server shut down")
