"""
Adaptador gRPC — envolve a classe `GRPCLayer` existente em `grpc_layer.py`.

Posicionamento na dissertação: gRPC é baseline de comparação, não parte
da solução proposta. Manter como adaptador isolado deixa esse papel
explícito no código.
"""
from __future__ import annotations

import time
from typing import Any

from .base import TransportAdapter, TransportResult


class GrpcTransportAdapter(TransportAdapter):
    def __init__(self, grpc_layer: Any) -> None:
        self._grpc = grpc_layer

    def name(self) -> str:
        return "grpc"

    def is_available(self) -> bool:
        return bool(self._grpc) and bool(getattr(self._grpc, "is_available", lambda: False)())

    async def dispatch(self, task: dict, agent: Any, timeout_ms: int) -> TransportResult:
        if not self.is_available():
            return TransportResult(
                success=False,
                agent_id=getattr(agent, "agent_id", ""),
                error_message="grpc transport unavailable",
            )
        address = getattr(agent, "grpc_address", "") or f"{getattr(agent, 'hostname', 'localhost')}:50051"
        t_start = time.perf_counter()
        try:
            resp = await self._grpc.dispatch_chat(
                address=address,
                task_id=task["task_id"],
                messages=task.get("messages", []),
                max_tokens=task.get("max_tokens", 256),
                temperature=task.get("temperature", 0.7),
                priority=task.get("priority", 5),
                timeout_ms=timeout_ms,
            )
            latency = int((time.perf_counter() - t_start) * 1000)
            return TransportResult(
                success=bool(resp) and getattr(resp, "success", True),
                content=getattr(resp, "content", "") or "",
                agent_id=getattr(agent, "agent_id", ""),
                latency_ms=latency,
                prompt_tokens=int(getattr(resp, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(resp, "completion_tokens", 0) or 0),
                finish_reason=getattr(resp, "finish_reason", "") or "",
                raw=resp,
            )
        except Exception as e:
            return TransportResult(
                success=False,
                agent_id=getattr(agent, "agent_id", ""),
                latency_ms=int((time.perf_counter() - t_start) * 1000),
                error_message=f"grpc dispatch failed: {e}",
            )
