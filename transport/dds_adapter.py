"""
Adaptador DDS — envolve a classe `DDSLayer` existente em `dds.py`.

Não reimplementa a lógica de tópicos, QoS e correlação por request_id;
delega para a camada DDS já construída em `main.py`. Serve como ponto
de entrada único para o dispatcher quando migrarmos incrementalmente os
blocos if/elif do `server.py`.
"""
from __future__ import annotations

import time
from typing import Any

from .base import TransportAdapter, TransportResult


class DdsTransportAdapter(TransportAdapter):
    def __init__(self, dds_layer: Any) -> None:
        self._dds = dds_layer

    def name(self) -> str:
        return "dds"

    def is_available(self) -> bool:
        return bool(self._dds) and bool(getattr(self._dds, "is_available", lambda: False)())

    async def dispatch(self, task: dict, agent: Any, timeout_ms: int) -> TransportResult:
        if not self.is_available():
            return TransportResult(
                success=False,
                agent_id=getattr(agent, "agent_id", ""),
                error_message="dds transport unavailable",
            )

        task_id = task["task_id"]
        t_start = time.perf_counter()
        try:
            await self._dds.publish_task(
                task_id=task_id,
                messages=task.get("messages", []),
                target_agent_id=getattr(agent, "agent_id", ""),
                max_tokens=task.get("max_tokens", 256),
                temperature=task.get("temperature", 0.7),
                priority=task.get("priority", 5),
                timeout_ms=timeout_ms,
            )
            resp = await self._dds.wait_for_agent_response(task_id, timeout_ms=timeout_ms)
            latency = int((time.perf_counter() - t_start) * 1000)
            if resp is None:
                return TransportResult(
                    success=False,
                    agent_id=getattr(agent, "agent_id", ""),
                    latency_ms=latency,
                    error_message="dds response timeout",
                )
            return TransportResult(
                success=True,
                content=getattr(resp, "content", "") or resp.get("content", "") if hasattr(resp, "get") else getattr(resp, "content", ""),
                agent_id=getattr(resp, "agent_id", "") or getattr(agent, "agent_id", ""),
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
                error_message=f"dds dispatch failed: {e}",
            )
