"""
Adaptador HTTP (baseline de comparação).

Encapsula o padrão aiohttp POST usado hoje em `server.py` como baseline.
Este arquivo é o ponto único onde endereçamento e serialização HTTP
devem viver a partir da migração incremental do dispatcher.
"""
from __future__ import annotations

import time
from typing import Any

import aiohttp

from .base import TransportAdapter, TransportResult


class HttpTransportAdapter(TransportAdapter):
    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._session = session

    def name(self) -> str:
        return "http"

    def is_available(self) -> bool:
        # HTTP é o baseline universal; sempre disponível.
        return True

    async def dispatch(self, task: dict, agent: Any, timeout_ms: int) -> TransportResult:
        hostname = getattr(agent, "hostname", "localhost")
        port = getattr(agent, "port", 8082)
        url = f"http://{hostname}:{port}/chat"

        session = self._session or aiohttp.ClientSession()
        own_session = self._session is None
        t_start = time.perf_counter()
        try:
            async with session.post(
                url,
                json=task,
                timeout=aiohttp.ClientTimeout(total=timeout_ms / 1000),
            ) as resp:
                data = await resp.json()
            latency = int((time.perf_counter() - t_start) * 1000)
            return TransportResult(
                success=resp.status == 200,
                content=data.get("content", ""),
                agent_id=getattr(agent, "agent_id", ""),
                latency_ms=latency,
                prompt_tokens=int(data.get("prompt_tokens", 0)),
                completion_tokens=int(data.get("completion_tokens", 0)),
                finish_reason=data.get("finish_reason", ""),
                raw=data,
            )
        except Exception as e:
            return TransportResult(
                success=False,
                agent_id=getattr(agent, "agent_id", ""),
                latency_ms=int((time.perf_counter() - t_start) * 1000),
                error_message=f"http dispatch failed: {e}",
            )
        finally:
            if own_session:
                await session.close()
