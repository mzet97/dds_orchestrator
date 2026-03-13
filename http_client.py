#!/usr/bin/env python3
"""
HTTP Client for Orchestrator
Fallback when DDS is not available
"""
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import aiohttp

from models import AgentInfo, AgentRegistration, AgentTaskRequest, AgentTaskResponse


class HTTPClient:
    """
    HTTP Client for communicating with agents
    Used as fallback when DDS is not available
    """

    def __init__(self, timeout_seconds: int = 120):
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def health_check(self, host: str, port: int) -> bool:
        """Check if agent is healthy"""
        if self.session is None:
            raise RuntimeError("HTTPClient not initialized. Use as context manager or call __aenter__.")
        try:
            url = f"http://{host}:{port}/health"
            async with self.session.get(url) as response:
                return response.status == 200
        except Exception:
            return False

    async def get_agent_status(self, host: str, port: int) -> Optional[Dict]:
        """Get agent status via /health endpoint"""
        if self.session is None:
            raise RuntimeError("HTTPClient not initialized. Use as context manager or call __aenter__.")
        try:
            url = f"http://{host}:{port}/health"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception:
            pass
        return None

    async def send_chat_request(
        self,
        host: str,
        port: int,
        messages: List[Dict],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> Optional[AgentTaskResponse]:
        """Send chat request to agent"""
        if self.session is None:
            raise RuntimeError("HTTPClient not initialized. Use as context manager or call __aenter__.")
        try:
            url = f"http://{host}:{port}/chat"
            payload = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            async with self.session.post(url, json=payload) as response:
                data = await response.json()
                if response.status == 200:
                    return AgentTaskResponse(
                        task_id=data.get("task_id", ""),
                        agent_id=data.get("agent_id", ""),
                        content=data.get("response", ""),
                        is_final=True,
                        processing_time_ms=data.get("processing_time_ms", 0),
                        success=data.get("success", True),
                        error_message=data.get("error", ""),
                    )
                else:
                    return AgentTaskResponse(
                        task_id="",
                        agent_id="",
                        content="",
                        success=False,
                        error_message=data.get("error", f"HTTP {response.status}"),
                    )
        except Exception as e:
            return AgentTaskResponse(
                task_id="",
                agent_id="",
                content="",
                success=False,
                error_message=str(e),
            )

    async def send_generate_request(
        self,
        host: str,
        port: int,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> Optional[AgentTaskResponse]:
        """Send generation request to agent"""
        if self.session is None:
            raise RuntimeError("HTTPClient not initialized. Use as context manager or call __aenter__.")
        try:
            url = f"http://{host}:{port}/v1/completions"
            payload = {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            async with self.session.post(url, json=payload) as response:
                data = await response.json()
                if response.status == 200:
                    # OpenAI-compatible /v1/completions format
                    choices = data.get("choices", [])
                    content = choices[0].get("text", "") if choices else data.get("response", "")
                    return AgentTaskResponse(
                        task_id=data.get("task_id", ""),
                        agent_id=data.get("agent_id", ""),
                        content=content,
                        is_final=True,
                        processing_time_ms=data.get("processing_time_ms", 0),
                        success=True,
                        error_message="",
                    )
                else:
                    return AgentTaskResponse(
                        task_id="",
                        agent_id="",
                        content="",
                        success=False,
                        error_message=data.get("error", f"HTTP {response.status}"),
                    )
        except Exception as e:
            return AgentTaskResponse(
                task_id="",
                agent_id="",
                content="",
                success=False,
                error_message=str(e),
            )


# ============================================
# ORCHESTRATOR HTTP CLIENT
# ============================================


class OrchestratorHTTPClient:
    """HTTP client for connecting to orchestrator as agent"""

    def __init__(self, orchestrator_url: str, timeout_seconds: int = 120):
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.session: Optional[aiohttp.ClientSession] = None
        self.agent_id = f"agent-http-{uuid.uuid4().hex[:8]}"

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def register(self, registration: AgentRegistration) -> bool:
        """Register with orchestrator"""
        if self.session is None:
            raise RuntimeError("HTTPClient not initialized. Use as context manager or call __aenter__.")
        try:
            url = f"{self.orchestrator_url}/agents/register"
            async with self.session.post(url, json=registration.model_dump()) as response:
                return response.status == 200
        except Exception as e:
            print(f"Failed to register: {e}")
            return False

    async def send_task(self, task: AgentTaskRequest, agent_id: Optional[str] = None) -> Optional[AgentTaskResponse]:
        """Send task to orchestrator for a specific agent"""
        if self.session is None:
            raise RuntimeError("HTTPClient not initialized. Use as context manager or call __aenter__.")
        target_id = agent_id or task.requester_id
        try:
            url = f"{self.orchestrator_url}/agents/{target_id}/task"
            async with self.session.post(url, json=task.model_dump()) as response:
                if response.status == 200:
                    data = await response.json()
                    return AgentTaskResponse(**data)
        except Exception as e:
            print(f"Failed to send task: {e}")
        return None
