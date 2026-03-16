"""
Agent Registry - manages registered agents in the system
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Information about a registered agent"""
    agent_id: str
    hostname: str
    port: int
    model: str
    model_path: str = ""
    specialization: str = "generic"  # text, vision, embedding, generic
    vram_available_mb: int = 0
    slots_idle: int = 1
    slots_total: int = 1
    vision_enabled: bool = False
    status: str = "idle"  # idle, busy, error
    last_heartbeat: float = field(default_factory=time.time)
    registered_at: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)
    llm_host: str = "localhost"  # Host da LLM que este agente gerencia
    llm_port: int = 8082  # Porta da LLM
    grpc_address: str = ""  # gRPC endpoint (host:port) if agent supports gRPC
    # Fuzzy decision metrics (updated after each response)
    avg_latency_ms: float = 0.0       # Exponential moving average of response latency
    error_rate: float = 0.0            # Error rate from accumulated counts
    success_count: int = 0             # Accumulated successful responses
    total_count: int = 0               # Accumulated total responses
    agent_profile: str = "balanced"    # "fast" | "quality" | "balanced" | "backup"
    gpu_type: str = ""                 # "rtx3080" | "rx6600m" | ""


class AgentRegistry:
    """Registry for managing agents in the orchestrator"""

    def __init__(self, config):
        self.config = config
        self.agents: Dict[str, AgentInfo] = {}
        self._lock = asyncio.Lock()

    async def register_agent(self, agent_info) -> str:
        """Register a new agent or update existing.

        Accepts an AgentInfo dataclass, a Pydantic model with agent_id, or a dict.
        Returns the agent_id of the registered agent.
        """
        # Convert dict or Pydantic model to AgentInfo dataclass if needed
        if isinstance(agent_info, dict):
            agent_info = self._agent_info_from_dict(agent_info)
        elif not isinstance(agent_info, AgentInfo):
            # Assume Pydantic model or similar object with attributes
            agent_info = self._agent_info_from_dict(
                agent_info.model_dump() if hasattr(agent_info, 'model_dump')
                else agent_info.__dict__
            )

        async with self._lock:
            action = "Updating" if agent_info.agent_id in self.agents else "Registering new"
            logger.info(f"{action} agent {agent_info.agent_id}")
            self.agents[agent_info.agent_id] = agent_info

            return agent_info.agent_id

    def _agent_info_from_dict(self, data: dict) -> AgentInfo:
        """Create an AgentInfo from a dictionary, ignoring unknown fields."""
        known_fields = {f.name for f in AgentInfo.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return AgentInfo(**filtered)

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        async with self._lock:
            if agent_id in self.agents:
                self.agents.pop(agent_id, None)
                logger.info(f"Unregistered agent {agent_id}")
                return True
            return False

    async def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent by ID"""
        async with self._lock:
            return self.agents.get(agent_id)

    async def get_all_agents(self) -> List[AgentInfo]:
        """Get all registered agents"""
        async with self._lock:
            return list(self.agents.values())

    async def get_available_agents(self) -> List[AgentInfo]:
        """Get agents that are idle and available"""
        async with self._lock:
            return [
                agent for agent in self.agents.values()
                if agent.status == "idle" and agent.slots_idle > 0
            ]

    async def update_heartbeat(self, agent_id: str, status: str = None,
                               slots_idle: int = None, memory_usage_mb: int = None) -> bool:
        """Update agent heartbeat"""
        async with self._lock:
            agent = self.agents.get(agent_id)
            if not agent:
                return False

            agent.last_heartbeat = time.time()

            if status is not None:
                agent.status = status
            if slots_idle is not None:
                agent.slots_idle = slots_idle
            if memory_usage_mb is not None:
                agent.vram_available_mb = memory_usage_mb

            return True

    async def adjust_slots(self, agent_id: str, delta: int, status: str = None) -> bool:
        """Atomically adjust slots_idle by delta (positive = release, negative = acquire).
        If status is not provided, auto-derives: 'idle' when slots_idle > 0, 'busy' when 0."""
        async with self._lock:
            agent = self.agents.get(agent_id)
            if not agent:
                return False
            agent.slots_idle = max(0, agent.slots_idle + delta)
            if status is not None:
                agent.status = status
            else:
                agent.status = "idle" if agent.slots_idle > 0 else "busy"
            agent.last_heartbeat = time.time()
            return True

    async def update_response_metrics(self, agent_id: str, latency_ms: float, success: bool):
        """Update running metrics after a response is received.

        Uses exponential moving average (alpha=0.1, ~50-sample window) for latency.
        """
        async with self._lock:
            agent = self.agents.get(agent_id)
            if not agent:
                return
            agent.total_count += 1
            if success:
                agent.success_count += 1
            # Exponential moving average for latency
            alpha = 0.1
            if agent.avg_latency_ms == 0.0:
                agent.avg_latency_ms = latency_ms  # First sample
            else:
                agent.avg_latency_ms = alpha * latency_ms + (1 - alpha) * agent.avg_latency_ms
            # Error rate from accumulated counts
            agent.error_rate = 1.0 - (agent.success_count / max(1, agent.total_count))

    async def remove_stale_agents(self, timeout_seconds: int = None) -> List[str]:
        """Remove agents that haven't sent heartbeat"""
        timeout = timeout_seconds or self.config.agent_timeout_seconds
        current_time = time.time()
        stale_ids = []

        async with self._lock:
            for agent_id, agent in list(self.agents.items()):
                if current_time - agent.last_heartbeat > timeout:
                    stale_ids.append(agent_id)
                    self.agents.pop(agent_id, None)

        if stale_ids:
            logger.warning(f"Removed {len(stale_ids)} stale agents")

        return stale_ids

    async def select_agent(self, requirements: dict = None) -> Optional[AgentInfo]:
        """Select the best available agent based on requirements"""
        available = await self.get_available_agents()

        if not available:
            return None

        # Simple selection: least loaded
        if requirements is None:
            return max(available, key=lambda a: a.slots_idle)

        # Filter by requirements
        filtered = available

        if requirements.get("vision_enabled"):
            filtered = [a for a in filtered if a.vision_enabled]

        if requirements.get("model"):
            filtered = [a for a in filtered if a.model == requirements["model"]]

        if not filtered:
            return None

        return max(filtered, key=lambda a: a.slots_idle)

    async def get_online_agents(self) -> List[AgentInfo]:
        """Get agents that are not stale (have recent heartbeats)"""
        timeout = self.config.agent_timeout_seconds
        current_time = time.time()
        async with self._lock:
            return [
                agent for agent in self.agents.values()
                if current_time - agent.last_heartbeat <= timeout
            ]

    async def find_agents_by_model(self, model_id: str) -> List[AgentInfo]:
        """Find agents whose model or model_path contains model_id"""
        async with self._lock:
            return [
                agent for agent in self.agents.values()
                if model_id in agent.model or model_id in getattr(agent, 'model_path', '')
            ]

    async def get_stats(self) -> dict:
        """Get registry statistics"""
        async with self._lock:
            total = len(self.agents)
            idle = sum(1 for a in self.agents.values() if a.status == "idle")
            busy = sum(1 for a in self.agents.values() if a.status == "busy")
            error = sum(1 for a in self.agents.values() if a.status == "error")

            return {
                "total_agents": total,
                "idle_agents": idle,
                "busy_agents": busy,
                "error_agents": error,
                "online_agents": idle + busy,
            }
