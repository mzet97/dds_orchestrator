"""
Agent Registry - manages registered agents in the system
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional
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


class AgentRegistry:
    """Registry for managing agents in the orchestrator"""

    def __init__(self, config):
        self.config = config
        self.agents: Dict[str, AgentInfo] = {}
        self._lock = asyncio.Lock()

    async def register_agent(self, agent_info: AgentInfo) -> bool:
        """Register a new agent or update existing"""
        async with self._lock:
            existing = self.agents.get(agent_info.agent_id)

            if existing:
                # Update existing agent
                logger.info(f"Updating agent {agent_info.agent_id}")
                self.agents[agent_info.agent_id] = agent_info
            else:
                # New registration
                logger.info(f"Registering new agent {agent_info.agent_id}")
                self.agents[agent_info.agent_id] = agent_info

            return True

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        async with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
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

            if status:
                agent.status = status
            if slots_idle is not None:
                agent.slots_idle = slots_idle
            if memory_usage_mb is not None:
                agent.vram_available_mb = max(0, agent.vram_available_mb - memory_usage_mb)

            return True

    async def remove_stale_agents(self, timeout_seconds: int = None) -> List[str]:
        """Remove agents that haven't sent heartbeat"""
        timeout = timeout_seconds or self.config.agent_timeout_seconds
        current_time = time.time()
        stale_ids = []

        async with self._lock:
            for agent_id, agent in list(self.agents.items()):
                if current_time - agent.last_heartbeat > timeout:
                    stale_ids.append(agent_id)
                    del self.agents[agent_id]

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
            return min(available, key=lambda a: a.slots_idle)

        # Filter by requirements
        filtered = available

        if requirements.get("vision_enabled"):
            filtered = [a for a in filtered if a.vision_enabled]

        if requirements.get("model"):
            filtered = [a for a in filtered if a.model == requirements["model"]]

        if not filtered:
            return None

        return min(filtered, key=lambda a: a.slots_idle)

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
