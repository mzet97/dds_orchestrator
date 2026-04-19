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


@dataclass(slots=True)
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
    # Transports this agent can actually serve requests on.
    # Filled from the agent's registration payload. Selectors MUST
    # filter on it — without this, the orchestrator will happily route
    # a gRPC request to an agent whose llama-server has no gRPC build
    # (observed: 97.8% failure rate on heterogeneous clusters).
    # Defaults to ["http"] since every agent speaks HTTP.
    transports: List[str] = field(default_factory=lambda: ["http"])
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

    _EMA_ALPHA = 0.1
    _EMA_COMPLEMENT = 0.9  # 1 - _EMA_ALPHA

    def __init__(self, config):
        self.config = config
        self.agents: Dict[str, AgentInfo] = {}
        # Dual-lock regime (intentional, not a bug):
        #   * ``_lock`` / ``agent_available_condition`` guards async call sites
        #     (aiohttp handlers, scheduler) and supports Condition.notify().
        #   * ``_thread_lock`` guards sync call sites running in thread pools
        #     (notably sync gRPC servicers in grpc_layer). Acquiring an
        #     asyncio.Lock from a thread would deadlock the loop.
        # The two locks do NOT serialize against each other. Treat the fields
        # they protect (slots_idle, status, last_heartbeat) as eventually
        # consistent across regimes — authoritative slot accounting lives in
        # the Redis InstancePool; this registry view is a cache. Do NOT mix
        # the two locks in one critical section.
        self._lock = asyncio.Lock()
        self.agent_available_condition = asyncio.Condition(self._lock)
        import threading
        self._thread_lock = threading.Lock()
        # Main asyncio loop used to wake async waiters after sync slot
        # adjustments. Set by the orchestrator after the loop starts.
        self._main_loop = None

    def bind_main_loop(self, loop) -> None:
        """Attach the main asyncio loop so sync slot releases can notify
        async waiters via call_soon_threadsafe."""
        self._main_loop = loop

    def _notify_async_waiters_from_thread(self) -> None:
        """Schedule a condition notify on the main loop. No-op if the loop
        is not bound yet (startup race). Safe to call from any thread."""
        loop = self._main_loop
        if loop is None or not loop.is_running():
            return

        async def _notify():
            async with self.agent_available_condition:
                self.agent_available_condition.notify_all()

        try:
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_notify()))
        except RuntimeError:
            # Loop closing; drop the notify — no one will be waiting anyway.
            pass

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

        async with self.agent_available_condition:
            action = "Updating" if agent_info.agent_id in self.agents else "Registering new"
            logger.info(f"{action} agent {agent_info.agent_id}")
            self.agents[agent_info.agent_id] = agent_info
            if agent_info.slots_idle > 0:
                self.agent_available_condition.notify(1)

            return agent_info.agent_id

    def _agent_info_from_dict(self, data: dict) -> AgentInfo:
        """Create an AgentInfo from a dictionary, ignoring unknown fields."""
        known_fields = {f.name for f in AgentInfo.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return AgentInfo(**filtered)

    def _infer_agent_profile(self, agent: AgentInfo) -> str:
        """
        Infer agent profile automatically based on metrics.

        Profiles:
        - "fast": latency < 200ms OR GPU type is high-end (M3, M2, A100)
        - "quality": latency > 800ms (slower but more capable models)
        - "backup": error_rate > 20% (fallback agent)
        - "balanced": default
        """
        if agent.error_rate > 0.2:
            return "backup"

        # Check GPU type for fast classification
        if agent.gpu_type:
            fast_gpus = ["M3", "M2", "M1", "A100", "A40", "A6000", "H100"]
            for gpu in fast_gpus:
                if gpu in agent.gpu_type:
                    return "fast"

        # Check latency-based classification
        if agent.avg_latency_ms > 0 and agent.avg_latency_ms < 200:
            return "fast"
        elif agent.avg_latency_ms > 800:
            return "quality"

        return "balanced"

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

    async def get_available_agents(self, transport: Optional[str] = None) -> List[AgentInfo]:
        """Get agents that are idle and available.

        When ``transport`` is given ("http"/"grpc"/"dds"), only agents whose
        ``transports`` list declares support for it are returned. Without this
        filter, heterogeneous clusters route requests to agents whose
        llama-server has no matching build, producing silent failures.
        """
        async with self._lock:
            return [
                agent for agent in self.agents.values()
                if agent.status == "idle" and agent.slots_idle > 0
                and (transport is None
                     or transport in getattr(agent, "transports", ["http"]))
            ]

    async def update_heartbeat(self, agent_id: str, status: str = None,
                               slots_idle: int = None, memory_usage_mb: int = None) -> bool:
        """Update agent heartbeat"""
        async with self.agent_available_condition:
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

            if agent.slots_idle > 0 and agent.status == "idle":
                self.agent_available_condition.notify(1)

            return True

    async def adjust_slots(self, agent_id: str, delta: int, status: str = None) -> bool:
        """Atomically adjust slots_idle by delta (positive = release, negative = acquire).
        If status is not provided, auto-derives: 'idle' when slots_idle > 0, 'busy' when 0."""
        async with self.agent_available_condition:
            agent = self.agents.get(agent_id)
            if not agent:
                return False
            agent.slots_idle = max(0, agent.slots_idle + delta)
            if status is not None:
                agent.status = status
            else:
                agent.status = "idle" if agent.slots_idle > 0 else "busy"
            agent.last_heartbeat = time.time()
            if agent.slots_idle > 0:
                self.agent_available_condition.notify(1)
            return True

    def adjust_slots_sync(self, agent_id: str, delta: int, status: str = None) -> bool:
        """Thread-safe slot adjustment callable from sync (thread-pool) handlers.

        Mirrors ``adjust_slots`` but uses ``_thread_lock`` so sync gRPC
        servicers can mutate slot state without switching to the asyncio
        loop. Callers must NOT mix this with ``adjust_slots`` for the same
        agent in the same critical section — use one regime per call site.
        """
        became_available = False
        with self._thread_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                return False
            prev_slots = agent.slots_idle
            new_val = agent.slots_idle + delta
            if delta >= 0:
                agent.slots_idle = min(new_val, getattr(agent, "slots_total", new_val))
            else:
                agent.slots_idle = max(0, new_val)
            if status is not None:
                agent.status = status
            else:
                agent.status = "idle" if agent.slots_idle > 0 else "busy"
            agent.last_heartbeat = time.time()
            became_available = prev_slots == 0 and agent.slots_idle > 0

        # Wake async waiters parked on agent_available_condition. Critical for
        # the sync gRPC path: without this, the fair-waiter only unblocks on
        # heartbeat/register and requests time out at the 60s wait cap while
        # slots actually sit idle.
        if became_available or delta > 0:
            self._notify_async_waiters_from_thread()
        return True

    async def update_response_metrics(self, agent_id: str, latency_ms: float, success: bool):
        """Update running metrics after a response is received.

        Uses exponential moving average (alpha=0.1, ~50-sample window) for latency.
        Automatically infers and updates agent_profile based on metrics.
        """
        async with self._lock:
            agent = self.agents.get(agent_id)
            if not agent:
                return
            agent.total_count += 1
            if success:
                agent.success_count += 1
            # Exponential moving average for latency
            if agent.avg_latency_ms == 0.0:
                agent.avg_latency_ms = latency_ms  # First sample
            else:
                agent.avg_latency_ms = self._EMA_ALPHA * latency_ms + self._EMA_COMPLEMENT * agent.avg_latency_ms
            # Error rate from accumulated counts
            agent.error_rate = 1.0 - (agent.success_count / max(1, agent.total_count))

            # Auto-infer agent profile based on current metrics
            old_profile = agent.agent_profile
            agent.agent_profile = self._infer_agent_profile(agent)
            if old_profile != agent.agent_profile:
                logger.debug(f"Agent {agent_id} profile changed: {old_profile} → {agent.agent_profile} "
                           f"(latency={agent.avg_latency_ms:.1f}ms, error_rate={agent.error_rate:.2%})")

    async def remove_stale_agents(self, timeout_seconds: int = None) -> List[tuple]:
        """Remove agents that haven't sent heartbeat.

        Returns list of (agent_id, grpc_address) tuples so callers can
        clean up transport-layer caches (e.g. gRPC channel pools) that
        are keyed by the agent URL, not the id.
        """
        timeout = timeout_seconds or self.config.agent_timeout_seconds
        current_time = time.time()
        stale = []

        async with self._lock:
            for agent_id, agent in list(self.agents.items()):
                if current_time - agent.last_heartbeat > timeout:
                    logger.warning(f"Agent {agent_id} stale. Last heartbeat: {current_time - agent.last_heartbeat:.1f}s ago")
                    stale.append((agent_id, getattr(agent, "grpc_address", "") or ""))
                    self.agents.pop(agent_id, None)

        if stale:
            logger.warning(f"Removed {len(stale)} stale agents")

        return stale

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
