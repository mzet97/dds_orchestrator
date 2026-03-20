"""
Instance Pool with 3 Routing Algorithms
Routes requests to 38 llama.cpp instances using Redis-backed atomic slots.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RoutingAlgorithm(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_SCORE = "weighted_score"


@dataclass
class InstanceInfo:
    port: int
    hostname: str = "192.168.1.61"
    inst_type: str = "gpu"        # "gpu" or "cpu"
    model: str = "Qwen3.5-2B"
    slots_total: int = 15         # 15 GPU, 4 CPU
    healthy: bool = True
    weight: float = 1.0           # GPU=1.0, CPU=0.3


class InstancePool:
    """Pool of 38 instances with O(38)=O(1) routing algorithms."""

    def __init__(self, redis, mongo=None,
                 algorithm: RoutingAlgorithm = RoutingAlgorithm.LEAST_LOADED,
                 w_load: float = 0.6, w_latency: float = 0.4):
        self._redis = redis
        self._mongo = mongo
        self._algorithm = algorithm
        self._w_load = w_load
        self._w_latency = w_latency
        self._instances: dict[int, InstanceInfo] = {}

    async def register_instance(self, info: InstanceInfo):
        """Register an instance in the pool."""
        self._instances[info.port] = info
        await self._redis.init_instance(info.port, info.slots_total, info.inst_type)
        if self._mongo:
            await self._mongo.register_instance({
                "port": info.port,
                "hostname": info.hostname,
                "type": info.inst_type,
                "model": info.model,
                "slots_total": info.slots_total,
                "weight": info.weight,
            })
        logger.info(f"Registered instance :{info.port} ({info.inst_type}, {info.slots_total} slots)")

    def set_algorithm(self, algo: RoutingAlgorithm):
        """Change routing algorithm at runtime."""
        self._algorithm = algo
        logger.info(f"Routing algorithm changed to {algo.value}")

    async def select_instance(self, prefer_type: str = None) -> Optional[InstanceInfo]:
        """Select best instance using current algorithm.

        Returns InstanceInfo with an acquired slot, or None if all full.
        """
        if self._algorithm == RoutingAlgorithm.ROUND_ROBIN:
            return await self._select_round_robin(prefer_type)
        elif self._algorithm == RoutingAlgorithm.LEAST_LOADED:
            return await self._select_least_loaded(prefer_type)
        elif self._algorithm == RoutingAlgorithm.WEIGHTED_SCORE:
            return await self._select_weighted_score(prefer_type)
        return None

    async def release_instance(self, port: int, latency_ms: float, success: bool):
        """Release a slot and update metrics."""
        await self._redis.release_slot(port)
        await self._redis.update_latency(port, latency_ms)
        await self._redis.update_health(port)

        if success:
            await self._redis.record_success(port)
        else:
            await self._redis.record_error(port)

    async def get_status(self) -> dict:
        """Get current status of all instances."""
        loads = await self._redis.get_all_loads()
        active = await self._redis.get_active()
        return {
            "algorithm": self._algorithm.value,
            "total_instances": len(self._instances),
            "active_requests": active,
            "instances": loads,
        }

    # === Routing Algorithms ===

    async def _select_round_robin(self, prefer_type: str = None) -> Optional[InstanceInfo]:
        """Round-robin: INCR index % len(healthy), try acquire."""
        loads = await self._redis.get_all_loads()
        healthy = self._filter_healthy(loads, prefer_type)
        if not healthy:
            if prefer_type:
                healthy = self._filter_healthy(loads, prefer_type=None)
            if not healthy:
                return None

        idx = await self._redis.get_rr_index()
        n = len(healthy)

        # Try each instance starting from rr index (INCR returns 1-based)
        for offset in range(n):
            inst = healthy[(idx - 1 + offset) % n]
            if await self._redis.acquire_slot(inst.port):
                return inst
        return None

    async def _select_least_loaded(self, prefer_type: str = None) -> Optional[InstanceInfo]:
        """Least-loaded: pick instance with lowest slots_used/slots_total ratio."""
        loads = await self._redis.get_all_loads()
        if not loads:
            return None

        # Filter healthy and by type preference
        candidates = []
        for load in loads:
            port = load["port"]
            inst = self._instances.get(port)
            if not inst or not load["healthy"]:
                continue
            if prefer_type and inst.inst_type != prefer_type:
                continue
            ratio = load["slots_used"] / max(load["slots_total"], 1)
            candidates.append((ratio, port, inst))

        if not candidates:
            # Fallback: try all types
            if prefer_type:
                return await self._select_least_loaded(prefer_type=None)
            return None

        # Sort by load ratio ascending
        candidates.sort(key=lambda x: x[0])

        for _, port, inst in candidates:
            if await self._redis.acquire_slot(port):
                return inst
        return None

    async def _select_weighted_score(self, prefer_type: str = None) -> Optional[InstanceInfo]:
        """Weighted score: score = w_load * (used/total) + w_latency * norm(latency).
        GPU weight=1.0 vs CPU weight=0.3.
        """
        loads = await self._redis.get_all_loads()
        if not loads:
            return None

        # First pass: collect eligible candidates
        eligible = []
        for load in loads:
            port = load["port"]
            inst = self._instances.get(port)
            if not inst or not load["healthy"]:
                continue
            if prefer_type and inst.inst_type != prefer_type:
                continue
            eligible.append((load, inst))

        if not eligible:
            if prefer_type:
                return await self._select_weighted_score(prefer_type=None)
            return None

        # Normalize latency across eligible candidates only
        max_latency = max((l["avg_latency"] for l, _ in eligible), default=1.0)
        if max_latency == 0:
            max_latency = 1.0

        candidates = []
        for load, inst in eligible:

            load_ratio = load["slots_used"] / max(load["slots_total"], 1)
            latency_norm = load["avg_latency"] / max_latency
            score = self._w_load * load_ratio + self._w_latency * latency_norm

            # Apply weight factor: lower weight = higher adjusted score
            # GPU (weight=1.0) is preferred over CPU (weight=0.3)
            adjusted_score = score / max(inst.weight, 0.01)
            # Tie-breaker: prefer higher weight when scores equal (e.g. at startup)
            candidates.append((adjusted_score, -inst.weight, load["port"], inst))

        candidates.sort(key=lambda x: (x[0], x[1]))

        for _, _, port, inst in candidates:
            if await self._redis.acquire_slot(port):
                return inst
        return None

    def _filter_healthy(self, loads: list[dict], prefer_type: str = None) -> list[InstanceInfo]:
        """Filter healthy instances from pre-fetched loads (0 Redis calls)."""
        result = []
        for load in loads:
            port = load["port"]
            inst = self._instances.get(port)
            if not inst or not load["healthy"]:
                continue
            if prefer_type and inst.inst_type != prefer_type:
                continue
            result.append(inst)
        return result
