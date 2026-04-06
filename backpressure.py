"""
Backpressure Manager
Rate limiting + circuit breaker for overload protection.
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class BackpressureManager:
    """Rate limiting + circuit breaker to protect against overload."""

    def __init__(self, redis, max_rps: int = 5000,
                 circuit_error_threshold: float = 0.8,
                 circuit_cooldown_s: int = 30):
        self._redis = redis
        self._max_rps = max_rps
        self._circuit_error_threshold = circuit_error_threshold
        self._circuit_cooldown_s = circuit_cooldown_s
        # Track circuit breaker state per instance (port -> open_until timestamp)
        self._circuit_open_until: dict[int, float] = {}

    async def allow_request(self) -> bool:
        """Check if the request is allowed under the rate limit."""
        return await self._redis.check_rate_limit(self._max_rps)

    async def record_result(self, port: int, success: bool):
        """Record request result for circuit breaker tracking."""
        if success:
            await self._redis.record_success(port)
        else:
            await self._redis.record_error(port)

    async def is_circuit_open(self, port: int) -> bool:
        """Check if circuit breaker is open for an instance.

        Opens when error rate > 80% in 10s window.
        Closes after 30s cooldown.
        """
        # Check cooldown first
        open_until = self._circuit_open_until.get(port, 0)
        now = time.time()
        if now < open_until:
            return True  # Still in cooldown

        # Check error rate
        error_rate = await self._redis.get_error_rate(port)
        if error_rate > self._circuit_error_threshold:
            self._circuit_open_until[port] = now + self._circuit_cooldown_s
            logger.warning(f"Circuit breaker OPEN for port {port} (error_rate={error_rate:.1%})")
            return True

        # Circuit is closed (or was closed after cooldown)
        if port in self._circuit_open_until and now >= open_until:
            logger.info(f"Circuit breaker CLOSED for port {port}")
            del self._circuit_open_until[port]

        return False

    async def get_pressure_level(self) -> str:
        """Get current system pressure level.

        Returns: "normal" | "elevated" | "critical"
        """
        # Get total available slots
        loads = await self._redis.get_all_loads()
        total_slots = sum(l.get("slots_total", 0) for l in loads)
        total_used = sum(l.get("slots_used", 0) for l in loads)

        if total_slots == 0:
            return "critical"

        utilization = total_used / total_slots

        if utilization > 0.9:
            return "critical"
        elif utilization > 0.7:
            return "elevated"
        return "normal"

    def get_open_circuits(self) -> list[int]:
        """Get list of ports with open circuit breakers."""
        now = time.time()
        return [port for port, until in self._circuit_open_until.items() if now < until]
