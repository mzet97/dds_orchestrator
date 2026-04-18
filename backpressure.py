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
        # Protect _circuit_open_until against concurrent readers/writers.
        # Without this, two requests arriving for the same port can race on
        # the error-rate check and cause the breaker to flap open/closed.
        self._circuit_lock = asyncio.Lock()

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
        # Fast path: read-only check under lock (cheap; no I/O).
        async with self._circuit_lock:
            open_until = self._circuit_open_until.get(port, 0)
            now = time.time()
            if now < open_until:
                return True  # Still in cooldown

        # Error-rate fetch is I/O (Redis) — do it OUTSIDE the lock.
        error_rate = await self._redis.get_error_rate(port)

        async with self._circuit_lock:
            now = time.time()
            if error_rate > self._circuit_error_threshold:
                # Re-check to avoid clobbering a breaker that another coroutine
                # just opened with a later deadline.
                existing = self._circuit_open_until.get(port, 0)
                new_deadline = now + self._circuit_cooldown_s
                if new_deadline > existing:
                    self._circuit_open_until[port] = new_deadline
                    logger.warning(f"Circuit breaker OPEN for port {port} (error_rate={error_rate:.1%})")
                return True

            if port in self._circuit_open_until and now >= self._circuit_open_until[port]:
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
        """Get list of ports with open circuit breakers.

        Note: snapshot read under GIL; a new asyncio lock cannot be held from
        sync context. Callers must treat the result as advisory.
        """
        now = time.time()
        # Copy .items() to tolerate concurrent mutation by async writers.
        return [port for port, until in list(self._circuit_open_until.items()) if now < until]
