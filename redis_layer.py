"""
Redis State Manager for the Orchestrator
Provides atomic slot management, EMA latency tracking, rate limiting,
and load test coordination via Redis.
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Lua script: atomic slot acquisition
# Returns 1 if acquired, 0 if full
_LUA_ACQUIRE_SLOT = """
local used = redis.call('INCR', KEYS[1])
local total = tonumber(redis.call('GET', KEYS[2]) or '0')
if used > total then
    redis.call('DECR', KEYS[1])
    return 0
end
return 1
"""

# Lua script: atomic EMA update
# Returns new EMA value as string
_LUA_EMA_UPDATE = """
local old = tonumber(redis.call('GET', KEYS[1]) or '0')
local alpha = tonumber(ARGV[2])
local new_val = tonumber(ARGV[1]) * alpha + old * (1 - alpha)
redis.call('SET', KEYS[1], tostring(new_val))
return tostring(new_val)
"""

# Lua script: atomic slot release (DECR with floor at 0)
_LUA_RELEASE_SLOT = """
local used = tonumber(redis.call('GET', KEYS[1]) or '0')
if used > 0 then
    redis.call('DECR', KEYS[1])
    return used - 1
end
return 0
"""


class RedisStateManager:
    """Atomic state management via Redis for instance routing."""

    def __init__(self, url: str = "redis://localhost:6379", password: str = ""):
        self._url = url
        self._password = password
        self._redis = None
        self._acquire_sha = None
        self._ema_sha = None

    async def connect(self):
        """Connect to Redis and register Lua scripts."""
        import redis.asyncio as aioredis

        if self._redis:
            await self.close()

        kwargs = {"decode_responses": True}
        if self._password:
            kwargs["password"] = self._password

        self._redis = aioredis.from_url(self._url, **kwargs)
        await self._redis.ping()
        logger.info(f"Connected to Redis at {self._url}")

        # Register Lua scripts
        self._acquire_sha = await self._redis.script_load(_LUA_ACQUIRE_SLOT)
        self._ema_sha = await self._redis.script_load(_LUA_EMA_UPDATE)
        self._release_sha = await self._redis.script_load(_LUA_RELEASE_SLOT)
        logger.info("Lua scripts registered")

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    # === Instance Initialization ===

    async def init_instance(self, port: int, slots_total: int, inst_type: str):
        """Initialize an instance's state in Redis."""
        pipe = self._redis.pipeline()
        pipe.set(f"inst:{port}:slots_used", 0)
        pipe.set(f"inst:{port}:slots_total", slots_total)
        pipe.set(f"inst:{port}:type", inst_type)
        pipe.set(f"inst:{port}:avg_latency", 0)
        pipe.set(f"inst:{port}:health", "alive", ex=30)
        await pipe.execute()

    # === Slot Management (Lua for atomicity) ===

    async def acquire_slot(self, port: int) -> bool:
        """Atomically acquire a slot. Returns True if successful."""
        result = await self._redis.evalsha(
            self._acquire_sha, 2,
            f"inst:{port}:slots_used",
            f"inst:{port}:slots_total",
        )
        return int(result) == 1

    async def release_slot(self, port: int):
        """Release a slot atomically (DECR with floor at 0 via Lua)."""
        await self._redis.evalsha(
            self._release_sha, 1,
            f"inst:{port}:slots_used",
        )

    async def get_slots(self, port: int) -> tuple[int, int]:
        """Get (slots_used, slots_total) for an instance."""
        pipe = self._redis.pipeline()
        pipe.get(f"inst:{port}:slots_used")
        pipe.get(f"inst:{port}:slots_total")
        used, total = await pipe.execute()
        return int(used or 0), int(total or 0)

    async def get_all_loads(self) -> list[dict]:
        """Get load info for all instances via pipeline."""
        # First get all instance ports
        keys = []
        cursor = "0"
        while True:
            cursor, batch = await self._redis.scan(cursor=cursor, match="inst:*:slots_total", count=100)
            keys.extend(batch)
            if cursor == "0" or cursor == 0:
                break

        if not keys:
            return []

        ports = [k.split(":")[1] for k in keys]
        pipe = self._redis.pipeline()
        for port in ports:
            pipe.get(f"inst:{port}:slots_used")
            pipe.get(f"inst:{port}:slots_total")
            pipe.get(f"inst:{port}:type")
            pipe.get(f"inst:{port}:avg_latency")
            pipe.exists(f"inst:{port}:health")

        results = await pipe.execute()
        loads = []
        for i, port in enumerate(ports):
            base = i * 5
            loads.append({
                "port": int(port),
                "slots_used": int(results[base] or 0),
                "slots_total": int(results[base + 1] or 0),
                "type": results[base + 2] or "unknown",
                "avg_latency": float(results[base + 3] or 0),
                "healthy": bool(results[base + 4]),
            })
        return loads

    # === EMA Latency ===

    async def update_latency(self, port: int, latency_ms: float, alpha: float = 0.1):
        """Update EMA latency for an instance atomically."""
        await self._redis.evalsha(
            self._ema_sha, 1,
            f"inst:{port}:avg_latency",
            str(latency_ms), str(alpha),
        )

    async def get_latency(self, port: int) -> float:
        """Get current EMA latency for an instance."""
        val = await self._redis.get(f"inst:{port}:avg_latency")
        return float(val or 0)

    # === Health (TTL-based) ===

    async def update_health(self, port: int):
        """Mark instance as healthy (30s TTL)."""
        await self._redis.set(f"inst:{port}:health", "alive", ex=30)

    async def is_healthy(self, port: int) -> bool:
        """Check if instance health key exists (TTL not expired)."""
        return bool(await self._redis.exists(f"inst:{port}:health"))

    # === Round-Robin ===

    async def get_rr_index(self) -> int:
        """Get and increment the round-robin index."""
        return await self._redis.incr("routing:rr_idx")

    # === Rate Limiting ===

    async def check_rate_limit(self, max_rps: int = 5000) -> bool:
        """Check if request is within rate limit. Returns True if allowed."""
        epoch_s = int(time.time())
        key = f"rate:{epoch_s}"
        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 2)
        results = await pipe.execute()
        count = results[0]
        return count <= max_rps

    # === Active Request Counter ===

    async def incr_active(self) -> int:
        """Increment active request count."""
        return await self._redis.incr("stats:active")

    async def decr_active(self) -> int:
        """Decrement active request count atomically (floor at 0)."""
        return int(await self._redis.evalsha(
            self._release_sha, 1, "stats:active",
        ))

    async def get_active(self) -> int:
        """Get active request count."""
        val = await self._redis.get("stats:active")
        return int(val or 0)

    # === Load Test Coordination ===

    async def signal_ready(self, worker_id: str):
        """Signal that a load test worker is ready."""
        await self._redis.sadd("load:ready", worker_id)
        await self._redis.expire("load:ready", 300)

    async def get_ready_count(self) -> int:
        """Get number of ready workers."""
        return await self._redis.scard("load:ready")

    async def signal_start(self):
        """Signal load test start."""
        await self._redis.rpush("load:start", "go")

    async def wait_start(self, timeout_s: int = 60) -> bool:
        """Wait for load test start signal. Returns True if received."""
        result = await self._redis.blpop("load:start", timeout=timeout_s)
        return result is not None

    async def push_result(self, result: dict):
        """Push a load test result."""
        import json
        await self._redis.rpush("load:results", json.dumps(result))

    async def collect_results(self, count: int, timeout_s: int = 60) -> list:
        """Collect load test results."""
        import json
        results = []
        for _ in range(count):
            item = await self._redis.blpop("load:results", timeout=timeout_s)
            if item is None:
                break
            results.append(json.loads(item[1]))
        return results

    # === Circuit Breaker State ===

    async def record_error(self, port: int):
        """Record an error for circuit breaker tracking."""
        key = f"inst:{port}:errors"
        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 10)  # 10s window
        await pipe.execute()

    async def record_success(self, port: int):
        """Record a success for circuit breaker tracking."""
        key = f"inst:{port}:successes"
        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 10)
        await pipe.execute()

    async def get_error_rate(self, port: int) -> float:
        """Get error rate in the last 10s window."""
        pipe = self._redis.pipeline()
        pipe.get(f"inst:{port}:errors")
        pipe.get(f"inst:{port}:successes")
        errors_str, successes_str = await pipe.execute()
        errors = int(errors_str or 0)
        successes = int(successes_str or 0)
        total = errors + successes
        if total == 0:
            return 0.0
        return errors / total

    # === Cleanup ===

    async def cleanup(self):
        """Remove all orchestrator keys (for test teardown)."""
        cursor = "0"
        while True:
            cursor, keys = await self._redis.scan(cursor=cursor, match="inst:*", count=100)
            if keys:
                await self._redis.delete(*keys)
            if cursor == "0" or cursor == 0:
                break
        for key in ("routing:rr_idx", "stats:active", "load:ready", "load:start", "load:results"):
            await self._redis.delete(key)
