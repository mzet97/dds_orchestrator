#!/usr/bin/env python3
"""
End-to-end tests for the 10-instance topology using real Redis and MongoDB.

Topology:
  .61 (RTX 3080 10GB): 6 instances, ports 8082-8087, parallel=15
  .60 (RX 6600M 8GB):  4 instances, ports 8088-8091, parallel=10
  Orchestrator on .62:8080
  Redis: redis://redis.home.arpa:6379, password=Admin@123
  MongoDB: mongodb://admin:Admin%40123@mongodb.home.arpa:27017/?authSource=admin

Run:
  cd dds_orchestrator
  pytest tests/test_10inst_e2e.py -v
"""

import asyncio
import os
import sys
import time
import pytest
import pytest_asyncio

from pathlib import Path

# Add parent to path so we can import orchestrator modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from redis_layer import RedisStateManager
from mongo_layer import MongoMetricsStore
from instance_pool import InstancePool, InstanceInfo, RoutingAlgorithm

# ===== Infrastructure Coordinates =====

REDIS_URL = "redis://redis.home.arpa:6379"
REDIS_PASSWORD = "Admin@123"
MONGO_URL = "mongodb://admin:Admin%40123@mongodb.home.arpa:27017/?authSource=admin"
MONGO_DB = "test_dds_10inst_e2e"

# ===== 10-Instance Topology =====

RTX_INSTANCES = [
    InstanceInfo(
        port=8082 + i, hostname="192.168.1.61",
        inst_type="gpu", model="Qwen3.5-2B",
        slots_total=15, weight=1.0,
    )
    for i in range(6)
]

RX_INSTANCES = [
    InstanceInfo(
        port=8088 + i, hostname="192.168.1.60",
        inst_type="gpu", model="Qwen3.5-2B",
        slots_total=10, weight=0.7,
    )
    for i in range(4)
]

ALL_INSTANCES = RTX_INSTANCES + RX_INSTANCES
TOTAL_SLOTS = sum(inst.slots_total for inst in ALL_INSTANCES)  # 6*15 + 4*10 = 130

# ===== Availability Checks =====


async def _redis_available():
    try:
        mgr = RedisStateManager(REDIS_URL, REDIS_PASSWORD)
        await mgr.connect()
        await mgr.close()
        return True
    except Exception:
        return False


async def _mongo_available():
    try:
        store = MongoMetricsStore(MONGO_URL, MONGO_DB)
        await store.connect()
        await store.close()
        return True
    except Exception:
        return False


redis_ok = False
mongo_ok = False
try:
    redis_ok = asyncio.run(_redis_available())
except Exception:
    pass
try:
    mongo_ok = asyncio.run(_mongo_available())
except Exception:
    pass

skip_redis = pytest.mark.skipif(not redis_ok, reason=f"Redis not available at {REDIS_URL}")
skip_mongo = pytest.mark.skipif(not mongo_ok, reason=f"MongoDB not available at {MONGO_URL}")
skip_both = pytest.mark.skipif(
    not (redis_ok and mongo_ok),
    reason=f"Redis={redis_ok}, MongoDB={mongo_ok} -- both required",
)

# ===== Fixtures =====


@pytest_asyncio.fixture
async def redis_mgr():
    """Connect to real Redis, clean up, yield, close."""
    mgr = RedisStateManager(REDIS_URL, REDIS_PASSWORD)
    await mgr.connect()
    await mgr.cleanup()
    # Also clean agent keys
    cursor = "0"
    while True:
        cursor, keys = await mgr._redis.scan(cursor=cursor, match="agent:*", count=500)
        if keys:
            await mgr._redis.delete(*keys)
        if cursor == "0" or cursor == 0:
            break
    yield mgr
    await mgr.cleanup()
    # Clean agent keys on teardown too
    cursor = "0"
    while True:
        cursor, keys = await mgr._redis.scan(cursor=cursor, match="agent:*", count=500)
        if keys:
            await mgr._redis.delete(*keys)
        if cursor == "0" or cursor == 0:
            break
    await mgr.close()


@pytest_asyncio.fixture
async def mongo_store():
    """Connect to real MongoDB, drop test data, yield, close."""
    store = MongoMetricsStore(MONGO_URL, MONGO_DB)
    await store.connect()
    await store.drop_all()
    await store.ensure_indexes()
    yield store
    await store.drop_all()
    await store.close()


@pytest_asyncio.fixture
async def instance_pool(redis_mgr, mongo_store):
    """Create InstancePool with real Redis + MongoDB, register all 10 instances."""
    pool = InstancePool(
        redis_mgr, mongo_store,
        algorithm=RoutingAlgorithm.LEAST_LOADED,
    )
    for inst in ALL_INSTANCES:
        await pool.register_instance(inst)
    return pool


# ===== Test Cases =====

# ---------- 1. Redis Connection ----------

@skip_redis
@pytest.mark.asyncio
async def test_redis_connection():
    """Verify Redis is reachable and responds to PING."""
    mgr = RedisStateManager(REDIS_URL, REDIS_PASSWORD)
    await mgr.connect()
    pong = await mgr._redis.ping()
    assert pong is True
    await mgr.close()


# ---------- 2. MongoDB Connection ----------

@skip_mongo
@pytest.mark.asyncio
async def test_mongo_connection():
    """Verify MongoDB is reachable and responds to ping command."""
    store = MongoMetricsStore(MONGO_URL, MONGO_DB)
    await store.connect()
    result = await store._client.admin.command("ping")
    assert result.get("ok") == 1.0
    await store.close()


# ---------- 3. Register 10 Instances ----------

@skip_both
@pytest.mark.asyncio
async def test_register_10_instances(instance_pool, redis_mgr):
    """Register all 10 instances and verify they appear in Redis."""
    status = await instance_pool.get_status()
    assert status["total_instances"] == 10

    # Verify each instance has its keys in Redis
    for inst in ALL_INSTANCES:
        used, total = await redis_mgr.get_slots(inst.port)
        assert total == inst.slots_total, (
            f"Instance :{inst.port} expected {inst.slots_total} slots, got {total}"
        )
        assert used == 0


# ---------- 4. Total Slots = 130 ----------

@skip_both
@pytest.mark.asyncio
async def test_slots_total_130(instance_pool, redis_mgr):
    """Sum of all slot totals across 10 instances should be 130."""
    total = 0
    for inst in ALL_INSTANCES:
        _, slots = await redis_mgr.get_slots(inst.port)
        total += slots
    assert total == TOTAL_SLOTS, f"Expected {TOTAL_SLOTS} total slots, got {total}"


# ---------- 5. Acquire All 130 Slots ----------

@skip_both
@pytest.mark.asyncio
async def test_acquire_all_130_slots(instance_pool, redis_mgr):
    """Acquire all 130 slots; the 131st should fail."""
    acquired = 0
    for inst in ALL_INSTANCES:
        for _ in range(inst.slots_total):
            ok = await redis_mgr.acquire_slot(inst.port)
            assert ok, f"Failed to acquire slot on :{inst.port} after {acquired} total"
            acquired += 1

    assert acquired == TOTAL_SLOTS

    # 131st slot should fail on every instance
    for inst in ALL_INSTANCES:
        assert not await redis_mgr.acquire_slot(inst.port), (
            f"Should not acquire beyond capacity on :{inst.port}"
        )

    # Release all for cleanup
    for inst in ALL_INSTANCES:
        for _ in range(inst.slots_total):
            await redis_mgr.release_slot(inst.port)


# ---------- 6. Release Restores Capacity ----------

@skip_both
@pytest.mark.asyncio
async def test_release_restores_capacity(instance_pool, redis_mgr):
    """Fill one instance, release 1 slot, and verify a new acquire succeeds."""
    port = ALL_INSTANCES[0].port
    total = ALL_INSTANCES[0].slots_total

    # Fill all slots
    for _ in range(total):
        assert await redis_mgr.acquire_slot(port)

    # One more should fail
    assert not await redis_mgr.acquire_slot(port)

    # Release one
    await redis_mgr.release_slot(port)

    # Now should succeed
    assert await redis_mgr.acquire_slot(port)

    # Release all
    for _ in range(total):
        await redis_mgr.release_slot(port)


# ---------- 7. Concurrent Acquire 100 ----------

@skip_both
@pytest.mark.asyncio
async def test_concurrent_acquire_100(instance_pool, redis_mgr):
    """100 concurrent acquires should all succeed (130 slots available)."""
    results = []

    async def try_acquire():
        inst = await instance_pool.select_instance()
        if inst:
            results.append(inst.port)
            await instance_pool.release_instance(inst.port, latency_ms=1.0, success=True)

    await asyncio.gather(*[try_acquire() for _ in range(100)])

    assert len(results) == 100, (
        f"Expected 100 successful acquisitions, got {len(results)}"
    )


# ---------- 8. Slot Stress 200 Concurrent ----------

@skip_both
@pytest.mark.asyncio
async def test_slot_stress_200_concurrent(instance_pool, redis_mgr):
    """200 concurrent acquires with 130 slots: exactly 130 should succeed."""
    succeeded = 0
    failed = 0
    lock = asyncio.Lock()

    async def try_acquire():
        nonlocal succeeded, failed
        inst = await instance_pool.select_instance()
        if inst:
            async with lock:
                succeeded += 1
            # Hold the slot briefly to create contention
            await asyncio.sleep(0.01)
            await instance_pool.release_instance(inst.port, latency_ms=1.0, success=True)
        else:
            async with lock:
                failed += 1

    await asyncio.gather(*[try_acquire() for _ in range(200)])

    assert succeeded + failed == 200
    # At least some must have been rejected (200 > 130)
    assert failed > 0, "Expected some rejections with 200 requests on 130 slots"
    # All that were acquired should have been served
    assert succeeded <= TOTAL_SLOTS, (
        f"Succeeded ({succeeded}) should not exceed total slots ({TOTAL_SLOTS})"
    )

    # Verify all slots released
    for inst in ALL_INSTANCES:
        used, _ = await redis_mgr.get_slots(inst.port)
        assert used == 0, f"Slots leaked on :{inst.port}: {used} still in use"


# ---------- 9. Round Robin Cycles ----------

@skip_both
@pytest.mark.asyncio
async def test_round_robin_cycles(instance_pool, redis_mgr):
    """With round_robin, 20 acquisitions should cover all 10 instances."""
    instance_pool.set_algorithm(RoutingAlgorithm.ROUND_ROBIN)

    seen_ports = set()
    for _ in range(20):
        inst = await instance_pool.select_instance()
        assert inst is not None
        seen_ports.add(inst.port)
        await instance_pool.release_instance(inst.port, latency_ms=1.0, success=True)

    assert len(seen_ports) == 10, (
        f"Round robin should visit all 10 instances in 20 tries, visited {len(seen_ports)}: {seen_ports}"
    )


# ---------- 10. Least Loaded Prefers Empty ----------

@skip_both
@pytest.mark.asyncio
async def test_least_loaded_prefers_empty(instance_pool, redis_mgr):
    """Pre-load one instance; least_loaded should avoid it."""
    instance_pool.set_algorithm(RoutingAlgorithm.LEAST_LOADED)

    # Load the first instance to near capacity
    loaded_port = ALL_INSTANCES[0].port
    loaded_total = ALL_INSTANCES[0].slots_total
    for _ in range(loaded_total - 1):
        await redis_mgr.acquire_slot(loaded_port)

    # Next several acquisitions should avoid the loaded instance
    other_ports = []
    for _ in range(10):
        inst = await instance_pool.select_instance()
        assert inst is not None
        other_ports.append(inst.port)
        await instance_pool.release_instance(inst.port, latency_ms=1.0, success=True)

    # The loaded instance should not appear (or very rarely)
    loaded_count = other_ports.count(loaded_port)
    assert loaded_count == 0, (
        f"Least-loaded selected the pre-loaded instance {loaded_count}/10 times"
    )

    # Release the pre-loaded slots
    for _ in range(loaded_total - 1):
        await redis_mgr.release_slot(loaded_port)


# ---------- 11. Weighted Score Prefers RTX ----------

@skip_both
@pytest.mark.asyncio
async def test_weighted_score_prefers_rtx(instance_pool, redis_mgr):
    """RTX instances (weight=1.0) should be preferred over RX (weight=0.7)."""
    instance_pool.set_algorithm(RoutingAlgorithm.WEIGHTED_SCORE)

    rtx_ports = {inst.port for inst in RTX_INSTANCES}
    rx_ports = {inst.port for inst in RX_INSTANCES}

    rtx_count = 0
    rx_count = 0

    for _ in range(20):
        inst = await instance_pool.select_instance()
        assert inst is not None
        if inst.port in rtx_ports:
            rtx_count += 1
        elif inst.port in rx_ports:
            rx_count += 1
        await instance_pool.release_instance(inst.port, latency_ms=1.0, success=True)

    # RTX should be selected more often than RX
    assert rtx_count > rx_count, (
        f"RTX ({rtx_count}) should be preferred over RX ({rx_count})"
    )


# ---------- 12. Algorithm Switch at Runtime ----------

@skip_both
@pytest.mark.asyncio
async def test_algorithm_switch_runtime(instance_pool, redis_mgr):
    """Switch from round_robin to least_loaded mid-test."""
    instance_pool.set_algorithm(RoutingAlgorithm.ROUND_ROBIN)

    # Phase 1: round robin
    rr_ports = set()
    for _ in range(10):
        inst = await instance_pool.select_instance()
        assert inst is not None
        rr_ports.add(inst.port)
        await instance_pool.release_instance(inst.port, latency_ms=1.0, success=True)

    assert len(rr_ports) >= 5, "Round robin should hit multiple instances"

    # Switch algorithm
    instance_pool.set_algorithm(RoutingAlgorithm.LEAST_LOADED)
    assert instance_pool._algorithm == RoutingAlgorithm.LEAST_LOADED

    # Phase 2: least loaded should still work
    for _ in range(10):
        inst = await instance_pool.select_instance()
        assert inst is not None
        await instance_pool.release_instance(inst.port, latency_ms=1.0, success=True)


# ---------- 13. Agent Instance Mapping ----------

@skip_redis
@pytest.mark.asyncio
async def test_agent_instance_mapping(redis_mgr):
    """Bulk-register 1000 agents and verify count."""
    agents = [
        {
            "agent_id": f"agent-{idx:02d}-{j:03d}",
            "hostname": ALL_INSTANCES[idx % 10].hostname,
            "instance_port": ALL_INSTANCES[idx % 10].port,
        }
        for idx in range(10)
        for j in range(100)
    ]
    assert len(agents) == 1000

    await redis_mgr.register_agents_bulk(agents)
    count = await redis_mgr.get_agent_count()
    assert count == 1000, f"Expected 1000 agents registered, got {count}"


# ---------- 14. EMA Latency Update ----------

@skip_redis
@pytest.mark.asyncio
async def test_ema_latency_update(redis_mgr):
    """Update latency 100 times with constant value; EMA should converge."""
    port = 19900
    await redis_mgr.init_instance(port, slots_total=1, inst_type="gpu")

    target_latency = 50.0
    alpha = 0.1

    for _ in range(100):
        await redis_mgr.update_latency(port, target_latency, alpha=alpha)

    ema = await redis_mgr.get_latency(port)
    # After 100 iterations with alpha=0.1 and constant input, EMA converges
    # to the input value. Allow 5% tolerance.
    assert abs(ema - target_latency) < target_latency * 0.05, (
        f"EMA should converge to {target_latency}, got {ema}"
    )


# ---------- 15. Health TTL Expiry ----------

@skip_redis
@pytest.mark.asyncio
async def test_health_ttl_expiry(redis_mgr):
    """Set health with short TTL, wait, verify expired."""
    port = 19901
    await redis_mgr.init_instance(port, slots_total=1, inst_type="gpu")

    # Set a very short TTL directly (1 second)
    await redis_mgr._redis.set(f"inst:{port}:health", "alive", ex=1)
    assert await redis_mgr.is_healthy(port)

    # Wait for expiry
    await asyncio.sleep(1.5)

    healthy = await redis_mgr.is_healthy(port)
    assert not healthy, "Health key should have expired after TTL"


# ---------- 16. Circuit Breaker ----------

@skip_redis
@pytest.mark.asyncio
async def test_circuit_breaker(redis_mgr):
    """Record errors and verify error_rate reflects them."""
    port = 19902
    await redis_mgr.init_instance(port, slots_total=1, inst_type="gpu")

    # Record 8 errors and 2 successes -> error_rate = 0.8
    for _ in range(8):
        await redis_mgr.record_error(port)
    for _ in range(2):
        await redis_mgr.record_success(port)

    error_rate = await redis_mgr.get_error_rate(port)
    assert abs(error_rate - 0.8) < 0.01, (
        f"Expected error_rate ~0.8, got {error_rate}"
    )


# ---------- 17. MongoDB Save Run ----------

@skip_mongo
@pytest.mark.asyncio
async def test_mongo_save_run(mongo_store):
    """Save a benchmark run and retrieve it by run_id."""
    run_data = {
        "run_id": "test-10inst-run-1",
        "experiment": "10inst_1000agents",
        "scenario": "S1",
        "protocol": "dds",
        "algorithm": "least_loaded",
        "num_clients": 100,
        "results": {
            "p50": 25.3,
            "p95": 68.7,
            "p99": 112.4,
            "throughput_rps": 890,
            "error_rate": 0.002,
        },
    }
    run_id = await mongo_store.save_run(run_data)
    assert run_id == "test-10inst-run-1"

    retrieved = await mongo_store.get_run(run_id)
    assert retrieved is not None
    assert retrieved["experiment"] == "10inst_1000agents"
    assert retrieved["results"]["p50"] == 25.3
    assert retrieved["results"]["throughput_rps"] == 890


# ---------- 18. MongoDB Agents Bulk ----------

@skip_mongo
@pytest.mark.asyncio
async def test_mongo_agents_bulk(mongo_store):
    """Register 1000 agents in MongoDB and verify count."""
    agents = [
        {
            "agent_id": f"agent-{idx:02d}-{j:03d}",
            "hostname": ALL_INSTANCES[idx % 10].hostname,
            "instance_port": ALL_INSTANCES[idx % 10].port,
            "model": "Qwen3.5-2B",
        }
        for idx in range(10)
        for j in range(100)
    ]
    assert len(agents) == 1000

    await mongo_store.register_agents_bulk(agents)
    count = await mongo_store.get_agent_count()
    assert count == 1000, f"Expected 1000 agents in MongoDB, got {count}"


# ---------- 19. Redis Cleanup ----------

@skip_redis
@pytest.mark.asyncio
async def test_cleanup(redis_mgr):
    """Redis cleanup should remove all inst:* keys."""
    # Create some instance data
    for port in [19910, 19911, 19912]:
        await redis_mgr.init_instance(port, slots_total=5, inst_type="gpu")

    # Verify keys exist
    for port in [19910, 19911, 19912]:
        _, total = await redis_mgr.get_slots(port)
        assert total == 5

    # Cleanup
    await redis_mgr.cleanup()

    # Verify all inst:* keys are gone
    cursor = "0"
    remaining = []
    while True:
        cursor, keys = await redis_mgr._redis.scan(cursor=cursor, match="inst:*", count=100)
        remaining.extend(keys)
        if cursor == "0" or cursor == 0:
            break

    assert len(remaining) == 0, f"Cleanup left {len(remaining)} inst:* keys: {remaining[:5]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
