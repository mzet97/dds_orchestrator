#!/usr/bin/env python3
"""
End-to-end tests for the 38-instance infrastructure.
Starts a real orchestrator server with Redis + MongoDB + InstancePool
and sends actual HTTP requests through it.
"""

import asyncio
import os
import sys
import time
import pytest
import pytest_asyncio
import aiohttp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REDIS_URL = "redis://192.168.1.51:30379"
REDIS_PASSWORD = "Admin@123"
MONGO_URL = "mongodb://admin:Admin%40123@192.168.1.51:27017/?authSource=admin"
MONGO_DB = "test_dds_38inst_e2e"

# Use a non-standard port to avoid conflicts
TEST_PORT = 18080


async def _redis_available():
    try:
        from redis_layer import RedisStateManager
        mgr = RedisStateManager(REDIS_URL, REDIS_PASSWORD)
        await mgr.connect()
        await mgr.close()
        return True
    except Exception:
        return False


async def _mongo_available():
    try:
        from mongo_layer import MongoMetricsStore
        store = MongoMetricsStore(MONGO_URL, MONGO_DB)
        await store.connect()
        await store.close()
        return True
    except Exception:
        return False


redis_ok = asyncio.run(_redis_available())
mongo_ok = asyncio.run(_mongo_available())
skip_e2e = pytest.mark.skipif(
    not (redis_ok and mongo_ok),
    reason=f"Redis={redis_ok}, MongoDB={mongo_ok} — both required for E2E"
)


@skip_e2e
class TestServerWithInstancePool:
    """E2E: start real orchestrator with InstancePool, send HTTP requests."""

    @pytest_asyncio.fixture
    async def server_url(self):
        """Start orchestrator server with Redis + MongoDB + InstancePool."""
        from config import OrchestratorConfig
        from registry import AgentRegistry
        from scheduler import TaskScheduler
        from selector import AgentSelector
        from dds import DDSLayer
        from redis_layer import RedisStateManager
        from mongo_layer import MongoMetricsStore
        from instance_pool import InstancePool, InstanceInfo, RoutingAlgorithm
        from backpressure import BackpressureManager
        from server import OrchestratorServer

        # Config with DDS disabled (no CycloneDDS needed for this test)
        config = OrchestratorConfig(
            host="127.0.0.1",
            port=TEST_PORT,
            dds_enabled=False,
            max_agents=1000,
            max_concurrent_tasks=2000,
        )

        # Components
        registry = AgentRegistry(config)
        scheduler = TaskScheduler(config)
        dds_layer = DDSLayer(config)  # DDS disabled, will be a no-op
        selector = AgentSelector()

        # Redis
        redis_mgr = RedisStateManager(REDIS_URL, REDIS_PASSWORD)
        await redis_mgr.connect()
        await redis_mgr.cleanup()

        # MongoDB
        mongo_store = MongoMetricsStore(MONGO_URL, MONGO_DB)
        await mongo_store.connect()
        await mongo_store.drop_all()
        await mongo_store.ensure_indexes()

        # InstancePool with fake instances (ports that don't exist, but
        # we're testing the routing/slot logic, not actual inference)
        instance_pool = InstancePool(
            redis_mgr, mongo_store,
            algorithm=RoutingAlgorithm.LEAST_LOADED,
        )
        # Register 3 fake GPU + 2 fake CPU instances
        for i in range(3):
            await instance_pool.register_instance(
                InstanceInfo(port=18082 + i, hostname="127.0.0.1",
                             inst_type="gpu", slots_total=4, weight=1.0))
        for i in range(2):
            await instance_pool.register_instance(
                InstanceInfo(port=18092 + i, hostname="127.0.0.1",
                             inst_type="cpu", slots_total=2, weight=0.3))

        backpressure = BackpressureManager(redis_mgr, max_rps=1000)

        # Create and start server
        server = OrchestratorServer(
            config=config,
            registry=registry,
            scheduler=scheduler,
            dds_layer=dds_layer,
            selector=selector,
            instance_pool=instance_pool,
            redis_mgr=redis_mgr,
            mongo_store=mongo_store,
            backpressure=backpressure,
        )
        await server.start()

        url = f"http://127.0.0.1:{TEST_PORT}"

        yield url

        # Cleanup
        await server.stop()
        await redis_mgr.cleanup()
        await redis_mgr.close()
        await mongo_store.drop_all()
        await mongo_store.close()

    # ---- Health & Status ----

    @pytest.mark.asyncio
    async def test_health_endpoint(self, server_url):
        """GET /health should return healthy."""
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{server_url}/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_status_endpoint(self, server_url):
        """GET /status should return registry + scheduler stats."""
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{server_url}/status") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert "registry" in data
                assert "scheduler" in data

    # ---- Pool Status ----

    @pytest.mark.asyncio
    async def test_pool_status(self, server_url):
        """GET /api/v1/pool/status should show 5 instances."""
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{server_url}/api/v1/pool/status") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["total_instances"] == 5
                assert data["algorithm"] == "least_loaded"
                assert "instances" in data
                assert len(data["instances"]) == 5
                assert "pressure_level" in data
                assert data["pressure_level"] in ("normal", "elevated", "critical")
                # Verify instance types
                types = {inst["type"] for inst in data["instances"]}
                assert "gpu" in types
                assert "cpu" in types

    # ---- Algorithm Switching ----

    @pytest.mark.asyncio
    async def test_set_algorithm_round_robin(self, server_url):
        """PUT /api/v1/routing/algorithm should switch algorithm."""
        async with aiohttp.ClientSession() as s:
            async with s.put(
                f"{server_url}/api/v1/routing/algorithm",
                json={"algorithm": "round_robin"}
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["success"] is True
                assert data["algorithm"] == "round_robin"

            # Verify it changed
            async with s.get(f"{server_url}/api/v1/pool/status") as resp:
                data = await resp.json()
                assert data["algorithm"] == "round_robin"

    @pytest.mark.asyncio
    async def test_set_algorithm_weighted_score(self, server_url):
        """Switch to weighted_score."""
        async with aiohttp.ClientSession() as s:
            async with s.put(
                f"{server_url}/api/v1/routing/algorithm",
                json={"algorithm": "weighted_score"}
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["algorithm"] == "weighted_score"

    @pytest.mark.asyncio
    async def test_set_algorithm_invalid(self, server_url):
        """Invalid algorithm name should return 400."""
        async with aiohttp.ClientSession() as s:
            async with s.put(
                f"{server_url}/api/v1/routing/algorithm",
                json={"algorithm": "invalid_algo"}
            ) as resp:
                assert resp.status == 400
                data = await resp.json()
                assert "valid" in data

    # ---- Chat via InstancePool (expects 503 since no real llama-server) ----

    @pytest.mark.asyncio
    async def test_chat_acquires_and_releases_slot(self, server_url):
        """POST /api/v1/chat/completions should acquire a slot,
        attempt dispatch, and release the slot even on failure.
        Since no real llama-server exists, DDS is disabled, and HTTP fallback
        will fail — but the slot management should still work correctly.
        """
        async with aiohttp.ClientSession() as s:
            # Check initial pool status
            async with s.get(f"{server_url}/api/v1/pool/status") as resp:
                before = await resp.json()
                total_used_before = sum(
                    inst.get("slots_used", 0) for inst in before["instances"]
                )

            # Send a chat request (will fail because no backend, but tests the flow)
            async with s.post(
                f"{server_url}/api/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}],
                      "max_tokens": 5},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                # May be 200 with empty content or 500 — either is fine
                # The important thing is it doesn't hang and slots are released
                status = resp.status
                data = await resp.json()

            # Wait a moment for async cleanup
            await asyncio.sleep(0.5)

            # Check slots are released
            async with s.get(f"{server_url}/api/v1/pool/status") as resp:
                after = await resp.json()
                total_used_after = sum(
                    inst.get("slots_used", 0) for inst in after["instances"]
                )
                assert total_used_after == 0, \
                    f"Slots not released: {total_used_after} still in use"

    @pytest.mark.asyncio
    async def test_concurrent_chat_slots_released(self, server_url):
        """Send 10 concurrent chat requests — all slots should be released after."""
        async with aiohttp.ClientSession() as s:
            tasks = []
            for i in range(10):
                tasks.append(
                    s.post(
                        f"{server_url}/api/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": f"test {i}"}],
                              "max_tokens": 5},
                        timeout=aiohttp.ClientTimeout(total=15),
                    )
                )
            # Gather — some may fail, that's OK
            responses = await asyncio.gather(
                *[self._do_request(s, server_url, i) for i in range(10)],
                return_exceptions=True,
            )

            # Wait for cleanup
            await asyncio.sleep(1)

            # All slots should be back to 0
            async with s.get(f"{server_url}/api/v1/pool/status") as resp:
                pool = await resp.json()
                total_used = sum(
                    inst.get("slots_used", 0) for inst in pool["instances"]
                )
                assert total_used == 0, \
                    f"Slots leaked after 10 concurrent requests: {total_used} in use"

    async def _do_request(self, session, url, idx):
        try:
            async with session.post(
                f"{url}/api/v1/chat/completions",
                json={"messages": [{"role": "user", "content": f"test {idx}"}],
                      "max_tokens": 5},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                return await resp.json()
        except Exception:
            return None

    # ---- Metrics in MongoDB ----

    @pytest.mark.asyncio
    async def test_chat_logs_metric_to_mongo(self, server_url):
        """Chat request should log a metric to MongoDB."""
        from mongo_layer import MongoMetricsStore
        store = MongoMetricsStore(MONGO_URL, MONGO_DB)
        await store.connect()

        # Clear metrics
        await store._db["metrics"].delete_many({})

        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{server_url}/api/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "metric test"}],
                      "max_tokens": 5, "scenario": "test_e2e"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                await resp.json()

        # Wait for fire-and-forget log
        await asyncio.sleep(1)

        metrics = await store.get_metrics(scenario="test_e2e")
        await store.close()

        assert len(metrics) >= 1, "No metric logged to MongoDB"
        m = metrics[0]
        assert "instance_port" in m
        assert "latency_ms" in m
        assert "algorithm" in m

    # ---- Metrics Summary ----

    @pytest.mark.asyncio
    async def test_metrics_summary_endpoint(self, server_url):
        """GET /api/v1/metrics/summary should return aggregated data."""
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{server_url}/api/v1/metrics/summary") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert "groups" in data

    # ---- Backpressure / Rate Limit ----

    @pytest.mark.asyncio
    async def test_rate_limit_not_triggered(self, server_url):
        """Single request should not be rate limited."""
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{server_url}/api/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "rate test"}],
                      "max_tokens": 5},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                # Should NOT be 429
                assert resp.status != 429

    # ---- Agent Registration (legacy path still works) ----

    @pytest.mark.asyncio
    async def test_legacy_agent_registration(self, server_url):
        """Agent registration endpoint should still work alongside instance pool."""
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{server_url}/api/v1/agents/register",
                json={
                    "agent_id": "legacy-agent-1",
                    "hostname": "127.0.0.1",
                    "port": 9999,
                    "model": "test-model",
                    "slots_idle": 1,
                    "slots_total": 1,
                }
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["success"] is True

            # Verify it's registered
            async with s.get(f"{server_url}/api/v1/agents") as resp:
                data = await resp.json()
                agents = data["agents"]
                assert any(a["agent_id"] == "legacy-agent-1" for a in agents)


@skip_e2e
class TestRedisSlotStress:
    """Stress test Redis slot atomicity with real Redis."""

    @pytest.mark.asyncio
    async def test_100_concurrent_acquire_release_real_redis(self):
        """100 concurrent acquire/release on real Redis should be atomic."""
        from redis_layer import RedisStateManager

        redis = RedisStateManager(REDIS_URL, REDIS_PASSWORD)
        await redis.connect()
        await redis.cleanup()
        await redis.init_instance(19999, slots_total=100, inst_type="gpu")

        results = {"acquired": 0, "released": 0}

        async def acquire_release():
            if await redis.acquire_slot(19999):
                results["acquired"] += 1
                await asyncio.sleep(0.001)  # simulate work
                await redis.release_slot(19999)
                results["released"] += 1

        await asyncio.gather(*[acquire_release() for _ in range(100)])

        used, total = await redis.get_slots(19999)
        assert used == 0, f"Slots leaked: {used} still in use after 100 acquire/release"
        assert results["acquired"] == 100
        assert results["released"] == 100

        await redis.cleanup()
        await redis.close()

    @pytest.mark.asyncio
    async def test_overcommit_slots(self):
        """More acquires than slots_total should be rejected atomically."""
        from redis_layer import RedisStateManager

        redis = RedisStateManager(REDIS_URL, REDIS_PASSWORD)
        await redis.connect()
        await redis.cleanup()
        await redis.init_instance(19998, slots_total=5, inst_type="cpu")

        results = {"acquired": 0, "rejected": 0}

        async def try_acquire():
            if await redis.acquire_slot(19998):
                results["acquired"] += 1
                await asyncio.sleep(0.01)  # hold slot
                await redis.release_slot(19998)
            else:
                results["rejected"] += 1

        # 50 concurrent attempts on 5 slots
        await asyncio.gather(*[try_acquire() for _ in range(50)])

        used, _ = await redis.get_slots(19998)
        assert used == 0
        assert results["acquired"] + results["rejected"] == 50
        # Not all 50 could acquire simultaneously (only 5 slots)
        # But all should eventually have been processed
        assert results["acquired"] > 0
        assert results["rejected"] > 0  # at least some must have been rejected

        await redis.cleanup()
        await redis.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
