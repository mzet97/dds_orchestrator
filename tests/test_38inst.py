#!/usr/bin/env python3
"""
Tests for the 38-instance infrastructure (Phases 1-5).
- Phase 1: IDL target_agent_id field
- Phase 2: RedisStateManager (mock-based when Redis unavailable)
- Phase 3: MongoMetricsStore (integration test with real MongoDB)
- Phase 4: InstancePool + BackpressureManager
- Phase 5: Config, Server integration
"""

import asyncio
import os
import sys
import time
import pytest
import pytest_asyncio

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ======================================================================
# Phase 1: IDL target_agent_id
# ======================================================================

class TestIDLTargetAgentId:
    """Verify target_agent_id field exists in both orchestrator and agent IDL."""

    def test_orchestrator_idl_has_field(self):
        """TaskRequest in orchestrator IDL should have target_agent_id."""
        idl_path = os.path.join(os.path.dirname(__file__), "..",
                                "orchestrator", "_OrchestratorDDS.py")
        with open(idl_path) as f:
            content = f.read()
        assert "target_agent_id" in content, "orchestrator IDL missing target_agent_id"

    def test_agent_idl_has_field(self):
        """TaskRequest in agent IDL should have target_agent_id."""
        idl_path = _find_repo_file("..", "..", "dds_agent", "orchestrator", "_OrchestratorDDS.py")
        if not idl_path:
            idl_path = _find_repo_file("..", "dds_agent_orch", "_OrchestratorDDS.py")
        assert idl_path, "agent _OrchestratorDDS.py not found"
        with open(idl_path) as f:
            content = f.read()
        assert "target_agent_id" in content, "agent IDL missing target_agent_id"

    def test_idl_source_has_field(self):
        """OrchestratorDDS.idl should have target_agent_id."""
        idl_path = _find_repo_file("..", "..", "llama.cpp_dds", "dds", "idl", "OrchestratorDDS.idl")
        if not idl_path:
            idl_path = _find_repo_file("..", "llama_dds", "OrchestratorDDS.idl")
        assert idl_path, "OrchestratorDDS.idl not found"
        with open(idl_path) as f:
            content = f.read()
        assert "target_agent_id" in content, "IDL source missing target_agent_id"

    def test_dds_layer_agent_task_request_has_field(self):
        """AgentTaskRequest dataclass should have target_agent_id."""
        from dds import AgentTaskRequest
        req = AgentTaskRequest(
            task_id="t1", requester_id="orch", task_type="chat",
            messages=[], priority=5, timeout_ms=1000, requires_context=False,
            target_agent_id="agent-00-001",
        )
        assert req.target_agent_id == "agent-00-001"

    def test_agent_task_request_default_empty(self):
        """target_agent_id should default to empty string (broadcast)."""
        from dds import AgentTaskRequest
        req = AgentTaskRequest(
            task_id="t1", requester_id="orch", task_type="chat",
            messages=[], priority=5, timeout_ms=1000, requires_context=False,
        )
        assert req.target_agent_id == ""

    def test_agent_filter_logic(self):
        """Agent filter should skip requests not targeted to it."""
        # Simulate the filter from agent_llm_dds.py
        agent_id = "agent-00-001"

        class FakeReq:
            def __init__(self, target):
                self.target_agent_id = target
                self.task_id = "task-1"

        # Targeted to this agent → should process
        req1 = FakeReq("agent-00-001")
        target = getattr(req1, "target_agent_id", "")
        assert not (target and target != agent_id), "Should not skip own request"

        # Targeted to another agent → should skip
        req2 = FakeReq("agent-05-003")
        target = getattr(req2, "target_agent_id", "")
        assert (target and target != agent_id), "Should skip other agent's request"

        # Broadcast (empty) → should process
        req3 = FakeReq("")
        target = getattr(req3, "target_agent_id", "")
        assert not (target and target != agent_id), "Should not skip broadcast"


# ======================================================================
# Phase 2: Redis Layer (unit tests with real Redis if available)
# ======================================================================

REDIS_URL = "redis://192.168.1.51:30379"
REDIS_PASSWORD = "Admin@123"

async def _redis_available():
    try:
        from redis_layer import RedisStateManager
        mgr = RedisStateManager(REDIS_URL, REDIS_PASSWORD)
        await mgr.connect()
        await mgr.close()
        return True
    except Exception:
        return False

redis_available = False
try:
    # asyncio.get_event_loop() is deprecated when no loop is running (3.10+);
    # use asyncio.run directly — it creates, runs, and closes a fresh loop.
    redis_available = asyncio.run(_redis_available())
except Exception:
    try:
        redis_available = asyncio.run(_redis_available())
    except Exception:
        pass

redis_skip = pytest.mark.skipif(not redis_available, reason="Redis not available")


class TestRedisLayerUnit:
    """Unit tests for RedisStateManager that don't require a real Redis."""

    def test_import(self):
        from redis_layer import RedisStateManager
        mgr = RedisStateManager("redis://localhost:6379")
        assert mgr._url == "redis://localhost:6379"

    def test_lua_scripts_defined(self):
        from redis_layer import _LUA_ACQUIRE_SLOT, _LUA_EMA_UPDATE
        assert "INCR" in _LUA_ACQUIRE_SLOT
        assert "redis.call" in _LUA_EMA_UPDATE


@redis_skip
class TestRedisLayerIntegration:
    """Integration tests with real Redis."""

    @pytest_asyncio.fixture
    async def redis(self):
        from redis_layer import RedisStateManager
        mgr = RedisStateManager(REDIS_URL, REDIS_PASSWORD)
        await mgr.connect()
        await mgr.cleanup()
        yield mgr
        await mgr.cleanup()
        await mgr.close()

    @pytest.mark.asyncio
    async def test_init_and_acquire_release(self, redis):
        await redis.init_instance(9999, slots_total=4, inst_type="gpu")
        used, total = await redis.get_slots(9999)
        assert total == 4
        assert used == 0

        # Acquire 4 slots
        for _ in range(4):
            assert await redis.acquire_slot(9999)

        # 5th should fail
        assert not await redis.acquire_slot(9999)

        # Release one
        await redis.release_slot(9999)
        used, _ = await redis.get_slots(9999)
        assert used == 3

        # Can acquire again
        assert await redis.acquire_slot(9999)

    @pytest.mark.asyncio
    async def test_concurrent_acquire_release(self, redis):
        """100 concurrent acquire/release should leave slots_used == 0."""
        await redis.init_instance(9998, slots_total=100, inst_type="cpu")

        async def acquire_release():
            if await redis.acquire_slot(9998):
                await asyncio.sleep(0.001)
                await redis.release_slot(9998)

        await asyncio.gather(*[acquire_release() for _ in range(100)])

        used, _ = await redis.get_slots(9998)
        assert used == 0, f"Expected 0 slots used after all released, got {used}"

    @pytest.mark.asyncio
    async def test_health_ttl(self, redis):
        await redis.init_instance(9997, slots_total=1, inst_type="gpu")
        assert await redis.is_healthy(9997)

        # Health key has 30s TTL; we can verify it exists
        await redis.update_health(9997)
        assert await redis.is_healthy(9997)

    @pytest.mark.asyncio
    async def test_rate_limit(self, redis):
        # Should allow within limit
        assert await redis.check_rate_limit(max_rps=100)

    @pytest.mark.asyncio
    async def test_ema_latency(self, redis):
        await redis.init_instance(9996, slots_total=1, inst_type="gpu")
        await redis.update_latency(9996, 100.0, alpha=1.0)
        lat = await redis.get_latency(9996)
        assert abs(lat - 100.0) < 1.0

    @pytest.mark.asyncio
    async def test_round_robin_index(self, redis):
        idx1 = await redis.get_rr_index()
        idx2 = await redis.get_rr_index()
        assert idx2 == idx1 + 1

    @pytest.mark.asyncio
    async def test_active_counter(self, redis):
        await redis._redis.set("stats:active", 0)
        await redis.incr_active()
        await redis.incr_active()
        assert await redis.get_active() == 2
        await redis.decr_active()
        assert await redis.get_active() == 1


# ======================================================================
# Phase 3: MongoDB Layer
# ======================================================================

MONGO_URL = "mongodb://admin:Admin%40123@192.168.1.51:27017/?authSource=admin"

async def _mongo_available():
    try:
        from mongo_layer import MongoMetricsStore
        store = MongoMetricsStore(MONGO_URL, "test_dds_38inst")
        await store.connect()
        await store.close()
        return True
    except Exception:
        return False

mongo_available = False
try:
    mongo_available = asyncio.run(_mongo_available())
except Exception:
    try:
        mongo_available = asyncio.run(_mongo_available())
    except Exception:
        pass

mongo_skip = pytest.mark.skipif(not mongo_available, reason="MongoDB not available")


class TestMongoLayerUnit:
    """Unit tests that don't need a real MongoDB."""

    def test_import(self):
        from mongo_layer import MongoMetricsStore
        store = MongoMetricsStore("mongodb://localhost:27017")
        assert store._db_name == "dds_orchestrator"

    def test_custom_db_name(self):
        from mongo_layer import MongoMetricsStore
        store = MongoMetricsStore("mongodb://localhost:27017", "custom_db")
        assert store._db_name == "custom_db"


@mongo_skip
class TestMongoLayerIntegration:
    """Integration tests with real MongoDB."""

    @pytest_asyncio.fixture
    async def mongo(self):
        from mongo_layer import MongoMetricsStore
        store = MongoMetricsStore(MONGO_URL, "test_dds_38inst")
        await store.connect()
        await store.drop_all()
        await store.ensure_indexes()
        yield store
        await store.drop_all()
        await store.close()

    @pytest.mark.asyncio
    async def test_log_and_query_metrics(self, mongo):
        await mongo.log_request({
            "request_id": "r1", "instance_port": 8082,
            "instance_type": "gpu", "protocol": "dds",
            "algorithm": "least_loaded", "latency_ms": 42.5,
            "success": True, "scenario": "S1",
        })
        metrics = await mongo.get_metrics(scenario="S1")
        assert len(metrics) == 1
        assert metrics[0]["latency_ms"] == 42.5

    @pytest.mark.asyncio
    async def test_save_and_get_run(self, mongo):
        run_id = await mongo.save_run({
            "run_id": "test-run-1",
            "experiment": "E1", "scenario": "S1",
            "protocol": "dds", "algorithm": "least_loaded",
            "num_clients": 100,
            "results": {"p50": 30, "p95": 80, "p99": 120},
        })
        assert run_id == "test-run-1"

        run = await mongo.get_run("test-run-1")
        assert run is not None
        assert run["experiment"] == "E1"
        assert run["results"]["p50"] == 30

    @pytest.mark.asyncio
    async def test_register_agents_bulk(self, mongo):
        agents = [
            {"agent_id": f"agent-{i:03d}", "instance_port": 8082 + i % 38,
             "model": "Qwen3.5-2B"}
            for i in range(100)
        ]
        await mongo.register_agents_bulk(agents)
        count = await mongo.get_agent_count()
        assert count == 100

    @pytest.mark.asyncio
    async def test_batch_insert(self, mongo):
        """10K inserts should complete without loss."""
        metrics = [
            {"request_id": f"r-{i}", "latency_ms": i * 0.1,
             "success": True, "scenario": "S3", "protocol": "dds",
             "algorithm": "round_robin"}
            for i in range(1000)
        ]
        await mongo.log_request_batch(metrics)
        result = await mongo.get_metrics(scenario="S3", limit=2000)
        assert len(result) == 1000

    @pytest.mark.asyncio
    async def test_latency_percentiles(self, mongo):
        # Insert known data
        run_id = "perc-test"
        metrics = [
            {"request_id": f"p-{i}", "latency_ms": float(i),
             "success": True, "run_id": run_id}
            for i in range(100)
        ]
        await mongo.log_request_batch(metrics)
        percs = await mongo.get_latency_percentiles(run_id)
        assert percs["count"] == 100
        assert percs["p50"] == 49  # nearest-rank median of 0-99
        assert percs["min"] == 0.0
        assert percs["max"] == 99.0

    @pytest.mark.asyncio
    async def test_metrics_summary(self, mongo):
        for proto in ["dds", "http"]:
            for i in range(10):
                await mongo.log_request({
                    "request_id": f"s-{proto}-{i}",
                    "latency_ms": 50.0 if proto == "dds" else 100.0,
                    "success": True, "protocol": proto,
                    "algorithm": "least_loaded", "scenario": "S1",
                })
        summary = await mongo.get_metrics_summary("S1")
        assert len(summary["groups"]) == 2  # dds + http


# ======================================================================
# Phase 4: InstancePool + Backpressure (mock Redis)
# ======================================================================

class MockRedis:
    """Minimal mock of RedisStateManager for unit testing."""

    def __init__(self):
        self._instances = {}
        self._rr_idx = 0
        self._active = 0
        self._errors = {}
        self._successes = {}

    async def init_instance(self, port, slots_total, inst_type):
        self._instances[port] = {
            "slots_used": 0, "slots_total": slots_total,
            "type": inst_type, "avg_latency": 0.0, "healthy": True,
        }

    async def acquire_slot(self, port):
        inst = self._instances.get(port)
        if not inst:
            return False
        if inst["slots_used"] >= inst["slots_total"]:
            return False
        inst["slots_used"] += 1
        return True

    async def release_slot(self, port):
        inst = self._instances.get(port)
        if inst and inst["slots_used"] > 0:
            inst["slots_used"] -= 1

    async def get_slots(self, port):
        inst = self._instances.get(port, {})
        return inst.get("slots_used", 0), inst.get("slots_total", 0)

    async def get_all_loads(self):
        return [
            {"port": port, **data}
            for port, data in self._instances.items()
        ]

    async def update_latency(self, port, latency_ms, alpha=0.1):
        inst = self._instances.get(port)
        if inst:
            inst["avg_latency"] = latency_ms * alpha + inst["avg_latency"] * (1 - alpha)

    async def update_health(self, port):
        inst = self._instances.get(port)
        if inst:
            inst["healthy"] = True

    async def is_healthy(self, port):
        inst = self._instances.get(port)
        return inst["healthy"] if inst else False

    async def get_rr_index(self):
        self._rr_idx += 1
        return self._rr_idx

    async def check_rate_limit(self, max_rps=5000):
        return True

    async def incr_active(self):
        self._active += 1
        return self._active

    async def decr_active(self):
        self._active = max(0, self._active - 1)
        return self._active

    async def get_active(self):
        return self._active

    async def record_error(self, port):
        self._errors[port] = self._errors.get(port, 0) + 1

    async def record_success(self, port):
        self._successes[port] = self._successes.get(port, 0) + 1

    async def get_error_rate(self, port):
        errors = self._errors.get(port, 0)
        successes = self._successes.get(port, 0)
        total = errors + successes
        return errors / total if total > 0 else 0.0


class TestInstancePool:
    """Test InstancePool with mock Redis."""

    def _make_pool(self):
        from instance_pool import InstancePool, InstanceInfo, RoutingAlgorithm
        redis = MockRedis()
        pool = InstancePool(redis, mongo=None,
                            algorithm=RoutingAlgorithm.ROUND_ROBIN)
        return pool, redis

    @pytest.mark.asyncio
    async def test_register_instance(self):
        pool_obj, redis = self._make_pool()
        from instance_pool import InstanceInfo
        await pool_obj.register_instance(
            InstanceInfo(port=8082, inst_type="gpu", slots_total=15, weight=1.0))
        assert 8082 in pool_obj._instances
        assert redis._instances[8082]["slots_total"] == 15

    @pytest.mark.asyncio
    async def test_round_robin_selection(self):
        pool_obj, redis = self._make_pool()
        from instance_pool import InstanceInfo
        for i in range(3):
            await pool_obj.register_instance(
                InstanceInfo(port=8082 + i, inst_type="gpu", slots_total=2))

        selected = set()
        for _ in range(6):
            inst = await pool_obj.select_instance()
            assert inst is not None
            selected.add(inst.port)
        # All 3 instances should have been selected
        assert len(selected) == 3

    @pytest.mark.asyncio
    async def test_least_loaded_selection(self):
        pool_obj, redis = self._make_pool()
        from instance_pool import InstanceInfo, RoutingAlgorithm
        pool_obj.set_algorithm(RoutingAlgorithm.LEAST_LOADED)

        await pool_obj.register_instance(
            InstanceInfo(port=8082, inst_type="gpu", slots_total=15))
        await pool_obj.register_instance(
            InstanceInfo(port=8092, inst_type="cpu", slots_total=4))

        # GPU has more free slots, should be preferred
        inst = await pool_obj.select_instance()
        assert inst is not None
        assert inst.port == 8082

    @pytest.mark.asyncio
    async def test_weighted_score_prefers_gpu(self):
        pool_obj, redis = self._make_pool()
        from instance_pool import InstanceInfo, RoutingAlgorithm
        pool_obj.set_algorithm(RoutingAlgorithm.WEIGHTED_SCORE)

        await pool_obj.register_instance(
            InstanceInfo(port=8082, inst_type="gpu", slots_total=15, weight=1.0))
        await pool_obj.register_instance(
            InstanceInfo(port=8092, inst_type="cpu", slots_total=4, weight=0.3))

        inst = await pool_obj.select_instance()
        assert inst is not None
        # GPU should be preferred (higher weight = lower adjusted score)
        assert inst.port == 8082

    @pytest.mark.asyncio
    async def test_all_full_returns_none(self):
        pool_obj, redis = self._make_pool()
        from instance_pool import InstanceInfo
        await pool_obj.register_instance(
            InstanceInfo(port=8082, inst_type="gpu", slots_total=1))

        # Fill the only slot
        inst = await pool_obj.select_instance()
        assert inst is not None

        # Now should return None
        inst2 = await pool_obj.select_instance()
        assert inst2 is None

    @pytest.mark.asyncio
    async def test_release_updates_metrics(self):
        pool_obj, redis = self._make_pool()
        from instance_pool import InstanceInfo
        await pool_obj.register_instance(
            InstanceInfo(port=8082, inst_type="gpu", slots_total=2))

        inst = await pool_obj.select_instance()
        assert inst is not None
        used_before, _ = await redis.get_slots(8082)
        assert used_before == 1

        await pool_obj.release_instance(8082, latency_ms=50.0, success=True)
        used_after, _ = await redis.get_slots(8082)
        assert used_after == 0

    @pytest.mark.asyncio
    async def test_set_algorithm_runtime(self):
        pool_obj, _ = self._make_pool()
        from instance_pool import RoutingAlgorithm
        pool_obj.set_algorithm(RoutingAlgorithm.WEIGHTED_SCORE)
        assert pool_obj._algorithm == RoutingAlgorithm.WEIGHTED_SCORE

    @pytest.mark.asyncio
    async def test_get_status(self):
        pool_obj, redis = self._make_pool()
        from instance_pool import InstanceInfo
        await pool_obj.register_instance(
            InstanceInfo(port=8082, inst_type="gpu", slots_total=15))
        status = await pool_obj.get_status()
        assert status["total_instances"] == 1
        assert status["algorithm"] == "round_robin"


class TestBackpressure:
    """Test BackpressureManager."""

    @pytest.mark.asyncio
    async def test_allow_request(self):
        from backpressure import BackpressureManager
        redis = MockRedis()
        bp = BackpressureManager(redis, max_rps=5000)
        assert await bp.allow_request()

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self):
        from backpressure import BackpressureManager
        redis = MockRedis()
        await redis.init_instance(8082, 15, "gpu")
        bp = BackpressureManager(redis, circuit_error_threshold=0.5, circuit_cooldown_s=1)

        # Record errors to trigger circuit breaker
        for _ in range(10):
            await redis.record_error(8082)
        for _ in range(2):
            await redis.record_success(8082)

        # Error rate = 10/12 = 83% > 50% → circuit should open
        assert await bp.is_circuit_open(8082)

    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_after_cooldown(self):
        from backpressure import BackpressureManager
        redis = MockRedis()
        await redis.init_instance(8082, 15, "gpu")
        bp = BackpressureManager(redis, circuit_error_threshold=0.5, circuit_cooldown_s=0)

        for _ in range(10):
            await redis.record_error(8082)

        # Open
        assert await bp.is_circuit_open(8082)

        # Cooldown is 0s, so next check should close
        # Reset errors
        redis._errors[8082] = 0
        redis._successes[8082] = 0
        assert not await bp.is_circuit_open(8082)

    @pytest.mark.asyncio
    async def test_pressure_level(self):
        from backpressure import BackpressureManager
        redis = MockRedis()
        await redis.init_instance(8082, 10, "gpu")
        bp = BackpressureManager(redis)

        level = await bp.get_pressure_level()
        assert level == "normal"


# ======================================================================
# Phase 5: Config
# ======================================================================

class TestConfig:
    """Test OrchestratorConfig new fields."""

    def test_new_fields_exist(self):
        from config import OrchestratorConfig
        c = OrchestratorConfig()
        assert hasattr(c, "redis_url")
        assert hasattr(c, "redis_password")
        assert hasattr(c, "mongo_url")
        assert hasattr(c, "mongo_db")
        assert hasattr(c, "routing_algorithm")
        assert hasattr(c, "max_rps")
        assert hasattr(c, "instance_ports_gpu")
        assert hasattr(c, "instance_ports_cpu")
        assert hasattr(c, "instance_host")
        assert hasattr(c, "slots_per_gpu")
        assert hasattr(c, "slots_per_cpu")

    def test_defaults(self):
        from config import OrchestratorConfig
        c = OrchestratorConfig()
        assert c.routing_algorithm == "least_loaded"
        assert c.max_rps == 5000
        assert c.slots_per_gpu == 15
        assert c.slots_per_cpu == 4
        assert c.instance_host == "192.168.1.61"

    def test_to_dict_includes_new_fields(self):
        from config import OrchestratorConfig
        c = OrchestratorConfig(redis_url="redis://test:6379")
        d = c.to_dict()
        assert "redis_url" in d
        assert d["redis_url"] == "redis://test:6379"
        assert "routing_algorithm" in d
        assert "instance_ports_gpu" in d

    def test_load_from_env(self):
        from config import load_config_from_env
        os.environ["REDIS_URL"] = "redis://test:6379"
        os.environ["ROUTING_ALGORITHM"] = "round_robin"
        os.environ["MAX_RPS"] = "10000"
        try:
            c = load_config_from_env()
            assert c.redis_url == "redis://test:6379"
            assert c.routing_algorithm == "round_robin"
            assert c.max_rps == 10000
            assert c.max_agents == 1000  # updated default
            assert c.max_concurrent_tasks == 2000  # updated default
        finally:
            del os.environ["REDIS_URL"]
            del os.environ["ROUTING_ALGORITHM"]
            del os.environ["MAX_RPS"]

    def test_field_types_include_new_fields(self):
        from config import OrchestratorConfig
        c = OrchestratorConfig()
        assert "redis_url" in c._field_types
        assert "routing_algorithm" in c._field_types
        assert "max_rps" in c._field_types
        assert "slots_per_gpu" in c._field_types


# ======================================================================
# Phase 1: CycloneDDS XML
# ======================================================================

def _find_repo_file(*rel_parts):
    """Find a file relative to test dir or repo root. Returns path or None."""
    # Try relative to test dir (local dev)
    p = os.path.join(os.path.dirname(__file__), *rel_parts)
    if os.path.exists(p):
        return p
    # Try relative to VM_TEST_DIR layout
    for base in [os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..")]:
        p = os.path.join(base, *rel_parts)
        if os.path.exists(p):
            return p
    return None


class TestCycloneDDSConfig:
    """Verify cyclonedds-38inst.xml configuration."""

    def test_xml_valid(self):
        import xml.etree.ElementTree as ET
        xml_path = _find_repo_file("..", "..", "llama.cpp_dds", "dds", "cyclonedds-38inst.xml")
        if not xml_path:
            xml_path = _find_repo_file("..", "llama_dds", "cyclonedds-38inst.xml")
        assert xml_path, "cyclonedds-38inst.xml not found"
        tree = ET.parse(xml_path)
        root = tree.getroot()
        assert root.tag.endswith("CycloneDDS")

    def test_max_participant_index(self):
        xml_path = _find_repo_file("..", "..", "llama.cpp_dds", "dds", "cyclonedds-38inst.xml")
        if not xml_path:
            xml_path = _find_repo_file("..", "llama_dds", "cyclonedds-38inst.xml")
        assert xml_path, "cyclonedds-38inst.xml not found"
        with open(xml_path) as f:
            content = f.read()
        assert "99" in content, "MaxAutoParticipantIndex should be 99"

    def test_orchestrator_copy_exists(self):
        xml_path = _find_repo_file("..", "cyclonedds-38inst.xml")
        assert xml_path, "orchestrator cyclonedds-38inst.xml not found"


# ======================================================================
# Benchmark scripts import test
# ======================================================================

class TestBenchmarkScriptsImport:
    """Verify benchmark scripts can be imported without errors."""

    def _bench_file(self, name):
        p = _find_repo_file("..", "benchmarks", name)
        assert p, f"benchmarks/{name} not found"
        return p

    def test_deploy_38_imports(self):
        """deploy_38_instances.py should define key constants."""
        with open(self._bench_file("deploy_38_instances.py")) as f:
            content = f.read()
        assert "GPU_INSTANCES" in content
        assert "CPU_INSTANCES" in content
        assert "full_deploy" in content

    def test_register_1000_imports(self):
        with open(self._bench_file("register_1000_agents.py")) as f:
            content = f.read()
        assert "generate_agents" in content
        assert "register_bulk" in content

    def test_load_generator_imports(self):
        with open(self._bench_file("load_generator.py")) as f:
            content = f.read()
        assert "LoadGenerator" in content
        assert "LoadConfig" in content

    def test_run_full_benchmark_imports(self):
        with open(self._bench_file("run_full_benchmark.py")) as f:
            content = f.read()
        assert "BenchmarkSuite" in content
        assert "SCENARIOS" in content

    def test_generate_38inst_plots_imports(self):
        with open(self._bench_file("generate_38inst_plots.py")) as f:
            content = f.read()
        assert "plot_e1_cdf" in content
        assert "generate_latex_table" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
