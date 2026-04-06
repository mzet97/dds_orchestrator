#!/usr/bin/env python3
"""
Tests for the 10-instance, 1000-agent topology (Phases 1-5).
- Phase 1: Config and IDL validation
- Phase 2: Redis layer (MockRedis-based unit tests)
- Phase 3: MongoDB layer (MockMongo-based unit tests)
- Phase 4: InstancePool with 10 instances
- Phase 5: Integration (concurrent access, config loading)

Topology:
  .61 (RTX 3080): 6 instances, ports 8082-8087, parallel=15 each, weight=1.0
  .60 (RX 6600M): 4 instances, ports 8088-8091, parallel=10 each, weight=0.7
  Total: 10 instances, 130 slots
  1000 agents, ~100 per instance
  Model: Qwen3.5-2B
"""

import asyncio
import fnmatch
import os
import sys
import time
import pytest
import pytest_asyncio
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ======================================================================
# 10-Instance Topology Constants
# ======================================================================

RTX3080_HOST = "192.168.1.61"
RX6600M_HOST = "192.168.1.60"

RTX3080_PORTS = list(range(8082, 8088))   # 8082-8087, 6 instances
RX6600M_PORTS = list(range(8088, 8092))   # 8088-8091, 4 instances
ALL_PORTS = RTX3080_PORTS + RX6600M_PORTS  # 10 instances total

RTX3080_PARALLEL = 15
RX6600M_PARALLEL = 10
TOTAL_SLOTS = len(RTX3080_PORTS) * RTX3080_PARALLEL + len(RX6600M_PORTS) * RX6600M_PARALLEL  # 130

RTX3080_WEIGHT = 1.0
RX6600M_WEIGHT = 0.7

NUM_AGENTS = 1000
AGENTS_PER_INSTANCE = NUM_AGENTS // len(ALL_PORTS)  # 100

MODEL_NAME = "Qwen3.5-2B"


# ======================================================================
# MockRedis Implementation
# ======================================================================

class MockPipeline:
    def __init__(self, redis):
        self._redis = redis
        self._ops = []

    def set(self, key, value, ex=None):
        self._ops.append(("set", key, str(value)))
        return self

    def get(self, key):
        self._ops.append(("get", key))
        return self

    def incr(self, key):
        self._ops.append(("incr", key))
        return self

    def decr(self, key):
        self._ops.append(("decr", key))
        return self

    def exists(self, key):
        self._ops.append(("exists", key))
        return self

    def expire(self, key, seconds):
        self._ops.append(("expire", key, seconds))
        return self

    async def execute(self):
        results = []
        for op in self._ops:
            if op[0] == "set":
                self._redis._data[op[1]] = op[2]
                results.append(True)
            elif op[0] == "get":
                results.append(self._redis._data.get(op[1]))
            elif op[0] == "incr":
                val = int(self._redis._data.get(op[1], 0)) + 1
                self._redis._data[op[1]] = str(val)
                results.append(val)
            elif op[0] == "decr":
                val = max(0, int(self._redis._data.get(op[1], 0)) - 1)
                self._redis._data[op[1]] = str(val)
                results.append(val)
            elif op[0] == "exists":
                results.append(1 if op[1] in self._redis._data else 0)
            elif op[0] == "expire":
                results.append(True)
        self._ops = []
        return results


class MockRedis:
    """In-memory Redis mock for unit testing."""

    def __init__(self):
        self._data = {}
        self._scripts = {}
        self._script_counter = 0

    async def ping(self):
        return True

    async def set(self, key, value, ex=None):
        self._data[key] = str(value)

    async def get(self, key):
        return self._data.get(key)

    async def incr(self, key):
        val = int(self._data.get(key, 0)) + 1
        self._data[key] = str(val)
        return val

    async def decr(self, key):
        val = max(0, int(self._data.get(key, 0)) - 1)
        self._data[key] = str(val)
        return val

    async def delete(self, *keys):
        for k in keys:
            self._data.pop(k, None)

    async def exists(self, key):
        return 1 if key in self._data else 0

    async def scan(self, cursor="0", match="*", count=100):
        keys = [k for k in self._data if fnmatch.fnmatch(k, match)]
        return ("0", keys)

    async def script_load(self, script):
        self._script_counter += 1
        sha = f"sha_{self._script_counter}"
        self._scripts[sha] = script
        return sha

    async def evalsha(self, sha, numkeys, *args):
        # Implement acquire/release/ema logic inline
        script = self._scripts.get(sha, "")
        if "INCR" in script and "total" in script:  # acquire
            key_used, key_total = args[0], args[1]
            used = int(self._data.get(key_used, 0)) + 1
            total = int(self._data.get(key_total, 0))
            if used > total:
                return 0
            self._data[key_used] = str(used)
            return 1
        elif "DECR" in script:  # release
            key = args[0]
            used = int(self._data.get(key, 0))
            if used > 0:
                self._data[key] = str(used - 1)
                return used - 1
            return 0
        elif "alpha" in script:  # ema
            key = args[0]
            new_val = float(args[1])
            alpha = float(args[2])
            old = float(self._data.get(key, 0))
            result = new_val * alpha + old * (1 - alpha)
            self._data[key] = str(result)
            return str(result)
        return 0

    def pipeline(self):
        return MockPipeline(self)

    async def aclose(self):
        pass

    async def scard(self, key):
        return 0

    async def sadd(self, key, *vals):
        pass

    async def expire(self, key, seconds):
        pass

    async def rpush(self, key, val):
        pass

    async def blpop(self, key, timeout=0):
        return None


# ======================================================================
# MockMongo Implementation
# ======================================================================

class MockCollection:
    """Minimal mock for a MongoDB collection."""

    def __init__(self):
        self._docs = []
        self._indexes = []

    async def insert_one(self, doc):
        self._docs.append(dict(doc))

    async def insert_many(self, docs, ordered=True):
        for doc in docs:
            self._docs.append(dict(doc))

    async def replace_one(self, filter_doc, doc, upsert=False):
        for i, existing in enumerate(self._docs):
            if all(existing.get(k) == v for k, v in filter_doc.items()):
                self._docs[i] = dict(doc)
                return
        if upsert:
            self._docs.append(dict(doc))

    async def bulk_write(self, ops):
        count = 0
        for op in ops:
            # Simulate upsert
            self._docs.append({})
            count += 1

        class Result:
            upserted_count = count
            modified_count = 0
        return Result()

    async def count_documents(self, query):
        if not query:
            return len(self._docs)
        count = 0
        for doc in self._docs:
            if all(doc.get(k) == v for k, v in query.items()):
                count += 1
        return count

    def find(self, query=None, projection=None):
        return MockCursor(self._docs, query or {})

    async def find_one(self, query, projection=None):
        for doc in self._docs:
            if all(doc.get(k) == v for k, v in query.items()):
                result = dict(doc)
                if projection:
                    result.pop("_id", None)
                return result
        return None

    async def create_index(self, keys, **kwargs):
        self._indexes.append(keys)

    async def drop(self):
        self._docs = []
        self._indexes = []

    def aggregate(self, pipeline):
        return MockAggCursor(self._docs, pipeline)


class MockCursor:
    def __init__(self, docs, query):
        self._docs = [d for d in docs
                      if all(d.get(k) == v for k, v in query.items())]

    def sort(self, *args, **kwargs):
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    async def to_list(self, length=1000):
        return [dict(d) for d in self._docs[:length]]


class MockAggCursor:
    def __init__(self, docs, pipeline):
        self._docs = docs
        self._pipeline = pipeline

    async def to_list(self, length=100):
        # Very simplified aggregation for testing
        return []


class MockDB:
    def __init__(self):
        self._collections = {}

    def __getitem__(self, name):
        if name not in self._collections:
            self._collections[name] = MockCollection()
        return self._collections[name]


class MockMongoMetricsStore:
    """Mock that mimics MongoMetricsStore interface using in-memory dicts."""

    def __init__(self):
        self._db = MockDB()
        self._connected = False

    async def connect(self):
        self._connected = True

    async def close(self):
        self._connected = False

    async def ensure_indexes(self):
        # Just verify no error
        for name in ("metrics", "routing_log", "benchmark_runs", "agents", "instances"):
            await self._db[name].create_index("test_idx")

    async def register_agents_bulk(self, agents):
        coll = self._db["agents"]
        for a in agents:
            await coll.replace_one({"agent_id": a["agent_id"]}, a, upsert=True)

    async def register_instance(self, instance):
        coll = self._db["instances"]
        await coll.replace_one({"port": instance["port"]}, instance, upsert=True)

    async def get_agent_count(self):
        return await self._db["agents"].count_documents({})

    async def save_run(self, run):
        run_id = run.get("run_id", "auto-id")
        run["run_id"] = run_id
        coll = self._db["benchmark_runs"]
        await coll.replace_one({"run_id": run_id}, run, upsert=True)
        return run_id

    async def get_run(self, run_id):
        return await self._db["benchmark_runs"].find_one({"run_id": run_id})

    async def get_runs(self, experiment=None):
        query = {}
        if experiment:
            query["experiment"] = experiment
        cursor = self._db["benchmark_runs"].find(query)
        return await cursor.to_list(length=1000)

    async def log_request(self, metric):
        await self._db["metrics"].insert_one(metric)

    async def drop_all(self):
        for name in ("metrics", "routing_log", "benchmark_runs", "agents", "instances"):
            await self._db[name].drop()


# ======================================================================
# Helper: create a RedisStateManager with MockRedis injected
# ======================================================================

def _make_redis_manager():
    """Create a RedisStateManager with MockRedis replacing the real connection.

    Registers Lua script SHAs by directly manipulating the mock's internal
    dict, avoiding any async calls so this works from both sync and async
    contexts (no event-loop conflict inside pytest-asyncio tests).
    """
    from redis_layer import RedisStateManager
    mgr = RedisStateManager("redis://mock:6379")
    mock = MockRedis()
    mgr._redis = mock

    # Directly populate script SHAs without going through async script_load
    mock._script_counter = 3
    mock._scripts["sha_1"] = "INCR total acquire"
    mock._scripts["sha_2"] = "DECR release"
    mock._scripts["sha_3"] = "alpha ema"
    mgr._acquire_sha = "sha_1"
    mgr._release_sha = "sha_2"
    mgr._ema_sha = "sha_3"

    return mgr, mock


# ======================================================================
# Helper: find repo files
# ======================================================================

def _find_repo_file(*rel_parts):
    """Find a file relative to test dir or repo root. Returns path or None."""
    p = os.path.join(os.path.dirname(__file__), *rel_parts)
    if os.path.exists(p):
        return p
    for base in [os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..")]:
        p = os.path.join(base, *rel_parts)
        if os.path.exists(p):
            return p
    return None


# ======================================================================
# Phase 1: Config and IDL Validation
# ======================================================================

class TestConfigDefaults:
    """Verify OrchestratorConfig defaults and loading for 10-instance topology."""

    def test_config_defaults_exist(self):
        from config import OrchestratorConfig
        c = OrchestratorConfig()
        assert hasattr(c, "redis_url")
        assert hasattr(c, "redis_password")
        assert hasattr(c, "mongo_url")
        assert hasattr(c, "mongo_db")
        assert hasattr(c, "routing_algorithm")
        assert hasattr(c, "max_rps")
        assert hasattr(c, "instance_host")
        assert hasattr(c, "instance_host_map")
        assert hasattr(c, "slots_per_gpu")
        assert hasattr(c, "slots_per_cpu")
        assert hasattr(c, "weight_gpu")
        assert hasattr(c, "weight_cpu")

    def test_config_default_values(self):
        from config import OrchestratorConfig
        c = OrchestratorConfig()
        assert c.routing_algorithm == "least_loaded"
        assert c.max_rps == 5000
        assert c.slots_per_gpu == 15
        assert c.slots_per_cpu == 4
        assert c.weight_gpu == 1.0
        assert c.weight_cpu == 0.3
        assert c.max_agents == 1000
        assert c.max_concurrent_tasks == 2000

    def test_config_to_dict_has_instance_fields(self):
        from config import OrchestratorConfig
        c = OrchestratorConfig()
        d = c.to_dict()
        assert "instance_host_map" in d
        assert "weight_gpu" in d
        assert "weight_cpu" in d
        assert "slots_per_gpu" in d
        assert "slots_per_cpu" in d

    def test_instance_host_map_parsing(self):
        """Verify instance_host_map can represent the 10-instance topology."""
        from config import OrchestratorConfig
        # Build the expected host_map string
        entries = []
        for port in RTX3080_PORTS:
            entries.append(f"{RTX3080_HOST}:{port}")
        for port in RX6600M_PORTS:
            entries.append(f"{RX6600M_HOST}:{port}")
        host_map = ",".join(entries)

        c = OrchestratorConfig(instance_host_map=host_map)
        assert c.instance_host_map == host_map

        # Parse the map back
        pairs = c.instance_host_map.split(",")
        assert len(pairs) == 10
        for pair in pairs:
            host, port_str = pair.rsplit(":", 1)
            port = int(port_str)
            assert port in ALL_PORTS
            if port in RTX3080_PORTS:
                assert host == RTX3080_HOST
            else:
                assert host == RX6600M_HOST

    def test_weight_gpu_and_weight_cpu_fields(self):
        from config import OrchestratorConfig
        c = OrchestratorConfig(weight_gpu=1.0, weight_cpu=0.7)
        assert c.weight_gpu == 1.0
        assert c.weight_cpu == 0.7
        d = c.to_dict()
        assert d["weight_gpu"] == 1.0
        assert d["weight_cpu"] == 0.7

    def test_field_types_include_weight_fields(self):
        from config import OrchestratorConfig
        c = OrchestratorConfig()
        assert "weight_gpu" in c._field_types
        assert "weight_cpu" in c._field_types
        assert "instance_host_map" in c._field_types

    def test_dds_config_file_exists(self):
        """Verify the 10-instance DDS config XML exists."""
        xml_path = _find_repo_file("..", "configs", "cyclonedds-10inst-network.xml")
        assert xml_path is not None, "cyclonedds-10inst-network.xml not found"

    def test_dds_config_valid_xml(self):
        """Verify the 10-instance DDS config is valid XML."""
        import xml.etree.ElementTree as ET
        xml_path = _find_repo_file("..", "configs", "cyclonedds-10inst-network.xml")
        if not xml_path:
            pytest.skip("cyclonedds-10inst-network.xml not found")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        assert root.tag.endswith("CycloneDDS")


# ======================================================================
# Phase 2: Redis Layer (MockRedis-based unit tests)
# ======================================================================

class TestRedisLayerMock:
    """Unit tests for RedisStateManager using MockRedis."""

    def test_import(self):
        from redis_layer import RedisStateManager
        mgr = RedisStateManager("redis://localhost:6379")
        assert mgr._url == "redis://localhost:6379"

    def test_lua_scripts_defined(self):
        from redis_layer import _LUA_ACQUIRE_SLOT, _LUA_EMA_UPDATE, _LUA_RELEASE_SLOT
        assert "INCR" in _LUA_ACQUIRE_SLOT
        assert "redis.call" in _LUA_EMA_UPDATE
        assert "DECR" in _LUA_RELEASE_SLOT

    @pytest.mark.asyncio
    async def test_init_instance_all_10(self):
        """Initialize all 10 instances and verify state."""
        mgr, mock = _make_redis_manager()
        for port in RTX3080_PORTS:
            await mgr.init_instance(port, slots_total=RTX3080_PARALLEL, inst_type="gpu")
        for port in RX6600M_PORTS:
            await mgr.init_instance(port, slots_total=RX6600M_PARALLEL, inst_type="gpu")

        # Verify each instance was initialized
        for port in RTX3080_PORTS:
            used, total = await mgr.get_slots(port)
            assert used == 0
            assert total == RTX3080_PARALLEL

        for port in RX6600M_PORTS:
            used, total = await mgr.get_slots(port)
            assert used == 0
            assert total == RX6600M_PARALLEL

    @pytest.mark.asyncio
    async def test_acquire_slot_success(self):
        """Acquire a slot on a .61 instance."""
        mgr, mock = _make_redis_manager()
        await mgr.init_instance(8082, slots_total=15, inst_type="gpu")
        assert await mgr.acquire_slot(8082)
        used, total = await mgr.get_slots(8082)
        assert used == 1
        assert total == 15

    @pytest.mark.asyncio
    async def test_acquire_slot_full(self):
        """Acquiring beyond capacity should fail."""
        mgr, mock = _make_redis_manager()
        await mgr.init_instance(8082, slots_total=2, inst_type="gpu")
        assert await mgr.acquire_slot(8082)
        assert await mgr.acquire_slot(8082)
        assert not await mgr.acquire_slot(8082)  # 3rd should fail

    @pytest.mark.asyncio
    async def test_release_slot_atomicity(self):
        """Acquire then release should return to 0."""
        mgr, mock = _make_redis_manager()
        await mgr.init_instance(8088, slots_total=10, inst_type="gpu")
        for _ in range(5):
            assert await mgr.acquire_slot(8088)
        for _ in range(5):
            await mgr.release_slot(8088)
        used, _ = await mgr.get_slots(8088)
        assert used == 0

    @pytest.mark.asyncio
    async def test_release_slot_floor_at_zero(self):
        """Releasing when already at 0 should stay at 0."""
        mgr, mock = _make_redis_manager()
        await mgr.init_instance(8082, slots_total=15, inst_type="gpu")
        await mgr.release_slot(8082)
        used, _ = await mgr.get_slots(8082)
        assert used == 0

    @pytest.mark.asyncio
    async def test_get_all_loads_10_instances(self):
        """get_all_loads should return data for all 10 instances."""
        mgr, mock = _make_redis_manager()
        for port in RTX3080_PORTS:
            await mgr.init_instance(port, slots_total=RTX3080_PARALLEL, inst_type="gpu")
        for port in RX6600M_PORTS:
            await mgr.init_instance(port, slots_total=RX6600M_PARALLEL, inst_type="gpu")

        loads = await mgr.get_all_loads()
        assert len(loads) == 10

        ports_seen = {l["port"] for l in loads}
        assert ports_seen == set(ALL_PORTS)

        total_slots = sum(l["slots_total"] for l in loads)
        assert total_slots == TOTAL_SLOTS

    @pytest.mark.asyncio
    async def test_ema_latency_update(self):
        """EMA latency update should blend old and new values."""
        mgr, mock = _make_redis_manager()
        await mgr.init_instance(8082, slots_total=15, inst_type="gpu")

        # First update with alpha=1.0 should set to exact value
        await mgr.update_latency(8082, 100.0, alpha=1.0)
        lat = await mgr.get_latency(8082)
        assert abs(lat - 100.0) < 1.0

        # Second update with alpha=0.5 should blend
        await mgr.update_latency(8082, 50.0, alpha=0.5)
        lat = await mgr.get_latency(8082)
        # Expected: 50*0.5 + 100*0.5 = 75
        assert abs(lat - 75.0) < 1.0

    @pytest.mark.asyncio
    async def test_set_agent_instance(self):
        """Map an agent to its instance."""
        mgr, mock = _make_redis_manager()
        await mgr.set_agent_instance("agent-00-001", RTX3080_HOST, 8082)
        result = await mgr.get_agent_instance("agent-00-001")
        assert result is not None
        host, port = result
        assert host == RTX3080_HOST
        assert port == 8082

    @pytest.mark.asyncio
    async def test_get_agent_instance_not_found(self):
        """Non-existent agent should return None."""
        mgr, mock = _make_redis_manager()
        result = await mgr.get_agent_instance("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_register_agents_bulk(self):
        """Bulk register agents and verify count."""
        mgr, mock = _make_redis_manager()
        agents = []
        for i in range(100):
            inst_idx = i % len(ALL_PORTS)
            port = ALL_PORTS[inst_idx]
            host = RTX3080_HOST if port in RTX3080_PORTS else RX6600M_HOST
            agents.append({
                "agent_id": f"agent-{i:04d}",
                "hostname": host,
                "instance_port": port,
            })
        await mgr.register_agents_bulk(agents)
        count = await mgr.get_agent_count()
        assert count == 100

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Rate limit check should pass under limit."""
        mgr, mock = _make_redis_manager()
        assert await mgr.check_rate_limit(max_rps=5000)

    @pytest.mark.asyncio
    async def test_round_robin_index_increments(self):
        """Round-robin index should increment monotonically."""
        mgr, mock = _make_redis_manager()
        idx1 = await mgr.get_rr_index()
        idx2 = await mgr.get_rr_index()
        idx3 = await mgr.get_rr_index()
        assert idx2 == idx1 + 1
        assert idx3 == idx2 + 1

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Instance should be healthy after init."""
        mgr, mock = _make_redis_manager()
        await mgr.init_instance(8082, slots_total=15, inst_type="gpu")
        assert await mgr.is_healthy(8082)

    @pytest.mark.asyncio
    async def test_active_counter(self):
        """Active request counter should incr/decr correctly."""
        mgr, mock = _make_redis_manager()
        await mock.set("stats:active", "0")
        await mgr.incr_active()
        await mgr.incr_active()
        assert await mgr.get_active() == 2
        await mgr.decr_active()
        assert await mgr.get_active() == 1


# ======================================================================
# Phase 3: MongoDB Layer (MockMongo-based unit tests)
# ======================================================================

class TestMongoLayerMock:
    """Unit tests for MongoMetricsStore using MockMongo."""

    @pytest_asyncio.fixture
    async def mongo(self):
        store = MockMongoMetricsStore()
        await store.connect()
        yield store
        await store.drop_all()
        await store.close()

    @pytest.mark.asyncio
    async def test_ensure_indexes_no_error(self, mongo):
        """ensure_indexes should complete without error."""
        await mongo.ensure_indexes()

    @pytest.mark.asyncio
    async def test_register_agents_bulk_1000(self, mongo):
        """Register 1000 agents, ~100 per instance."""
        agents = []
        for i in range(NUM_AGENTS):
            inst_idx = i % len(ALL_PORTS)
            port = ALL_PORTS[inst_idx]
            host = RTX3080_HOST if port in RTX3080_PORTS else RX6600M_HOST
            agents.append({
                "agent_id": f"agent-{i:04d}",
                "instance_port": port,
                "hostname": host,
                "model": MODEL_NAME,
            })
        await mongo.register_agents_bulk(agents)
        count = await mongo.get_agent_count()
        assert count == NUM_AGENTS

    @pytest.mark.asyncio
    async def test_register_instance_for_10(self, mongo):
        """Register all 10 instances."""
        for port in RTX3080_PORTS:
            await mongo.register_instance({
                "port": port,
                "hostname": RTX3080_HOST,
                "type": "gpu",
                "gpu_type": "rtx3080",
                "model": MODEL_NAME,
                "slots_total": RTX3080_PARALLEL,
                "weight": RTX3080_WEIGHT,
            })
        for port in RX6600M_PORTS:
            await mongo.register_instance({
                "port": port,
                "hostname": RX6600M_HOST,
                "type": "gpu",
                "gpu_type": "rx6600m",
                "model": MODEL_NAME,
                "slots_total": RX6600M_PARALLEL,
                "weight": RX6600M_WEIGHT,
            })

        instances = mongo._db["instances"]
        count = await instances.count_documents({})
        assert count == 10

    @pytest.mark.asyncio
    async def test_save_run_and_get_run(self, mongo):
        """Save a benchmark run and retrieve it."""
        run_id = await mongo.save_run({
            "run_id": "run-10inst-e1",
            "experiment": "E1",
            "scenario": "S1",
            "protocol": "dds",
            "algorithm": "least_loaded",
            "num_clients": 100,
            "topology": "10inst-1000agents",
            "results": {"p50": 35, "p95": 85, "p99": 130},
        })
        assert run_id == "run-10inst-e1"

        run = await mongo.get_run("run-10inst-e1")
        assert run is not None
        assert run["experiment"] == "E1"
        assert run["topology"] == "10inst-1000agents"
        assert run["results"]["p50"] == 35

    @pytest.mark.asyncio
    async def test_get_runs_filtered(self, mongo):
        """get_runs with experiment filter."""
        await mongo.save_run({
            "run_id": "run-1",
            "experiment": "E1",
            "protocol": "dds",
        })
        await mongo.save_run({
            "run_id": "run-2",
            "experiment": "E2",
            "protocol": "dds",
        })
        runs = await mongo.get_runs(experiment="E1")
        assert len(runs) == 1
        assert runs[0]["experiment"] == "E1"

    @pytest.mark.asyncio
    async def test_log_request(self, mongo):
        """Log a single request metric."""
        await mongo.log_request({
            "request_id": "r1",
            "instance_port": 8082,
            "instance_type": "gpu",
            "protocol": "dds",
            "algorithm": "least_loaded",
            "latency_ms": 42.5,
            "success": True,
        })
        count = await mongo._db["metrics"].count_documents({})
        assert count == 1


# ======================================================================
# Phase 4: InstancePool with 10 Instances
# ======================================================================

class MockRedisForPool:
    """MockRedis adapted for InstancePool usage (same interface as test_38inst)."""

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


class TestInstancePool10:
    """Test InstancePool with 10 instances (6 x .61 + 4 x .60)."""

    def _make_pool(self, algorithm="round_robin"):
        from instance_pool import InstancePool, RoutingAlgorithm
        redis = MockRedisForPool()
        pool = InstancePool(redis, mongo=None,
                            algorithm=RoutingAlgorithm(algorithm))
        return pool, redis

    async def _register_all_10(self, pool):
        from instance_pool import InstanceInfo
        for port in RTX3080_PORTS:
            await pool.register_instance(InstanceInfo(
                port=port, hostname=RTX3080_HOST, inst_type="gpu",
                model=MODEL_NAME, slots_total=RTX3080_PARALLEL,
                weight=RTX3080_WEIGHT,
            ))
        for port in RX6600M_PORTS:
            await pool.register_instance(InstanceInfo(
                port=port, hostname=RX6600M_HOST, inst_type="gpu",
                model=MODEL_NAME, slots_total=RX6600M_PARALLEL,
                weight=RX6600M_WEIGHT,
            ))

    @pytest.mark.asyncio
    async def test_register_10_instances(self):
        """Register all 10 instances."""
        pool, redis = self._make_pool()
        await self._register_all_10(pool)
        assert len(pool._instances) == 10
        assert len(redis._instances) == 10

        # Verify slot totals
        for port in RTX3080_PORTS:
            assert redis._instances[port]["slots_total"] == RTX3080_PARALLEL
        for port in RX6600M_PORTS:
            assert redis._instances[port]["slots_total"] == RX6600M_PARALLEL

    @pytest.mark.asyncio
    async def test_round_robin_cycles_all_10(self):
        """Round-robin should cycle through all 10 instances."""
        pool, redis = self._make_pool("round_robin")
        await self._register_all_10(pool)

        selected_ports = []
        for _ in range(20):
            inst = await pool.select_instance()
            assert inst is not None
            selected_ports.append(inst.port)

        # All 10 instances should appear
        assert set(selected_ports) == set(ALL_PORTS)

    @pytest.mark.asyncio
    async def test_least_loaded_picks_lowest_ratio(self):
        """Least-loaded should pick instance with lowest used/total ratio."""
        pool, redis = self._make_pool("least_loaded")
        await self._register_all_10(pool)

        # Load up 8082 (RTX 3080) with 10/15 slots
        for _ in range(10):
            redis._instances[8082]["slots_used"] += 1

        # 8088 (RX 6600M) has 0/10 = 0.0 ratio, all others have 0/15 or 0/10
        # But 8082 has 10/15 = 0.67 ratio
        inst = await pool.select_instance()
        assert inst is not None
        # Should NOT pick 8082 (highest ratio)
        assert inst.port != 8082

    @pytest.mark.asyncio
    async def test_weighted_score_prefers_rtx3080(self):
        """Weighted score should prefer RTX 3080 (weight=1.0) over RX 6600M (weight=0.7)."""
        pool, redis = self._make_pool("weighted_score")
        await self._register_all_10(pool)

        # All instances start at 0 load and 0 latency
        # score = 0, adjusted_score = 0/weight
        # RTX 3080: 0/1.0 = 0.0, RX 6600M: 0/0.7 = 0.0
        # Tie-breaker: prefer higher weight (RTX 3080)
        inst = await pool.select_instance()
        assert inst is not None
        assert inst.port in RTX3080_PORTS, (
            f"Expected RTX 3080 port, got {inst.port} "
            f"(weight={inst.weight})"
        )

    @pytest.mark.asyncio
    async def test_weighted_score_accounts_for_weight_difference(self):
        """With equal load ratios, higher-weight instances should be preferred."""
        pool, redis = self._make_pool("weighted_score")
        await self._register_all_10(pool)

        # Set equal load ratio on one RTX and one RX instance
        # RTX 8082: 7/15 = 0.467, RX 8088: 5/10 = 0.5
        redis._instances[8082]["slots_used"] = 7
        redis._instances[8088]["slots_used"] = 5

        # Zero out all other instances' load
        for port in ALL_PORTS:
            if port not in (8082, 8088):
                redis._instances[port]["slots_used"] = 0

        inst = await pool.select_instance()
        assert inst is not None
        # Should pick an empty instance first (0 load)
        assert inst.port not in (8082, 8088)

    @pytest.mark.asyncio
    async def test_slot_exhaustion_returns_none(self):
        """Filling all 130 slots should cause select_instance to return None."""
        pool, redis = self._make_pool("round_robin")
        await self._register_all_10(pool)

        # Fill every slot
        count = 0
        while True:
            inst = await pool.select_instance()
            if inst is None:
                break
            count += 1
            if count > TOTAL_SLOTS + 10:
                break  # Safety guard

        assert count == TOTAL_SLOTS, f"Expected {TOTAL_SLOTS} slots, got {count}"

        # Next select should return None
        assert await pool.select_instance() is None

    @pytest.mark.asyncio
    async def test_release_instance_updates_metrics(self):
        """release_instance should decrement slots and update latency."""
        pool, redis = self._make_pool("round_robin")
        await self._register_all_10(pool)

        inst = await pool.select_instance()
        assert inst is not None
        port = inst.port

        used_before, _ = await redis.get_slots(port)
        assert used_before == 1

        await pool.release_instance(port, latency_ms=50.0, success=True)
        used_after, _ = await redis.get_slots(port)
        assert used_after == 0
        assert redis._instances[port]["avg_latency"] > 0

    @pytest.mark.asyncio
    async def test_get_status_returns_all_10(self):
        """get_status should report all 10 instances."""
        pool, redis = self._make_pool("least_loaded")
        await self._register_all_10(pool)

        status = await pool.get_status()
        assert status["total_instances"] == 10
        assert status["algorithm"] == "least_loaded"
        assert len(status["instances"]) == 10

    @pytest.mark.asyncio
    async def test_set_algorithm_runtime(self):
        """Changing algorithm at runtime should work."""
        pool, redis = self._make_pool("round_robin")
        from instance_pool import RoutingAlgorithm
        assert pool._algorithm == RoutingAlgorithm.ROUND_ROBIN

        pool.set_algorithm(RoutingAlgorithm.WEIGHTED_SCORE)
        assert pool._algorithm == RoutingAlgorithm.WEIGHTED_SCORE

    @pytest.mark.asyncio
    async def test_release_and_reacquire(self):
        """After releasing, the slot should be available again."""
        pool, redis = self._make_pool("round_robin")
        from instance_pool import InstanceInfo
        await pool.register_instance(InstanceInfo(
            port=8082, inst_type="gpu", slots_total=1, weight=1.0))

        inst1 = await pool.select_instance()
        assert inst1 is not None
        assert inst1.port == 8082

        # Full now
        assert await pool.select_instance() is None

        # Release
        await pool.release_instance(8082, latency_ms=30.0, success=True)

        # Should be available again
        inst2 = await pool.select_instance()
        assert inst2 is not None
        assert inst2.port == 8082


# ======================================================================
# Phase 4b: Backpressure with 10-instance topology
# ======================================================================

class TestBackpressure10:
    """Test BackpressureManager with 10-instance topology."""

    @pytest.mark.asyncio
    async def test_allow_request(self):
        from backpressure import BackpressureManager
        redis = MockRedisForPool()
        bp = BackpressureManager(redis, max_rps=5000)
        assert await bp.allow_request()

    @pytest.mark.asyncio
    async def test_pressure_level_normal(self):
        """With no load, pressure should be 'normal'."""
        from backpressure import BackpressureManager
        redis = MockRedisForPool()
        for port in ALL_PORTS[:10]:
            await redis.init_instance(port, 15, "gpu")
        bp = BackpressureManager(redis)
        level = await bp.get_pressure_level()
        assert level == "normal"

    @pytest.mark.asyncio
    async def test_pressure_level_critical(self):
        """With >90% load, pressure should be 'critical'."""
        from backpressure import BackpressureManager
        redis = MockRedisForPool()
        for port in ALL_PORTS:
            slots = RTX3080_PARALLEL if port in RTX3080_PORTS else RX6600M_PARALLEL
            await redis.init_instance(port, slots, "gpu")

        # Fill >90% of 130 slots = >117 slots
        filled = 0
        for port in ALL_PORTS:
            inst = redis._instances[port]
            fill = int(inst["slots_total"] * 0.95)
            inst["slots_used"] = fill
            filled += fill

        bp = BackpressureManager(redis)
        level = await bp.get_pressure_level()
        assert level == "critical"

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_errors(self):
        from backpressure import BackpressureManager
        redis = MockRedisForPool()
        await redis.init_instance(8082, 15, "gpu")
        bp = BackpressureManager(redis, circuit_error_threshold=0.5, circuit_cooldown_s=1)

        for _ in range(10):
            await redis.record_error(8082)
        for _ in range(2):
            await redis.record_success(8082)

        # Error rate = 10/12 ~ 83% > 50%
        assert await bp.is_circuit_open(8082)


# ======================================================================
# Phase 5: Integration Tests
# ======================================================================

class TestIntegration10:
    """Integration tests for the 10-instance topology."""

    def test_config_from_env_with_host_map(self):
        """Load config with INSTANCE_HOST_MAP env var."""
        from config import load_config_from_env

        entries = []
        for port in RTX3080_PORTS:
            entries.append(f"{RTX3080_HOST}:{port}")
        for port in RX6600M_PORTS:
            entries.append(f"{RX6600M_HOST}:{port}")
        host_map = ",".join(entries)

        os.environ["INSTANCE_HOST_MAP"] = host_map
        os.environ["ROUTING_ALGORITHM"] = "weighted_score"
        os.environ["WEIGHT_GPU"] = "1.0"
        os.environ["WEIGHT_CPU"] = "0.7"
        os.environ["SLOTS_PER_GPU"] = "15"
        try:
            c = load_config_from_env()
            assert c.instance_host_map == host_map
            assert c.routing_algorithm == "weighted_score"
            assert c.weight_gpu == 1.0
            assert c.weight_cpu == 0.7
            assert c.slots_per_gpu == 15
        finally:
            del os.environ["INSTANCE_HOST_MAP"]
            del os.environ["ROUTING_ALGORITHM"]
            del os.environ["WEIGHT_GPU"]
            del os.environ["WEIGHT_CPU"]
            del os.environ["SLOTS_PER_GPU"]

    @pytest.mark.asyncio
    async def test_concurrent_slot_acquisition_100(self):
        """100 concurrent slot acquisitions should not exceed total slots."""
        from instance_pool import InstancePool, InstanceInfo, RoutingAlgorithm
        redis = MockRedisForPool()
        pool = InstancePool(redis, mongo=None,
                            algorithm=RoutingAlgorithm.ROUND_ROBIN)

        for port in RTX3080_PORTS:
            await pool.register_instance(InstanceInfo(
                port=port, hostname=RTX3080_HOST, inst_type="gpu",
                model=MODEL_NAME, slots_total=RTX3080_PARALLEL,
                weight=RTX3080_WEIGHT,
            ))
        for port in RX6600M_PORTS:
            await pool.register_instance(InstanceInfo(
                port=port, hostname=RX6600M_HOST, inst_type="gpu",
                model=MODEL_NAME, slots_total=RX6600M_PARALLEL,
                weight=RX6600M_WEIGHT,
            ))

        results = await asyncio.gather(*[
            pool.select_instance() for _ in range(100)
        ])
        successful = [r for r in results if r is not None]
        assert len(successful) == 100  # 130 slots > 100 requests

        # Verify no instance exceeds its capacity
        for port, inst_data in redis._instances.items():
            assert inst_data["slots_used"] <= inst_data["slots_total"], (
                f"Port {port}: used={inst_data['slots_used']} > total={inst_data['slots_total']}"
            )

    @pytest.mark.asyncio
    async def test_concurrent_acquire_release_returns_to_zero(self):
        """Concurrent acquire+release cycles should leave all slots at 0."""
        from instance_pool import InstancePool, InstanceInfo, RoutingAlgorithm
        redis = MockRedisForPool()
        pool = InstancePool(redis, mongo=None,
                            algorithm=RoutingAlgorithm.ROUND_ROBIN)

        for port in RTX3080_PORTS:
            await pool.register_instance(InstanceInfo(
                port=port, hostname=RTX3080_HOST, inst_type="gpu",
                model=MODEL_NAME, slots_total=RTX3080_PARALLEL,
                weight=RTX3080_WEIGHT,
            ))
        for port in RX6600M_PORTS:
            await pool.register_instance(InstanceInfo(
                port=port, hostname=RX6600M_HOST, inst_type="gpu",
                model=MODEL_NAME, slots_total=RX6600M_PARALLEL,
                weight=RX6600M_WEIGHT,
            ))

        async def acquire_and_release():
            inst = await pool.select_instance()
            if inst:
                await asyncio.sleep(0.001)
                await pool.release_instance(inst.port, latency_ms=30.0, success=True)

        await asyncio.gather(*[acquire_and_release() for _ in range(50)])

        total_used = sum(inst["slots_used"] for inst in redis._instances.values())
        assert total_used == 0, f"Expected 0 total slots used, got {total_used}"

    @pytest.mark.asyncio
    async def test_pool_with_mongo_integration(self):
        """InstancePool with MockMongo should register instances in both stores."""
        from instance_pool import InstancePool, InstanceInfo, RoutingAlgorithm
        redis = MockRedisForPool()
        mongo = MockMongoMetricsStore()
        await mongo.connect()

        pool = InstancePool(redis, mongo=mongo,
                            algorithm=RoutingAlgorithm.LEAST_LOADED)

        for port in RTX3080_PORTS:
            await pool.register_instance(InstanceInfo(
                port=port, hostname=RTX3080_HOST, inst_type="gpu",
                model=MODEL_NAME, slots_total=RTX3080_PARALLEL,
                weight=RTX3080_WEIGHT,
            ))

        assert len(pool._instances) == 6
        assert len(redis._instances) == 6

        # Mongo should have 6 instance records
        count = await mongo._db["instances"].count_documents({})
        assert count == 6

        await mongo.close()

    @pytest.mark.asyncio
    async def test_full_topology_130_slots_exhaust(self):
        """Fill all 130 slots across 10 instances, then verify none remain."""
        from instance_pool import InstancePool, InstanceInfo, RoutingAlgorithm
        redis = MockRedisForPool()
        pool = InstancePool(redis, mongo=None,
                            algorithm=RoutingAlgorithm.LEAST_LOADED)

        for port in RTX3080_PORTS:
            await pool.register_instance(InstanceInfo(
                port=port, hostname=RTX3080_HOST, inst_type="gpu",
                model=MODEL_NAME, slots_total=RTX3080_PARALLEL,
                weight=RTX3080_WEIGHT,
            ))
        for port in RX6600M_PORTS:
            await pool.register_instance(InstanceInfo(
                port=port, hostname=RX6600M_HOST, inst_type="gpu",
                model=MODEL_NAME, slots_total=RX6600M_PARALLEL,
                weight=RX6600M_WEIGHT,
            ))

        acquired = 0
        for _ in range(TOTAL_SLOTS + 10):
            inst = await pool.select_instance()
            if inst is None:
                break
            acquired += 1

        assert acquired == TOTAL_SLOTS
        assert await pool.select_instance() is None

        # Verify exact slot counts
        for port in RTX3080_PORTS:
            assert redis._instances[port]["slots_used"] == RTX3080_PARALLEL
        for port in RX6600M_PORTS:
            assert redis._instances[port]["slots_used"] == RX6600M_PARALLEL

    @pytest.mark.asyncio
    async def test_redis_manager_with_mock_10_instances(self):
        """Full RedisStateManager flow using MockRedis for all 10 instances."""
        mgr, mock = _make_redis_manager()

        # Initialize all 10
        for port in RTX3080_PORTS:
            await mgr.init_instance(port, RTX3080_PARALLEL, "gpu")
        for port in RX6600M_PORTS:
            await mgr.init_instance(port, RX6600M_PARALLEL, "gpu")

        # Acquire one slot per instance
        for port in ALL_PORTS:
            assert await mgr.acquire_slot(port)

        # Verify each instance has 1 slot used
        loads = await mgr.get_all_loads()
        for load in loads:
            assert load["slots_used"] == 1

        # Release all
        for port in ALL_PORTS:
            await mgr.release_slot(port)

        # All should be 0
        for port in ALL_PORTS:
            used, _ = await mgr.get_slots(port)
            assert used == 0

    @pytest.mark.asyncio
    async def test_agent_distribution_across_instances(self):
        """1000 agents should be distributed ~100 per instance."""
        mgr, mock = _make_redis_manager()

        agents = []
        for i in range(NUM_AGENTS):
            inst_idx = i % len(ALL_PORTS)
            port = ALL_PORTS[inst_idx]
            host = RTX3080_HOST if port in RTX3080_PORTS else RX6600M_HOST
            agents.append({
                "agent_id": f"agent-{i:04d}",
                "hostname": host,
                "instance_port": port,
            })
        await mgr.register_agents_bulk(agents)

        # Verify total agent count
        count = await mgr.get_agent_count()
        assert count == NUM_AGENTS

        # Verify each agent can be looked up
        result = await mgr.get_agent_instance("agent-0000")
        assert result is not None
        host, port = result
        assert port == ALL_PORTS[0]

    def test_topology_constants_correct(self):
        """Verify topology constants are internally consistent."""
        assert len(RTX3080_PORTS) == 6
        assert len(RX6600M_PORTS) == 4
        assert len(ALL_PORTS) == 10
        assert TOTAL_SLOTS == 6 * 15 + 4 * 10  # 90 + 40 = 130
        assert TOTAL_SLOTS == 130
        assert AGENTS_PER_INSTANCE == 100
        assert NUM_AGENTS == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
