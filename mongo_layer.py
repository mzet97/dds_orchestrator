"""
MongoDB Metrics Store for the Orchestrator
Provides async persistence for request metrics, routing decisions,
benchmark runs, and agent/instance registration.
"""

import asyncio
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class MongoMetricsStore:
    """Async MongoDB persistence for metrics and benchmark results."""

    def __init__(self, url: str, db_name: str = "dds_orchestrator"):
        self._url = url
        self._db_name = db_name
        self._client = None
        self._db = None

    async def connect(self):
        """Connect to MongoDB."""
        import motor.motor_asyncio as motor

        self._client = motor.AsyncIOMotorClient(self._url)
        self._db = self._client[self._db_name]
        # Verify connection
        await self._client.admin.command("ping")
        logger.info(f"Connected to MongoDB at {self._url}, db={self._db_name}")

    async def close(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None

    async def ensure_indexes(self):
        """Create indexes and TTL policies."""
        # metrics collection
        metrics = self._db["metrics"]
        await metrics.create_index([("scenario", 1), ("protocol", 1), ("timestamp", -1)])
        await metrics.create_index("run_id")
        await metrics.create_index(
            "timestamp", expireAfterSeconds=30 * 24 * 3600  # 30 days TTL
        )

        # routing_log collection
        routing = self._db["routing_log"]
        await routing.create_index([("algorithm", 1), ("timestamp", -1)])
        await routing.create_index(
            "timestamp", expireAfterSeconds=7 * 24 * 3600  # 7 days TTL
        )

        # benchmark_runs collection
        runs = self._db["benchmark_runs"]
        await runs.create_index("run_id", unique=True)
        await runs.create_index([("experiment", 1), ("protocol", 1)])

        # agents collection
        agents = self._db["agents"]
        await agents.create_index("agent_id", unique=True)
        await agents.create_index("instance_port")

        # instances collection
        instances = self._db["instances"]
        await instances.create_index("port", unique=True)

        logger.info("MongoDB indexes ensured")

    # === Metrics (fire-and-forget on hot path) ===

    async def log_request(self, metric: dict):
        """Log a single request metric.

        Schema: {request_id, instance_port, instance_type, protocol,
                 algorithm, latency_ms, success, error, timestamp, scenario, run_id}
        """
        metric.setdefault("timestamp", datetime.now(timezone.utc))
        try:
            await self._db["metrics"].insert_one(metric)
        except Exception as e:
            logger.debug(f"Failed to log metric: {e}")

    async def log_request_batch(self, metrics: list[dict]):
        """Log a batch of request metrics."""
        if not metrics:
            return
        for m in metrics:
            m.setdefault("timestamp", datetime.now(timezone.utc))
        try:
            await self._db["metrics"].insert_many(metrics, ordered=False)
        except Exception as e:
            logger.debug(f"Failed to log metric batch: {e}")

    # === Routing Decisions ===

    async def log_routing(self, decision: dict):
        """Log a routing decision for analysis.

        Schema: {request_id, algorithm, chosen_port, score,
                 all_scores, load_snapshot, decision_time_us, timestamp}
        """
        decision.setdefault("timestamp", datetime.now(timezone.utc))
        try:
            await self._db["routing_log"].insert_one(decision)
        except Exception as e:
            logger.debug(f"Failed to log routing decision: {e}")

    # === Benchmark Runs ===

    async def save_run(self, run: dict) -> str:
        """Save a benchmark run. Returns the run_id.

        Schema: {run_id, experiment, scenario, protocol, algorithm,
                 num_clients, config, results{p50,p95,p99,mean,max,
                 throughput_rps,error_rate,total_requests}, timestamp}
        """
        run_id = run.get("run_id") or str(uuid.uuid4())
        run["run_id"] = run_id
        run.setdefault("timestamp", datetime.now(timezone.utc))
        await self._db["benchmark_runs"].replace_one(
            {"run_id": run_id}, run, upsert=True
        )
        logger.info(f"Saved benchmark run: {run_id}")
        return run_id

    async def get_runs(self, experiment: str = None, scenario: str = None,
                       protocol: str = None, algorithm: str = None) -> list:
        """Query benchmark runs with optional filters."""
        query = {}
        if experiment:
            query["experiment"] = experiment
        if scenario:
            query["scenario"] = scenario
        if protocol:
            query["protocol"] = protocol
        if algorithm:
            query["algorithm"] = algorithm
        cursor = self._db["benchmark_runs"].find(query, {"_id": 0}).sort("timestamp", -1)
        return await cursor.to_list(length=1000)

    async def get_run(self, run_id: str) -> Optional[dict]:
        """Get a single benchmark run by run_id."""
        return await self._db["benchmark_runs"].find_one(
            {"run_id": run_id}, {"_id": 0}
        )

    # === Agent Registry (bulk) ===

    async def register_agents_bulk(self, agents: list[dict]):
        """Register multiple agents (upsert by agent_id)."""
        if not agents:
            return
        from pymongo import UpdateOne
        ops = [
            UpdateOne(
                {"agent_id": a["agent_id"]},
                {"$set": a},
                upsert=True,
            )
            for a in agents
        ]
        result = await self._db["agents"].bulk_write(ops)
        logger.info(f"Registered {result.upserted_count + result.modified_count} agents")

    async def register_instance(self, instance: dict):
        """Register or update an instance."""
        await self._db["instances"].replace_one(
            {"port": instance["port"]}, instance, upsert=True
        )

    async def get_agents(self, instance_port: int = None) -> list:
        """Get agents, optionally filtered by instance port."""
        query = {}
        if instance_port is not None:
            query["instance_port"] = instance_port
        cursor = self._db["agents"].find(query, {"_id": 0})
        return await cursor.to_list(length=2000)

    async def get_agent_count(self) -> int:
        """Get total number of registered agents."""
        return await self._db["agents"].count_documents({})

    # === Queries ===

    async def get_metrics(self, scenario: str = None, protocol: str = None,
                          algorithm: str = None, run_id: str = None,
                          limit: int = 10000) -> list:
        """Query request metrics with filters."""
        query = {}
        if scenario:
            query["scenario"] = scenario
        if protocol:
            query["protocol"] = protocol
        if algorithm:
            query["algorithm"] = algorithm
        if run_id:
            query["run_id"] = run_id
        cursor = self._db["metrics"].find(query, {"_id": 0}).sort("timestamp", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def get_latency_percentiles(self, run_id: str) -> dict:
        """Compute latency percentiles for a benchmark run using aggregation."""
        pipeline = [
            {"$match": {"run_id": run_id, "success": True}},
            {"$group": {
                "_id": None,
                "latencies": {"$push": "$latency_ms"},
                "count": {"$sum": 1},
                "mean": {"$avg": "$latency_ms"},
                "max": {"$max": "$latency_ms"},
                "min": {"$min": "$latency_ms"},
            }},
        ]
        results = await self._db["metrics"].aggregate(pipeline).to_list(length=1)
        if not results:
            return {}

        r = results[0]
        latencies = sorted(r["latencies"])
        n = len(latencies)

        def percentile(p):
            if n == 0:
                return 0
            idx = max(0, int(math.ceil(n * p / 100)) - 1)
            return latencies[min(idx, n - 1)]

        return {
            "count": r["count"],
            "mean": r["mean"],
            "min": r["min"],
            "max": r["max"],
            "p50": percentile(50),
            "p95": percentile(95),
            "p99": percentile(99),
        }

    async def get_metrics_summary(self, scenario: str = None) -> dict:
        """Get aggregated metrics summary."""
        match = {}
        if scenario:
            match["scenario"] = scenario

        pipeline = [
            {"$match": match} if match else {"$match": {}},
            {"$group": {
                "_id": {"protocol": "$protocol", "algorithm": "$algorithm"},
                "count": {"$sum": 1},
                "avg_latency": {"$avg": "$latency_ms"},
                "max_latency": {"$max": "$latency_ms"},
                "errors": {"$sum": {"$cond": [{"$eq": ["$success", False]}, 1, 0]}},
            }},
        ]
        results = await self._db["metrics"].aggregate(pipeline).to_list(length=100)
        return {
            "groups": [
                {
                    "protocol": r["_id"]["protocol"],
                    "algorithm": r["_id"]["algorithm"],
                    "count": r["count"],
                    "avg_latency_ms": r["avg_latency"],
                    "max_latency_ms": r["max_latency"],
                    "error_count": r["errors"],
                    "error_rate": r["errors"] / r["count"] if r["count"] > 0 else 0,
                }
                for r in results
            ]
        }

    # === Cleanup ===

    async def drop_all(self):
        """Drop all collections (for test teardown)."""
        for name in ("metrics", "routing_log", "benchmark_runs", "agents", "instances"):
            await self._db[name].drop()
