#!/usr/bin/env python3
"""
Register 1000 logical agents distributed across 10 GPU instances.

Topology:
  .61 (RTX 3080): 6 instances, ports 8082-8087, parallel=15 → 600 agents
  .60 (RX 6600M): 4 instances, ports 8088-8091, parallel=10 → 400 agents
  Total: 1000 agents, 100 per instance

Warmup strategy:
  Phase 1: 5 requests/instance sequentially (cold → warm VRAM)
  Phase 2: 50 requests/instance concurrent (warm KV cache + DDS topics)
  Phase 3: 100 parallel requests across all instances (validate routing)
"""

import argparse
import asyncio
import time

import aiohttp


# ===== 10-Instance Topology =====

INSTANCES = [
    # .61 RTX 3080: 6 instances
    {"host": "192.168.1.61", "port": 8082 + i, "type": "gpu",
     "slots": 15, "gpu": "rtx3080", "vram_mb": 1500}
    for i in range(6)
] + [
    # .60 RX 6600M: 4 instances
    {"host": "192.168.1.60", "port": 8088 + i, "type": "gpu",
     "slots": 10, "gpu": "rx6600m", "vram_mb": 1500}
    for i in range(4)
]

ORCHESTRATOR_DEFAULT = "http://192.168.1.62:8080"


def generate_agents(total: int = 1000) -> list[dict]:
    """Generate agent definitions distributed across 10 instances.

    Distributes proportionally: RTX instances get more agents (higher slots).
    """
    agents = []
    # Weighted distribution: proportional to slots
    total_slots = sum(inst["slots"] for inst in INSTANCES)

    assigned = 0
    for idx, inst in enumerate(INSTANCES):
        # Proportional allocation, last instance gets remainder
        if idx == len(INSTANCES) - 1:
            count = total - assigned
        else:
            count = round(total * inst["slots"] / total_slots)
            assigned += count

        for j in range(count):
            agents.append({
                "agent_id": f"agent-{idx:02d}-{j:03d}",
                "hostname": inst["host"],
                "port": inst["port"],
                "model": "Qwen3.5-2B",
                "slots_total": inst["slots"],
                "slots_idle": inst["slots"],
                "instance_type": inst["type"],
                "instance_port": inst["port"],
                "gpu_type": inst["gpu"],
                "vision_enabled": False,
                "reasoning_enabled": False,
                "vram_available_mb": inst["vram_mb"],
            })

    return agents


async def register_bulk(orchestrator_url: str, agents: list,
                        batch_size: int = 100) -> tuple[int, int]:
    """Register agents in batches via HTTP POST."""
    registered = 0
    errors = 0

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(agents), batch_size):
            batch = agents[i:i + batch_size]
            tasks = [_register_one(session, orchestrator_url, a) for a in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    errors += 1
                else:
                    registered += 1

            print(f"  Registered {registered}/{len(agents)} "
                  f"(errors: {errors})", flush=True)

    return registered, errors


async def _register_one(session: aiohttp.ClientSession, url: str, agent: dict):
    """Register a single agent."""
    async with session.post(
        f"{url}/api/v1/agents/register",
        json=agent,
        timeout=aiohttp.ClientTimeout(total=10),
    ) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Registration failed: {resp.status}")
        return await resp.json()


async def _send_chat(session: aiohttp.ClientSession, url: str,
                     messages: list, max_tokens: int = 10):
    """Send a single chat request."""
    async with session.post(
        f"{url}/api/v1/chat/completions",
        json={"messages": messages, "max_tokens": max_tokens,
              "model": "Qwen3.5-2B"},
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        return await resp.json()


async def warmup_phased(orchestrator_url: str):
    """3-phase warmup to progressively warm all instances.

    Phase 1: Sequential warmup (5 req/instance) — loads model into VRAM
    Phase 2: Concurrent warmup (50 req/instance) — warms KV cache, DDS topics
    Phase 3: Routing validation (100 parallel) — tests orchestrator routing
    """
    prompt = [{"role": "user", "content": "Say hello."}]

    # Phase 1: Sequential (cold start)
    print("\n[Warmup Phase 1] Sequential cold-start (5 req/instance)...")
    async with aiohttp.ClientSession() as session:
        for inst in INSTANCES:
            successes = 0
            for _ in range(5):
                try:
                    await _send_chat(session, orchestrator_url, prompt)
                    successes += 1
                except Exception as e:
                    print(f"  [{inst['host']}:{inst['port']}] warmup error: {e}")
            print(f"  [{inst['host']}:{inst['port']}] {successes}/5 OK")

    # Phase 2: Concurrent per-instance (warm caches)
    print("\n[Warmup Phase 2] Concurrent warmup (50 req/instance)...")
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100)
    ) as session:
        for inst in INSTANCES:
            tasks = [_send_chat(session, orchestrator_url, prompt)
                     for _ in range(50)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successes = sum(1 for r in results if not isinstance(r, Exception))
            print(f"  [{inst['host']}:{inst['port']}] {successes}/50 OK")

    # Phase 3: Routing validation (all instances at once)
    print("\n[Warmup Phase 3] Routing validation (100 parallel)...")
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=200)
    ) as session:
        tasks = [_send_chat(session, orchestrator_url, prompt)
                 for _ in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successes = sum(1 for r in results if not isinstance(r, Exception))
        errors = sum(1 for r in results if isinstance(r, Exception))
        print(f"  {successes}/100 OK, {errors} errors")

    print("\nWarmup complete!")


async def save_to_mongo(mongo_url: str, agents: list):
    """Save agents to MongoDB."""
    try:
        import sys
        sys.path.insert(0, "..")
        from mongo_layer import MongoMetricsStore

        store = MongoMetricsStore(mongo_url)
        await store.connect()
        await store.ensure_indexes()

        # Bulk insert
        col = store._db["agents"]
        await col.delete_many({})  # Clean slate
        if agents:
            await col.insert_many(agents)
        count = await col.count_documents({})
        print(f"MongoDB: {count} agents stored")
        await store.close()
    except Exception as e:
        print(f"MongoDB save failed: {e}")


async def save_to_redis(redis_url: str, password: str, agents: list):
    """Save agent-to-instance mapping in Redis."""
    try:
        import sys
        sys.path.insert(0, "..")
        from redis_layer import RedisStateManager

        mgr = RedisStateManager(redis_url, password)
        await mgr.connect()

        import redis.asyncio as aioredis
        pipe = mgr._redis.pipeline()
        for agent in agents:
            key = f"agent:{agent['agent_id']}:instance"
            value = f"{agent['hostname']}:{agent['instance_port']}"
            pipe.set(key, value)
        await pipe.execute()
        print(f"Redis: {len(agents)} agent mappings stored")
        await mgr.close()
    except Exception as e:
        print(f"Redis save failed: {e}")


async def main_async(args):
    agents = generate_agents(args.total)
    print(f"Generated {len(agents)} agents across {len(INSTANCES)} instances")

    # Print distribution
    from collections import Counter
    dist = Counter(a["instance_port"] for a in agents)
    for inst in INSTANCES:
        n = dist.get(inst["port"], 0)
        print(f"  {inst['host']}:{inst['port']} ({inst['gpu']}): {n} agents")

    # Register via HTTP
    print(f"\nRegistering with orchestrator at {args.url}...")
    registered, errors = await register_bulk(args.url, agents, args.batch_size)
    print(f"\nRegistered: {registered}, Errors: {errors}")

    # Save to MongoDB
    if args.mongo_url:
        await save_to_mongo(args.mongo_url, agents)

    # Save to Redis
    if args.redis_url:
        await save_to_redis(args.redis_url, args.redis_password, agents)

    # Warmup
    if args.warmup:
        await warmup_phased(args.url)

    # Verify
    if args.verify:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.url}/api/v1/agents") as resp:
                data = await resp.json()
                count = len(data.get("agents", []))
                print(f"\nVerification: {count} agents registered in orchestrator")


def main():
    parser = argparse.ArgumentParser(
        description="Register 1000 agents across 10 GPU instances"
    )
    parser.add_argument("--url", type=str, default=ORCHESTRATOR_DEFAULT,
                       help="Orchestrator URL")
    parser.add_argument("--total", type=int, default=1000,
                       help="Total agents to register")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Registration batch size")
    parser.add_argument("--warmup", action="store_true",
                       help="Run 3-phase warmup after registration")
    parser.add_argument("--mongo-url", type=str, default="",
                       help="MongoDB URL for persistence")
    parser.add_argument("--redis-url", type=str, default="",
                       help="Redis URL for agent mapping")
    parser.add_argument("--redis-password", type=str, default="Admin@123",
                       help="Redis password")
    parser.add_argument("--verify", action="store_true", default=True,
                       help="Verify registration count")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
