#!/usr/bin/env python3
"""
Register 1000 logical agents distributed across 38 instances.
~26 agents per instance, named agent-{instance_idx:02d}-{agent_idx:03d}.
"""

import argparse
import asyncio
import aiohttp
import time


GPU_PORTS = list(range(8082, 8092))   # 10 GPU instances
CPU_PORTS = list(range(8092, 8120))   # 28 CPU instances
ALL_PORTS = GPU_PORTS + CPU_PORTS     # 38 instances


def generate_agents(total=1000):
    """Generate agent definitions distributed across instances."""
    agents = []
    per_instance = total // len(ALL_PORTS)   # 26
    remainder = total % len(ALL_PORTS)        # 12

    for idx, port in enumerate(ALL_PORTS):
        count = per_instance + (1 if idx < remainder else 0)
        inst_type = "gpu" if port < 8092 else "cpu"
        slots = 15 if inst_type == "gpu" else 4

        for j in range(count):
            agents.append({
                "agent_id": f"agent-{idx:02d}-{j:03d}",
                "hostname": "192.168.1.61",
                "port": port,
                "model": "Qwen3.5-2B",
                "slots_total": slots,
                "slots_idle": slots,
                "instance_type": inst_type,
                "instance_port": port,
                "vision_enabled": False,
                "reasoning_enabled": False,
                "vram_available_mb": 1000 if inst_type == "gpu" else 0,
            })

    return agents


async def register_bulk(orchestrator_url: str, agents: list, batch_size: int = 100):
    """Register agents in batches via HTTP POST."""
    registered = 0
    errors = 0

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(agents), batch_size):
            batch = agents[i:i + batch_size]
            tasks = []
            for agent in batch:
                tasks.append(_register_one(session, orchestrator_url, agent))
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    errors += 1
                else:
                    registered += 1

            print(f"  Registered {registered}/{len(agents)} (errors: {errors})", flush=True)

    return registered, errors


async def _register_one(session, url, agent):
    """Register a single agent."""
    async with session.post(
        f"{url}/api/v1/agents/register",
        json=agent,
        timeout=aiohttp.ClientTimeout(total=10),
    ) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Registration failed: {resp.status}")
        return await resp.json()


async def warmup(orchestrator_url: str, prompts_per_instance: int = 5):
    """Send warmup requests to load models into memory/VRAM."""
    print(f"\nWarming up {len(ALL_PORTS)} instances ({prompts_per_instance} requests each)...")
    warmup_prompt = [{"role": "user", "content": "Say hello."}]

    async with aiohttp.ClientSession() as session:
        for port in ALL_PORTS:
            tasks = []
            for _ in range(prompts_per_instance):
                tasks.append(_send_chat(session, orchestrator_url, warmup_prompt))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successes = sum(1 for r in results if not isinstance(r, Exception))
            print(f"  Instance :{port} — {successes}/{prompts_per_instance} warmup OK")

    print("Warmup complete")


async def _send_chat(session, url, messages, max_tokens=10):
    """Send a single chat request."""
    async with session.post(
        f"{url}/api/v1/chat/completions",
        json={"messages": messages, "max_tokens": max_tokens},
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        return await resp.json()


async def save_to_mongo(mongo_url: str, agents: list):
    """Save agents to MongoDB."""
    try:
        import sys
        sys.path.insert(0, "..")
        from mongo_layer import MongoMetricsStore

        store = MongoMetricsStore(mongo_url)
        await store.connect()
        await store.ensure_indexes()
        await store.register_agents_bulk(agents)
        count = await store.get_agent_count()
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

        redis = RedisStateManager(redis_url, password)
        await redis.connect()

        import redis.asyncio as aioredis
        pipe = redis._redis.pipeline()
        for agent in agents:
            pipe.set(f"agent:{agent['agent_id']}:instance_port", agent["instance_port"])
        await pipe.execute()
        print(f"Redis: {len(agents)} agent mappings stored")
        await redis.close()
    except Exception as e:
        print(f"Redis save failed: {e}")


async def main_async(args):
    agents = generate_agents(args.total)
    print(f"Generated {len(agents)} agents across {len(ALL_PORTS)} instances")
    print(f"Distribution: {args.total // len(ALL_PORTS)} per instance "
          f"(+1 for first {args.total % len(ALL_PORTS)})")

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
        await warmup(args.url, args.warmup_per_instance)

    # Verify
    if args.verify:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.url}/api/v1/agents") as resp:
                data = await resp.json()
                count = len(data.get("agents", []))
                print(f"\nVerification: {count} agents registered in orchestrator")


def main():
    parser = argparse.ArgumentParser(description="Register 1000 logical agents")
    parser.add_argument("--url", type=str, default="http://192.168.1.61:8080",
                       help="Orchestrator URL")
    parser.add_argument("--total", type=int, default=1000,
                       help="Total agents to register")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Registration batch size")
    parser.add_argument("--warmup", action="store_true",
                       help="Run warmup after registration")
    parser.add_argument("--warmup-per-instance", type=int, default=5,
                       help="Warmup requests per instance")
    parser.add_argument("--mongo-url", type=str, default="",
                       help="MongoDB URL for persistence")
    parser.add_argument("--redis-url", type=str, default="",
                       help="Redis URL for agent mapping")
    parser.add_argument("--redis-password", type=str, default="Admin@123",
                       help="Redis password")
    parser.add_argument("--verify", action="store_true", default=True,
                       help="Verify registration")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
