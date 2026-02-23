#!/usr/bin/env python3
"""
Test script for orchestrator
Tests the core components without requiring a real LLM
"""
import asyncio
import sys
import os

# Add current dir to path
sys.path.insert(0, os.path.dirname(__file__))

from config import OrchestratorConfig
from models import AgentRegistration, AgentTaskRequest, TaskType, ChatMessage
from registry import AgentRegistry
from scheduler import TaskScheduler, Task, TaskPriority
from context import ContextManager
from dds_client import MockDDSClient
from http_client import HTTPClient


async def test_registry():
    """Test agent registry"""
    print("\n=== Testing Agent Registry ===")

    dds_client = MockDDSClient()
    registry = AgentRegistry(dds_client)

    # Register agent
    reg = AgentRegistration(
        agent_id="test-agent-1",
        hostname="localhost",
        port=8081,
        model="tinyllama-1.1b",
        model_path="/models/tinyllama.gguf",
        vram_available_mb=6000,
        vram_total_mb=8000,
        slots_idle=1,
        slots_total=1,
    )

    agent_id = await registry.register_agent(reg)
    print(f"Registered agent: {agent_id}")

    # Get agent
    agent = await registry.get_agent(agent_id)
    print(f"Got agent: {agent.agent_id}, model: {agent.model}")

    # Find agent
    found = await registry.find_agent(model="tinyllama-1.1b")
    print(f"Found agent: {found.agent_id if found else 'None'}")

    # Get all agents
    all_agents = await registry.get_all_agents()
    print(f"Total agents: {len(all_agents)}")

    print("[OK] Registry test passed!")
    return True


async def test_scheduler():
    """Test task scheduler"""
    print("\n=== Testing Task Scheduler ===")

    config = OrchestratorConfig()
    scheduler = TaskScheduler(config)

    # Submit task using scheduler's Task type
    task = Task(
        task_id="task-1",
        task_type="chat",
        messages=[{"role": "user", "content": "Hello!"}],
        priority=TaskPriority.NORMAL,
    )

    task_id = await scheduler.submit_task(task)
    print(f"Submitted task: {task_id}")

    # Get stats
    stats = await scheduler.get_stats()
    print(f"Queue stats: {stats}")

    print("[OK] Scheduler test passed!")
    return True


async def test_context_manager():
    """Test context manager"""
    print("\n=== Testing Context Manager ===")

    context_manager = ContextManager()

    # Create context
    context_id = await context_manager.create_context(
        user_id="user-1",
        initial_message=ChatMessage(role="user", content="Hello!"),
    )
    print(f"Created context: {context_id}")

    # Get messages
    messages = await context_manager.get_messages(context_id)
    print(f"Messages in context: {len(messages)}")

    # Add message
    await context_manager.add_message(
        context_id,
        ChatMessage(role="assistant", content="Hi there!")
    )

    messages = await context_manager.get_messages(context_id)
    print(f"Messages after add: {len(messages)}")

    print("[OK] Context manager test passed!")
    return True


async def test_http_client():
    """Test HTTP client"""
    print("\n=== Testing HTTP Client ===")

    client = HTTPClient(timeout_seconds=10)

    # Just test instantiation
    async with client:
        print("HTTP Client created successfully")

    print("[OK] HTTP Client test passed!")
    return True


async def main():
    """Run all tests"""
    print("=" * 50)
    print("DDS-LLM-Orchestrator Tests")
    print("=" * 50)

    tests = [
        ("Registry", test_registry),
        ("Scheduler", test_scheduler),
        ("Context Manager", test_context_manager),
        ("HTTP Client", test_http_client),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name} test failed: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
