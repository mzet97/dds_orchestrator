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

# Set DDS config for real DDS
os.environ.setdefault('CYCLONEDDS_URI', 'file:///mnt/e/TI/git/tese/llama.cpp_dds/dds/cyclonedds-local.xml')

from config import OrchestratorConfig
from models import AgentRegistration, AgentTaskRequest, TaskType, ChatMessage
from registry import AgentRegistry, AgentInfo
from scheduler import TaskScheduler, Task, TaskPriority
from context import ContextManager
from http_client import HTTPClient


async def test_registry():
    """Test agent registry"""
    print("\n=== Testing Agent Registry ===")

    config = OrchestratorConfig()
    registry = AgentRegistry(config)

    # Create AgentInfo (the type that registry.register_agent expects)
    agent_info = AgentInfo(
        agent_id="test-agent-1",
        hostname="localhost",
        port=8081,
        model="tinyllama-1.1b",
        specialization="generic",
        vram_available_mb=6000,
        slots_idle=1,
        slots_total=1,
    )

    result = await registry.register_agent(agent_info)
    assert result is True, "register_agent should return True"
    print(f"  Registered agent: {agent_info.agent_id}")

    # Get agent by ID
    agent = await registry.get_agent("test-agent-1")
    assert agent is not None, "get_agent should return the registered agent"
    assert agent.agent_id == "test-agent-1", f"Expected agent_id 'test-agent-1', got '{agent.agent_id}'"
    assert agent.model == "tinyllama-1.1b", f"Expected model 'tinyllama-1.1b', got '{agent.model}'"
    print(f"  Got agent: {agent.agent_id}, model: {agent.model}")

    # Select agent (registry has select_agent, not find_agent)
    found = await registry.select_agent(requirements={"model": "tinyllama-1.1b"})
    assert found is not None, "select_agent should find an agent matching the model"
    assert found.agent_id == "test-agent-1", f"Expected agent 'test-agent-1', got '{found.agent_id}'"
    print(f"  Selected agent: {found.agent_id}")

    # Get all agents
    all_agents = await registry.get_all_agents()
    assert len(all_agents) == 1, f"Expected 1 agent, got {len(all_agents)}"
    print(f"  Total agents: {len(all_agents)}")

    print("[PASS] Registry test passed!")


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
    assert task_id == "task-1", f"Expected task_id 'task-1', got '{task_id}'"
    print(f"  Submitted task: {task_id}")

    # Get stats
    stats = await scheduler.get_stats()
    assert stats["total_tasks"] == 1, f"Expected 1 total task, got {stats['total_tasks']}"
    assert stats["queue_size"] == 1, f"Expected queue_size 1, got {stats['queue_size']}"
    print(f"  Queue stats: {stats}")

    # Get next task
    next_task = await scheduler.get_next_task()
    assert next_task is not None, "get_next_task should return the submitted task"
    assert next_task.task_id == "task-1", f"Expected task_id 'task-1', got '{next_task.task_id}'"
    print(f"  Got next task: {next_task.task_id}")

    print("[PASS] Scheduler test passed!")


async def test_context_manager():
    """Test context manager"""
    print("\n=== Testing Context Manager ===")

    context_manager = ContextManager()

    # Create context
    context_id = await context_manager.create_context(
        user_id="user-1",
        initial_message=ChatMessage(role="user", content="Hello!"),
    )
    assert context_id is not None, "create_context should return a context_id"
    assert len(context_id) > 0, "context_id should not be empty"
    print(f"  Created context: {context_id}")

    # Get messages
    messages = await context_manager.get_messages(context_id)
    assert len(messages) >= 1, f"Expected at least 1 message, got {len(messages)}"
    print(f"  Messages in context: {len(messages)}")

    # Add message
    await context_manager.add_message(
        context_id,
        ChatMessage(role="assistant", content="Hi there!")
    )

    messages = await context_manager.get_messages(context_id)
    assert len(messages) >= 2, f"Expected at least 2 messages after add, got {len(messages)}"
    print(f"  Messages after add: {len(messages)}")

    print("[PASS] Context manager test passed!")


async def test_http_client():
    """Test HTTP client"""
    print("\n=== Testing HTTP Client ===")

    client = HTTPClient(timeout_seconds=10)

    # Just test instantiation
    async with client:
        assert client is not None, "HTTP Client should be instantiated"
        print("  HTTP Client created successfully")

    print("[PASS] HTTP Client test passed!")


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
        except AssertionError as e:
            print(f"[FAIL] {name} test failed (assertion): {e}")
            failed += 1
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
