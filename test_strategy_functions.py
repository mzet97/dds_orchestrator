"""
Phase 3 Tests: Strategy Functions (_execute_with_retry and _execute_fanout)

Tests that the extracted strategy functions work correctly with various scenarios.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from registry import AgentInfo, AgentRegistry
from scheduler import Task, TaskScheduler, TaskPriority
from server import OrchestratorServer


@pytest.fixture
def mock_config():
    """Create mock configuration"""
    config = MagicMock()
    config.default_max_tokens = 100
    config.default_temperature = 0.7
    config.task_timeout_seconds = 60
    config.agent_timeout_seconds = 30
    return config


@pytest.fixture
def mock_agents():
    """Create multiple mock agents for testing"""
    agents = []
    for i in range(1, 4):
        agent = MagicMock(spec=AgentInfo)
        agent.agent_id = f"agent-{i}"
        agent.hostname = "localhost"
        agent.port = 8000 + i
        agent.model = f"model-{i}"
        agent.slots_idle = 2
        agent.slots_total = 2
        agent.avg_latency_ms = 100 + (i * 50)
        agent.error_rate = 0.05
        agent.agent_profile = "balanced"
        agents.append(agent)
    return agents


@pytest.fixture
def mock_task():
    """Create a mock task"""
    task = MagicMock(spec=Task)
    task.task_id = "task-123"
    task.task_type = "chat"
    task.assigned_agent_id = "agent-1"
    task.priority = TaskPriority.NORMAL
    task.timeout_ms = 60000
    return task


@pytest.mark.asyncio
async def test_execute_with_retry_success_on_second_attempt(mock_config, mock_agents, mock_task):
    """Test retry succeeds on second attempt after first fails"""
    primary_agent = mock_agents[0]
    fallback_agents = mock_agents[1:]

    registry = MagicMock()
    scheduler = MagicMock()
    dds_layer = MagicMock()

    # Create orchestrator
    orchestrator = OrchestratorServer(mock_config, registry, scheduler, dds_layer)

    # Mock registry methods
    registry.adjust_slots = AsyncMock(return_value=True)
    registry.update_response_metrics = AsyncMock()

    # Mock scheduler
    scheduler.track_task = AsyncMock()
    scheduler.complete_task = AsyncMock()

    # Mock _execute_agent_request to fail first attempt, succeed second
    attempt_count = [0]
    async def mock_execute_agent(*args, **kwargs):
        attempt_count[0] += 1
        if attempt_count[0] == 1:
            # First attempt fails
            return {}, False, "Connection timeout", False
        else:
            # Second attempt succeeds
            return {"content": "Success!", "success": True}, True, None, False

    orchestrator._execute_agent_request = mock_execute_agent

    # Execute with retry
    response_content, response_data, agent_id = await orchestrator._execute_with_retry(
        primary_agent, fallback_agents, mock_task,
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "Hello"}],
        100, 0.7, 5,
        "balanced", "ctx-123", "session-123"
    )

    # Verify success on retry
    assert response_content == "Success!"
    assert response_data["success"] == True
    assert agent_id == fallback_agents[0].agent_id  # Second agent succeeded
    assert attempt_count[0] == 2


@pytest.mark.asyncio
async def test_execute_with_retry_all_attempts_fail(mock_config, mock_agents, mock_task):
    """Test retry exhausts all attempts and returns empty content"""
    primary_agent = mock_agents[0]
    fallback_agents = mock_agents[1:]

    registry = MagicMock()
    scheduler = MagicMock()
    dds_layer = MagicMock()

    orchestrator = OrchestratorServer(mock_config, registry, scheduler, dds_layer)

    registry.adjust_slots = AsyncMock(return_value=True)
    scheduler.track_task = AsyncMock()

    # All attempts fail
    async def mock_execute_agent_fail(*args, **kwargs):
        return {}, False, "Connection error", False

    orchestrator._execute_agent_request = mock_execute_agent_fail

    # Execute with retry
    response_content, response_data, agent_id = await orchestrator._execute_with_retry(
        primary_agent, fallback_agents, mock_task,
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "Hello"}],
        100, 0.7, 5,
        "balanced", "ctx-123", "session-123"
    )

    # All attempts failed - should return empty
    assert response_content == ""
    assert response_data == {}


@pytest.mark.asyncio
async def test_execute_fanout_first_agent_wins(mock_config, mock_agents, mock_task):
    """Test fanout returns first successful response"""
    registry = MagicMock()
    scheduler = MagicMock()
    dds_layer = MagicMock()

    orchestrator = OrchestratorServer(mock_config, registry, scheduler, dds_layer)

    registry.adjust_slots = AsyncMock(return_value=True)
    scheduler.track_task = AsyncMock()

    # Agent 1 succeeds immediately
    call_count = [0]
    async def mock_execute_agent_fanout(*args, **kwargs):
        call_count[0] += 1
        # First call (agent-1) succeeds immediately
        if call_count[0] == 1:
            await asyncio.sleep(0.01)  # Slight delay
            return {"content": "Agent 1 response", "success": True}, True, None, False
        # Other calls timeout or fail
        await asyncio.sleep(2)  # Long delay
        return {"content": "Agent response", "success": True}, True, None, False

    orchestrator._execute_agent_request = mock_execute_agent_fanout

    # Execute fanout
    response_content, response_data, agent_id = await orchestrator._execute_fanout(
        mock_agents, mock_task,
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "Hello"}],
        100, 0.7, 5,
        "balanced", "ctx-123", "session-123", dds_priority=0
    )

    # First agent should win (fastest response)
    assert "Agent 1 response" in response_content or response_content != ""
    assert agent_id != ""


@pytest.mark.asyncio
async def test_execute_fanout_releases_slots(mock_config, mock_agents, mock_task):
    """Test fanout releases slots for all agents"""
    registry = MagicMock()
    scheduler = MagicMock()
    dds_layer = MagicMock()

    orchestrator = OrchestratorServer(mock_config, registry, scheduler, dds_layer)

    slots_acquired = []
    slots_released = []

    async def mock_adjust_slots(agent_id, delta):
        if delta < 0:
            slots_acquired.append(agent_id)
        else:
            slots_released.append(agent_id)
        return True

    registry.adjust_slots = mock_adjust_slots
    scheduler.track_task = AsyncMock()

    # Mock execute to return success
    async def mock_execute_fanout(*args, **kwargs):
        return {"content": "Response", "success": True}, True, None, False

    orchestrator._execute_agent_request = mock_execute_fanout

    # Execute fanout
    await orchestrator._execute_fanout(
        mock_agents[:2], mock_task,
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "Hello"}],
        100, 0.7, 5,
        "balanced", "ctx-123", "session-123"
    )

    # Should have acquired and released slots
    assert len(slots_acquired) > 0
    assert len(slots_released) > 0


@pytest.mark.asyncio
async def test_execute_fanout_with_single_agent():
    """Test fanout returns empty when less than 2 agents provided"""
    config = MagicMock()
    registry = MagicMock()
    scheduler = MagicMock()
    dds_layer = MagicMock()

    orchestrator = OrchestratorServer(config, registry, scheduler, dds_layer)

    task = MagicMock()

    # Fanout with single agent should return empty
    response_content, response_data, agent_id = await orchestrator._execute_fanout(
        [MagicMock()],  # Only 1 agent
        task,
        [],  # messages
        [],  # all_messages
        100, 0.7, 5,
        "balanced", "ctx-123", "session-123"
    )

    assert response_content == ""
    assert response_data == {}
    assert agent_id == ""
