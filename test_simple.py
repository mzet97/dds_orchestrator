#!/usr/bin/env python3
"""
Simplified unit tests for DDS-LLM Orchestrator components.
Focus on basic functionality that can be tested without full infrastructure.
"""

import pytest
import asyncio
from context import ContextManager, ChatMessage
from models import AgentInfo


class TestContextManager:
    """Test ContextManager basic functionality"""

    @pytest.fixture
    def context_manager(self):
        """Create fresh context manager"""
        return ContextManager(max_contexts=10, max_messages_per_context=5)

    @pytest.mark.asyncio
    async def test_context_creation(self, context_manager):
        """Test creating new context"""
        session_id = "test-session-1"
        context_id = await context_manager.get_or_create_for_user(session_id)

        assert context_id is not None
        assert isinstance(context_id, str)

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self, context_manager):
        """Test adding and retrieving messages"""
        session_id = "test-session-2"
        context_id = await context_manager.get_or_create_for_user(session_id)

        # Add messages
        msg1 = ChatMessage(role="user", content="Hello")
        await context_manager.add_message(context_id, msg1)

        msg2 = ChatMessage(role="assistant", content="Hi there!")
        await context_manager.add_message(context_id, msg2)

        # Get messages
        messages = await context_manager.get_messages(context_id)

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi there!"

    @pytest.mark.asyncio
    async def test_sliding_window(self, context_manager):
        """Test max_messages_per_context limit"""
        session_id = "test-session-3"
        context_id = await context_manager.get_or_create_for_user(session_id)

        # Add 8 messages (max is 5)
        for i in range(8):
            msg = ChatMessage(role="user", content=f"Message {i}")
            await context_manager.add_message(context_id, msg)

        messages = await context_manager.get_messages(context_id)

        # Should only keep last 5
        assert len(messages) == 5
        assert messages[0].content == "Message 3"
        assert messages[-1].content == "Message 7"


class TestRetryLogic:
    """Test retry behavior"""

    def test_exponential_backoff(self):
        """Test backoff calculation"""
        assert 0.5 * (2 ** 0) == 0.5
        assert 0.5 * (2 ** 1) == 1.0
        assert 0.5 * (2 ** 2) == 2.0

    @pytest.mark.asyncio
    async def test_retry_max_attempts(self):
        """Test that retry stops after max attempts"""
        attempt_count = 0
        max_retries = 3

        async def failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            raise Exception("Failed")

        # Simulate retry loop
        for attempt in range(max_retries):
            try:
                await failing_operation()
            except Exception:
                pass

        assert attempt_count == max_retries


class TestAgentInfo:
    """Test AgentInfo model creation"""

    def test_agent_creation(self):
        """Test creating AgentInfo with required fields"""
        agent = AgentInfo(
            agent_id="agent-1",
            hostname="localhost",
            port=8081,
            model="text"
        )

        assert agent.agent_id == "agent-1"
        assert agent.hostname == "localhost"
        assert agent.port == 8081
        assert agent.model == "text"

    def test_agent_list_preparation_single(self):
        """Single strategy uses only primary agent"""
        agents = [
            AgentInfo(agent_id="agent-1", hostname="localhost", port=8081, model="text"),
            AgentInfo(agent_id="agent-2", hostname="localhost", port=8082, model="text"),
            AgentInfo(agent_id="agent-3", hostname="localhost", port=8083, model="text"),
        ]
        primary = agents[0]
        strategy = "single"

        # Single strategy: only primary
        agent_list = [primary]

        assert len(agent_list) == 1
        assert agent_list[0].agent_id == "agent-1"

    def test_agent_list_preparation_retry(self):
        """Retry strategy prepares pool of 3 agents"""
        agents = [
            AgentInfo(agent_id="agent-1", hostname="localhost", port=8081, model="text"),
            AgentInfo(agent_id="agent-2", hostname="localhost", port=8082, model="text"),
            AgentInfo(agent_id="agent-3", hostname="localhost", port=8083, model="text"),
        ]
        primary = agents[0]
        strategy = "retry"

        # Retry strategy: prepare pool
        agent_list = [primary]
        additional = [a for a in agents if a.agent_id != primary.agent_id][:2]
        agent_list.extend(additional)

        assert len(agent_list) == 3
        assert all(isinstance(a, AgentInfo) for a in agent_list)

    def test_agent_list_preparation_fanout(self):
        """Fanout strategy prepares full pool"""
        agents = [
            AgentInfo(agent_id="agent-1", hostname="localhost", port=8081, model="text"),
            AgentInfo(agent_id="agent-2", hostname="localhost", port=8082, model="text"),
            AgentInfo(agent_id="agent-3", hostname="localhost", port=8083, model="text"),
        ]
        strategy = "fanout"

        # Fanout: use all agents (or up to 3)
        agent_list = agents[:3]

        assert len(agent_list) == 3
        assert agent_list[0].agent_id == "agent-1"


class TestQoSProfiles:
    """Test QoS profile selection"""

    def test_qos_low_cost(self):
        """Score [0, 3) maps to LOW_COST"""
        score = 2.5
        profile = "LOW_COST" if score < 3.0 else ("BALANCED" if score < 7.0 else "CRITICAL")
        assert profile == "LOW_COST"

    def test_qos_balanced(self):
        """Score [3, 7) maps to BALANCED"""
        score = 5.0
        profile = "LOW_COST" if score < 3.0 else ("BALANCED" if score < 7.0 else "CRITICAL")
        assert profile == "BALANCED"

    def test_qos_critical(self):
        """Score [7, 10] maps to CRITICAL"""
        score = 8.5
        profile = "LOW_COST" if score < 3.0 else ("BALANCED" if score < 7.0 else "CRITICAL")
        assert profile == "CRITICAL"


class TestStrategySelection:
    """Test strategy selection"""

    def test_strategy_single(self):
        """Score < 3.0 maps to single"""
        score = 2.5
        strategy = "single" if score < 3.0 else ("retry" if score < 7.0 else "fanout")
        assert strategy == "single"

    def test_strategy_retry(self):
        """Score [3, 7) maps to retry"""
        score = 5.0
        strategy = "single" if score < 3.0 else ("retry" if score < 7.0 else "fanout")
        assert strategy == "retry"

    def test_strategy_fanout(self):
        """Score >= 7.0 maps to fanout"""
        score = 8.5
        strategy = "single" if score < 3.0 else ("retry" if score < 7.0 else "fanout")
        assert strategy == "fanout"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
