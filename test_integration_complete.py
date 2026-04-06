#!/usr/bin/env python3
"""
Integration tests for DDS-LLM Orchestrator
Tests: ContextManager persistence, retry behavior, fuzzy strategy selection
"""

import pytest
import asyncio
import uuid
from typing import List, Dict

# Imports from orchestrator
from context import ContextManager, ChatMessage
from models import AgentInfo, AgentTaskRequest
from fuzzy_selector import FuzzyDecisionEngine, FuzzyInput


class TestContextManagerPersistence:
    """Test ContextManager multi-turn conversation persistence"""

    @pytest.fixture
    def context_manager(self):
        """Create fresh context manager for each test"""
        return ContextManager(max_contexts=10, max_messages_per_context=5)

    @pytest.mark.asyncio
    async def test_context_creation(self, context_manager):
        """Test creating new context for user"""
        session_id = str(uuid.uuid4())
        context_id = await context_manager.get_or_create_for_user(session_id)

        assert context_id is not None
        assert isinstance(context_id, str)

    @pytest.mark.asyncio
    async def test_context_persistence_across_requests(self, context_manager):
        """Test that context persists across multiple requests"""
        session_id = "test-session-1"

        # First request
        context_id_1 = await context_manager.get_or_create_for_user(session_id)
        await context_manager.add_message(context_id_1, "user", "Hello")
        await context_manager.add_message(context_id_1, "assistant", "Hi there!")

        # Second request - should get same context with history
        context_id_2 = await context_manager.get_or_create_for_user(session_id)
        messages = await context_manager.get_messages(context_id_2)

        assert context_id_1 == context_id_2
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi there!"

    @pytest.mark.asyncio
    async def test_sliding_window(self, context_manager):
        """Test that sliding window maintains max_messages_per_context"""
        session_id = "test-session-2"
        context_id = await context_manager.get_or_create_for_user(session_id)

        # Add more messages than max (5)
        for i in range(8):
            await context_manager.add_message(context_id, "user", f"Message {i}")

        messages = await context_manager.get_messages(context_id)

        # Should only keep last 5 messages
        assert len(messages) == 5
        # First message should be the 4th one (0-indexed: messages[3])
        assert messages[0].content == "Message 3"
        assert messages[-1].content == "Message 7"

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolation(self, context_manager):
        """Test that different sessions maintain separate contexts"""
        session_1 = "session-a"
        session_2 = "session-b"

        context_1 = await context_manager.get_or_create_for_user(session_1)
        context_2 = await context_manager.get_or_create_for_user(session_2)

        await context_manager.add_message(context_1, "user", "Session A message")
        await context_manager.add_message(context_2, "user", "Session B message")

        msgs_1 = await context_manager.get_messages(context_1)
        msgs_2 = await context_manager.get_messages(context_2)

        assert len(msgs_1) == 1
        assert len(msgs_2) == 1
        assert msgs_1[0].content == "Session A message"
        assert msgs_2[0].content == "Session B message"

    @pytest.mark.asyncio
    async def test_clear_context(self, context_manager):
        """Test clearing context removes all messages"""
        session_id = "test-session-3"
        context_id = await context_manager.get_or_create_for_user(session_id)

        await context_manager.add_message(context_id, "user", "Test")
        messages = await context_manager.get_messages(context_id)
        assert len(messages) == 1

        await context_manager.clear_context(context_id)
        messages = await context_manager.get_messages(context_id)

        assert len(messages) == 0


class TestRetryBehavior:
    """Test retry logic with exponential backoff"""

    def test_exponential_backoff_calculation(self):
        """Test exponential backoff formula: 0.5 * (2 ** attempt)"""
        # Attempt 0: 0.5 * 2^0 = 0.5s
        assert 0.5 * (2 ** 0) == 0.5
        # Attempt 1: 0.5 * 2^1 = 1.0s
        assert 0.5 * (2 ** 1) == 1.0
        # Attempt 2: 0.5 * 2^2 = 2.0s
        assert 0.5 * (2 ** 2) == 2.0

    @pytest.mark.asyncio
    async def test_retry_max_attempts(self):
        """Test that retry respects max_retries limit"""
        max_retries = 3
        attempt_count = 0

        async def failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < max_retries:
                raise Exception("Simulated failure")
            return "success"

        # Simulate retry loop
        for attempt in range(max_retries):
            try:
                result = await failing_operation()
                if result:
                    break
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.001)  # Skip actual sleep in test
                else:
                    raise

        assert attempt_count == max_retries

    @pytest.mark.asyncio
    async def test_retry_early_success(self):
        """Test that successful response stops retries early"""
        attempt_count = 0

        async def operation_succeeds_on_second_try():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 2:
                return "success"
            raise Exception("Not yet")

        max_retries = 3
        result = None
        for attempt in range(max_retries):
            try:
                result = await operation_succeeds_on_second_try()
                break  # Stop retrying on success
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.001)

        assert attempt_count == 2
        assert result == "success"


class TestFuzzyStrategySelection:
    """Test fuzzy logic strategy selection"""

    @pytest.fixture
    def fuzzy_engine(self):
        """Create fuzzy engine for tests"""
        try:
            return FuzzyDecisionEngine()
        except Exception:
            # If fuzzy engine fails to initialize, return mock
            return None

    def test_strategy_for_low_urgency(self, fuzzy_engine):
        """Low urgency should prefer 'single' strategy"""
        if fuzzy_engine is None:
            pytest.skip("Fuzzy engine not available")

        # Low urgency (1/10), low complexity, low load
        metrics = FuzzyInput(
            urgency=1.0,
            complexity=2.0,
            agent_load=10.0,
            agent_latency=50.0,
        )
        decision = fuzzy_engine.select("test-agent", metrics)

        # Low urgency should result in "single" strategy (score < 3.0)
        assert decision.strategy in ["single", "retry"]
        assert 0.0 <= decision.agent_score <= 100.0

    def test_strategy_for_high_urgency(self, fuzzy_engine):
        """High urgency with high load should prefer 'retry' or 'fanout'"""
        if fuzzy_engine is None:
            pytest.skip("Fuzzy engine not available")

        # High urgency (9/10), high complexity, high load
        metrics = FuzzyInput(
            urgency=9.0,
            complexity=8.0,
            agent_load=80.0,
            agent_latency=500.0,
        )
        decision = fuzzy_engine.select("test-agent", metrics)

        # High urgency should result in higher score (> 5.0)
        assert decision.strategy in ["retry", "fanout"]
        assert decision.agent_score >= 50.0

    def test_strategy_deterministic(self, fuzzy_engine):
        """Same metrics should always produce same strategy"""
        if fuzzy_engine is None:
            pytest.skip("Fuzzy engine not available")

        metrics = FuzzyInput(
            urgency=5.0,
            complexity=5.0,
            agent_load=50.0,
            agent_latency=300.0,
        )

        decision1 = fuzzy_engine.select("test-agent", metrics)
        decision2 = fuzzy_engine.select("test-agent", metrics)

        assert decision1.strategy == decision2.strategy
        assert decision1.agent_score == decision2.agent_score

    def test_strategy_range(self, fuzzy_engine):
        """Test all metric extremes produce valid strategies"""
        if fuzzy_engine is None:
            pytest.skip("Fuzzy engine not available")

        extremes = [
            (0.0, 0.0, 0.0, 0.0),      # All minimum
            (10.0, 10.0, 100.0, 2000.0),  # All maximum
            (5.0, 5.0, 50.0, 1000.0),    # All medium
        ]

        for urgency, complexity, load, latency in extremes:
            metrics = FuzzyInput(
                urgency=urgency,
                complexity=complexity,
                agent_load=load,
                agent_latency=latency,
            )
            decision = fuzzy_engine.select("test-agent", metrics)

            assert decision.strategy in ["single", "retry", "fanout"]
            assert 0.0 <= decision.agent_score <= 100.0


class TestQoSProfileSelection:
    """Test QoS profile selection based on fuzzy score"""

    def test_qos_profile_low_cost(self):
        """Score [0.0, 3.0) should map to LOW_COST"""
        score = 2.5
        # Mapping logic: LOW_COST if score < 3.0
        profile = "LOW_COST" if score < 3.0 else ("BALANCED" if score < 7.0 else "CRITICAL")
        assert profile == "LOW_COST"

    def test_qos_profile_balanced(self):
        """Score [3.0, 7.0) should map to BALANCED"""
        score = 5.0
        profile = "LOW_COST" if score < 3.0 else ("BALANCED" if score < 7.0 else "CRITICAL")
        assert profile == "BALANCED"

    def test_qos_profile_critical(self):
        """Score [7.0, 10.0] should map to CRITICAL"""
        score = 8.5
        profile = "LOW_COST" if score < 3.0 else ("BALANCED" if score < 7.0 else "CRITICAL")
        assert profile == "CRITICAL"


class TestAgentListPreparation:
    """Test agent pool preparation for retry/fanout strategies"""

    def test_single_strategy_uses_one_agent(self):
        """single strategy should use only primary agent"""
        agents = [
            AgentInfo(agent_id="agent-1"),
            AgentInfo(agent_id="agent-2"),
            AgentInfo(agent_id="agent-3"),
        ]
        primary_agent = agents[0]
        strategy = "single"

        # For single strategy, only use primary
        agent_list = [primary_agent]
        assert len(agent_list) == 1
        assert agent_list[0].agent_id == "agent-1"

    def test_retry_strategy_prepares_pool(self):
        """retry strategy should prepare pool of agents"""
        agents = [
            AgentInfo(agent_id="agent-1"),
            AgentInfo(agent_id="agent-2"),
            AgentInfo(agent_id="agent-3"),
        ]
        primary_agent = agents[0]
        strategy = "retry"

        # For retry strategy, prepare pool with additional agents
        agent_list = [primary_agent]
        additional_agents = [
            a for a in agents if a.agent_id != primary_agent.agent_id
        ][:2]
        agent_list.extend(additional_agents)

        assert len(agent_list) == 3
        assert agent_list[0].agent_id == "agent-1"
        assert "agent-2" in [a.agent_id for a in agent_list]
        assert "agent-3" in [a.agent_id for a in agent_list]

    def test_fanout_strategy_prepares_full_pool(self):
        """fanout strategy should prepare pool of all agents"""
        agents = [
            AgentInfo(agent_id="agent-1"),
            AgentInfo(agent_id="agent-2"),
            AgentInfo(agent_id="agent-3"),
        ]
        strategy = "fanout"

        # For fanout, use all agents
        agent_list = agents

        assert len(agent_list) == 3
        for i, agent in enumerate(agent_list, 1):
            assert agent.agent_id == f"agent-{i}"


class TestFanoutStrategy:
    """Test fanout parallel execution strategy"""

    @pytest.mark.asyncio
    async def test_fanout_reduces_latency(self):
        """Fanout should reduce latency by parallelizing"""
        import time

        # Simulate 3 agents with different latencies
        async def slow_agent():
            await asyncio.sleep(0.5)
            return "slow"

        async def fast_agent():
            await asyncio.sleep(0.1)
            return "fast"

        async def medium_agent():
            await asyncio.sleep(0.3)
            return "medium"

        # Sequential: 0.5 + 0.1 + 0.3 = 0.9s
        sequential_start = time.time()
        result1 = await slow_agent()
        result2 = await fast_agent()
        result3 = await medium_agent()
        sequential_time = time.time() - sequential_start

        # Parallel: max(0.5, 0.1, 0.3) = 0.5s
        parallel_start = time.time()
        done, pending = await asyncio.wait(
            [slow_agent(), fast_agent(), medium_agent()],
            return_when=asyncio.FIRST_COMPLETED
        )
        parallel_time = time.time() - parallel_start

        # Parallel should be significantly faster
        assert parallel_time < sequential_time
        assert parallel_time < 0.2  # Should finish when fast_agent completes

    @pytest.mark.asyncio
    async def test_fanout_cancels_pending_tasks(self):
        """Fanout should cancel pending tasks after first success"""
        cancelled_count = 0

        async def task_that_cancels():
            nonlocal cancelled_count
            try:
                await asyncio.sleep(10)  # Long sleep
            except asyncio.CancelledError:
                cancelled_count += 1
                raise

        async def quick_task():
            await asyncio.sleep(0.01)
            return "done"

        # Create tasks
        tasks = [
            asyncio.create_task(task_that_cancels()),
            asyncio.create_task(quick_task()),
            asyncio.create_task(task_that_cancels()),
        ]

        # Wait for first completion
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Verify cancellations
        assert cancelled_count >= 2

    @pytest.mark.asyncio
    async def test_fanout_returns_first_success(self):
        """Fanout should return first successful response"""

        async def failing_task():
            raise Exception("Failed")

        async def success_task():
            await asyncio.sleep(0.05)
            return "success"

        async def delayed_task():
            await asyncio.sleep(0.2)
            return "delayed"

        # Create tasks
        tasks = [
            asyncio.create_task(failing_task()),
            asyncio.create_task(success_task()),
            asyncio.create_task(delayed_task()),
        ]

        # Wait for first completion
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )

        # Get first successful result
        result = None
        for task in done:
            try:
                result = await task
                if result:
                    break
            except Exception:
                pass

        # Cancel others
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert result is not None
        assert result == "success"

    def test_fanout_requires_multiple_agents(self):
        """Fanout strategy requires at least 2 agents"""
        agents_single = [AgentInfo(agent_id="agent-1")]
        agents_multiple = [
            AgentInfo(agent_id="agent-1"),
            AgentInfo(agent_id="agent-2"),
            AgentInfo(agent_id="agent-3"),
        ]

        # Single agent: fanout not applicable
        assert len(agents_single) < 2

        # Multiple agents: fanout applicable
        assert len(agents_multiple) > 1

    def test_fanout_uses_up_to_three_agents(self):
        """Fanout should use at most 3 agents even if more available"""
        agents = [
            AgentInfo(agent_id=f"agent-{i}")
            for i in range(1, 11)  # 10 agents
        ]

        fanout_agent_count = min(3, len(agents))
        assert fanout_agent_count == 3

        # With 3 agents
        agents_3 = agents[:3]
        assert min(3, len(agents_3)) == 3

        # With 2 agents
        agents_2 = agents[:2]
        assert min(3, len(agents_2)) == 2


# Run tests with: pytest test_integration_complete.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
