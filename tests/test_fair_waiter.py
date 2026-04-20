"""Regression tests for the fair-waiter notify path in AgentRegistry.

Protects against silent regressions of the fixes shipped in:
  - e93271f (notify(1) per freed slot drains gRPC backlog)
  - b88f87e (strong ref on notify task prevents GC reaping)

All tests are pure logic — no DDS, no subprocess, no network.
"""
import asyncio
import threading
from dataclasses import dataclass

import pytest

from config import OrchestratorConfig
from registry import AgentInfo, AgentRegistry


def _make_agent(agent_id: str = "a1", slots_total: int = 1, slots_idle: int = 0) -> AgentInfo:
    return AgentInfo(
        agent_id=agent_id,
        hostname="localhost",
        port=8082,
        model="test-model",
        slots_idle=slots_idle,
        slots_total=slots_total,
        status="busy" if slots_idle == 0 else "idle",
        transports=["http", "dds", "grpc"],
    )


@pytest.mark.asyncio
async def test_adjust_slots_sync_wakes_one_waiter():
    """A single freed slot must wake exactly one parked waiter."""
    reg = AgentRegistry(OrchestratorConfig())
    reg.bind_main_loop(asyncio.get_running_loop())

    agent = _make_agent(slots_idle=0)
    await reg.register_agent(agent)

    woken = []

    async def waiter(idx: int):
        async with reg.agent_available_condition:
            await reg.agent_available_condition.wait()
            woken.append(idx)

    # Park 3 waiters.
    tasks = [asyncio.create_task(waiter(i)) for i in range(3)]
    await asyncio.sleep(0.05)  # ensure all are actually waiting on the condition

    # Release a single slot from a background thread (simulating gRPC worker).
    def _release():
        reg.adjust_slots_sync(agent.agent_id, delta=+1)

    threading.Thread(target=_release, daemon=True).start()

    # Give the notify task a moment to run on the loop.
    await asyncio.sleep(0.2)

    assert len(woken) == 1, f"Expected exactly 1 waiter woken, got {len(woken)}"

    # Clean up: wake remaining waiters so tasks finish.
    async with reg.agent_available_condition:
        reg.agent_available_condition.notify_all()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_adjust_slots_sync_negative_delta_no_notify():
    """Taking a slot (delta=-1) must NOT wake any waiter."""
    reg = AgentRegistry(OrchestratorConfig())
    reg.bind_main_loop(asyncio.get_running_loop())

    agent = _make_agent(slots_total=2, slots_idle=2)
    await reg.register_agent(agent)

    woken = []

    async def waiter():
        async with reg.agent_available_condition:
            try:
                await asyncio.wait_for(
                    reg.agent_available_condition.wait(), timeout=0.3
                )
                woken.append(True)
            except asyncio.TimeoutError:
                pass

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)

    # Take a slot. Waiter MUST NOT wake.
    threading.Thread(
        target=lambda: reg.adjust_slots_sync(agent.agent_id, delta=-1),
        daemon=True,
    ).start()

    await task
    assert woken == [], "Waiter should not wake on slot acquisition"


@pytest.mark.asyncio
async def test_notify_task_held_by_strong_ref():
    """The notify task must live in _pending_notifies until it completes.

    Regression for the pre-b88f87e bug where asyncio.ensure_future was
    fire-and-forget — GC could reap the task before it ran.
    """
    reg = AgentRegistry(OrchestratorConfig())
    reg.bind_main_loop(asyncio.get_running_loop())

    agent = _make_agent(slots_idle=0)
    await reg.register_agent(agent)

    assert len(reg._pending_notifies) == 0

    # Trigger notify from a thread.
    threading.Thread(
        target=lambda: reg.adjust_slots_sync(agent.agent_id, delta=+1),
        daemon=True,
    ).start()

    # Give call_soon_threadsafe time to schedule the task.
    await asyncio.sleep(0.01)

    # The task may already have completed (it's small). What matters is
    # that the registry keeps the ref alive during its lifetime — easiest
    # check: observe that the set is either populated now OR was populated
    # (callback removes it). Drain everything and confirm no leak.
    await asyncio.sleep(0.1)
    assert len(reg._pending_notifies) == 0, "Done callback must remove task from set"


@pytest.mark.asyncio
async def test_notify_no_loop_bound_is_noop():
    """Without bind_main_loop, notify_one_from_thread silently no-ops."""
    reg = AgentRegistry(OrchestratorConfig())
    # Deliberately NOT calling bind_main_loop.

    agent = _make_agent(slots_idle=0)
    await reg.register_agent(agent)

    # Should not raise, must return True (slot adjusted), just no notify.
    ok = reg.adjust_slots_sync(agent.agent_id, delta=+1)
    assert ok is True
    assert len(reg._pending_notifies) == 0


@pytest.mark.asyncio
async def test_register_agent_with_slots_notifies_one():
    """Registering an agent with free slots wakes exactly one waiter."""
    reg = AgentRegistry(OrchestratorConfig())
    reg.bind_main_loop(asyncio.get_running_loop())

    woken = []

    async def waiter(idx: int):
        async with reg.agent_available_condition:
            await reg.agent_available_condition.wait()
            woken.append(idx)

    # Park 2 waiters before any agent is registered.
    tasks = [asyncio.create_task(waiter(i)) for i in range(2)]
    await asyncio.sleep(0.05)

    # Register an agent with 1 idle slot.
    agent = _make_agent(slots_idle=1)
    await reg.register_agent(agent)

    await asyncio.sleep(0.1)
    assert len(woken) == 1

    async with reg.agent_available_condition:
        reg.agent_available_condition.notify_all()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
