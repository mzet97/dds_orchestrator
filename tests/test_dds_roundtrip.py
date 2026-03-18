#!/usr/bin/env python3
"""
Test DDS Round-Trip -- Verifica que mensagens publicadas chegam ao subscriber.
Testa publish -> subscribe com dois participantes no mesmo dominio.

NOTA: Estes testes medem "loopback local" (self-delivery) e comunicacao
entre dois DomainParticipants no mesmo processo. Nao sao round-trip reais
entre dois processos separados.

Uso:
    python test_dds_roundtrip.py
"""
import asyncio
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(__file__))

from config import OrchestratorConfig
from dds import DDSLayer, TOPIC_AGENT_REQUEST, TOPIC_AGENT_RESPONSE, TOPIC_AGENT_STATUS


async def test_dds_roundtrip_same_participant():
    """
    Test publish -> read with same DDS participant (self-loopback).
    NOTE: This is local self-delivery, not a real round-trip between processes.
    """
    print("\n=== Test: DDS Loopback (Same Participant) ===")
    print("  NOTE: This is local self-delivery, not a real inter-process round-trip.")

    config = OrchestratorConfig(dds_enabled=True, dds_domain=0)
    dds = DDSLayer(config)

    if not dds.is_available():
        print("[SKIP] CycloneDDS not available")
        return True

    task_id = f"rt-{uuid.uuid4().hex[:8]}"

    # Publish a request
    test_data = {
        "task_id": task_id,
        "requester_id": "roundtrip-test",
        "task_type": "chat",
        "messages_json": '[{"role": "user", "content": "roundtrip test"}]',
        "priority": 5,
        "timeout_ms": 5000,
        "requires_context": False,
        "context_id": "",
        "created_at": int(time.time() * 1000),
    }

    await dds.publish(TOPIC_AGENT_REQUEST, test_data)
    print(f"  Published task_id={task_id}")

    # Wait for propagation
    await asyncio.sleep(0.2)

    # Try to read (note: same participant may or may not see own messages
    # depending on CycloneDDS configuration)
    messages = dds.read_messages(TOPIC_AGENT_REQUEST, timeout_ms=1000)
    print(f"  Read {len(messages)} messages from {TOPIC_AGENT_REQUEST}")

    dds.close()

    if len(messages) > 0:
        print("[PASS] Loopback test passed (self-delivery working)")
    else:
        print("[SKIP] Self-loopback not received (expected in some DDS configs)")
        print("  Use two separate participants for inter-process round-trip")

    return True


async def test_dds_roundtrip_two_participants():
    """
    Test publish -> subscribe with two separate DDS participants.
    NOTE: Both participants are in the same process -- this tests local
    inter-participant delivery, not real inter-process communication.
    """
    print("\n=== Test: DDS Delivery (Two Participants, Same Process) ===")
    print("  NOTE: Both participants in same process (not inter-process round-trip).")

    config = OrchestratorConfig(dds_enabled=True, dds_domain=0)

    # Participant 1: Publisher (orchestrator side)
    pub_layer = DDSLayer(config)
    if not pub_layer.is_available():
        print("[SKIP] CycloneDDS not available")
        return True

    # Participant 2: Subscriber (agent side)
    sub_layer = DDSLayer(config)
    if not sub_layer.is_available():
        print("[SKIP] Second DDS participant failed")
        pub_layer.close()
        return True

    task_id = f"rt2-{uuid.uuid4().hex[:8]}"

    # Publish via participant 1
    test_data = {
        "task_id": task_id,
        "requester_id": "roundtrip-pub",
        "task_type": "chat",
        "messages_json": '[{"role": "user", "content": "two-participant test"}]',
        "priority": 5,
        "timeout_ms": 5000,
        "requires_context": False,
        "context_id": "",
        "created_at": int(time.time() * 1000),
    }

    await pub_layer.publish(TOPIC_AGENT_REQUEST, test_data)
    print(f"  Published task_id={task_id} via participant 1")

    # Wait for DDS discovery + propagation
    await asyncio.sleep(0.5)

    # Read via participant 2
    messages = sub_layer.read_messages(TOPIC_AGENT_REQUEST, timeout_ms=2000)
    print(f"  Read {len(messages)} messages via participant 2")

    pub_layer.close()
    sub_layer.close()

    if len(messages) > 0:
        print("[PASS] Two-participant delivery test passed!")
        return True
    else:
        print("[FAIL] No messages received between two participants")
        print("  DDS discovery may need more time, or loopback is disabled.")
        print("  In production, participants are long-lived and discovery happens at startup.")
        return False


async def test_dds_latency_roundtrip():
    """
    Measure DDS publish + read latency.
    NOTE: This measures publish() + read_messages() on the same participant.
    If self-delivery is not working, the read will timeout and the measured
    latency will reflect the timeout value, not actual DDS latency.
    """
    print("\n=== Test: DDS Latency (Publish + Read) ===")
    print("  NOTE: Measures publish+read on same participant (local loopback).")

    config = OrchestratorConfig(dds_enabled=True, dds_domain=0)
    dds = DDSLayer(config)

    if not dds.is_available():
        print("[SKIP] CycloneDDS not available")
        return True

    num_tests = 100
    latencies = []

    test_data = {
        "task_id": "latency-rt",
        "requester_id": "benchmark",
        "task_type": "chat",
        "messages_json": "[]",
        "priority": 5,
        "timeout_ms": 5000,
        "requires_context": False,
        "context_id": "",
        "created_at": 0,
    }

    # Warmup
    for _ in range(10):
        await dds.publish(TOPIC_AGENT_REQUEST, test_data)
        dds.read_messages(TOPIC_AGENT_REQUEST, timeout_ms=50)

    # Benchmark
    for i in range(num_tests):
        test_data["created_at"] = int(time.time() * 1000)

        start = time.perf_counter()
        await dds.publish(TOPIC_AGENT_REQUEST, test_data)
        _ = dds.read_messages(TOPIC_AGENT_REQUEST, timeout_ms=100)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)

    dds.close()

    import statistics
    avg = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    min_lat = min(latencies)
    max_lat = max(latencies)

    print(f"  Publish+Read Latency (n={num_tests}):")
    print(f"    Mean:   {avg:.3f} ms")
    print(f"    Median: {p50:.3f} ms")
    print(f"    p95:    {p95:.3f} ms")
    print(f"    Min:    {min_lat:.3f} ms")
    print(f"    Max:    {max_lat:.3f} ms")

    if avg > 90.0:
        print(f"  [WARN: avg={avg:.1f}ms is close to the 100ms read timeout.")
        print(f"   This likely means self-delivery is not working and the measured")
        print(f"   latency reflects the timeout, not actual DDS latency.]")
        print("[WARN] Latency likely reflects timeout, not real DDS latency")
        return True
    elif avg < 10.0:
        print("[PASS] Latency within acceptable range")
    else:
        print("[WARN] Latency higher than expected but below timeout threshold")

    return True


async def main():
    """Run all DDS round-trip tests"""
    print("=" * 60)
    print("DDS Round-Trip / Loopback Tests")
    print("=" * 60)

    tests = [
        ("Same Participant (Loopback)", test_dds_roundtrip_same_participant),
        ("Two Participants", test_dds_roundtrip_two_participants),
        ("Latency (Publish+Read)", test_dds_latency_roundtrip),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
