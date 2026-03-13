#!/usr/bin/env python3
"""
Test script for orchestrator DDS layer
Tests the DDS communication with real CycloneDDS
"""
import asyncio
import sys
import os
import time
import uuid

sys.path.insert(0, os.path.dirname(__file__))

from config import OrchestratorConfig
from dds import DDSLayer, TOPIC_AGENT_REQUEST, TOPIC_AGENT_STATUS


async def test_dds_layer():
    """Test DDS layer with real CycloneDDS"""
    print("\n=== Testing DDS Layer ===")

    config = OrchestratorConfig(dds_enabled=True, dds_domain=0)
    dds = DDSLayer(config)

    if not dds.is_available():
        print("[SKIP] CycloneDDS not available")
        return True

    assert dds.is_available(), "DDS layer should be available after initialization"
    print("[OK] DDS Layer initialized")

    # Test publishing
    test_data = {
        "task_id": str(uuid.uuid4()),
        "requester_id": "test-client",
        "task_type": "chat",
        "messages_json": '[{"role": "user", "content": "test"}]',
        "priority": 5,
        "timeout_ms": 30000,
        "requires_context": False,
        "context_id": "",
        "created_at": int(time.time() * 1000),
    }

    await dds.publish(TOPIC_AGENT_REQUEST, test_data)
    print("[OK] Published message to topic")

    # Test reading (should be empty for now)
    status = await dds.read_status_updates(timeout_ms=100)
    assert isinstance(status, list), "read_status_updates should return a list"
    print(f"[OK] Read {len(status)} status updates")

    # Cleanup
    dds.close()
    print("[PASS] DDS Layer test passed!")
    return True


async def test_dds_publish_overhead():
    """
    Measure DDS publish overhead (serialization + local delivery).
    NOTE: This measures only the time to call publish() without a subscriber.
    It reflects serialization and kernel handoff overhead, NOT round-trip latency.
    For round-trip latency, see test_dds_roundtrip.py.
    """
    print("\n=== Testing DDS Publish Overhead ===")
    print("  NOTE: Measures publish() call time only (no subscriber).")
    print("  This reflects serialization + local delivery overhead.")

    config = OrchestratorConfig(dds_enabled=True, dds_domain=0)
    dds = DDSLayer(config)

    if not dds.is_available():
        print("[SKIP] CycloneDDS not available")
        return True

    # Measure publish overhead
    num_tests = 100
    latencies = []

    test_data = {
        "task_id": "latency-test",
        "requester_id": "test",
        "task_type": "chat",
        "messages_json": "[]",
        "priority": 5,
        "timeout_ms": 30000,
        "requires_context": False,
        "context_id": "",
        "created_at": 0,
    }

    # Warmup
    for _ in range(10):
        await dds.publish(TOPIC_AGENT_REQUEST, test_data)

    for i in range(num_tests):
        start = time.perf_counter()
        await dds.publish(TOPIC_AGENT_REQUEST, test_data)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print(f"  Publish Overhead (n={num_tests}):")
    print(f"    Avg: {avg_latency:.3f} ms")
    print(f"    Min: {min_latency:.3f} ms")
    print(f"    Max: {max_latency:.3f} ms")

    # Assertions: publish overhead should be reasonable
    assert avg_latency < 50.0, f"Average publish overhead too high: {avg_latency:.3f} ms (expected < 50ms)"
    assert min_latency >= 0, "Latency should not be negative"

    dds.close()
    print("[PASS] DDS Publish Overhead test passed!")
    return True


async def test_dds_publish_rate():
    """
    Measure DDS publish rate (messages per second).
    NOTE: This measures the rate at which publish() can be called without
    a subscriber consuming messages. It reflects the maximum publish throughput
    of the local DDS stack, not end-to-end throughput.
    """
    print("\n=== Testing DDS Publish Rate ===")
    print("  NOTE: Measures publish rate without subscriber.")
    print("  This is the maximum local publish throughput, not end-to-end throughput.")

    config = OrchestratorConfig(dds_enabled=True, dds_domain=0)
    dds = DDSLayer(config)

    if not dds.is_available():
        print("[SKIP] CycloneDDS not available")
        return True

    # Measure publish rate
    duration_seconds = 5
    test_data = {
        "task_id": "throughput-test",
        "requester_id": "test",
        "task_type": "chat",
        "messages_json": "[]",
        "priority": 5,
        "timeout_ms": 30000,
        "requires_context": False,
        "context_id": "",
        "created_at": 0,
    }

    start_time = time.time()
    count = 0

    while time.time() - start_time < duration_seconds:
        await dds.publish(TOPIC_AGENT_REQUEST, test_data)
        count += 1

    elapsed = time.time() - start_time
    rate = count / elapsed

    print(f"  Publish Rate:")
    print(f"    Messages: {count}")
    print(f"    Time: {elapsed:.2f}s")
    print(f"    Rate: {rate:.1f} msg/s")

    # Assertions
    assert count > 0, "Should have published at least one message"
    assert rate > 10.0, f"Publish rate too low: {rate:.1f} msg/s (expected > 10 msg/s)"

    dds.close()
    print("[PASS] DDS Publish Rate test passed!")
    return True


async def main():
    """Run all tests"""
    print("=" * 50)
    print("DDS Layer Integration Tests")
    print("=" * 50)

    tests = [
        ("DDS Layer Init", test_dds_layer),
        ("DDS Publish Overhead", test_dds_publish_overhead),
        ("DDS Publish Rate", test_dds_publish_rate),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
            else:
                print(f"[FAIL] {name} returned False")
                failed += 1
        except AssertionError as e:
            print(f"[FAIL] {name} (assertion): {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
