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
    print(f"[OK] Read {len(status)} status updates")

    # Cleanup
    dds.close()
    print("[OK] DDS Layer test passed!")
    return True


async def test_dds_latency():
    """Test DDS publish/subscribe latency"""
    print("\n=== Testing DDS Latency ===")

    config = OrchestratorConfig(dds_enabled=True, dds_domain=0)
    dds = DDSLayer(config)

    if not dds.is_available():
        print("[SKIP] CycloneDDS not available")
        return True

    # Measure latency
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

    for i in range(num_tests):
        start = time.perf_counter()
        await dds.publish(TOPIC_AGENT_REQUEST, test_data)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print(f"Publish Latency (n={num_tests}):")
    print(f"  Avg: {avg_latency:.3f} ms")
    print(f"  Min: {min_latency:.3f} ms")
    print(f"  Max: {max_latency:.3f} ms")

    dds.close()
    return True


async def test_dds_throughput():
    """Test DDS throughput"""
    print("\n=== Testing DDS Throughput ===")

    config = OrchestratorConfig(dds_enabled=True, dds_domain=0)
    dds = DDSLayer(config)

    if not dds.is_available():
        print("[SKIP] CycloneDDS not available")
        return True

    # Measure throughput
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
    throughput = count / elapsed

    print(f"Throughput:")
    print(f"  Messages: {count}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Rate: {throughput:.1f} msg/s")

    dds.close()
    return True


async def main():
    """Run all tests"""
    print("=" * 50)
    print("DDS Layer Integration Tests")
    print("=" * 50)

    tests = [
        ("DDS Layer Init", test_dds_layer),
        ("DDS Latency", test_dds_latency),
        ("DDS Throughput", test_dds_throughput),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
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
