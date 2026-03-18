#!/usr/bin/env python3
"""
Test gRPC Round-Trip -- Verifica comunicação gRPC nativa com llama-server.
Testa Chat (unário), StreamChat (streaming) e GetStatus RPCs.

Mirrors test_dds_roundtrip.py for fair comparison.

Uso:
    # Com llama-server rodando com --enable-grpc:
    python test_grpc_roundtrip.py --endpoint localhost:50051

    # Com proxy _grpc_server.py:
    python test_grpc_roundtrip.py --endpoint localhost:50051 --use-proxy
"""
import argparse
import asyncio
import os
import statistics
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmarks"))

import grpc
from benchmarks.proto import llama_service_pb2
from benchmarks.proto import llama_service_pb2_grpc


async def test_grpc_connectivity(endpoint: str):
    """Test basic gRPC channel connectivity."""
    print("\n=== Test: gRPC Connectivity ===")

    channel = grpc.insecure_channel(endpoint)
    stub = llama_service_pb2_grpc.LlamaServiceStub(channel)

    try:
        status = stub.GetStatus(llama_service_pb2.Empty(), timeout=5.0)
        print(f"  Server ID:        {status.server_id}")
        print(f"  Model loaded:     {status.model_loaded}")
        print(f"  Ready:            {status.ready}")
        print(f"  Slots idle:       {status.slots_idle}")
        print(f"  Slots processing: {status.slots_processing}")
        channel.close()
        print("[PASS] gRPC connectivity OK")
        return True
    except grpc.RpcError as e:
        print(f"[FAIL] gRPC connectivity failed: {e.code()} - {e.details()}")
        channel.close()
        return False


async def test_grpc_chat_unary(endpoint: str):
    """Test unary Chat RPC (non-streaming)."""
    print("\n=== Test: gRPC Chat (Unary) ===")

    channel = grpc.insecure_channel(
        endpoint,
        options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)],
    )
    stub = llama_service_pb2_grpc.LlamaServiceStub(channel)

    request = llama_service_pb2.ChatCompletionRequest(
        request_id=str(uuid.uuid4()),
        model="default",
        messages=[llama_service_pb2.ChatMessage(role="user", content="What is 2+2?")],
        temperature=0.7,
        max_tokens=50,
        stream=False,
    )

    start = time.perf_counter()
    try:
        response = stub.Chat(request, timeout=60)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"  Request ID:        {response.request_id}")
        print(f"  Content:           {response.content[:100]}...")
        print(f"  Finish reason:     {response.finish_reason}")
        print(f"  Is final:          {response.is_final}")
        print(f"  Prompt tokens:     {response.prompt_tokens}")
        print(f"  Completion tokens: {response.completion_tokens}")
        print(f"  Latency:           {elapsed:.1f}ms")

        channel.close()

        if response.content:
            print("[PASS] Unary Chat RPC OK")
            return True
        else:
            print("[FAIL] Empty response content")
            return False

    except grpc.RpcError as e:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[FAIL] Chat RPC failed after {elapsed:.1f}ms: {e.code()} - {e.details()}")
        channel.close()
        return False


async def test_grpc_stream_chat(endpoint: str):
    """Test server-streaming StreamChat RPC."""
    print("\n=== Test: gRPC StreamChat (Server Streaming) ===")

    channel = grpc.insecure_channel(
        endpoint,
        options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)],
    )
    stub = llama_service_pb2_grpc.LlamaServiceStub(channel)

    request = llama_service_pb2.ChatCompletionRequest(
        request_id=str(uuid.uuid4()),
        model="default",
        messages=[llama_service_pb2.ChatMessage(role="user", content="Count from 1 to 5.")],
        temperature=0.7,
        max_tokens=100,
        stream=True,
    )

    accumulated = ""
    chunk_count = 0
    ttft = None

    start = time.perf_counter()
    try:
        stream = stub.StreamChat(request, timeout=60)

        for response in stream:
            current = time.perf_counter()

            if response.content and ttft is None:
                ttft = (current - start) * 1000

            accumulated += response.content
            chunk_count += 1

            if response.is_final:
                break

        elapsed = (time.perf_counter() - start) * 1000

        print(f"  Chunks received:   {chunk_count}")
        print(f"  Accumulated text:  {accumulated[:100]}...")
        print(f"  TTFT:              {ttft:.1f}ms" if ttft else "  TTFT: N/A")
        print(f"  Total time:        {elapsed:.1f}ms")

        channel.close()

        if chunk_count > 0 and accumulated:
            print("[PASS] StreamChat RPC OK")
            return True
        else:
            print("[FAIL] No streaming chunks received")
            return False

    except grpc.RpcError as e:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[FAIL] StreamChat RPC failed after {elapsed:.1f}ms: {e.code()} - {e.details()}")
        channel.close()
        return False


async def test_grpc_latency_roundtrip(endpoint: str, n: int = 100):
    """Measure gRPC Chat RPC round-trip latency."""
    print(f"\n=== Test: gRPC Latency (Chat RPC, n={n}) ===")

    channel = grpc.insecure_channel(
        endpoint,
        options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)],
    )
    stub = llama_service_pb2_grpc.LlamaServiceStub(channel)

    latencies = []

    # Warmup
    for _ in range(5):
        try:
            req = llama_service_pb2.ChatCompletionRequest(
                request_id=str(uuid.uuid4()),
                model="default",
                messages=[llama_service_pb2.ChatMessage(role="user", content="hi")],
                max_tokens=5,
                stream=False,
            )
            stub.Chat(req, timeout=30)
        except grpc.RpcError:
            pass

    # Benchmark
    for i in range(n):
        req = llama_service_pb2.ChatCompletionRequest(
            request_id=str(uuid.uuid4()),
            model="default",
            messages=[llama_service_pb2.ChatMessage(role="user", content="hi")],
            max_tokens=5,
            stream=False,
        )

        start = time.perf_counter()
        try:
            stub.Chat(req, timeout=30)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        except grpc.RpcError as e:
            print(f"  Iteration {i+1}: gRPC error {e.code()}")
            continue

    channel.close()

    if not latencies:
        print("[FAIL] No successful requests")
        return False

    avg = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    min_lat = min(latencies)
    max_lat = max(latencies)

    print(f"  Chat RPC Latency (n={len(latencies)}):")
    print(f"    Mean:   {avg:.3f} ms")
    print(f"    Median: {p50:.3f} ms")
    print(f"    p95:    {p95:.3f} ms")
    print(f"    Min:    {min_lat:.3f} ms")
    print(f"    Max:    {max_lat:.3f} ms")

    # Serialization overhead
    req = llama_service_pb2.ChatCompletionRequest(
        request_id="size-test",
        model="default",
        messages=[llama_service_pb2.ChatMessage(role="user", content="hi")],
        max_tokens=5,
    )
    proto_size = len(req.SerializeToString())
    print(f"    Proto msg size: {proto_size} bytes")

    print("[PASS] Latency measurement complete")
    return True


async def main(args):
    """Run all gRPC round-trip tests."""
    print("=" * 60)
    print("gRPC Round-Trip / Connectivity Tests")
    print(f"Endpoint: {args.endpoint}")
    print("=" * 60)

    tests = [
        ("Connectivity (GetStatus)", lambda: test_grpc_connectivity(args.endpoint)),
        ("Chat (Unary)", lambda: test_grpc_chat_unary(args.endpoint)),
        ("StreamChat (Streaming)", lambda: test_grpc_stream_chat(args.endpoint)),
        ("Latency (Chat RPC)", lambda: test_grpc_latency_roundtrip(args.endpoint, args.n)),
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
    parser = argparse.ArgumentParser(description="gRPC Round-Trip Tests")
    parser.add_argument("--endpoint", default="localhost:50051",
                        help="gRPC server endpoint (host:port)")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of latency iterations")

    args = parser.parse_args()
    success = asyncio.run(main(args))
    sys.exit(0 if success else 1)
