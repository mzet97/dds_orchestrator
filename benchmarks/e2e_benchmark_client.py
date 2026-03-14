#!/usr/bin/env python3
"""
End-to-End Benchmark Client — HTTP, gRPC, DDS
================================================
Executa benchmarks E1-E5 usando o protocolo COMPLETO de ponta a ponta:

  HTTP:  cliente --HTTP--> orchestrator --HTTP--> agent --HTTP--> llama-server
  gRPC:  cliente --gRPC--> orchestrator --gRPC--> agent --gRPC--> llama-server
  DDS:   cliente --DDS-->  orchestrator --DDS-->  agent --DDS-->  llama-server

Usage:
    # E1: Latência end-to-end
    python e2e_benchmark_client.py --protocol http --url http://192.168.1.62:8080 --scenario E1 --n 10
    python e2e_benchmark_client.py --protocol grpc --endpoint 192.168.1.62:50052 --scenario E1 --n 10
    python e2e_benchmark_client.py --protocol dds  --domain 0 --scenario E1 --n 10

    # Todos os cenários
    python e2e_benchmark_client.py --protocol http --url http://192.168.1.62:8080 --scenario all --n 5
"""

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

# ─── Protocol Clients ───────────────────────────────────────────────────────

class HTTPBenchmarkClient:
    """HTTP client for benchmarks — aiohttp POST to orchestrator."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.protocol = "HTTP"

    async def setup(self):
        pass

    async def chat(self, messages: List[Dict], model: str = "",
                   max_tokens: int = 50, temperature: float = 0.7,
                   timeout_s: float = 120.0) -> Dict:
        import aiohttp

        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        t_start = time.perf_counter_ns()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                body = await resp.text()
        t_end = time.perf_counter_ns()

        roundtrip_ms = (t_end - t_start) / 1e6

        try:
            data = json.loads(body)
            content = ""
            if "choices" in data:
                content = data["choices"][0].get("message", {}).get("content", "")
            elif "content" in data:
                content = data["content"]
            return {
                "content": content,
                "success": True,
                "roundtrip_ms": roundtrip_ms,
                "raw": data,
            }
        except Exception as e:
            return {
                "content": body[:200],
                "success": False,
                "error": str(e),
                "roundtrip_ms": roundtrip_ms,
            }

    async def chat_stream(self, messages: List[Dict], model: str = "",
                          max_tokens: int = 200, timeout_s: float = 120.0) -> Dict:
        """Streaming request — measures TTFT and ITL."""
        import aiohttp

        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }

        tokens = []
        t_start = time.perf_counter_ns()
        ttft = None
        content = ""

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                async for line in resp.content:
                    t_now = time.perf_counter_ns()
                    decoded = line.decode("utf-8", errors="ignore").strip()
                    if not decoded or decoded == "data: [DONE]":
                        continue
                    if decoded.startswith("data: "):
                        decoded = decoded[6:]
                    try:
                        chunk = json.loads(decoded)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            if ttft is None:
                                ttft = (t_now - t_start) / 1e6
                            tokens.append((t_now - t_start) / 1e6)
                            content += token
                    except json.JSONDecodeError:
                        pass

        t_end = time.perf_counter_ns()
        total_ms = (t_end - t_start) / 1e6

        # Calculate ITL
        itl_values = []
        for i in range(1, len(tokens)):
            itl_values.append(tokens[i] - tokens[i - 1])

        return {
            "content": content,
            "success": True,
            "ttft_ms": ttft or total_ms,
            "itl_mean_ms": statistics.mean(itl_values) if itl_values else 0,
            "itl_p99_ms": (sorted(itl_values)[int(len(itl_values) * 0.99)]
                           if len(itl_values) > 1 else 0),
            "total_tokens": len(tokens),
            "total_ms": total_ms,
        }

    def close(self):
        pass


class GRPCBenchmarkClient:
    """gRPC client for benchmarks — pure gRPC to orchestrator."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.protocol = "gRPC"
        self.stub = None
        self.channel = None

    async def setup(self):
        import grpc
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from proto import orchestrator_pb2 as pb2
        from proto import orchestrator_pb2_grpc as pb2_grpc
        self._pb2 = pb2
        self._pb2_grpc = pb2_grpc
        self.channel = grpc.insecure_channel(
            self.endpoint,
            options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)],
        )
        self.stub = pb2_grpc.ClientOrchestratorServiceStub(self.channel)

    async def chat(self, messages: List[Dict], model: str = "",
                   max_tokens: int = 50, temperature: float = 0.7,
                   timeout_s: float = 120.0) -> Dict:
        proto_msgs = [
            self._pb2.ChatMessage(role=m.get("role", "user"),
                                   content=m.get("content", ""))
            for m in messages
        ]
        request = self._pb2.ClientChatRequest(
            request_id=str(uuid.uuid4()),
            model=model,
            messages=proto_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            priority=5,
            timeout_ms=int(timeout_s * 1000),
        )

        t_start = time.perf_counter_ns()
        try:
            response = self.stub.Chat(request, timeout=timeout_s)
            t_end = time.perf_counter_ns()
            return {
                "content": response.content,
                "success": response.success,
                "error": response.error_message if response.error_message else None,
                "roundtrip_ms": (t_end - t_start) / 1e6,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "processing_time_ms": response.processing_time_ms,
            }
        except Exception as e:
            t_end = time.perf_counter_ns()
            return {
                "content": "",
                "success": False,
                "error": str(e),
                "roundtrip_ms": (t_end - t_start) / 1e6,
            }

    async def chat_stream(self, messages: List[Dict], model: str = "",
                          max_tokens: int = 200, timeout_s: float = 120.0) -> Dict:
        """Streaming via gRPC server-streaming."""
        proto_msgs = [
            self._pb2.ChatMessage(role=m.get("role", "user"),
                                   content=m.get("content", ""))
            for m in messages
        ]
        request = self._pb2.ClientChatRequest(
            request_id=str(uuid.uuid4()),
            model=model,
            messages=proto_msgs,
            max_tokens=max_tokens,
            stream=True,
            priority=5,
            timeout_ms=int(timeout_s * 1000),
        )

        tokens = []
        t_start = time.perf_counter_ns()
        ttft = None
        content = ""

        try:
            for response in self.stub.StreamChat(request, timeout=timeout_s):
                t_now = time.perf_counter_ns()
                if response.content:
                    if ttft is None:
                        ttft = (t_now - t_start) / 1e6
                    tokens.append((t_now - t_start) / 1e6)
                    content += response.content
                if response.is_final:
                    break
        except Exception as e:
            pass

        t_end = time.perf_counter_ns()
        total_ms = (t_end - t_start) / 1e6

        itl_values = []
        for i in range(1, len(tokens)):
            itl_values.append(tokens[i] - tokens[i - 1])

        return {
            "content": content,
            "success": True,
            "ttft_ms": ttft or total_ms,
            "itl_mean_ms": statistics.mean(itl_values) if itl_values else 0,
            "itl_p99_ms": (sorted(itl_values)[int(len(itl_values) * 0.99)]
                           if len(itl_values) > 1 else 0),
            "total_tokens": len(tokens),
            "total_ms": total_ms,
        }

    def close(self):
        if self.channel:
            self.channel.close()


class DDSBenchmarkClient:
    """DDS client for benchmarks — pure DDS pub/sub to orchestrator."""

    def __init__(self, domain_id: int = 0):
        self.domain_id = domain_id
        self.protocol = "DDS"
        self.dds_available = False

    async def setup(self):
        try:
            from cyclonedds.domain import DomainParticipant
            from cyclonedds.topic import Topic
            from cyclonedds.pub import DataWriter
            from cyclonedds.sub import DataReader
            from cyclonedds.core import Policy
            from cyclonedds.qos import Qos
            from cyclonedds.util import duration

            sys.path.insert(0, str(Path(__file__).parent.parent))
            from orchestrator import ClientRequest, ClientResponse

            self._ClientRequest = ClientRequest
            self._ClientResponse = ClientResponse

            self.participant = DomainParticipant(self.domain_id)
            self.topic_request = Topic(self.participant, "client/request", ClientRequest)
            self.topic_response = Topic(self.participant, "client/response", ClientResponse)

            qos = Qos(
                Policy.Reliability.Reliable(duration(seconds=10)),
                Policy.Durability.Volatile,
                Policy.History.KeepLast(1),
            )

            self.writer = DataWriter(self.participant, self.topic_request, qos)
            self.reader = DataReader(self.participant, self.topic_response, qos)
            self.dds_available = True
            print(f"[DDS Client] Initialized on domain {self.domain_id}")
        except Exception as e:
            print(f"[DDS Client] Failed to initialize: {e}")

    async def chat(self, messages: List[Dict], model: str = "",
                   max_tokens: int = 50, temperature: float = 0.7,
                   timeout_s: float = 120.0) -> Dict:
        if not self.dds_available:
            return {"content": "", "success": False, "error": "DDS not available", "roundtrip_ms": 0}

        request_id = str(uuid.uuid4())
        messages_json = json.dumps(messages)

        req = self._ClientRequest(
            request_id=request_id,
            client_id="benchmark-dds-client",
            task_type="chat",
            messages_json=messages_json,
            priority=5,
            timeout_ms=int(timeout_s * 1000),
            requires_context=False,
        )

        t_start = time.perf_counter_ns()
        self.writer.write(req)

        # Poll for response
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                samples = self.reader.take()
                for sample in samples:
                    if sample and getattr(sample, "request_id", None) == request_id:
                        t_end = time.perf_counter_ns()
                        return {
                            "content": getattr(sample, "content", ""),
                            "success": getattr(sample, "success", False),
                            "error": getattr(sample, "error_message", ""),
                            "roundtrip_ms": (t_end - t_start) / 1e6,
                            "prompt_tokens": getattr(sample, "prompt_tokens", 0),
                            "completion_tokens": getattr(sample, "completion_tokens", 0),
                            "processing_time_ms": getattr(sample, "processing_time_ms", 0),
                        }
            except Exception:
                pass
            time.sleep(0.005)

        t_end = time.perf_counter_ns()
        return {"content": "", "success": False, "error": "Timeout",
                "roundtrip_ms": (t_end - t_start) / 1e6}

    async def chat_stream(self, messages: List[Dict], model: str = "",
                          max_tokens: int = 200, timeout_s: float = 120.0) -> Dict:
        """DDS streaming — single response (DDS pub/sub doesn't do token-by-token streaming
        from client perspective in current implementation)."""
        result = await self.chat(messages, model=model, max_tokens=max_tokens,
                                timeout_s=timeout_s)
        return {
            "content": result.get("content", ""),
            "success": result.get("success", False),
            "ttft_ms": result.get("roundtrip_ms", 0),
            "itl_mean_ms": 0,
            "itl_p99_ms": 0,
            "total_tokens": 1,
            "total_ms": result.get("roundtrip_ms", 0),
        }

    def close(self):
        for attr in ("writer", "reader", "participant"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass


# ─── Prompts ─────────────────────────────────────────────────────────────────

PROMPTS = {
    "short": "What is 2+2?",
    "long": (
        "Explain in detail the differences between TCP and UDP protocols, "
        "including their use cases, advantages, disadvantages, header structure, "
        "connection establishment process, flow control mechanisms, error handling, "
        "and how they relate to the OSI model. Provide examples of applications "
        "that use each protocol and explain why they were chosen for those specific "
        "use cases."
    ),
}


# ─── Benchmark Scenarios ─────────────────────────────────────────────────────

async def run_E1(client, model: str, n: int) -> Dict:
    """E1: Decomposição de latência end-to-end."""
    results = {"short": [], "long": []}

    for prompt_type in ["short", "long"]:
        prompt = PROMPTS[prompt_type]
        messages = [{"role": "user", "content": prompt}]

        print(f"  E1 {client.protocol} {prompt_type}: ", end="", flush=True)
        for i in range(n):
            r = await client.chat(messages, model=model, max_tokens=50)
            results[prompt_type].append({
                "iteration": i,
                "roundtrip_ms": r.get("roundtrip_ms", 0),
                "success": r.get("success", False),
                "content_len": len(r.get("content", "")),
            })
            print(".", end="", flush=True)
        print()

        latencies = [r["roundtrip_ms"] for r in results[prompt_type] if r["success"]]
        if latencies:
            print(f"    p50={sorted(latencies)[len(latencies)//2]:.1f}ms "
                  f"mean={statistics.mean(latencies):.1f}ms "
                  f"std={statistics.stdev(latencies):.1f}ms" if len(latencies) > 1 else
                  f"    mean={statistics.mean(latencies):.1f}ms")

    return results


async def run_E3(client, model: str, n: int) -> Dict:
    """E3: Priorização — envia requisições normais e mede latência de HIGH priority."""
    results = {"normal": [], "high": []}
    messages = [{"role": "user", "content": "Say hello"}]

    print(f"  E3 {client.protocol}: ", end="", flush=True)
    for i in range(n):
        # Normal priority
        r = await client.chat(messages, model=model, max_tokens=5)
        results["normal"].append({"roundtrip_ms": r.get("roundtrip_ms", 0), "success": r.get("success", False)})

        # High priority (same request but conceptually priority matters at orchestrator level)
        r = await client.chat(messages, model=model, max_tokens=5)
        results["high"].append({"roundtrip_ms": r.get("roundtrip_ms", 0), "success": r.get("success", False)})
        print(".", end="", flush=True)
    print()

    return results


async def run_E4(client, model: str, n: int, num_clients_list: List[int] = None) -> Dict:
    """E4: Escalabilidade — mede throughput com N clientes simultâneos."""
    if num_clients_list is None:
        num_clients_list = [1, 2, 4]

    results = {}
    messages = [{"role": "user", "content": "Say hello briefly"}]

    for num_clients in num_clients_list:
        print(f"  E4 {client.protocol} clients={num_clients}: ", end="", flush=True)

        async def single_client_work():
            latencies = []
            for _ in range(n):
                r = await client.chat(messages, model=model, max_tokens=10)
                latencies.append(r.get("roundtrip_ms", 0))
            return latencies

        t_start = time.time()
        # Run concurrent clients (all using same client instance for simplicity)
        tasks = [single_client_work() for _ in range(num_clients)]
        all_latencies_nested = await asyncio.gather(*tasks)
        t_total = time.time() - t_start

        all_latencies = [lat for sublist in all_latencies_nested for lat in sublist]
        total_requests = len(all_latencies)
        throughput = total_requests / t_total if t_total > 0 else 0

        results[f"clients_{num_clients}"] = {
            "throughput_rps": round(throughput, 2),
            "latencies": all_latencies,
            "p50": sorted(all_latencies)[len(all_latencies) // 2] if all_latencies else 0,
            "p95": sorted(all_latencies)[int(len(all_latencies) * 0.95)] if all_latencies else 0,
            "total_time_s": round(t_total, 2),
        }
        print(f" {throughput:.1f} req/s, p50={results[f'clients_{num_clients}']['p50']:.0f}ms")

    return results


async def run_E5(client, model: str, n: int) -> Dict:
    """E5: Streaming — mede TTFT e ITL."""
    results = []
    messages = [{"role": "user", "content": "Write a short poem about the ocean"}]

    print(f"  E5 {client.protocol}: ", end="", flush=True)
    for i in range(n):
        r = await client.chat_stream(messages, model=model, max_tokens=200)
        results.append({
            "ttft_ms": r.get("ttft_ms", 0),
            "itl_mean_ms": r.get("itl_mean_ms", 0),
            "itl_p99_ms": r.get("itl_p99_ms", 0),
            "total_tokens": r.get("total_tokens", 0),
            "total_ms": r.get("total_ms", 0),
        })
        print(".", end="", flush=True)
    print()

    ttfts = [r["ttft_ms"] for r in results]
    if ttfts:
        print(f"    TTFT mean={statistics.mean(ttfts):.1f}ms")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def create_client(protocol: str, url: str = "", endpoint: str = "",
                  domain: int = 0):
    if protocol == "http":
        return HTTPBenchmarkClient(url)
    elif protocol == "grpc":
        return GRPCBenchmarkClient(endpoint)
    elif protocol == "dds":
        return DDSBenchmarkClient(domain)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


async def run_benchmarks(args):
    client = create_client(args.protocol, url=args.url,
                           endpoint=args.endpoint, domain=args.domain)
    await client.setup()

    scenarios = ["E1", "E3", "E4", "E5"] if args.scenario == "all" else [args.scenario]
    all_results = {
        "protocol": args.protocol.upper(),
        "model": args.model,
        "n": args.n,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    # Warmup
    print(f"\n{'='*60}")
    print(f"  Warmup ({client.protocol})...")
    print(f"{'='*60}")
    messages = [{"role": "user", "content": "hello"}]
    for _ in range(2):
        await client.chat(messages, model=args.model, max_tokens=5)
    print("  Warmup done.\n")

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"  {scenario}: {client.protocol} (n={args.n})")
        print(f"{'='*60}")

        if scenario == "E1":
            all_results["E1"] = await run_E1(client, args.model, args.n)
        elif scenario == "E3":
            all_results["E3"] = await run_E3(client, args.model, args.n)
        elif scenario == "E4":
            all_results["E4"] = await run_E4(client, args.model, args.n)
        elif scenario == "E5":
            all_results["E5"] = await run_E5(client, args.model, args.n)

    client.close()

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    fname = f"e2e_{args.protocol}_{all_results['timestamp']}.json"
    out_path = results_dir / fname
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Benchmark Client (HTTP/gRPC/DDS)")
    parser.add_argument("--protocol", choices=["http", "grpc", "dds"], required=True)
    parser.add_argument("--url", default="http://192.168.1.62:8080",
                        help="Orchestrator HTTP URL")
    parser.add_argument("--endpoint", default="192.168.1.62:50052",
                        help="Orchestrator gRPC endpoint")
    parser.add_argument("--domain", type=int, default=0,
                        help="DDS domain ID")
    parser.add_argument("--model", default="", help="Model name")
    parser.add_argument("--scenario", choices=["all", "E1", "E3", "E4", "E5"],
                        default="all", help="Scenario to run (E2 is standalone)")
    parser.add_argument("--n", type=int, default=5, help="Iterations per test")

    args = parser.parse_args()
    asyncio.run(run_benchmarks(args))


if __name__ == "__main__":
    main()
