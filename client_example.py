"""
DDS-LLM Orchestrator Client Examples
Demonstra como conectar ao orchestrator via HTTP ou DDS puro
"""

import asyncio
import json
import sys
import time
import uuid
import argparse
from typing import List, Dict, Any

import aiohttp
import requests


# ============================================
# HTTP Clients (legacy)
# ============================================

class OrchestratorClient:
    """Cliente async HTTP para o orchestrator"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    async def health_check(self) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as resp:
                return await resp.json()

    async def chat(self, messages: List[Dict], **kwargs) -> dict:
        payload = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.7),
            "task_type": kwargs.get("task_type", "chat"),
            "priority": kwargs.get("priority", 5),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/chat", json=payload) as resp:
                return await resp.json()

    async def generate(self, prompt: str, **kwargs) -> dict:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)

    async def list_agents(self) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/agents") as resp:
                return await resp.json()


class SyncOrchestratorClient:
    """Cliente síncrono HTTP"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    def chat(self, messages: List[Dict], **kwargs) -> dict:
        payload = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.7),
        }
        resp = requests.post(f"{self.base_url}/chat", json=payload)
        return resp.json()

    def generate(self, prompt: str, **kwargs) -> dict:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


# ============================================
# DDS Client (100% DDS, no HTTP)
# ============================================

class DDSOrchestratorClient:
    """
    Pure DDS client for the orchestrator.
    Communicates via client/request and client/response topics.
    No HTTP involved in the data path.
    """

    def __init__(self, domain_id: int = 0):
        self.domain_id = domain_id
        self.dds_available = False
        self._init_dds()

    def _init_dds(self):
        try:
            from cyclonedds.domain import DomainParticipant
            from cyclonedds.topic import Topic
            from cyclonedds.pub import DataWriter
            from cyclonedds.sub import DataReader
            from cyclonedds.core import Policy
            from cyclonedds.qos import Qos
            from cyclonedds.util import duration

            from orchestrator import ClientRequest, ClientResponse

            self._ClientRequest = ClientRequest
            self._ClientResponse = ClientResponse

            self.participant = DomainParticipant(self.domain_id)

            # Topics
            self.topic_request = Topic(self.participant, "client/request", ClientRequest)
            self.topic_response = Topic(self.participant, "client/response", ClientResponse)

            # QoS matching orchestrator (Reliable, Volatile, KeepLast 1)
            qos = Qos(
                Policy.Reliability.Reliable(duration(seconds=10)),
                Policy.Durability.Volatile,
                Policy.History.KeepLast(1),
            )

            self.writer = DataWriter(self.participant, self.topic_request, qos)
            self.reader = DataReader(self.participant, self.topic_response, qos)

            self.dds_available = True
            print(f"[DDS Client] Initialized on domain {self.domain_id}")
            print(f"[DDS Client] Topics: client/request (write), client/response (read)")

        except ImportError as e:
            print(f"CycloneDDS not available: {e}")
        except Exception as e:
            print(f"Failed to initialize DDS client: {e}")
            import traceback
            traceback.print_exc()

    def chat(self, messages: List[Dict], timeout_s: float = 120.0) -> dict:
        """Send chat request via DDS and wait for response"""
        if not self.dds_available:
            return {"error": "DDS not available"}

        request_id = str(uuid.uuid4())
        messages_json = json.dumps(messages)

        req = self._ClientRequest(
            request_id=request_id,
            client_id="dds-client",
            task_type="chat",
            messages_json=messages_json,
            priority=2,
            timeout_ms=int(timeout_s * 1000),
            requires_context=False,
        )

        print(f"[DDS Client] Publishing request {request_id}")
        self.writer.write(req)

        # Poll for response matching our request_id
        start = time.time()
        while time.time() - start < timeout_s:
            try:
                samples = self.reader.take()
                for sample in samples:
                    if sample and getattr(sample, "request_id", None) == request_id:
                        elapsed = time.time() - start
                        return {
                            "request_id": request_id,
                            "content": getattr(sample, "content", ""),
                            "is_final": getattr(sample, "is_final", False),
                            "prompt_tokens": getattr(sample, "prompt_tokens", 0),
                            "completion_tokens": getattr(sample, "completion_tokens", 0),
                            "processing_time_ms": getattr(sample, "processing_time_ms", 0),
                            "success": getattr(sample, "success", False),
                            "error_message": getattr(sample, "error_message", ""),
                            "dds_roundtrip_ms": round(elapsed * 1000, 2),
                        }
            except Exception:
                pass
            time.sleep(0.005)  # 5ms poll

        return {"error": "Timeout", "request_id": request_id}

    def generate(self, prompt: str, **kwargs) -> dict:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self):
        if hasattr(self, "participant"):
            del self.participant


# ============================================
# CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(description="DDS-LLM Client")
    parser.add_argument("prompt", nargs="*", help="Prompt text")
    parser.add_argument("--dds", action="store_true", help="Use pure DDS (no HTTP)")
    parser.add_argument("--http", action="store_true", help="Use HTTP client")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain ID")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="Orchestrator URL (HTTP mode)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout in seconds")
    args = parser.parse_args()

    if not args.prompt:
        parser.print_help()
        return

    prompt = " ".join(args.prompt)
    messages = [{"role": "user", "content": prompt}]

    if args.dds:
        print(f"=== DDS Client (domain {args.domain}) ===")
        client = DDSOrchestratorClient(domain_id=args.domain)
        t_start = time.time()
        response = client.chat(messages, timeout_s=args.timeout)
        t_total = time.time() - t_start
        print(f"\nResponse ({t_total*1000:.0f}ms):")
        if "error" in response and response.get("content", "") == "":
            print(f"  ERROR: {response['error']}")
        else:
            print(f"  Content: {response.get('content', '')}")
            print(f"  Success: {response.get('success')}")
            print(f"  Tokens: {response.get('prompt_tokens', 0)} prompt + {response.get('completion_tokens', 0)} completion")
            print(f"  DDS roundtrip: {response.get('dds_roundtrip_ms', 0)}ms")
        client.close()
    else:
        print(f"=== HTTP Client ({args.url}) ===")
        client = SyncOrchestratorClient(args.url)
        t_start = time.time()
        response = client.generate(prompt)
        t_total = time.time() - t_start
        print(f"\nResponse ({t_total*1000:.0f}ms):")
        print(f"  {json.dumps(response, indent=2)}")


if __name__ == "__main__":
    main()
