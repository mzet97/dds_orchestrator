#!/usr/bin/env python3
"""
DDS Client Benchmark - 100% DDS Communication
Envia requisições para o orquestrador usando DDS (sem HTTP)
"""

import asyncio
import json
import time
import uuid
import os
import sys
from datetime import datetime

# Adicionar path para importar dds_types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import Publisher
from cyclonedds.sub import Subscriber
from cyclonedds.topic import Topic
from cyclonedds.qos import QoS, Policy
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import bounded_str, int32, bool_


class ClientRequestType(IdlStruct):
    """Client request message type"""
    request_id: bounded_str[256]
    client_id: bounded_str[256]
    task_type: bounded_str[64]
    messages_json: bounded_str[16384]
    priority: int32
    timeout_ms: int32
    requires_context: bool_
    created_at: int32


class ClientResponseType(IdlStruct):
    """Client response message type"""
    request_id: bounded_str[256]
    client_id: bounded_str[256]
    content: bounded_str[16384]
    is_final: bool_
    prompt_tokens: int32
    completion_tokens: int32
    processing_time_ms: int32
    success: bool_
    error_message: bounded_str[1024]


class DDSClientBenchmark:
    """Cliente DDS para benchmark"""

    def __init__(self, client_id: str = None, domain: int = 0):
        self.client_id = client_id or f"benchmark-client-{uuid.uuid4().hex[:8]}"
        self.domain = domain
        self.participant = None
        self.publisher = None
        self.subscriber = None
        self.request_topic = None
        self.response_topic = None
        self.request_writer = None
        self.response_reader = None
        self.responses = {}
        self.loop = None

    def initialize(self):
        """Inicializa a conexão DDS"""
        print(f"Inicializando cliente DDS: {self.client_id}")

        # Criar participant
        self.participant = DomainParticipant(self.domain)

        # QoS para comunicação confiável
        qos_reliable = QoS(
            Policy.Reliability.RELIABLE,
            Policy.Durability.TRANSIENT_LOCAL
        )

        # Publisher para enviar requisições
        self.publisher = Publisher(self.participant)

        # Subscriber para receber respostas
        self.subscriber = Subscriber(self.participant)

        # Tópicos
        self.request_topic = Topic(
            self.participant, "client/request",
            ClientRequestType, qos=qos_reliable
        )
        self.response_topic = Topic(
            self.participant, "client/response",
            ClientResponseType, qos=qos_reliable
        )

        # DataWriters e DataReaders
        from cyclonedds.pub import DataWriter
        from cyclonedds.sub import DataReader

        self.request_writer = DataWriter(self.publisher, self.request_topic)
        self.response_reader = DataReader(self.subscriber, self.response_topic)

        print(f"Cliente DDS inicializado no domínio {self.domain}")

    def send_request(self, messages: list, task_type: str = "chat",
                    max_tokens: int = 10, priority: int = 1,
                    timeout_ms: int = 30000) -> dict:
        """Envia uma requisição via DDS e aguarda resposta"""

        request_id = uuid.uuid4().hex[:16]
        messages_json = json.dumps({
            "messages": messages,
            "max_tokens": max_tokens
        })

        # Criar mensagem de requisição
        request = ClientRequestType(
            request_id=request_id,
            client_id=self.client_id,
            task_type=task_type,
            messages_json=messages_json,
            priority=priority,
            timeout_ms=timeout_ms,
            requires_context=False,
            created_at=int(time.time())
        )

        # Enviar requisição
        self.request_writer.write(request)

        # Aguardar resposta
        start_time = time.time()
        while time.time() - start_time < (timeout_ms / 1000):
            # Verificar se há respostas
            msgs = self.response_reader.take()
            for msg in msgs:
                if msg.request_id == request_id:
                    return {
                        "request_id": msg.request_id,
                        "content": msg.content,
                        "is_final": msg.is_final,
                        "success": msg.success,
                        "prompt_tokens": msg.prompt_tokens,
                        "completion_tokens": msg.completion_tokens,
                        "processing_time_ms": msg.processing_time_ms,
                        "error_message": msg.error_message
                    }
            time.sleep(0.01)  # Pequeno sleep para evitar busy loop

        return {
            "request_id": request_id,
            "success": False,
            "error_message": "Timeout aguardando resposta"
        }

    def close(self):
        """Fecha a conexão DDS"""
        if self.participant:
            self.participant.close()


async def run_benchmark(prompts: list, model: str = "phi4-mini",
                       domain: int = 0, num_runs: int = 5) -> list:
    """Executa o benchmark com os prompts fornecidos"""

    client = DDSClientBenchmark(domain=domain)
    client.initialize()

    results = []

    # Warmup
    print("[Warmup]")
    for i in range(3):
        client.send_request(
            [{"role": "user", "content": "hi"}],
            max_tokens=5
        )

    # Benchmark
    print(f"\n[Benchmark] {len(prompts)} prompts, {num_runs} execuções cada")

    for prompt_idx, prompt in enumerate(prompts):
        prompt_name = prompt["name"]
        prompt_text = prompt["content"]
        max_tokens = prompt.get("max_tokens", 10)

        print(f"\n{prompt_name}:")
        prompt_results = []

        for i in range(num_runs):
            start = time.time()
            response = client.send_request(
                [{"role": "user", "content": prompt_text}],
                max_tokens=max_tokens
            )
            elapsed_ms = (time.time() - start) * 1000

            if response["success"]:
                print(f"  Run {i+1}: {elapsed_ms:.1f}ms (processing: {response['processing_time_ms']}ms)")
                prompt_results.append({
                    "latency_ms": elapsed_ms,
                    "processing_time_ms": response["processing_time_ms"],
                    "success": True
                })
            else:
                print(f"  Run {i+1}: FAILED - {response.get('error_message', 'Unknown error')}")
                prompt_results.append({
                    "latency_ms": elapsed_ms,
                    "success": False
                })

        results.append({
            "prompt": prompt_name,
            "runs": prompt_results
        })

    client.close()
    return results


def main():
    """Função principal"""
    import argparse

    parser = argparse.ArgumentParser(description="DDS Client Benchmark")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain")
    parser.add_argument("--runs", type=int, default=5, help="Número de execuções por prompt")
    parser.add_argument("--model", type=str, default="phi4-mini", help="Modelo a usar")
    args = parser.parse_args()

    # Prompts de teste (B1: Simple, B2: Medium, B3: Complex)
    prompts = [
        {"name": "Simple (B1)", "content": "What is 2+2?", "max_tokens": 10},
        {"name": "Medium (B2)", "content": "Explain machine learning in a few sentences.", "max_tokens": 20},
        {"name": "Complex (B3)", "content": "Write a detailed technical explanation of how neural networks work, including backpropagation.", "max_tokens": 30},
    ]

    print("=" * 50)
    print("DDS Client Benchmark - Fluxo 100% DDS")
    print(f"Domain: {args.domain}, Runs: {args.runs}")
    print("=" * 50)

    results = asyncio.run(run_benchmark(prompts, args.model, args.domain, args.runs))

    # Imprimir resumo
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)

    for result in results:
        name = result["prompt"]
        runs = [r["latency_ms"] for r in result["runs"] if r["success"]]
        if runs:
            avg = sum(runs) / len(runs)
            print(f"{name}: {avg:.1f}ms avg ({len(runs)}/{args.runs} successful)")
        else:
            print(f"{name}: FAILED")

    return results


if __name__ == "__main__":
    main()
