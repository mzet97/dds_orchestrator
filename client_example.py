"""
DDS-LLM Orchestrator Client Examples
Demonstra como conectar ao orchestrator via diferentes protocolos
"""

import asyncio
import aiohttp
import requests
from typing import List, Dict, Any


class OrchestratorClient:
    """Cliente base para连接到 orchestrator"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    async def health_check(self) -> dict:
        """Verifica saúde do orchestrator"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as resp:
                return await resp.json()

    async def chat(self, messages: List[Dict], **kwargs) -> dict:
        """Envia mensagem de chat"""
        payload = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.7),
            "task_type": kwargs.get("task_type", "chat"),
            "priority": kwargs.get("priority", 5),
            "requires_vision": kwargs.get("requires_vision", False),
            "requires_embedding": kwargs.get("requires_embedding", False),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/chat", json=payload) as resp:
                return await resp.json()

    async def generate(self, prompt: str, **kwargs) -> dict:
        """Gera texto a partir de prompt"""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)

    async def get_task_status(self, task_id: str) -> dict:
        """Verifica status de uma tarefa"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/tasks/{task_id}") as resp:
                return await resp.json()

    async def list_agents(self) -> dict:
        """Lista agentes disponíveis"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/agents") as resp:
                return await resp.json()


class SyncOrchestratorClient:
    """Cliente síncrono (sem async)"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    def health_check(self) -> dict:
        """Verifica saúde do orchestrator"""
        resp = requests.get(f"{self.base_url}/health")
        return resp.json()

    def chat(self, messages: List[Dict], **kwargs) -> dict:
        """Envia mensagem de chat"""
        payload = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.7),
            "task_type": kwargs.get("task_type", "chat"),
        }
        resp = requests.post(f"{self.base_url}/chat", json=payload)
        return resp.json()

    def generate(self, prompt: str, **kwargs) -> dict:
        """Gera texto a partir de prompt"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def list_agents(self) -> dict:
        """Lista agentes disponíveis"""
        resp = requests.get(f"{self.base_url}/agents")
        return resp.json()


# Exemplos de uso
async def example_async():
    """Exemplo de uso assíncrono"""
    client = OrchestratorClient("http://localhost:8080")

    # Check health
    health = await client.health_check()
    print(f"Health: {health}")

    # Chat simple
    response = await client.chat([
        {"role": "user", "content": "Olá, como você está?"}
    ])
    print(f"Chat response: {response}")

    # List agents
    agents = await client.list_agents()
    print(f"Agents: {agents}")


def example_sync():
    """Exemplo de uso síncrono"""
    client = SyncOrchestratorClient("http://localhost:8080")

    # Check health
    health = client.health_check()
    print(f"Health: {health}")

    # Chat simple
    response = client.chat([
        {"role": "user", "content": "Olá, como você está?"}
    ])
    print(f"Chat response: {response}")


# CLI Example
def example_cli():
    """Exemplo de CLI"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python client_example.py <prompt>")
        return

    prompt = " ".join(sys.argv[1:])
    client = SyncOrchestratorClient()

    print(f"Sending prompt: {prompt}")
    response = client.generate(prompt)
    print(f"Response: {response}")


if __name__ == "__main__":
    # example_sync()
    example_cli()
