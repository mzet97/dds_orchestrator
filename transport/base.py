"""
Camada de Transporte — interface abstrata.

Esta classe formaliza a "camada de transporte abstrata" descrita na
Seção sec:impl_servidor da dissertação v3: uma interface única com três
realizações intercambiáveis (HTTP, gRPC, DDS). O dispatcher do servidor
(server.py) deve consultar esta interface em vez de ramificar por
protocolo com if/elif — uma migração que acontecerá em lotes a partir do
scaffolding introduzido aqui (padrão strangler fig).

A assinatura é intencionalmente minimalista: cada método representa
exatamente um contrato que os três transportes já entregam no código
histórico. Toda a especificidade (endereçamento, serialização, controle
de fluxo) fica encapsulada nos adaptadores concretos.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional


@dataclass
class TransportResult:
    """Resultado normalizado de um dispatch entre os três transportes."""
    success: bool
    content: str = ""
    agent_id: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    error_message: str = ""
    raw: Optional[Any] = None  # objeto cru do transporte (TaskResponse, dict, etc.)


class TransportAdapter(ABC):
    """Interface única para os três transportes (HTTP, gRPC, DDS).

    Implementações concretas estão em `http_adapter.py`, `grpc_adapter.py`
    e `dds_adapter.py`. A migração completa do dispatcher do `server.py`
    para consultar este protocolo será feita em passadas incrementais.
    """

    @abstractmethod
    def name(self) -> str:
        """Nome curto do transporte: 'http', 'grpc' ou 'dds'."""

    @abstractmethod
    def is_available(self) -> bool:
        """True se o transporte está configurado e pronto para despachar."""

    @abstractmethod
    async def dispatch(self, task: dict, agent: Any, timeout_ms: int) -> TransportResult:
        """Despacha uma TaskRequest ao `agent` via este transporte.

        Args:
            task: dicionário com as chaves mínimas: `task_id`, `messages`,
                  `max_tokens`, `temperature`, `priority`.
            agent: AgentInfo ou objeto equivalente com endereçamento do agente.
            timeout_ms: timeout total para a operação.

        Returns:
            TransportResult preenchido com o resultado normalizado.
        """

    async def stream_dispatch(self, task: dict, agent: Any,
                              timeout_ms: int) -> AsyncIterator[TransportResult]:
        """Variante streaming opcional; emite chunks incrementais.

        Implementação default: converte `dispatch` em um iterador de um
        único elemento. Transportes que suportam streaming nativo (DDS,
        gRPC server streaming, HTTP SSE) devem sobrescrever.
        """
        result = await self.dispatch(task, agent, timeout_ms)
        yield result
