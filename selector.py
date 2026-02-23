"""
Agent Selector - Seleciona o agente especializado correto para cada requisição
"""

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Tipos de tarefa suportados"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    VISION = "vision"
    TOOL_CALL = "tool_call"


@dataclass
class SelectionCriteria:
    """Critérios para seleção de agente"""
    task_type: TaskType = TaskType.CHAT
    requires_vision: bool = False
    requires_embedding: bool = False
    preferred_model: Optional[str] = None
    priority: int = 5  # 1 = alta, 10 = baixa


@dataclass
class AgentMetrics:
    """Métricas do agente para seleção"""
    agent_id: str
    specialization: str
    current_load: int = 0
    max_load: int = 10
    success_rate: float = 1.0
    avg_response_time_ms: float = 0.0
    is_healthy: bool = True


class AgentSelector:
    """
    Seleciona o melhor agente para uma requisição baseado em:
    - Especialização do agente (text, vision, embedding)
    - Carga atual do agente
    - Taxa de sucesso
    - Tempo médio de resposta
    """

    def __init__(self):
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self._lock = asyncio.Lock()

    async def register_agent(self, agent_id: str, specialization: str, max_load: int = 10):
        """Registra um agente no seletor"""
        async with self._lock:
            self.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                specialization=specialization.lower(),
                max_load=max_load
            )
            logger.info(f"Agente {agent_id} registrado com especialização {specialization}")

    async def unregister_agent(self, agent_id: str):
        """Remove um agente do seletor"""
        async with self._lock:
            if agent_id in self.agent_metrics:
                del self.agent_metrics[agent_id]
                logger.info(f"Agente {agent_id} removido do seletor")

    async def update_metrics(self, agent_id: str, **kwargs):
        """Atualiza métricas de um agente"""
        async with self._lock:
            if agent_id not in self.agent_metrics:
                logger.warning(f"Tentativa de atualizar métricas de agente desconhecido: {agent_id}")
                return

            metrics = self.agent_metrics[agent_id]
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)

    async def select_agent(self, criteria: SelectionCriteria) -> Optional[str]:
        """
        Seleciona o melhor agente para os critérios fornecidos

        Retorna:
            agent_id do agente selecionado, ou None se nenhum disponível
        """
        async with self._lock:
            # Filtra agentes por especialização
            suitable_agents = []

            for agent_id, metrics in self.agent_metrics.items():
                if not metrics.is_healthy:
                    continue

                if metrics.current_load >= metrics.max_load:
                    # Agente sobrecarregado
                    continue

                # Verifica especialização
                if self._matches_specialization(metrics.specialization, criteria):
                    suitable_agents.append(metrics)

            if not suitable_agents:
                logger.warning(f"Nenhum agente disponível para critérios: {criteria}")
                return None

            # Seleciona o melhor agente usando estratégia
            selected = self._select_best_agent(suitable_agents, criteria)

            if selected:
                # Incrementa carga
                selected.current_load += 1
                logger.info(f"Agente selecionado: {selected.agent_id} (carga: {selected.current_load}/{selected.max_load})")

            return selected.agent_id if selected else None

    def _matches_specialization(self, agent_spec: str, criteria: SelectionCriteria) -> bool:
        """Verifica se especialização do agente corresponde aos critérios"""
        # Mapeamento de tipos de tarefa para especialização
        task_to_spec = {
            TaskType.CHAT: ["text", "generic"],
            TaskType.COMPLETION: ["text", "generic"],
            TaskType.EMBEDDING: ["embedding"],
            TaskType.VISION: ["vision"],
            TaskType.TOOL_CALL: ["text", "generic"],
        }

        required_specs = task_to_spec.get(criteria.task_type, ["generic"])

        # Verifica flags específicas
        if criteria.requires_vision and "vision" not in required_specs:
            required_specs.append("vision")

        if criteria.requires_embedding and "embedding" not in required_specs:
            required_specs.append("embedding")

        return agent_spec in required_specs

    def _select_best_agent(self, agents: List[AgentMetrics], criteria: SelectionCriteria) -> Optional[AgentMetrics]:
        """
        Seleciona o melhor agente usando estratégia de least-loaded

        Critérios de seleção (em ordem):
        1. Carga menor (mais disponível)
        2. Taxa de sucesso maior
        3. Tempo de resposta menor
        """
        if not agents:
            return None

        # Ordena por múltiplos critérios
        sorted_agents = sorted(
            agents,
            key=lambda a: (
                a.current_load,  # Menor carga primeiro
                -a.success_rate,  # Maior taxa de sucesso primeiro
                a.avg_response_time_ms  # Menor tempo primeiro
            )
        )

        return sorted_agents[0]

    async def release_agent(self, agent_id: str):
        """Libera um agente (decrementa carga)"""
        async with self._lock:
            if agent_id in self.agent_metrics:
                metrics = self.agent_metrics[agent_id]
                if metrics.current_load > 0:
                    metrics.current_load -= 1
                    logger.debug(f"Agente {agent_id} liberado (carga: {metrics.current_load})")

    def get_available_agents(self) -> List[dict]:
        """Retorna lista de agentes disponíveis"""
        result = []
        for agent_id, metrics in self.agent_metrics.items():
            if metrics.is_healthy and metrics.current_load < metrics.max_load:
                result.append({
                    "agent_id": agent_id,
                    "specialization": metrics.specialization,
                    "available_slots": metrics.max_load - metrics.current_load,
                    "success_rate": metrics.success_rate,
                    "avg_response_time_ms": metrics.avg_response_time_ms,
                })
        return result


class LoadBalancer:
    """Balanceador de carga para distribuição de requisições"""

    def __init__(self, strategy: str = "least_loaded"):
        """
        Args:
            strategy: least_loaded, round_robin, weighted
        """
        self.strategy = strategy
        self._round_robin_index = 0
        self._round_robin_lock = asyncio.Lock()

    async def select_agent(self, agents: List[dict]) -> Optional[dict]:
        """Seleciona agente usando estratégia de balanceamento"""
        if not agents:
            return None

        if self.strategy == "least_loaded":
            return min(agents, key=lambda a: a.get("current_load", 0))

        elif self.strategy == "round_robin":
            async with self._round_robin_lock:
                agent = agents[self._round_robin_index % len(agents)]
                self._round_robin_index += 1
                return agent

        elif self.strategy == "weighted":
            # Agentes com mais slots disponíveis têm maior peso
            total_slots = sum(a.get("available_slots", 1) for a in agents)
            import random
            r = random.uniform(0, total_slots)
            cumulative = 0
            for agent in agents:
                cumulative += agent.get("available_slots", 1)
                if cumulative >= r:
                    return agent

        # Default: least_loaded
        return min(agents, key=lambda a: a.get("current_load", 0))
