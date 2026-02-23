#!/usr/bin/env python3
"""
DDS Client for Orchestrator
Communicates with agents via CycloneDDS
"""
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from models import (
    AgentRegistration,
    AgentState,
    AgentStatus,
    AgentTaskRequest,
    AgentTaskResponse,
)


# Try to import cyclonedds, fallback to mock if not available
try:
    import cyclonedds
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.core import Qos, Policy
    from cyclonedds.pub import DataWriter
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from cyclonedds.util import duration
    DDS_AVAILABLE = True
except ImportError:
    DDS_AVAILABLE = False


# ============================================
# DDS MESSAGE TYPES
# ============================================


@dataclass
class DDSAgentRegistration:
    """DDS type for agent registration"""
    agent_id: str = ""
    hostname: str = ""
    port: int = 0
    model: str = ""
    model_path: str = ""
    vram_available_mb: int = 0
    vram_total_mb: int = 0
    slots_idle: int = 0
    slots_total: int = 0
    vision_enabled: bool = False
    reasoning_enabled: bool = False
    registered_at: int = 0

    def to_model(self) -> AgentRegistration:
        return AgentRegistration(
            agent_id=self.agent_id,
            hostname=self.hostname,
            port=self.port,
            model=self.model,
            model_path=self.model_path,
            vram_available_mb=self.vram_available_mb,
            vram_total_mb=self.vram_total_mb,
            slots_idle=self.slots_idle,
            slots_total=self.slots_total,
            vision_enabled=self.vision_enabled,
            reasoning_enabled=self.reasoning_enabled,
            registered_at=self.registered_at,
        )

    @classmethod
    def from_model(cls, model: AgentRegistration) -> "DDSAgentRegistration":
        return cls(
            agent_id=model.agent_id,
            hostname=model.hostname,
            port=model.port,
            model=model.model,
            model_path=model.model_path,
            vram_available_mb=model.vram_available_mb,
            vram_total_mb=model.vram_total_mb,
            slots_idle=model.slots_idle,
            slots_total=model.slots_total,
            vision_enabled=model.vision_enabled,
            reasoning_enabled=model.reasoning_enabled,
            registered_at=model.registered_at,
        )


@dataclass
class DDSAgentStatus:
    """DDS type for agent status"""
    agent_id: str = ""
    state: str = "idle"
    current_slots: int = 0
    idle_slots: int = 1
    memory_usage_mb: int = 0
    vram_usage_mb: int = 0
    current_model: str = ""
    last_heartbeat: int = 0

    def to_model(self) -> AgentStatus:
        return AgentStatus(
            agent_id=self.agent_id,
            state=AgentState(self.state),
            current_slots=self.current_slots,
            idle_slots=self.idle_slots,
            memory_usage_mb=self.memory_usage_mb,
            vram_usage_mb=self.vram_usage_mb,
            current_model=self.current_model,
            last_heartbeat=self.last_heartbeat,
        )


@dataclass
class DDSTaskRequest:
    """DDS type for task request"""
    task_id: str = ""
    requester_id: str = ""
    task_type: str = "chat"
    messages_json: str = "[]"
    priority: int = 5
    timeout_ms: int = 30000
    requires_context: bool = False
    context_id: str = ""
    created_at: int = 0

    def to_model(self) -> AgentTaskRequest:
        from .models import ChatMessage, TaskType
        messages = [ChatMessage(**m) for m in json.loads(self.messages_json)]
        return AgentTaskRequest(
            task_id=self.task_id,
            requester_id=self.requester_id,
            task_type=TaskType(self.task_type),
            messages=messages,
            priority=self.priority,
            timeout_ms=self.timeout_ms,
            requires_context=self.requires_context,
            context_id=self.context_id if self.context_id else None,
            created_at=self.created_at,
        )

    @classmethod
    def from_model(cls, model: AgentTaskRequest) -> "DDSTaskRequest":
        messages_json = json.dumps([m.model_dump() for m in model.messages])
        return cls(
            task_id=model.task_id,
            requester_id=model.requester_id,
            task_type=model.task_type.value,
            messages_json=messages_json,
            priority=model.priority,
            timeout_ms=model.timeout_ms,
            requires_context=model.requires_context,
            context_id=model.context_id or "",
            created_at=model.created_at,
        )


@dataclass
class DDSTaskResponse:
    """DDS type for task response"""
    task_id: str = ""
    agent_id: str = ""
    content: str = ""
    is_final: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    processing_time_ms: int = 0
    success: bool = True
    error_message: str = ""
    created_at: int = 0

    def to_model(self) -> AgentTaskResponse:
        return AgentTaskResponse(
            task_id=self.task_id,
            agent_id=self.agent_id,
            content=self.content,
            is_final=self.is_final,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            processing_time_ms=self.processing_time_ms,
            success=self.success,
            error_message=self.error_message,
            created_at=self.created_at,
        )

    @classmethod
    def from_model(cls, model: AgentTaskResponse) -> "DDSTaskResponse":
        return cls(
            task_id=model.task_id,
            agent_id=model.agent_id,
            content=model.content,
            is_final=model.is_final,
            prompt_tokens=model.prompt_tokens,
            completion_tokens=model.completion_tokens,
            processing_time_ms=model.processing_time_ms,
            success=model.success,
            error_message=model.error_message,
            created_at=model.created_at,
        )


# ============================================
# DDS CLIENT
# ============================================


class DDSClient:
    """
    DDS Client for communicating with agents
    Uses CycloneDDS for pub/sub communication
    """

    def __init__(self, domain_id: int = 0):
        self.domain_id = domain_id
        self.dds_available = DDS_AVAILABLE

        if not self.dds_available:
            print("Warning: CycloneDDS not available, using HTTP fallback mode")
            self.participant = None
            return

        # Initialize DDS
        try:
            self._init_dds()
        except Exception as e:
            print(f"Warning: Failed to initialize DDS: {e}")
            self.dds_available = False
            self.participant = None

    def _init_dds(self):
        """Initialize DDS entities"""
        # DomainParticipant
        self.participant = DomainParticipant(self.domain_id)

        # Topics
        self.topic_register = Topic(
            self.participant, "agent/register", DDSAgentRegistration
        )
        self.topic_status = Topic(
            self.participant, "agent/status", DDSAgentStatus
        )
        self.topic_request = Topic(
            self.participant, "agent/request", DDSTaskRequest
        )
        self.topic_response = Topic(
            self.participant, "agent/response", DDSTaskResponse
        )

        # QoS - Reliable for requests/responses
        self.qos_reliable = Qos(
            Policy.Reliability.Reliable(duration(seconds=10)),
            Policy.Durability.Volatile,
            Policy.History.KeepLast(1),
        )

        # QoS - BestEffort for status (high frequency)
        self.qos_best_effort = Qos(
            Policy.Reliability.BestEffort,
            Policy.Durability.Volatile,
            Policy.History.KeepLast(5),
        )

        # Writers
        self.writer_request = DataWriter(self.participant, self.topic_request, self.qos_reliable)
        self.writer_register = DataWriter(self.participant, self.topic_register, self.qos_reliable)

        # Readers
        self.reader_status = DataReader(self.participant, self.topic_status, self.qos_best_effort)
        self.reader_response = DataReader(self.participant, self.topic_response, self.qos_reliable)

        print(f"[OK] DDS Client initialized (domain: {self.domain_id})")

    def publish_registration(self, registration: AgentRegistration):
        """Publish agent registration"""
        if not self.dds_available or not self.writer_register:
            return

        dds_data = DDSAgentRegistration.from_model(registration)
        self.writer_register.write(dds_data)

    def publish_task_request(self, task: AgentTaskRequest):
        """Publish task request to agents"""
        if not self.dds_available or not self.writer_request:
            return

        dds_data = DDSTaskRequest.from_model(task)
        self.writer_request.write(dds_data)

    def get_status_updates(self, timeout_ms: int = 100) -> List[AgentStatus]:
        """Get status updates from agents"""
        if not self.dds_available or not self.reader_status:
            return []

        try:
            samples = self.reader_status.take(timeout=duration(milliseconds=timeout_ms))
            return [s.data.to_model() for s in samples]
        except Exception:
            return []

    def get_responses(self, timeout_ms: int = 100) -> List[AgentTaskResponse]:
        """Get task responses"""
        if not self.dds_available or not self.reader_response:
            return []

        try:
            samples = self.reader_response.take(timeout=duration(milliseconds=timeout_ms))
            return [s.data.to_model() for s in samples]
        except Exception:
            return []

    def close(self):
        """Close DDS connections"""
        if self.participant:
            self.participant.close()


# ============================================
# MOCK DDS CLIENT (Fallback)
# ============================================


class MockDDSClient:
    """Mock DDS client when CycloneDDS is not available"""

    def __init__(self, domain_id: int = 0):
        self.domain_id = domain_id
        self.dds_available = False
        self.registrations: Dict[str, AgentRegistration] = {}
        self.statuses: Dict[str, AgentStatus] = {}
        self.pending_requests: Dict[str, AgentTaskRequest] = {}
        self.responses: Dict[str, AgentTaskResponse] = {}
        print("Mock DDS Client initialized (fallback mode)")

    def publish_registration(self, registration: AgentRegistration):
        """Store registration locally"""
        self.registrations[registration.agent_id] = registration
        print(f"Mock: Agent {registration.agent_id} registered")

    def publish_task_request(self, task: AgentTaskRequest):
        """Store task request"""
        self.pending_requests[task.task_id] = task
        print(f"Mock: Task {task.task_id} published")

    def get_status_updates(self, timeout_ms: int = 100) -> List[AgentStatus]:
        """Get cached statuses"""
        return list(self.statuses.values())

    def get_responses(self, timeout_ms: int = 100) -> List[AgentTaskResponse]:
        """Get cached responses"""
        return list(self.responses.values())

    def close(self):
        """No-op for mock"""
        pass


# ============================================
# FACTORY
# ============================================


def create_dds_client(domain_id: int = 0, mock: bool = False) -> Any:
    """Factory function to create DDS client"""
    if mock or not DDS_AVAILABLE:
        return MockDDSClient(domain_id)
    return DDSClient(domain_id)
