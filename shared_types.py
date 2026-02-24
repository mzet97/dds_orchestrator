# Shared DDS types for Orchestrator-Agent communication
# This module should be imported by both orchestrator and agent

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import bounded_str, int32, int64, uint8
from dataclasses import dataclass

@dataclass
class TaskRequestType(IdlStruct):
    """Task request from orchestrator to agent"""
    task_id: bounded_str[256] = ""
    requester_id: bounded_str[256] = ""
    task_type: bounded_str[64] = ""
    messages_json: bounded_str[16384] = ""
    priority: int32 = 0
    timeout_ms: int32 = 0
    requires_context: int = 0
    context_id: bounded_str[256] = ""
    created_at: int64 = 0


@dataclass
class TaskResponseType(IdlStruct):
    """Task response from agent to orchestrator"""
    task_id: bounded_str[256] = ""
    agent_id: bounded_str[256] = ""
    content: bounded_str[16384] = ""
    is_final: int = 0
    prompt_tokens: int32 = 0
    completion_tokens: int32 = 0
    processing_time_ms: int32 = 0
    success: int = 0
    error_message: bounded_str[1024] = ""


@dataclass
class AgentRegistrationType(IdlStruct):
    """Agent registration message"""
    agent_id: bounded_str[256] = ""
    hostname: bounded_str[256] = ""
    port: int32 = 0
    model: bounded_str[256] = ""
    vram_available_mb: int32 = 0
    slots_idle: int32 = 0
    vision_enabled: bool = False
    reasoning_enabled: bool = False
    registered_at: int64 = 0


@dataclass
class AgentStatusType(IdlStruct):
    """Agent status message"""
    agent_id: bounded_str[256] = ""
    state: bounded_str[64] = ""
    current_slots: int32 = 0
    idle_slots: int32 = 0
    memory_usage_mb: int32 = 0
    vram_usage_mb: int32 = 0
    current_model: bounded_str[256] = ""
    last_heartbeat: int64 = 0


@dataclass
class ClientRequestType(IdlStruct):
    """Client request message type"""
    request_id: bounded_str[256] = ""
    client_id: bounded_str[256] = ""
    task_type: bounded_str[64] = ""
    messages_json: bounded_str[16384] = ""
    priority: int32 = 0
    timeout_ms: int32 = 0
    requires_context: int = 0
    created_at: int64 = 0


@dataclass
class ClientResponseType(IdlStruct):
    """Client response message type"""
    request_id: bounded_str[256] = ""
    client_id: bounded_str[256] = ""
    content: bounded_str[16384] = ""
    is_final: int = 0
    prompt_tokens: int32 = 0
    completion_tokens: int32 = 0
    processing_time_ms: int32 = 0
    success: int = 0
    error_message: bounded_str[1024] = ""
