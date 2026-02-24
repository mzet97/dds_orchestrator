# Shared DDS types for Client-Orchestrator communication
# This module should be imported by both client and orchestrator

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import bounded_str, int32
from dataclasses import dataclass

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
    created_at: int32 = 0


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
