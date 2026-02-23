# Shared DDS types for Client-Orchestrator communication
# This module should be imported by both client and orchestrator

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import bounded_str, int32

class ClientRequestType(IdlStruct):
    """Client request message type"""
    request_id: bounded_str[256]
    client_id: bounded_str[256]
    task_type: bounded_str[64]
    messages_json: bounded_str[16384]
    priority: int32
    timeout_ms: int32
    requires_context: bool
    created_at: int32


class ClientResponseType(IdlStruct):
    """Client response message type"""
    request_id: bounded_str[256]
    client_id: bounded_str[256]
    content: bounded_str[16384]
    is_final: bool
    prompt_tokens: int32
    completion_tokens: int32
    processing_time_ms: int32
    success: bool
    error_message: bounded_str[1024]
