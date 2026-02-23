#!/usr/bin/env python3
"""
Pydantic models for DDS-LLM-Orchestrator
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================
# ENUMS
# ============================================


class TaskType(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"


class AgentState(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


# ============================================
# CHAT MODELS (OpenAI Compatible)
# ============================================


class ChatMessage(BaseModel):
    """Chat message"""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """Chat completion request (OpenAI compatible)"""
    model: str = Field(..., description="Model to use")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(256, ge=1, le=4096)
    stream: bool = Field(False, description="Enable streaming")
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    n: int = Field(1, ge=1)
    stop: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    """Choice in chat completion"""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """Chat completion response (OpenAI compatible)"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    )


# ============================================
# AGENT MODELS
# ============================================


class AgentRegistration(BaseModel):
    """Agent registration information"""
    agent_id: str
    hostname: str
    port: int
    model: str
    model_path: str
    vram_available_mb: int
    vram_total_mb: int = 0
    slots_idle: int
    slots_total: int = 1
    vision_enabled: bool = False
    reasoning_enabled: bool = False
    capabilities: List[str] = Field(default_factory=list)
    registered_at: int = 0


class AgentStatus(BaseModel):
    """Agent status"""
    agent_id: str
    state: AgentState = AgentState.IDLE
    current_slots: int = 0
    idle_slots: int = 1
    memory_usage_mb: int = 0
    vram_usage_mb: int = 0
    current_model: str = ""
    last_heartbeat: int = 0


class AgentInfo(BaseModel):
    """Full agent information"""
    agent_id: str
    hostname: str
    port: int
    model: str
    state: AgentState = AgentState.IDLE
    idle_slots: int = 1
    total_slots: int = 1
    vram_available_mb: int = 0
    vram_total_mb: int = 0
    vision_enabled: bool = False
    reasoning_enabled: bool = False
    registered_at: int = 0
    last_heartbeat: int = 0


# ============================================
# TASK MODELS
# ============================================


class AgentTaskRequest(BaseModel):
    """Task request to an agent"""
    task_id: str
    requester_id: str
    task_type: TaskType = TaskType.CHAT
    messages: List[ChatMessage]
    priority: int = Field(5, ge=1, le=10)
    timeout_ms: int = 30000
    requires_context: bool = False
    context_id: Optional[str] = None
    created_at: int = 0


class AgentTaskResponse(BaseModel):
    """Task response from an agent"""
    task_id: str
    agent_id: str
    content: str = ""
    is_final: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    processing_time_ms: int = 0
    success: bool = True
    error_message: str = ""
    created_at: int = 0


# ============================================
# ORCHESTRATOR MODELS
# ============================================


class OrchestratorCommand(BaseModel):
    """Command to orchestrator"""
    command_id: str
    target_agent_id: str = "all"
    action: str  # load_model, unload, priority, etc.
    payload: str = ""


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    version: str = "1.0.0"
    agents_registered: int = 0
    agents_online: int = 0
    uptime_seconds: int = 0


class ModelInfo(BaseModel):
    """Model information"""
    id: str
    object: str = "model"
    owned_by: str = "local"
    permission: List[Any] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    """List models response (OpenAI compatible)"""
    object: str = "list"
    data: List[ModelInfo] = Field(default_factory=list)


# ============================================
# STREAMS
# ============================================


class StreamChunk(BaseModel):
    """Streaming chunk"""
    task_id: str
    content: str
    is_final: bool = False
    completion_tokens: int = 0
