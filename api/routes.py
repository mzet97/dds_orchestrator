#!/usr/bin/env python3
"""
API Routes for DDS-LLM-Orchestrator
"""
import asyncio
import json
import time
import uuid
from typing import List, Optional

import aiohttp
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from models import (
    AgentInfo,
    AgentRegistration,
    AgentState,
    AgentTaskRequest,
    AgentTaskResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    StreamChunk,
    TaskType,
)
from registry import AgentRegistry
from scheduler import TaskScheduler
from context import ContextManager
from dds_client import DDSClient
from http_client import HTTPClient


# Create router
router = APIRouter()


# ============================================
# STATE (will be injected by main.py)
# ============================================

registry: AgentRegistry = None
scheduler: TaskScheduler = None
context_manager: ContextManager = None
dds_client: DDSClient = None
http_client: HTTPClient = None
start_time: int = 0


def init_routes(
    reg: AgentRegistry,
    sched: TaskScheduler,
    ctx: ContextManager,
    dds: DDSClient,
    http: HTTPClient,
):
    """Initialize routes with dependencies"""
    global registry, scheduler, context_manager, dds_client, http_client, start_time
    registry = reg
    scheduler = sched
    context_manager = ctx
    dds_client = dds
    http_client = http
    start_time = int(time.time())


# ============================================
# HEALTH
# ============================================


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    stats = await registry.get_stats()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agents_registered=stats["total_agents"],
        agents_online=stats["online_agents"],
        uptime_seconds=int(time.time()) - start_time,
    )


# ============================================
# AGENTS
# ============================================


@router.post("/agents/register")
async def register_agent(registration: AgentRegistration):
    """Register a new agent"""
    registration.registered_at = int(time.time())
    agent_id = await registry.register_agent(registration)
    return {"agent_id": agent_id, "status": "registered"}


@router.get("/agents", response_model=List[AgentInfo])
async def list_agents():
    """List all registered agents"""
    return await registry.get_all_agents()


@router.get("/agents/{agent_id}", response_model=AgentInfo)
async def get_agent(agent_id: str):
    """Get agent by ID"""
    agent = await registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.delete("/agents/{agent_id}")
async def unregister_agent(agent_id: str):
    """Unregister an agent"""
    await registry.unregister_agent(agent_id)
    return {"status": "unregistered"}


@router.post("/agents/{agent_id}/task", response_model=AgentTaskResponse)
async def send_task_to_agent(agent_id: str, task: AgentTaskRequest):
    """Send task to specific agent"""
    # Get agent
    agent = await registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Check if agent is available
    if agent.state != AgentState.IDLE or agent.idle_slots == 0:
        raise HTTPException(status_code=503, detail="Agent is busy")

    # Send task via HTTP (fallback)
    async with http_client:
        # Build messages
        messages = [m.model_dump() for m in task.messages]

        response = await http_client.send_chat_request(
            host=agent.hostname,
            port=agent.port,
            messages=messages,
            model=agent.model,
        )

        if response:
            return response

    raise HTTPException(status_code=500, detail="Failed to send task")


# ============================================
# CHAT COMPLETIONS (OpenAI Compatible)
# ============================================


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completions endpoint (OpenAI compatible)"""
    # Handle streaming
    if request.stream:
        return await chat_completions_stream(request)

    # Find available agent
    agent = await registry.select_agent(requirements={"model": request.model})

    if not agent:
        raise HTTPException(status_code=503, detail="No available agents")

    # Create task
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    task = AgentTaskRequest(
        task_id=task_id,
        requester_id="api",
        task_type=TaskType.CHAT,
        messages=request.messages,
        priority=5,
        timeout_ms=request.max_tokens * 100,  # Estimate
    )

    # Send via HTTP
    async with http_client:
        messages_dict = [m.model_dump() for m in request.messages]
        response = await http_client.send_chat_request(
            host=agent.hostname,
            port=agent.port,
            messages=messages_dict,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

    if not response or not response.success:
        raise HTTPException(status_code=500, detail=response.error_message if response else "Task failed")

    # Build response
    response_message = ChatMessage(role="assistant", content=response.content)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=response_message,
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.prompt_tokens + response.completion_tokens,
        },
    )


@router.websocket("/v1/chat/completions")
async def chat_completions_websocket(websocket: WebSocket):
    """WebSocket endpoint for streaming chat completions"""
    await websocket.accept()

    try:
        # Receive the request
        data = await websocket.receive_json()
        request = ChatCompletionRequest(**data)

        # Find available agent from registry
        agent = await registry.select_agent(requirements={"model": request.model})
        if not agent:
            await websocket.send_json({
                "error": {"message": "No agents available", "type": "server_error"}
            })
            return

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # Send initial response with role
        await websocket.send_json({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "",
                },
                "finish_reason": None,
            }],
        })

        # Send request to agent and relay response
        async with http_client:
            messages_dict = [m.model_dump() for m in request.messages]
            response = await http_client.send_chat_request(
                host=agent.hostname,
                port=agent.port,
                messages=messages_dict,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

        if response and response.success:
            content = response.content
            # Stream content in chunks
            chunk_size = 4  # words per chunk
            words = content.split()
            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i:i + chunk_size]) + " "
                await websocket.send_json({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": chunk_text,
                        },
                        "finish_reason": None,
                    }],
                })
                await asyncio.sleep(0.02)
        else:
            error_msg = response.error_message if response else "Agent request failed"
            await websocket.send_json({
                "error": {"message": error_msg, "type": "agent_error"}
            })
            return

        # Send final message
        await websocket.send_json({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "error": {
                    "message": str(e),
                    "type": "streaming_error",
                }
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.post("/v1/chat/completions_stream")
async def chat_completions_stream(request: ChatCompletionRequest):
    """HTTP endpoint for streaming chat completions (SSE)"""
    import json
    from fastapi.responses import StreamingResponse

    # Find available agent
    agent = await registry.select_agent(requirements={"model": request.model})
    if not agent:
        raise HTTPException(status_code=503, detail="No available agents")

    async def generate():
        """Generate streaming response using SSE"""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # Send initial chunk with role
        yield "data: " + json.dumps({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }],
        }) + "\n\n"

        # Try real SSE streaming from agent first
        try:
            url = f"http://{agent.hostname}:{agent.port}/v1/chat/completions"
            messages_dict = [m.model_dump() for m in request.messages]
            payload = {
                "messages": messages_dict,
                "stream": True,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload,
                                       timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status == 200 and resp.content_type == "text/event-stream":
                        # Real SSE pass-through from agent
                        async for line in resp.content:
                            decoded = line.decode().strip()
                            if decoded.startswith("data:"):
                                yield decoded + "\n\n"
                            elif decoded == "":
                                continue
                    else:
                        # Agent doesn't support SSE streaming, fall back to chunked response
                        data = await resp.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if not content:
                            content = data.get("response", "")

                        # Send content as word chunks
                        words = content.split()
                        for word in words:
                            yield "data: " + json.dumps({
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": word + " "},
                                    "finish_reason": None,
                                }],
                            }) + "\n\n"

        except Exception as e:
            # Fallback: request via HTTP client (non-streaming), then chunk it
            try:
                async with http_client:
                    messages_dict = [m.model_dump() for m in request.messages]
                    response = await http_client.send_chat_request(
                        host=agent.hostname,
                        port=agent.port,
                        messages=messages_dict,
                        model=request.model,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    )

                if response and response.success:
                    content = response.content
                    words = content.split()
                    for word in words:
                        yield "data: " + json.dumps({
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": word + " "},
                                "finish_reason": None,
                            }],
                        }) + "\n\n"
                else:
                    yield "data: " + json.dumps({
                        "error": {"message": str(e), "type": "streaming_error"}
                    }) + "\n\n"
            except Exception as inner_e:
                yield "data: " + json.dumps({
                    "error": {"message": str(inner_e), "type": "streaming_error"}
                }) + "\n\n"

        # Send final chunk
        yield "data: " + json.dumps({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }) + "\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ============================================
# MODELS
# ============================================


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models"""
    agents = await registry.get_online_agents()

    # Get unique models
    models_seen = set()
    model_list = []

    for agent in agents:
        if agent.model not in models_seen:
            models_seen.add(agent.model)
            model_list.append(
                ModelInfo(
                    id=agent.model,
                    owned_by="local",
                )
            )

    return ModelListResponse(data=model_list)


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get model info"""
    agents = await registry.find_agents_by_model(model_id)

    if not agents:
        raise HTTPException(status_code=404, detail="Model not found")

    agent = agents[0]

    return {
        "id": model_id,
        "object": "model",
        "owned_by": "local",
        "agent": {
            "agent_id": agent.agent_id,
            "hostname": agent.hostname,
            "state": agent.state.value,
            "idle_slots": agent.idle_slots,
        },
    }


# ============================================
# CONTEXTS
# ============================================


@router.post("/contexts")
async def create_context(user_id: str, initial_message: Optional[ChatMessage] = None):
    """Create a new conversation context"""
    context_id = await context_manager.create_context(
        user_id=user_id,
        initial_message=initial_message,
    )
    return {"context_id": context_id}


@router.get("/contexts/{context_id}/messages")
async def get_context_messages(context_id: str):
    """Get messages in context"""
    messages = await context_manager.get_messages(context_id)
    return {"context_id": context_id, "messages": [m.model_dump() for m in messages]}


@router.post("/contexts/{context_id}/messages")
async def add_context_message(context_id: str, message: ChatMessage):
    """Add message to context"""
    success = await context_manager.add_message(context_id, message)
    if not success:
        raise HTTPException(status_code=404, detail="Context not found")
    return {"status": "added"}


@router.delete("/contexts/{context_id}")
async def delete_context(context_id: str):
    """Delete context"""
    success = await context_manager.clear_context(context_id)
    if not success:
        raise HTTPException(status_code=404, detail="Context not found")
    return {"status": "deleted"}


