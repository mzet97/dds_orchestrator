"""
gRPC Communication Layer for the Orchestrator.
Mirrors DDSLayer (dds.py) interface for fair protocol comparison.

Uses grpc.aio for native asyncio integration with the aiohttp server.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass

import grpc
import grpc.aio

logger = logging.getLogger(__name__)

# Import generated stubs lazily
_stubs_loaded = False
_pb2 = None
_pb2_grpc = None


def _ensure_stubs():
    """Load generated protobuf stubs on first use."""
    global _stubs_loaded, _pb2, _pb2_grpc
    if _stubs_loaded:
        return True
    try:
        # Add proto/ dir to sys.path so generated stubs can find each other
        # (orchestrator_pb2_grpc.py does 'import orchestrator_pb2' without prefix)
        import os, sys
        proto_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto")
        if proto_dir not in sys.path:
            sys.path.insert(0, proto_dir)
        from proto import orchestrator_pb2 as pb2
        from proto import orchestrator_pb2_grpc as pb2_grpc
        _pb2 = pb2
        _pb2_grpc = pb2_grpc
        _stubs_loaded = True
        return True
    except ImportError as e:
        logger.warning(f"gRPC stubs not available: {e}. Run: "
                       "python -m grpc_tools.protoc -I proto --python_out=proto "
                       "--grpc_python_out=proto proto/orchestrator.proto")
        return False


def _copy_messages_to_proto(messages, proto_req) -> None:
    """Copy chat messages (dict or obj with role/content) into proto_req.messages."""
    for msg in messages:
        pm = proto_req.messages.add()
        if isinstance(msg, dict):
            pm.role = msg.get("role", "user")
            pm.content = msg.get("content", "")
        else:
            pm.role = getattr(msg, "role", "user")
            pm.content = getattr(msg, "content", "")


@dataclass
class AgentTaskRequest:
    """Task request to agent — mirrors dds.AgentTaskRequest"""
    task_id: str
    requester_id: str
    task_type: str
    messages: list
    priority: int
    timeout_ms: int
    requires_context: bool
    context_id: str = ""
    stream: bool = False
    max_tokens: int = 50
    temperature: float = 0.7


@dataclass
class AgentTaskResponse:
    """Task response from agent — mirrors dds.AgentTaskResponse"""
    task_id: str
    agent_id: str
    content: str
    is_final: bool
    prompt_tokens: int
    completion_tokens: int
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None


class GRPCLayer:
    """gRPC communication layer — same interface as DDSLayer.

    The orchestrator uses this to communicate with agents via gRPC
    instead of DDS pub/sub topics.
    """

    def __init__(self, config):
        self.config = config
        self.grpc_available = False

        # Per-task waiters: task_id -> (asyncio.Event, [result])
        self._pending_agent_responses: Dict[str, Any] = {}

        # Agent channels: agent_id -> grpc.aio.Channel
        self._agent_channels: Dict[str, grpc.aio.Channel] = {}
        # Agent stubs: agent_id -> stub
        self._agent_stubs: Dict[str, Any] = {}

        # Async channel pool keyed by address for forward_to_instance round-robin.
        # N channels per address mitigates HTTP/2 head-of-line blocking under
        # concurrent inflight calls to the same agent.
        self._aio_channel_pool: Dict[str, list] = {}
        self._aio_rr_idx: Dict[str, int] = {}
        self._aio_pool_lock = asyncio.Lock()
        self._AIO_CHANNELS_PER_ADDR = 4

        # Server for receiving connections from agents
        self._server = None
        self._port = getattr(config, "grpc_port", 50052)

        # Client request handler — set by OrchestratorServer to route client
        # gRPC requests through the same registry/selector/agent pipeline.
        self._client_request_handler: Optional[Callable] = None
        # Reference to the main asyncio event loop (set during start())
        self._event_loop = None

        if getattr(config, "grpc_enabled", False):
            self._init_grpc()

    def _init_grpc(self):
        """Initialize gRPC"""
        if not _ensure_stubs():
            logger.warning("gRPC stubs not available, gRPC layer disabled")
            return

        self.grpc_available = True
        logger.info(f"gRPC layer initialized (port {self._port})")

    async def start(self):
        """Start gRPC server (sync, in dedicated thread) for agent AND client connections.

        Uses synchronous grpc.server in a background thread instead of grpc.aio
        to avoid event loop conflicts between the gRPC server and outgoing gRPC
        calls to agents (grpc.aio hangs when mixing server and client in same process).
        """
        if not self.grpc_available:
            return

        import concurrent.futures
        import asyncio

        self._event_loop = asyncio.get_running_loop()
        # 128 workers: each handles one sync gRPC call to agent (~0.5-2s blocking).
        # gRPC queues excess requests internally. Higher values waste stack memory
        # (~8MB/thread on Linux). If needed, switch to grpc.aio for true async.
        self._server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=128))

        # Agent-facing service
        agent_servicer_cls = _make_servicer_class()
        _pb2_grpc.add_OrchestratorAgentServiceServicer_to_server(
            agent_servicer_cls(self), self._server
        )

        # Client-facing service (uses sync handler with thread executor)
        client_servicer_cls = _make_client_servicer_class_sync()
        _pb2_grpc.add_ClientOrchestratorServiceServicer_to_server(
            client_servicer_cls(self), self._server
        )

        listen_addr = f"0.0.0.0:{self._port}"
        self._server.add_insecure_port(listen_addr)
        self._server.start()
        logger.info(f"gRPC server started on {listen_addr} (sync, agent + client services)")

    async def stop(self):
        """Stop gRPC server and close channels."""
        if self._server:
            self._server.stop(grace=5)
            self._server = None

        for channel in self._agent_channels.values():
            await channel.close()
        self._agent_channels.clear()
        self._agent_stubs.clear()

        # Close pooled async channels used by forward_to_instance
        async with self._aio_pool_lock:
            for addr, channels in self._aio_channel_pool.items():
                for ch in channels:
                    try:
                        await ch.close()
                    except Exception as e:
                        logger.warning(f"Error closing pooled aio channel for {addr}: {e}")
            self._aio_channel_pool.clear()
            self._aio_rr_idx.clear()

        self.grpc_available = False
        logger.info("gRPC layer stopped")

    async def _get_or_create_aio_channel(self, addr: str):
        """Round-robin pooled grpc.aio.Channel for the given address.

        Channels are created lazily and reused across calls. N channels per
        address (configured via ``_AIO_CHANNELS_PER_ADDR``) keep concurrent
        streams from serializing on HTTP/2 HOL blocking.
        """
        async with self._aio_pool_lock:
            channels = self._aio_channel_pool.get(addr)
            if channels is None:
                opts = [
                    ("grpc.max_send_message_length", 64 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                    ("grpc.keepalive_time_ms", 30000),
                ]
                channels = [
                    grpc.aio.insecure_channel(addr, options=opts)
                    for _ in range(self._AIO_CHANNELS_PER_ADDR)
                ]
                self._aio_channel_pool[addr] = channels
                self._aio_rr_idx[addr] = 0
            idx = self._aio_rr_idx[addr]
            self._aio_rr_idx[addr] = (idx + 1) % len(channels)
            return channels[idx]

    async def unregister_agent(self, agent_id: str, grpc_url: str = None):
        """Remove agent and close its gRPC channel(s) to prevent resource leaks.

        If grpc_url is provided, also purge the sync channel pool entries
        used by server.py for unary forwarding, plus the aio pool used by
        forward_to_instance.
        """
        if agent_id in self._agent_channels:
            try:
                await self._agent_channels[agent_id].close()
            except Exception as e:
                logger.warning(f"Error closing gRPC channel for agent {agent_id}: {e}")
            del self._agent_channels[agent_id]
        self._agent_stubs.pop(agent_id, None)

        if not grpc_url:
            return

        # Purge async pool entry (used by forward_to_instance)
        async with self._aio_pool_lock:
            aio_channels = self._aio_channel_pool.pop(grpc_url, [])
            self._aio_rr_idx.pop(grpc_url, None)
        for ch in aio_channels:
            try:
                await ch.close()
            except Exception as e:
                logger.warning(f"Error closing pooled aio channel for {grpc_url}: {e}")

        # Purge sync pool entry (used by _process_grpc_client_request_sync in server.py)
        srv = getattr(self, '_orchestrator_server', None)
        pool = getattr(srv, '_grpc_channel_pool', None)
        if pool is None or grpc_url not in pool:
            return
        pool_lock = getattr(srv, '_grpc_pool_lock', None)
        if pool_lock is not None:
            with pool_lock:
                channels = pool.pop(grpc_url, [])
                getattr(srv, '_grpc_rr_idx', {}).pop(grpc_url, None)
        else:
            channels = pool.pop(grpc_url, [])
        for ch in channels:
            try:
                ch.close()
            except Exception as e:
                logger.warning(f"Error closing pooled gRPC channel for {grpc_url}: {e}")

    def _get_or_create_stub(self, agent_id: str, agent_url: str):
        """Get or create a gRPC stub for an agent."""
        if agent_id in self._agent_stubs:
            return self._agent_stubs[agent_id]

        # Extract host:port from agent URL
        # agent_url is like "http://host:port" or "host:port"
        addr = agent_url.replace("http://", "").replace("https://", "")
        # Use gRPC port (agent's gRPC port, typically agent_http_port + 1000 or configured)
        # For now, use the address as-is (agent should expose gRPC on a known port)

        channel = grpc.aio.insecure_channel(addr)
        stub = _pb2_grpc.OrchestratorAgentServiceStub(channel)
        self._agent_channels[agent_id] = channel
        self._agent_stubs[agent_id] = stub
        logger.info(f"Created gRPC stub for agent {agent_id} at {addr}")
        return stub

    def prepare_agent_response_waiter(self, task_id: str):
        """Pre-register a waiter for the given task_id.
        Mirrors DDSLayer.prepare_agent_response_waiter().
        """
        if task_id not in self._pending_agent_responses:
            event = asyncio.Event()
            result_container = [None]
            self._pending_agent_responses[task_id] = (event, result_container)

    def prepare_stream_waiter(self, task_id: str):
        """Pre-register a streaming waiter for the given task_id.
        Mirrors DDSLayer.prepare_stream_waiter().
        """
        key = f"stream_{task_id}"
        if key not in self._pending_agent_responses:
            self._pending_agent_responses[key] = asyncio.Queue()

    async def stream_agent_response(self, task_id: str, timeout_ms: int = 120000):
        """Async generator that yields individual response chunks from agent.

        Chunks are pushed into an ``asyncio.Queue`` by ``publish_agent_request``
        as they arrive from the agent, so this generator blocks on ``queue.get``
        instead of 1 ms-polling a list.
        """
        key = f"stream_{task_id}"
        if key not in self._pending_agent_responses:
            self.prepare_stream_waiter(task_id)

        queue = self._pending_agent_responses[key]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (timeout_ms / 1000)

        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    yield {"content": "", "is_final": True, "error": "timeout"}
                    return
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    yield {"content": "", "is_final": True, "error": "timeout"}
                    return
                yield chunk
                if chunk.get("is_final", False):
                    return
        finally:
            # Ensures queue is released even if the consumer abandons the
            # generator before is_final or timeout (e.g. SSE client disconnect).
            self._pending_agent_responses.pop(key, None)

    async def publish_agent_request(self, request: AgentTaskRequest,
                                     agent_id: str = None,
                                     agent_grpc_url: str = None):
        """Send task request to agent via gRPC.

        Unlike DDS pub/sub, gRPC requires a direct connection to the agent.
        The agent_id and agent_grpc_url are used to find/create the stub.
        """
        if not self.grpc_available:
            logger.warning("gRPC not available, cannot send request")
            return

        if not agent_grpc_url:
            logger.error("agent_grpc_url required for gRPC publish_agent_request")
            return

        stub = self._get_or_create_stub(agent_id or "unknown", agent_grpc_url)

        # Convert to protobuf
        proto_req = _pb2.AgentTaskRequest(
            task_id=request.task_id,
            requester_id=request.requester_id,
            task_type=request.task_type,
            priority=request.priority,
            timeout_ms=request.timeout_ms,
            requires_context=request.requires_context,
            stream=request.stream,
        )
        _copy_messages_to_proto(request.messages, proto_req)

        try:
            if request.stream:
                # Server-streaming — forward chunks to stream waiter if present
                stream_key = f"stream_{request.task_id}"
                response_stream = stub.StreamTask(proto_req)

                if stream_key in self._pending_agent_responses:
                    # Per-chunk forwarding (SSE streaming path) via asyncio.Queue
                    queue = self._pending_agent_responses.get(stream_key)
                    async for proto_resp in response_stream:
                        chunk = {
                            "content": proto_resp.content,
                            "is_final": bool(proto_resp.is_final),
                            "prompt_tokens": proto_resp.prompt_tokens,
                            "completion_tokens": proto_resp.completion_tokens,
                            "processing_time_ms": proto_resp.processing_time_ms,
                        }
                        if queue is not None:
                            queue.put_nowait(chunk)
                        if proto_resp.is_final:
                            break
                else:
                    # Accumulate all content (non-SSE path)
                    accumulated_content = ""
                    async for proto_resp in response_stream:
                        accumulated_content += proto_resp.content
                        if proto_resp.is_final:
                            result = AgentTaskResponse(
                                task_id=proto_resp.task_id,
                                agent_id=proto_resp.agent_id,
                                content=accumulated_content,
                                is_final=True,
                                prompt_tokens=proto_resp.prompt_tokens,
                                completion_tokens=proto_resp.completion_tokens,
                                processing_time_ms=proto_resp.processing_time_ms,
                                success=proto_resp.success,
                                error_message=proto_resp.error_message or None,
                            )
                            self._resolve_waiter(request.task_id, result)
            else:
                # Unary
                proto_resp = await stub.SubmitTask(proto_req)
                result = AgentTaskResponse(
                    task_id=proto_resp.task_id,
                    agent_id=proto_resp.agent_id,
                    content=proto_resp.content,
                    is_final=proto_resp.is_final,
                    prompt_tokens=proto_resp.prompt_tokens,
                    completion_tokens=proto_resp.completion_tokens,
                    processing_time_ms=proto_resp.processing_time_ms,
                    success=proto_resp.success,
                    error_message=proto_resp.error_message or None,
                )
                self._resolve_waiter(request.task_id, result)

        except grpc.aio.AioRpcError as e:
            logger.error(f"gRPC request failed for task {request.task_id}: {e.code()} {e.details()}")
            result = AgentTaskResponse(
                task_id=request.task_id,
                agent_id=agent_id or "unknown",
                content="",
                is_final=True,
                prompt_tokens=0,
                completion_tokens=0,
                processing_time_ms=0,
                success=False,
                error_message=f"gRPC error: {e.code()} {e.details()}",
            )
            self._resolve_waiter(request.task_id, result)

    def _resolve_waiter(self, task_id: str, result):
        """Resolve a pending waiter with a result."""
        if task_id in self._pending_agent_responses:
            event, container = self._pending_agent_responses[task_id]
            container[0] = result
            event.set()

    async def wait_for_agent_response(self, task_id: str, timeout_ms: int = 60000) -> dict:
        """Wait for a specific agent response by task_id.
        Mirrors DDSLayer.wait_for_agent_response().
        """
        if not self.grpc_available:
            logger.warning("gRPC not available, cannot wait for response")
            return {"content": "", "error": "gRPC not available"}

        if task_id in self._pending_agent_responses:
            event, result_container = self._pending_agent_responses[task_id]
        else:
            event = asyncio.Event()
            result_container = [None]
            self._pending_agent_responses[task_id] = (event, result_container)

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_ms / 1000.0)
            result = result_container[0]
            if result is None:
                return {"content": "", "error": "No response received"}
            # Convert AgentTaskResponse to dict for compatibility
            if isinstance(result, AgentTaskResponse):
                return {
                    "task_id": result.task_id,
                    "agent_id": result.agent_id,
                    "content": result.content,
                    "is_final": result.is_final,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "processing_time_ms": result.processing_time_ms,
                    "success": result.success,
                    "error_message": result.error_message,
                }
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for gRPC response for task {task_id}")
            return {"content": "", "error": "Timeout waiting for response"}
        finally:
            self._pending_agent_responses.pop(task_id, None)

    async def forward_to_instance(self, agent_grpc_addr: str,
                                   request: AgentTaskRequest, task_id: str,
                                   timeout_s: int = 120) -> dict:
        """Forward a request to an agent via gRPC (OrchestratorAgentService.SubmitTask).

        Uses a per-address pooled async channel so concurrent requests do not
        pay the ~20-50 ms channel-setup cost and are not serialized by HTTP/2
        HOL blocking on a single channel.

        Flow: Orchestrator → gRPC → Agent (agent_llm_grpc.py) → gRPC → llama-server
        """
        import grpc as _grpc

        try:
            channel = await self._get_or_create_aio_channel(agent_grpc_addr)
            stub = _pb2_grpc.OrchestratorAgentServiceStub(channel)

            proto_req = _pb2.AgentTaskRequest(
                task_id=task_id,
                max_tokens=request.max_tokens if hasattr(request, 'max_tokens') else 20,
                temperature=request.temperature if hasattr(request, 'temperature') else 0.7,
                stream=False,
            )
            msgs = request.messages if isinstance(request.messages, list) else json.loads(request.messages_json)
            _copy_messages_to_proto(msgs, proto_req)

            proto_resp = await stub.SubmitTask(proto_req, timeout=timeout_s)

            content = proto_resp.content if proto_resp.content else ""
            logger.info(f"gRPC forward response: task={task_id} content_len={len(content)} "
                        f"success={proto_resp.success} error={proto_resp.error_message}")

            return {
                "task_id": proto_resp.task_id,
                "content": content,
                "is_final": True,
                "prompt_tokens": proto_resp.prompt_tokens,
                "completion_tokens": proto_resp.completion_tokens,
                "success": proto_resp.success if hasattr(proto_resp, 'success') else bool(content),
            }
        except _grpc.aio.AioRpcError as e:
            if e.code() == _grpc.StatusCode.DEADLINE_EXCEEDED:
                logger.error(f"gRPC timeout for task {task_id} to {agent_grpc_addr}")
                return {"content": "", "error": "gRPC timeout"}
            logger.error(f"gRPC forward to {agent_grpc_addr} failed: {e.code()} {e.details()}")
            return {"content": "", "error": f"{e.code()}: {e.details()}"}
        except Exception as e:
            logger.error(f"gRPC forward to {agent_grpc_addr} failed: {e}")
            return {"content": "", "error": str(e)}

    async def dispatch_agent_responses(self):
        """Background loop — for gRPC this is a no-op since responses arrive
        via direct RPC calls (not pub/sub polling). Kept for interface compatibility.
        """
        while True:
            await asyncio.sleep(3600)  # effectively idle

    def is_available(self) -> bool:
        """Check if gRPC is available"""
        return self.grpc_available

    def close(self):
        """Synchronous no-op — grpc.aio channels require `await self.stop()`.

        Kept so that legacy sync shutdown paths do not crash; real cleanup
        must go through `await self.stop()`.
        """
        self.grpc_available = False


def _make_servicer_class():
    """Create the agent-facing servicer class after stubs are loaded."""
    class _OrchestratorServicer(_pb2_grpc.OrchestratorAgentServiceServicer):
        """gRPC server-side handler for agents connecting to orchestrator."""

        def __init__(self, layer: GRPCLayer):
            self._layer = layer

        async def Heartbeat(self, request, context):
            """Handle agent heartbeat."""
            logger.debug(f"Heartbeat from agent {request.agent_id}: state={request.state}")
            return _pb2.HeartbeatResponse(
                acknowledged=True,
                message="OK"
            )

    return _OrchestratorServicer


def _make_client_servicer_class_sync():
    """Create sync client-facing servicer for use with grpc.server (not grpc.aio).

    The handler (_client_request_handler) is an async function in the main event loop.
    We schedule it from the gRPC thread pool using run_coroutine_threadsafe.
    """
    class _ClientServicerSync(_pb2_grpc.ClientOrchestratorServiceServicer):
        def __init__(self, layer: GRPCLayer):
            self._layer = layer

        def Chat(self, request, context):
            """Handle unary chat request from client (sync, runs in thread pool)."""
            handler = self._layer._client_request_handler
            if handler is None:
                return _pb2.ClientChatResponse(
                    request_id=request.request_id, content="",
                    is_final=True, success=False,
                    error_message="Client handler not configured",
                )

            try:
                messages = [{"role": m.role, "content": m.content} for m in request.messages]

                # Handler is now fully synchronous — call directly
                result = handler(
                    request_id=request.request_id,
                    messages=messages,
                    model=request.model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    priority=request.priority,
                    timeout_ms=request.timeout_ms or 120000,
                    stream=False,
                )

                return _pb2.ClientChatResponse(
                    request_id=request.request_id,
                    content=result.get("content", ""),
                    is_final=True,
                    prompt_tokens=result.get("prompt_tokens", 0),
                    completion_tokens=result.get("completion_tokens", 0),
                    processing_time_ms=result.get("processing_time_ms", 0),
                    success=result.get("success", False),
                    error_message=result.get("error", ""),
                    model=request.model,
                )
            except Exception as e:
                logger.error(f"Error processing gRPC client Chat: {e}", exc_info=True)
                return _pb2.ClientChatResponse(
                    request_id=request.request_id, content="",
                    is_final=True, success=False,
                    error_message=str(e),
                )

        def StreamChat(self, request, context):
            """Handle streaming chat by forwarding to agent's StreamTask."""
            import grpc as _grpc
            import os as _os
            import sys as _sys

            # Find an agent
            layer = self._layer
            registry = layer._orchestrator_server.registry if hasattr(layer, "_orchestrator_server") else None
            if registry is None:
                # fallback: try via global handler context — get from server
                yield _pb2.ClientChatResponse(
                    request_id=request.request_id, content="",
                    is_final=True, success=False,
                    error_message="No registry on grpc layer",
                )
                return

            with registry._thread_lock:
                # Prefer agents serving the requested model (when given);
                # fall back to any agent with idle slots. Pick the least
                # loaded to spread streaming load across agents instead of
                # piling onto agents[0] (previous behavior caused skewed
                # benchmarks and ignored the `model` field entirely).
                requested_model = (getattr(request, "model", "") or "").strip()
                all_available = [a for a in registry.agents.values()
                                 if a.slots_idle > 0 and a.status in ("idle", "busy")]
                agents = [a for a in all_available if a.model == requested_model] \
                         if requested_model else all_available
                if not agents:
                    agents = all_available  # no model match, fall back
                if not agents:
                    yield _pb2.ClientChatResponse(
                        request_id=request.request_id, content="",
                        is_final=True, success=False,
                        error_message="No agents available",
                    )
                    return
                agent = max(agents, key=lambda a: a.slots_idle)
                agent.slots_idle = max(0, agent.slots_idle - 1)
                if agent.slots_idle == 0:
                    agent.status = "busy"

            agent_url = getattr(agent, "grpc_address", None) or f"{agent.hostname}:50053"

            try:
                # Build agent task request
                _proto_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "proto")
                if _proto_dir not in _sys.path:
                    _sys.path.insert(0, _proto_dir)
                from proto import orchestrator_pb2 as _opb2
                from proto import orchestrator_pb2_grpc as _opb2_grpc

                proto_req = _opb2.AgentTaskRequest(
                    task_id=request.request_id,
                    requester_id="orchestrator",
                    task_type="chat",
                    priority=request.priority or 5,
                    timeout_ms=request.timeout_ms or 120000,
                    requires_context=False,
                    stream=True,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
                for msg in request.messages:
                    pm = proto_req.messages.add()
                    pm.role = msg.role
                    pm.content = msg.content

                # Reuse the orchestrator's sync gRPC channel pool so we do
                # not pay channel-setup cost per stream and avoid HTTP/2 HOL
                # blocking on a single per-stream channel.
                srv = getattr(layer, '_orchestrator_server', None)
                pool = getattr(srv, '_grpc_channel_pool', None) if srv else None
                pool_lock = getattr(srv, '_grpc_pool_lock', None) if srv else None
                rr_idx = getattr(srv, '_grpc_rr_idx', None) if srv else None
                channels_per = getattr(srv, '_GRPC_CHANNELS_PER_AGENT', 4) if srv else 4

                if pool is not None and pool_lock is not None:
                    with pool_lock:
                        if agent_url not in pool:
                            opts = [
                                ("grpc.max_send_message_length", 64 * 1024 * 1024),
                                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                                ("grpc.keepalive_time_ms", 30000),
                            ]
                            pool[agent_url] = [
                                _grpc.insecure_channel(agent_url, options=opts)
                                for _ in range(channels_per)
                            ]
                            rr_idx[agent_url] = 0
                        channels = pool[agent_url]
                        idx = rr_idx[agent_url]
                        channel = channels[idx % len(channels)]
                        rr_idx[agent_url] = idx + 1
                    owns_channel = False
                else:
                    channel = _grpc.insecure_channel(agent_url)
                    owns_channel = True

                stub = _opb2_grpc.OrchestratorAgentServiceStub(channel)
                try:
                    for agent_resp in stub.StreamTask(proto_req, timeout=120):
                        yield _pb2.ClientChatResponse(
                            request_id=request.request_id,
                            content=agent_resp.content,
                            is_final=bool(agent_resp.is_final),
                            prompt_tokens=agent_resp.prompt_tokens,
                            completion_tokens=agent_resp.completion_tokens,
                            processing_time_ms=agent_resp.processing_time_ms,
                            success=bool(agent_resp.success),
                            error_message=agent_resp.error_message or "",
                            model=request.model,
                        )
                        if agent_resp.is_final:
                            break
                finally:
                    if owns_channel:
                        channel.close()
            except Exception as e:
                logger.error(f"StreamChat agent forward failed: {e}", exc_info=True)
                yield _pb2.ClientChatResponse(
                    request_id=request.request_id, content="",
                    is_final=True, success=False,
                    error_message=str(e),
                )
            finally:
                registry.adjust_slots_sync(agent.agent_id, delta=+1)

    return _ClientServicerSync
