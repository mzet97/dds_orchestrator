"""
Orchestrator HTTP Server
Provides REST API for clients to interact with the orchestration system
"""

import asyncio
import json
import logging
import sys
import time
import uuid
from typing import Dict, List, Optional
from aiohttp import web

from config import OrchestratorConfig
from registry import AgentRegistry, AgentInfo
from scheduler import TaskScheduler, Task, TaskPriority
from selector import AgentSelector, SelectionCriteria, TaskType
from dds import DDSLayer, AgentTaskRequest, ClientTaskRequest

logger = logging.getLogger(__name__)


class OrchestratorServer:
    """Main orchestrator server"""

    def __init__(self, config: OrchestratorConfig,
                 registry: AgentRegistry,
                 scheduler: TaskScheduler,
                 dds_layer: DDSLayer,
                 selector: AgentSelector = None,
                 grpc_layer=None,
                 fuzzy_engine=None):
        self.config = config
        self.registry = registry
        self.scheduler = scheduler
        self.dds = dds_layer
        self.grpc = grpc_layer  # optional gRPC layer (mirrors DDS layer interface)
        self.selector = selector or AgentSelector()
        self.fuzzy = fuzzy_engine  # optional fuzzy decision engine

        self.app = None
        self.runner = None
        self.site = None

        # Background tasks
        self._heartbeat_task = None
        self._cleanup_task = None
        self._dds_client_task = None
        self._response_dispatch_task = None
        self._grpc_dispatch_task = None
        self._client_response_dispatch_task = None
        self._cleanup_lock = asyncio.Lock()
        self._inflight_dds_tasks: set = set()

    async def start(self):
        """Start the orchestrator server"""
        # Setup routes
        self.app = web.Application()

        # Health and status
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/status', self.handle_status)

        # Agent management
        self.app.router.add_get('/agents', self.handle_list_agents)
        self.app.router.add_get('/agents/{agent_id}', self.handle_get_agent)
        self.app.router.add_post('/agents/register', self.handle_register_agent)
        self.app.router.add_post('/agents/{agent_id}/heartbeat', self.handle_agent_heartbeat)
        self.app.router.add_delete('/agents/{agent_id}', self.handle_unregister_agent)

        # API v1 compatibility
        self.app.router.add_get('/api/v1/agents', self.handle_list_agents)
        self.app.router.add_get('/api/v1/agents/{agent_id}', self.handle_get_agent)
        self.app.router.add_post('/api/v1/agents/register', self.handle_register_agent)
        self.app.router.add_post('/api/v1/agents/{agent_id}/heartbeat', self.handle_agent_heartbeat)
        self.app.router.add_delete('/api/v1/agents/{agent_id}', self.handle_unregister_agent)

        # Task management
        self.app.router.add_post('/chat', self.handle_chat)
        self.app.router.add_post('/generate', self.handle_generate)
        self.app.router.add_get('/tasks/{task_id}', self.handle_get_task)
        self.app.router.add_delete('/tasks/{task_id}', self.handle_cancel_task)
        self.app.router.add_get('/tasks', self.handle_list_tasks)

        # API v1 routes (OpenAI compatible)
        self.app.router.add_post('/v1/chat/completions', self.handle_chat)
        self.app.router.add_post('/v1/completions', self.handle_generate)
        self.app.router.add_post('/api/v1/chat/completions', self.handle_chat)

        # DDS topics (if enabled)
        if self.dds.is_available():
            self.app.router.add_post('/dds/publish', self.handle_dds_publish)

        # Start HTTP server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, self.config.host, self.config.port)
        await self.site.start()

        logger.info(f"Server started on {self.config.host}:{self.config.port}")

        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        # DDS client request handler
        self._dds_client_task = asyncio.create_task(self._dds_client_loop())
        # Start WaitSet-based dispatchers (zero-polling, event-driven)
        # Falls back to legacy polling if WaitSet not available
        self.dds.start_waitset_dispatchers()
        # Legacy dispatch tasks kept as idle fallbacks (sleep forever)
        self._response_dispatch_task = asyncio.create_task(self.dds.dispatch_agent_responses())
        if hasattr(self.dds, 'dispatch_client_responses'):
            self._client_response_dispatch_task = asyncio.create_task(self.dds.dispatch_client_responses())

        # Start gRPC layer if available
        if self.grpc and self.grpc.is_available():
            # Wire client request handler so gRPC clients can submit tasks
            self.grpc._client_request_handler = self._process_grpc_client_request_sync
            await self.grpc.start()
            self._grpc_dispatch_task = asyncio.create_task(self.grpc.dispatch_agent_responses())

    async def stop(self):
        """Stop the orchestrator server"""
        for task in (self._heartbeat_task, self._cleanup_task,
                     self._dds_client_task, self._response_dispatch_task,
                     self._grpc_dispatch_task, self._client_response_dispatch_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Cancel inflight DDS client tasks
        for t in list(self._inflight_dds_tasks):
            if not t.done():
                t.cancel()
        if self._inflight_dds_tasks:
            await asyncio.gather(*self._inflight_dds_tasks, return_exceptions=True)
            self._inflight_dds_tasks.clear()

        if self.grpc:
            await self.grpc.stop()

        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

        logger.info("Server stopped")

    async def _heartbeat_loop(self):
        """Background task to check agent heartbeats"""
        while True:
            try:
                await asyncio.sleep(10)
                stale_ids = await self.registry.remove_stale_agents()
                for agent_id in (stale_ids or []):
                    try:
                        await self.selector.unregister_agent(agent_id)
                    except Exception as e:
                        logger.warning(f"Failed to unregister stale agent {agent_id} from selector: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    async def _dds_client_loop(self):
        """Background task to process DDS client requests"""
        logger.info("DDS client loop started")
        while True:
            try:
                await asyncio.sleep(0.001)  # Poll every 1ms

                if not self.dds.is_available():
                    continue

                # Read client requests from DDS
                client_requests = await self.dds.read_client_requests(timeout_ms=100)

                if client_requests:
                    logger.debug(f"Received {len(client_requests)} client requests via DDS")

                for req in client_requests:
                    # Filter out empty/phantom DDS samples (dispose notifications, etc.)
                    request_id = getattr(req, "request_id", "")
                    if not request_id:
                        logger.debug("Skipping empty DDS client request (no request_id)")
                        continue
                    messages_json = getattr(req, "messages_json", "")
                    if not messages_json or messages_json == "[]":
                        logger.debug(f"Skipping empty DDS client request {request_id} (no messages)")
                        continue
                    # Process valid requests concurrently (tracked for clean shutdown)
                    task = asyncio.create_task(self._process_dds_client_request(req))
                    self._inflight_dds_tasks.add(task)
                    task.add_done_callback(lambda t, rid=request_id: self._handle_task_completion(t, rid))
                    task.add_done_callback(self._inflight_dds_tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in DDS client loop: {e}")

    def _handle_task_completion(self, task, request_id):
        """Handle completion of a background task"""
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info(f"Task for request {request_id} was cancelled")
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}", exc_info=True)

    async def _process_dds_client_request(self, req):
        """Process a DDS client request (req is an IDL struct, not a dict)"""
        import json

        request_id = getattr(req, "request_id", "unknown")
        client_id = getattr(req, "client_id", "unknown")
        messages_json = getattr(req, "messages_json", "[]")

        try:
            messages = json.loads(messages_json)
        except Exception as e:
            logger.warning(f"Failed to parse messages_json, using raw content: {e}")
            messages = [{"role": "user", "content": messages_json}]

        logger.info(f"Processing DDS client request: {request_id}")

        # Get available agent
        agents = await self.registry.get_available_agents()
        if not agents:
            # Send error response
            await self._send_dds_client_response(request_id, client_id, "", success=0, error="No agents available")
            return

        # Select agent: fuzzy if enabled, otherwise max idle slots
        agent, fuzzy_qos, fuzzy_strategy = self._select_with_fuzzy(
            agents, messages=messages, priority=5)
        logger.info(f"Selected agent: {agent.agent_id} (fuzzy_qos={fuzzy_qos}, strategy={fuzzy_strategy})")

        # Acquire agent slot (status auto-derived: idle if slots remain, busy if 0)
        if not await self.registry.adjust_slots(agent.agent_id, delta=-1):
            # Agent was removed between selection and slot acquisition
            await self._send_dds_client_response(request_id, client_id, "", success=0, error="Agent no longer available")
            return

        try:
            # Send task to agent via DDS
            dds_request = AgentTaskRequest(
                task_id=request_id,
                requester_id="orchestrator",
                task_type="chat",
                messages=messages,
                priority=5,
                timeout_ms=60000,
                requires_context=False,
                max_tokens=50,
                temperature=0.7,
            )
            # Register waiter BEFORE publishing to avoid race where response arrives first
            self.dds.prepare_agent_response_waiter(request_id)
            try:
                await self.dds.publish_agent_request(dds_request, qos_profile=fuzzy_qos)

                # Wait for agent response
                agent_response = await self.dds.wait_for_agent_response(request_id, timeout_ms=120000)
            except Exception:
                # Clean up waiter on failure
                self.dds._pending_agent_responses.pop(request_id, None)
                raise

            # Send response back to client via DDS
            # agent_response can be dict (timeout error) or IdlStruct (DDS response)
            if isinstance(agent_response, dict):
                content = agent_response.get("content", "")
                success = agent_response.get("success", False)
                prompt_tokens = agent_response.get("prompt_tokens", 0)
                completion_tokens = agent_response.get("completion_tokens", 0)
                processing_time_ms = agent_response.get("processing_time_ms", 0)
                error = agent_response.get("error", "")
            else:
                content = getattr(agent_response, "content", "")
                success = getattr(agent_response, "success", False)
                prompt_tokens = getattr(agent_response, "prompt_tokens", 0)
                completion_tokens = getattr(agent_response, "completion_tokens", 0)
                processing_time_ms = getattr(agent_response, "processing_time_ms", 0)
                error = getattr(agent_response, "error_message", "")

            await self._send_dds_client_response(
                request_id, client_id, content,
                success=success, error=error,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                processing_time_ms=processing_time_ms,
            )

            # Update agent metrics for fuzzy feedback
            await self.registry.update_response_metrics(
                agent.agent_id,
                latency_ms=processing_time_ms or 0,
                success=bool(content),
            )
        finally:
            # Always release agent slot
            await self.registry.adjust_slots(agent.agent_id, delta=+1)

    async def _send_dds_client_response(self, request_id, client_id, content, success=True, error="", prompt_tokens=0, completion_tokens=0, processing_time_ms=0):
        """Send response to client via DDS using IDL-generated ClientResponse type"""
        from orchestrator import ClientResponse

        response = ClientResponse(
            request_id=request_id,
            client_id=client_id,
            content=content,
            is_final=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            processing_time_ms=processing_time_ms,
            success=success,
            error_message=error,
        )

        await self.dds.publish_client_response(response)

    def _process_grpc_client_request_sync(self, request_id, messages, model="",
                                          max_tokens=50, temperature=0.7,
                                          priority=5, timeout_ms=120000,
                                          stream=False) -> dict:
        """Process a client request arriving via gRPC — fully synchronous.

        Called from the sync gRPC server thread pool.
        Routes: Client --gRPC--> Orchestrator --gRPC--> Agent --HTTP--> llama-server
        """
        import grpc as _grpc
        import os as _os

        logger.info(f"Processing gRPC client request (sync): {request_id}")

        # Thread-safe agent selection (lock pre-created in AgentRegistry.__init__)
        with self.registry._thread_lock:
            agents = [a for a in self.registry.agents.values()
                      if a.slots_idle > 0 and a.status in ("idle", "busy")]
            if not agents:
                return {"content": "", "success": False, "error": "No agents available"}

            agent, fuzzy_qos, fuzzy_strategy = self._select_with_fuzzy(
                agents, messages=messages, priority=priority)
            # Atomic slot decrement under lock
            agent.slots_idle = max(0, agent.slots_idle - 1)
            if agent.slots_idle == 0:
                agent.status = "busy"

        logger.info(f"Selected agent for gRPC: {agent.agent_id} (fuzzy_qos={fuzzy_qos}, strategy={fuzzy_strategy})")

        # Build gRPC request to agent
        _proto_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "proto")
        if _proto_dir not in sys.path:
            sys.path.insert(0, _proto_dir)
        from proto import orchestrator_pb2 as _pb2
        from proto import orchestrator_pb2_grpc as _pb2_grpc

        agent_grpc_url = getattr(agent, "grpc_address", None)
        if not agent_grpc_url:
            agent_grpc_url = f"{agent.hostname}:50053"

        proto_req = _pb2.AgentTaskRequest(
            task_id=request_id,
            requester_id="orchestrator",
            task_type="chat",
            priority=priority,
            timeout_ms=timeout_ms,
            requires_context=False,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for msg in messages:
            proto_msg = proto_req.messages.add()
            proto_msg.role = msg.get("role", "user")
            proto_msg.content = msg.get("content", "")

        logger.info(f"gRPC calling agent at {agent_grpc_url}")

        try:
            import grpc as _grpc

            # Direct sync gRPC call — works because we're in a ThreadPoolExecutor
            # thread from the sync grpc.server (not grpc.aio)
            channel = _grpc.insecure_channel(agent_grpc_url)
            stub = _pb2_grpc.OrchestratorAgentServiceStub(channel)

            resp = stub.SubmitTask(proto_req, timeout=timeout_ms / 1000)
            channel.close()

            logger.info(f"gRPC response: success={resp.success}, len={len(resp.content)}")
            return {
                "content": resp.content,
                "success": resp.success,
                "error": resp.error_message or "",
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "processing_time_ms": resp.processing_time_ms,
            }

        except Exception as e:
            logger.error(f"gRPC agent call failed: {e}")
            return {"content": "", "success": False, "error": str(e)}
        finally:
            # Release agent slot under thread lock (same lock as selection)
            try:
                import threading
                if not hasattr(self.registry, '_thread_lock'):
                    self.registry._thread_lock = threading.Lock()
                with self.registry._thread_lock:
                    a = self.registry.agents.get(agent.agent_id)
                    if a:
                        a.slots_idle = min(a.slots_idle + 1, getattr(a, 'slots_total', 1))
                        if a.slots_idle > 0:
                            a.status = "idle"
            except Exception as e:
                logger.warning(f"Failed to release agent slot: {e}")

    async def _cleanup_loop(self):
        """Background task to cleanup old tasks"""
        while True:
            try:
                await asyncio.sleep(60)
                await self.scheduler.cleanup_old_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    # === Request Handlers ===

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "timestamp": time.time(),
            "dds_available": self.dds.is_available(),
        })

    async def handle_status(self, request: web.Request) -> web.Response:
        """Get system status"""
        registry_stats = await self.registry.get_stats()
        scheduler_stats = await self.scheduler.get_stats()
        return web.json_response({
            "registry": registry_stats,
            "scheduler": scheduler_stats,
            "dds_available": self.dds.is_available(),
        })

    async def handle_list_agents(self, request: web.Request) -> web.Response:
        """List all registered agents"""
        agents = await self.registry.get_all_agents()
        return web.json_response({
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "hostname": a.hostname,
                    "port": a.port,
                    "model": a.model,
                    "status": a.status,
                    "slots_idle": a.slots_idle,
                }
                for a in agents
            ]
        })

    async def handle_get_agent(self, request: web.Request) -> web.Response:
        """Get specific agent"""
        agent_id = request.match_info['agent_id']
        agent = await self.registry.get_agent(agent_id)

        if not agent:
            return web.json_response({"error": "Agent not found"}, status=404)

        return web.json_response(asdict(agent))

    async def handle_register_agent(self, request: web.Request) -> web.Response:
        """Register a new agent"""
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

        # Generate agent_id if not provided
        import uuid
        agent_id = data.get("agent_id") or str(uuid.uuid4())

        agent_info = AgentInfo(
            agent_id=agent_id,
            hostname=data.get("hostname", "unknown"),
            port=data.get("port", 8080),
            model=data.get("model", "unknown"),
            vram_available_mb=data.get("vram_available_mb", 0),
            slots_idle=data.get("slots_idle", 1),
            slots_total=data.get("slots_total", data.get("slots_idle", 1)),
            vision_enabled=data.get("vision_enabled", False),
            capabilities=data.get("capabilities", []),
            grpc_address=data.get("grpc_address", ""),
            agent_profile=data.get("agent_profile", "balanced"),
            gpu_type=data.get("gpu_type", ""),
        )

        # Auto-assign agent profile based on port if not provided
        if not agent_info.agent_profile or agent_info.agent_profile == "balanced":
            port = agent_info.port
            if port == 8081:
                agent_info.agent_profile = "fast"
            elif port == 8091:
                agent_info.agent_profile = "quality"
            elif port == 8092:
                agent_info.agent_profile = "balanced"
            else:
                agent_info.agent_profile = "balanced"

        await self.registry.register_agent(agent_info)
        logger.info(f"Agent {agent_id} registered with profile={agent_info.agent_profile}")

        # Registrar também no selector para seleção inteligente
        try:
            await self.selector.register_agent(
                agent_id=agent_info.agent_id,
                specialization="generic",
                max_load=agent_info.slots_total
            )
        except Exception as e:
            logger.warning(f"Failed to register in selector: {e}")

        return web.json_response({
            "success": True,
            "agent_id": agent_info.agent_id
        })

    async def handle_agent_heartbeat(self, request: web.Request) -> web.Response:
        """Handle agent heartbeat"""
        agent_id = request.match_info['agent_id']
        try:
            data = await request.json()
            status = data.get("status", "idle")
            slots_idle = data.get("slots_idle", 1)
        except Exception as e:
            logger.warning(f"Failed to parse heartbeat JSON, using defaults: {e}")
            status = "idle"
            slots_idle = 1

        success = await self.registry.update_heartbeat(agent_id, status=status, slots_idle=slots_idle)
        return web.json_response({"success": success})

    async def handle_unregister_agent(self, request: web.Request) -> web.Response:
        """Unregister an agent"""
        agent_id = request.match_info['agent_id']
        success = await self.registry.unregister_agent(agent_id)

        if not success:
            return web.json_response({"error": "Agent not found"}, status=404)

        # Remover também do selector
        await self.selector.unregister_agent(agent_id)

        return web.json_response({"success": True})

    async def handle_chat(self, request: web.Request) -> web.Response:
        """Handle chat request"""
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)
        messages = data.get("messages", [])

        if not messages:
            return web.json_response({"error": "No messages provided"}, status=400)

        # Get parameters
        max_tokens = data.get("max_tokens", self.config.default_max_tokens)
        temperature = data.get("temperature", self.config.default_temperature)
        priority = data.get("priority", TaskPriority.NORMAL.value)
        task_type = data.get("task_type", "chat")

        # Wait for an available agent (queue instead of 503 rejection)
        # Fuzzy engine selects best agent + QoS profile + strategy
        fuzzy_urgency = data.get("urgency")
        fuzzy_complexity = data.get("complexity")
        agent, fuzzy_qos, fuzzy_strategy = await self._wait_for_available_agent(
            timeout_s=300, messages=messages, priority=priority,
            urgency=fuzzy_urgency, complexity=fuzzy_complexity,
        )
        if not agent:
            return web.json_response({
                "error": "No agents available after timeout",
                "code": "NO_AGENTS"
            }, status=503)
        logger.info(f"Selected agent: {agent.agent_id} (qos={fuzzy_qos}, strategy={fuzzy_strategy})")

        # Create task
        task = Task(
            task_id=f"task-{str(uuid.uuid4())}",
            task_type="chat",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=TaskScheduler._map_priority(priority),
            assigned_agent_id=agent.agent_id,
        )

        # Update agent status (atomic slot decrement, status auto-derived)
        # Retry if another request grabbed the slot between wait and decrement
        for _retry in range(3):
            if await self.registry.adjust_slots(agent.agent_id, delta=-1):
                break
            agent, fuzzy_qos, fuzzy_strategy = await self._wait_for_available_agent(
                timeout_s=300, messages=messages, priority=priority)
            if not agent:
                return web.json_response({"error": "Agent no longer available", "code": "AGENT_UNAVAILABLE"}, status=503)
        else:
            return web.json_response({"error": "Agent no longer available", "code": "AGENT_UNAVAILABLE"}, status=503)

        # Track task (don't queue — we dispatch inline via DDS/HTTP below)
        await self.scheduler.track_task(task)

        # Try gRPC first (if agent has grpc_address), then DDS, then HTTP fallback
        transport_success = False
        response_data = {}
        _streaming_handled = False  # Set by SSE streaming path to prevent double slot release

        try:
            # === gRPC path (agent has grpc_address and orchestrator has grpc_layer) ===
            stream_requested = data.get("stream", False)
            if (not transport_success
                    and self.grpc and self.grpc.is_available()
                    and getattr(agent, "grpc_address", "")):
                try:
                    from grpc_layer import AgentTaskRequest as GRPCAgentTaskRequest
                    grpc_request = GRPCAgentTaskRequest(
                        task_id=task.task_id,
                        requester_id="orchestrator",
                        task_type=task.task_type,
                        messages=messages,
                        priority=priority,
                        timeout_ms=task.timeout_ms,
                        requires_context=task.requires_context,
                        stream=stream_requested,
                    )

                    if stream_requested:
                        # SSE streaming path via gRPC: forward chunks to client
                        self.grpc.prepare_stream_waiter(task.task_id)
                        # publish_agent_request will forward chunks to stream waiter
                        asyncio.create_task(self.grpc.publish_agent_request(
                            grpc_request,
                            agent_id=agent.agent_id,
                            agent_grpc_url=agent.grpc_address,
                        ))

                        sse_response = web.StreamResponse(
                            status=200,
                            reason='OK',
                            headers={
                                'Content-Type': 'text/event-stream',
                                'Cache-Control': 'no-cache',
                                'Connection': 'keep-alive',
                            }
                        )
                        await sse_response.prepare(request)

                        try:
                            async for chunk in self.grpc.stream_agent_response(task.task_id, timeout_ms=120000):
                                content = chunk.get("content", "")
                                is_final = chunk.get("is_final", False)

                                if is_final and not content:
                                    await sse_response.write(b"data: [DONE]\n\n")
                                    break

                                sse_data = json.dumps({
                                    "id": task.task_id,
                                    "object": "chat.completion.chunk",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": "stop" if is_final else None,
                                    }],
                                })
                                await sse_response.write(f"data: {sse_data}\n\n".encode())

                                if is_final:
                                    await sse_response.write(b"data: [DONE]\n\n")
                                    break
                        finally:
                            await self.registry.adjust_slots(agent.agent_id, delta=+1)
                            await self.scheduler.complete_task(task.task_id, response="streaming")
                            _streaming_handled = True

                        return sse_response

                    else:
                        # Non-streaming gRPC path
                        self.grpc.prepare_agent_response_waiter(task.task_id)
                        await self.grpc.publish_agent_request(
                            grpc_request,
                            agent_id=agent.agent_id,
                            agent_grpc_url=agent.grpc_address,
                        )
                        response_data = await self.grpc.wait_for_agent_response(task.task_id, timeout_ms=120000)
                        if isinstance(response_data, dict) and response_data.get("error"):
                            logger.warning(f"gRPC response error: {response_data.get('error')}")
                        else:
                            transport_success = True

                except Exception as e:
                    logger.warning(f"gRPC communication failed: {e}, trying DDS")

            # === DDS path ===
            if not transport_success:
                try:
                    dds_request = AgentTaskRequest(
                        task_id=task.task_id,
                        requester_id="orchestrator",
                        task_type=task.task_type,
                        messages=messages,
                        priority=priority,
                        timeout_ms=task.timeout_ms,
                        requires_context=task.requires_context,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=stream_requested,
                    )

                    # Map task priority to DDS TRANSPORT_PRIORITY
                    priority_map = {1: 0, 2: 5, 5: 5, 10: 10, 20: 20}
                    dds_priority = priority_map.get(priority, 0)

                    if stream_requested and self.dds.is_available():
                        # SSE streaming path: forward individual chunks to client
                        self.dds.prepare_stream_waiter(task.task_id)
                        await self.dds.publish_agent_request(dds_request, priority=dds_priority,
                                                              qos_profile=fuzzy_qos)

                        sse_response = web.StreamResponse(
                            status=200,
                            reason='OK',
                            headers={
                                'Content-Type': 'text/event-stream',
                                'Cache-Control': 'no-cache',
                                'Connection': 'keep-alive',
                            }
                        )
                        await sse_response.prepare(request)

                        try:
                            async for chunk in self.dds.stream_agent_response(task.task_id, timeout_ms=120000):
                                content = chunk.get("content", "")
                                is_final = chunk.get("is_final", False)

                                if is_final and not content:
                                    await sse_response.write(b"data: [DONE]\n\n")
                                    break

                                sse_data = json.dumps({
                                    "id": task.task_id,
                                    "object": "chat.completion.chunk",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": "stop" if is_final else None,
                                    }],
                                })
                                await sse_response.write(f"data: {sse_data}\n\n".encode())

                                if is_final:
                                    await sse_response.write(b"data: [DONE]\n\n")
                                    break
                        finally:
                            # Slot released here; outer finally skipped via _streaming_handled flag
                            await self.registry.adjust_slots(agent.agent_id, delta=+1)
                            await self.scheduler.complete_task(task.task_id, response="streaming")
                            _streaming_handled = True

                        return sse_response

                    else:
                        # Non-streaming DDS path
                        self.dds.prepare_agent_response_waiter(task.task_id)
                        await self.dds.publish_agent_request(dds_request, priority=dds_priority,
                                                              qos_profile=fuzzy_qos)
                        response_data = await self.dds.wait_for_agent_response(task.task_id, timeout_ms=120000)

                        content = ""
                        if isinstance(response_data, dict):
                            content = response_data.get("content", "")
                        else:
                            content = getattr(response_data, "content", "")

                        if isinstance(response_data, dict) and response_data.get("error") == "DDS not available":
                            logger.info("DDS not available, will try HTTP fallback")
                        else:
                            transport_success = True

                except Exception as e:
                    logger.warning(f"DDS communication failed: {e}, falling back to HTTP")
                    self.dds._pending_agent_responses.pop(task.task_id, None)
                    self.dds._pending_agent_responses.pop(f"stream_{task.task_id}", None)

            # If transport failed, use HTTP fallback
            if not transport_success:
                logger.info(f"Using HTTP fallback for task {task.task_id}")
                agent_url = f"http://{agent.hostname}:{agent.port}"
                logger.info(f"HTTP fallback calling agent at {agent_url}/chat")
                try:
                    import aiohttp as _aiohttp
                    async with _aiohttp.ClientSession() as _session:
                        async with _session.post(
                            f"{agent_url}/chat",
                            json={
                                "messages": messages,
                                "max_tokens": max_tokens,
                                "temperature": temperature,
                            },
                            timeout=_aiohttp.ClientTimeout(total=max(1, task.timeout_ms // 1000))
                        ) as _resp:
                            http_result = await _resp.json()

                    # Extract content from agent response
                    content = http_result.get("content", "")
                    if not content and "choices" in http_result:
                        content = http_result["choices"][0].get("message", {}).get("content", "")
                    if not content and "response" in http_result:
                        content = http_result["response"]

                    logger.info(f"HTTP result: content_len={len(content)}")
                    response_data = {
                        "content": content,
                        "processing_time_ms": http_result.get("processing_time_ms", 0),
                        "prompt_tokens": http_result.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": http_result.get("usage", {}).get("completion_tokens", 0),
                        "success": True,
                    }
                    transport_success = True
                except Exception as e:
                    logger.error(f"HTTP fallback also failed: {e}")
                    response_data = {"content": "", "error": str(e), "success": False}

            # Complete task with result - handle both dict and IdlStruct
            response_content = ""
            response_error = None
            if isinstance(response_data, dict):
                response_content = response_data.get("content", "")
                processing_time = response_data.get("processing_time_ms", 0)
                if not response_data.get("success", 1):
                    response_error = response_data.get("error", "Unknown error")
            else:
                response_content = getattr(response_data, "content", "")
                processing_time = getattr(response_data, "processing_time_ms", 0)
                if not getattr(response_data, "success", True):
                    response_error = getattr(response_data, "error_message", "Unknown error")
            logger.debug(f"Transport result: success={transport_success}, content_len={len(response_content)}")

            await self.scheduler.complete_task(
                task.task_id,
                response=response_content,
                error=response_error,
                processing_time_ms=processing_time
            )

            # Update agent metrics for fuzzy decision feedback
            await self.registry.update_response_metrics(
                agent.agent_id,
                latency_ms=processing_time,
                success=bool(response_content),
            )
        except asyncio.CancelledError:
            # Client disconnected — mark task as cancelled so it doesn't stay "running"
            await self.scheduler.cancel_task(task.task_id)
            raise
        finally:
            # Always restore slot — unless streaming path already handled it
            if not _streaming_handled:
                await self.registry.adjust_slots(agent.agent_id, delta=+1)

        # Return OpenAI-compatible response for /api/v1/chat/completions
        return web.json_response({
            "id": task.task_id,
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content,
                },
                "finish_reason": "stop" if response_content else "error",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "agent_id": agent.agent_id,
            "processing_time_ms": processing_time,
        })

    def _select_with_fuzzy(self, agents: list, messages: list = None,
                           priority: int = 5, urgency: int = None,
                           complexity: int = None) -> tuple:
        """Select agent using fuzzy engine if available, otherwise max(slots_idle).

        Returns: (agent, qos_profile, strategy)
          - qos_profile: "low_cost" | "balanced" | "critical" | None
          - strategy: "single" | "retry" | "fanout"
        """
        if self.fuzzy and agents:
            task_input = {
                "urgency": urgency or priority,
                "complexity": complexity,
                "messages": messages or [],
                "priority": priority,
            }
            decision = self.fuzzy.select(task_input, agents)
            agent = next((a for a in agents if a.agent_id == decision.agent_id), agents[0])
            return agent, decision.qos_profile, decision.strategy

        # Fallback: baseline selection
        agent = max(agents, key=lambda a: a.slots_idle)
        return agent, None, "single"

    async def _wait_for_available_agent(self, timeout_s=300, messages=None,
                                         priority=5, urgency=None, complexity=None):
        """Wait for an agent with idle slots, then select using fuzzy or baseline.

        Returns: (agent, qos_profile, strategy) or (None, None, None)
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        # First check without delay
        agents = await self.registry.get_available_agents()
        if agents:
            return self._select_with_fuzzy(agents, messages, priority, urgency, complexity)
        # Poll with exponential backoff: 5ms → 10ms → 20ms → 40ms → 50ms cap
        delay = 0.005
        while loop.time() < deadline:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 0.05)
            agents = await self.registry.get_available_agents()
            if agents:
                return self._select_with_fuzzy(agents, messages, priority, urgency, complexity)
        return None, None, None

    async def handle_generate(self, request: web.Request) -> web.Response:
        """Handle generate request"""
        # NOTE: este endpoint bypassa DDS, usa HTTP direto ao agente
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        prompt = data.get("prompt", "")

        if not prompt:
            return web.json_response({"error": "No prompt provided"}, status=400)

        # Converter para formato chat e processar
        data["messages"] = [{"role": "user", "content": prompt}]

        # Reusar lógica do handle_chat com dados convertidos
        messages = data["messages"]
        max_tokens = data.get("max_tokens", self.config.default_max_tokens)
        temperature = data.get("temperature", self.config.default_temperature)

        # Selecionar agente e processar
        available = await self.registry.get_available_agents()
        if not available:
            return web.json_response({"error": "No agents available"}, status=503)

        agent = max(available, key=lambda a: a.slots_idle)
        agent_url = f"http://{agent.hostname}:{agent.port}"

        if not await self.registry.adjust_slots(agent.agent_id, delta=-1):
            return web.json_response({"error": "Agent no longer available", "code": "AGENT_UNAVAILABLE"}, status=503)

        try:
            result = await self.dds.send_request_via_http(
                agent_url,
                {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
            )
        except Exception as e:
            return web.json_response({"error": f"Agent communication failed: {e}"}, status=502)
        finally:
            await self.registry.adjust_slots(agent.agent_id, delta=+1)

        return web.json_response(result)

    async def handle_get_task(self, request: web.Request) -> web.Response:
        """Get task status"""
        task_id = request.match_info['task_id']
        task = await self.scheduler.get_task(task_id)

        if not task:
            return web.json_response({"error": "Task not found"}, status=404)

        return web.json_response({
            "task_id": task.task_id,
            "status": task.status.value,
            "response": task.response,
            "error": task.error,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
        })

    async def handle_cancel_task(self, request: web.Request) -> web.Response:
        """Cancel a task"""
        task_id = request.match_info['task_id']
        success = await self.scheduler.cancel_task(task_id)

        if not success:
            return web.json_response({"error": "Task not found or cannot be cancelled"}, status=404)

        return web.json_response({"success": True})

    async def handle_list_tasks(self, request: web.Request) -> web.Response:
        """List tasks"""
        status_filter = request.query.get('status')

        if status_filter == 'running':
            tasks = await self.scheduler.get_running_tasks()
        elif status_filter == 'pending':
            tasks = await self.scheduler.get_pending_tasks()
        else:
            tasks = await self.scheduler.get_completed_tasks()

        return web.json_response({
            "tasks": [
                {
                    "task_id": t.task_id,
                    "status": t.status.value,
                    "task_type": t.task_type,
                    "created_at": t.created_at,
                }
                for t in tasks
            ]
        })

    async def handle_dds_publish(self, request: web.Request) -> web.Response:
        """Publish to DDS topic"""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        topic = data.get("topic")
        message = data.get("message")

        if not topic or not message:
            return web.json_response({"error": "topic and message required"}, status=400)

        await self.dds.publish(topic, message)

        return web.json_response({"success": True})


# Helper for agent info dict
def asdict(obj):
    """Convert dataclass to dict, handling Enum values for JSON serialization"""
    from enum import Enum
    if hasattr(obj, '__dataclass_fields__'):
        return {k: asdict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, list):
        return [asdict(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: asdict(v) for k, v in obj.items()}
    else:
        return obj
