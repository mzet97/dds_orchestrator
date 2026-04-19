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
from context import ContextManager
from models import ChatMessage
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
                 fuzzy_engine=None,
                 instance_pool=None,
                 redis_mgr=None,
                 mongo_store=None,
                 backpressure=None):
        self.config = config
        self.registry = registry
        self.scheduler = scheduler
        self.dds = dds_layer
        self.grpc = grpc_layer  # optional gRPC layer (mirrors DDS layer interface)
        if self.grpc is not None:
            # Allow grpc_layer servicers to call back into the orchestrator
            # (used by StreamChat for agent selection / slot bookkeeping)
            try:
                self.grpc._orchestrator_server = self
            except Exception as e:
                logger.warning(f"Failed to set orchestrator on grpc layer: {e}")
        self.selector = selector or AgentSelector()
        self.fuzzy = fuzzy_engine  # optional fuzzy decision engine
        self.instance_pool = instance_pool
        self.redis_mgr = redis_mgr
        self.mongo_store = mongo_store
        self.backpressure = backpressure
        self.ctx = ContextManager(max_contexts=5000, max_messages_per_context=20)

        self.app = None
        self.runner = None
        self.site = None

        # gRPC channel pool: N channels per agent to mitigate HTTP/2 HOL blocking.
        # Thread-safe, initialized eagerly to avoid TOCTOU race.
        import threading as _threading
        self._grpc_channel_pool: dict[str, list] = {}  # url -> [ch1, ch2, ...]
        self._grpc_pool_lock = _threading.Lock()
        self._grpc_rr_idx: dict[str, int] = {}  # round-robin index per url
        self._GRPC_CHANNELS_PER_AGENT = 4

        # Background tasks
        self._heartbeat_task = None
        self._cleanup_task = None
        self._dds_client_task = None
        self._response_dispatch_task = None
        self._grpc_dispatch_task = None
        self._client_response_dispatch_task = None
        self._cleanup_lock = asyncio.Lock()
        self._inflight_dds_tasks: set = set()
        # Shared aiohttp session for pool HTTP calls (avoids per-request session overhead)
        self._pool_session = None

    async def start(self):
        """Start the orchestrator server"""
        # Store reference to the main aiohttp/asyncio loop. The sync gRPC
        # thread-pool handlers need this to schedule Redis-backed coroutines
        # on the SAME loop that owns the Redis connections. Creating a new
        # loop per call (via asyncio.run) would break Redis state and cause
        # a fixed ~10s spin in _process_grpc_client_request_sync.
        self._main_loop = asyncio.get_running_loop()

        # Let the registry wake async fair-waiters from sync slot releases
        # (gRPC sync servicer path). Without this, adjust_slots_sync adds
        # slots without notifying asyncio.Condition, so requests time out
        # at the 60s wait cap while slots actually sit idle.
        if hasattr(self.registry, "bind_main_loop"):
            self.registry.bind_main_loop(self._main_loop)

        # Shared aiohttp session — created once and reused across handlers.
        # Per-request ClientSession() costs ~1-3 ms of connector setup and
        # prevents HTTP keep-alive across requests, which is significant on
        # the hot path.
        import aiohttp as _aiohttp
        self._pool_session = _aiohttp.ClientSession(
            connector=_aiohttp.TCPConnector(limit=0, ttl_dns_cache=300),
            timeout=_aiohttp.ClientTimeout(total=self.config.task_timeout_seconds),
        )

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

        # Instance pool endpoints
        self.app.router.add_put('/api/v1/routing/algorithm', self.handle_set_algorithm)
        self.app.router.add_get('/api/v1/pool/status', self.handle_pool_status)
        self.app.router.add_get('/api/v1/metrics/summary', self.handle_metrics_summary)

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
        # Close shared pool session
        if self._pool_session:
            await self._pool_session.close()
            self._pool_session = None

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

        if hasattr(self, '_grpc_channel_pool'):
            # Pool stores a LIST of channels per URL (HTTP/2 HOL mitigation),
            # so iterate each channel, not the list itself.
            for url, channels in self._grpc_channel_pool.items():
                for ch in channels:
                    try:
                        ch.close()
                    except Exception as e:
                        logger.warning(f"Error closing channel {url}: {e}")
            self._grpc_channel_pool.clear()

        logger.info("Server stopped")

    async def _heartbeat_loop(self):
        """Background task to check agent heartbeats"""
        while True:
            try:
                await asyncio.sleep(10)
                stale = await self.registry.remove_stale_agents()
                for agent_id, grpc_url in (stale or []):
                    try:
                        await self.selector.unregister_agent(agent_id)
                        if self.grpc and self.grpc.is_available():
                            await self.grpc.unregister_agent(agent_id, grpc_url=grpc_url or None)
                    except Exception as e:
                        logger.warning(f"Failed to unregister stale agent {agent_id}: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    async def _dds_client_loop(self):
        """Background task to process DDS client requests.

        Event-driven via listener when possible; falls back to 1ms polling
        if listener attach fails. The listener pushes samples into an
        asyncio.Queue and this loop awaits the queue.
        """
        logger.info("DDS client loop started")
        if not hasattr(self, "_dds_client_req_queue"):
            self._dds_client_req_queue: asyncio.Queue = asyncio.Queue()
            # Try to attach a listener on the client/request reader
            try:
                if self.dds.is_available():
                    from dds import TOPIC_CLIENT_REQUEST
                    from cyclonedds.core import Listener as _Listener
                    reader = self.dds.subscribers.get(TOPIC_CLIENT_REQUEST)
                    if reader is not None:
                        loop = asyncio.get_running_loop()
                        queue_ref = self._dds_client_req_queue
                        def _on_client_request(_reader, _q=queue_ref, _l=loop, _r=reader):
                            try:
                                for s in _r.take():
                                    if s and not _l.is_closed():
                                        _l.call_soon_threadsafe(_q.put_nowait, s)
                            except Exception as e:
                                logger.exception(f"Error in _on_client_request: {e}")
                        lst = _Listener(on_data_available=_on_client_request)
                        reader.set_listener(lst)
                        if not hasattr(self.dds, "_dds_listeners"):
                            self.dds._dds_listeners = []
                        self.dds._dds_listeners.append(lst)
                        logger.info("Attached event-driven listener for client/request")
                        self._dds_client_event_driven = True
            except Exception as e:
                logger.warning(f"client/request listener attach failed: {e}; falling back to polling")

        event_driven = getattr(self, "_dds_client_event_driven", False)

        while True:
            try:
                if event_driven:
                    try:
                        first = await asyncio.wait_for(
                            self._dds_client_req_queue.get(), timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue
                    client_requests = [first]
                    while True:
                        try:
                            client_requests.append(self._dds_client_req_queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break
                else:
                    await asyncio.sleep(0.001)  # Poll every 1ms
                    if not self.dds.is_available():
                        continue
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
        priority = getattr(req, "priority", 5)

        try:
            messages = json.loads(messages_json)
        except Exception as e:
            logger.warning(f"Failed to parse messages_json, using raw content: {e}")
            messages = [{"role": "user", "content": messages_json}]

        logger.info(f"Processing DDS client request: {request_id} (priority={priority})")

        # Get available agent with retry+backoff (aligns with gRPC/HTTP behavior).
        # Deadline mirrors gRPC path: honor client-supplied timeout_ms, capped
        # by the orchestrator's task_timeout_seconds. Previously hardcoded 10s,
        # which caused DDS to drop requests while HTTP/gRPC (120-300s deadlines)
        # absorbed the same backlog — biasing comparison success rates.
        import asyncio as _asyncio
        agents = None
        loop = _asyncio.get_running_loop()
        _req_timeout_ms = getattr(req, "timeout_ms", 0) or 120000
        _acquire_deadline_s = min(
            _req_timeout_ms / 1000.0,
            float(self.config.task_timeout_seconds),
        )
        deadline = loop.time() + _acquire_deadline_s
        delay = 0.005
        while loop.time() < deadline:
            # Transport filter: DDS entrypoint only picks agents that can
            # actually serve DDS requests (requires CycloneDDS + DDS-enabled
            # llama-server). Without this filter, an HTTP-only agent would be
            # selected and the request would stall forever waiting on a DDS
            # response that never comes.
            agents = await self.registry.get_available_agents(transport="dds")
            if agents:
                break
            await _asyncio.sleep(delay)
            delay = min(delay * 2, 0.1)
        if not agents:
            await self._send_dds_client_response(request_id, client_id, "", success=0, error="No agents available")
            return

        # Select agent: fuzzy if enabled, otherwise max idle slots
        # Priority maps to urgency; complexity is auto-estimated from messages
        agent, fuzzy_qos, fuzzy_strategy = self._select_with_fuzzy(
            agents, messages=messages, priority=priority)
        logger.info(f"Selected agent: {agent.agent_id} (fuzzy_qos={fuzzy_qos}, strategy={fuzzy_strategy})")

        # Acquire agent slot (status auto-derived: idle if slots remain, busy if 0)
        if not await self.registry.adjust_slots(agent.agent_id, delta=-1):
            # Agent was removed between selection and slot acquisition
            await self._send_dds_client_response(request_id, client_id, "", success=0, error="Agent no longer available")
            return

        try:
            # Send task to agent via DDS — honor fields from the DDS client
            # request when provided; fall back to BENCH env then defaults.
            import os as _os
            _bench_mt = int(_os.environ.get("BENCH_DEFAULT_MAX_TOKENS", "50"))
            _bench_temp = float(_os.environ.get("BENCH_DEFAULT_TEMPERATURE", "0.7"))
            req_max_tokens = getattr(req, "max_tokens", 0) or _bench_mt
            req_temp_raw = getattr(req, "temperature", -1.0)
            req_temperature = req_temp_raw if req_temp_raw is not None and req_temp_raw >= 0 else _bench_temp
            req_stream = bool(getattr(req, "stream", False))
            dds_request = AgentTaskRequest(
                task_id=request_id,
                requester_id="orchestrator",
                task_type="chat",
                messages=messages,
                priority=priority,
                timeout_ms=60000,
                requires_context=False,
                max_tokens=req_max_tokens,
                temperature=req_temperature,
                stream=req_stream,
                target_agent_id=agent.agent_id,
            )
            # Streaming path: forward each agent chunk to the DDS client.
            if req_stream:
                self.dds.prepare_stream_waiter(request_id)
                try:
                    await self.dds.publish_agent_request(dds_request, qos_profile=fuzzy_qos)
                    last_prompt = 0
                    last_completion = 0
                    async for chunk in self.dds.stream_agent_response(request_id, timeout_ms=_req_timeout_ms):
                        c_content = chunk.get("content", "") or ""
                        c_final = bool(chunk.get("is_final", False))
                        last_prompt = chunk.get("prompt_tokens", last_prompt)
                        last_completion = chunk.get("completion_tokens", last_completion)
                        await self._send_dds_client_response(
                            request_id, client_id, c_content,
                            success=1 if not chunk.get("error") else 0,
                            error=chunk.get("error", ""),
                            prompt_tokens=last_prompt,
                            completion_tokens=last_completion,
                            processing_time_ms=chunk.get("processing_time_ms", 0),
                            is_final=c_final,
                        )
                        if c_final:
                            break
                finally:
                    await self.registry.adjust_slots(agent.agent_id, delta=+1)
                return  # streaming path is fully handled

            # Non-streaming path
            self.dds.prepare_agent_response_waiter(request_id)
            try:
                await self.dds.publish_agent_request(dds_request, qos_profile=fuzzy_qos)
                agent_response = await self.dds.wait_for_agent_response(request_id, timeout_ms=_req_timeout_ms)
            except Exception:
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

    async def _send_dds_client_response(self, request_id, client_id, content, success=True, error="", prompt_tokens=0, completion_tokens=0, processing_time_ms=0, is_final=True):
        """Send response to client via DDS using IDL-generated ClientResponse type.

        is_final=False is used by the streaming path to forward intermediate chunks;
        the last chunk must be sent with is_final=True so the client stops listening.
        """
        from orchestrator import ClientResponse

        response = ClientResponse(
            request_id=request_id,
            client_id=client_id,
            content=content,
            is_final=is_final,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            processing_time_ms=processing_time_ms,
            success=bool(success),
            error_message=error,
        )

        logger.info(f"[DDS_CLI_RESP] rid={request_id} client={client_id} success={success} "
                    f"error={error!r} content_len={len(content)}")
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

        # Slot acquisition: InstancePool (if configured) or fair registry wait.
        instance = None
        import asyncio as _asyncio

        # Fair slot acquisition: delegate to the same notification-based
        # waiter the HTTP path uses (_wait_for_available_agent). The previous
        # polling retry loop with 5ms→100ms exponential backoff caused
        # starvation under c≥50 — ~11% of requests stuck 10-60s while others
        # completed in 500-1000ms (bimodal distribution, spread 133× at c=50).
        # Redis BLPOP / asyncio.Condition gives fair FIFO ordering instead.
        main_loop = getattr(self, "_main_loop", None)
        total_timeout = min(timeout_ms / 1000.0, float(self.config.task_timeout_seconds))
        # Cap the slot-acquisition wait to fail fast when the cluster is broken
        # (no agents, deadlocked queue) instead of waiting the full 120s task
        # budget. 60s covers legitimate queue drain at c=250 with 8 slots
        # (~250/8*700ms = 22s), while still cutting off pathological hangs.
        wait_timeout = min(total_timeout, 60.0)
        agent = None
        fuzzy_qos = None
        fuzzy_strategy = None

        # (1) InstancePool (Redis-backed) first if available — it already
        # blocks on Redis BLPOP, so no polling needed.
        if self.instance_pool and main_loop is not None and main_loop.is_running():
            try:
                future = _asyncio.run_coroutine_threadsafe(
                    self.instance_pool.select_instance(), main_loop)
                instance = future.result(timeout=wait_timeout)
            except Exception:
                instance = None

        # (2) Registry fair-wait — mirrors HTTP path, filters on transports=grpc
        # so agents without a gRPC build (e.g. AMD ROCm 6.3 bfloat16 dup-symbols
        # on RX6600M) are never selected.
        if instance is None and main_loop is not None and main_loop.is_running():
            try:
                future = _asyncio.run_coroutine_threadsafe(
                    self._wait_for_available_agent(
                        timeout_s=wait_timeout,
                        messages=messages,
                        priority=priority,
                        transport="grpc",
                    ),
                    main_loop,
                )
                agent, fuzzy_qos, fuzzy_strategy = future.result(timeout=wait_timeout + 1.0)
            except Exception as e:
                logger.warning(f"gRPC fair-wait failed: {e}")
                agent, fuzzy_qos, fuzzy_strategy = None, None, None

            # Decrement the chosen agent's slot atomically, mirroring handle_chat.
            if agent is not None:
                with self.registry._thread_lock:
                    real_agent = self.registry.agents.get(agent.agent_id)
                    if real_agent and real_agent.slots_idle > 0:
                        real_agent.slots_idle = max(0, real_agent.slots_idle - 1)
                        if real_agent.slots_idle == 0:
                            real_agent.status = "busy"
                        agent = real_agent
                    else:
                        # Raced with another consumer — fall through to failure.
                        agent = None

        if instance is None and agent is None:
            return {"content": "", "success": False, "error": "No agents available"}

        # Resolve agent for logging + gRPC address
        if instance is not None:
            # InstancePool path: find agent by hostname for gRPC address
            _hostname = instance.hostname
            agent = None
            for _a in self.registry.agents.values():
                if _a.hostname == _hostname:
                    agent = _a
                    break
            if agent is None:
                # Create a minimal agent-like object for gRPC routing
                class _FakeAgent:
                    agent_id = f"pool-{_hostname}"
                    hostname = _hostname
                    grpc_address = f"{_hostname}:59201"
                agent = _FakeAgent()
            logger.info(f"Selected agent for gRPC (via pool): {agent.agent_id}")
        else:
            logger.info(f"Selected agent for gRPC (via registry): {agent.agent_id}")

        # Import proto stubs (sys.path already configured at module load via grpc_layer)
        from proto import orchestrator_pb2 as _pb2
        from proto import orchestrator_pb2_grpc as _pb2_grpc

        agent_grpc_url = getattr(agent, "grpc_address", None)
        if not agent_grpc_url:
            agent_grpc_url = f"{agent.hostname}:59201"

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

            # Connection pooling: N channels per agent to mitigate HTTP/2 HOL blocking.
            # Round-robin across channels reduces stream contention on a single TCP conn.
            with self._grpc_pool_lock:
                if agent_grpc_url not in self._grpc_channel_pool:
                    opts = [
                        ("grpc.max_send_message_length", 64 * 1024 * 1024),
                        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                        ("grpc.keepalive_time_ms", 30000),
                    ]
                    self._grpc_channel_pool[agent_grpc_url] = [
                        _grpc.insecure_channel(agent_grpc_url, options=opts)
                        for _ in range(self._GRPC_CHANNELS_PER_AGENT)
                    ]
                    self._grpc_rr_idx[agent_grpc_url] = 0
                channels = self._grpc_channel_pool[agent_grpc_url]
                idx = self._grpc_rr_idx[agent_grpc_url]
                channel = channels[idx % len(channels)]
                self._grpc_rr_idx[agent_grpc_url] = idx + 1
            stub = _pb2_grpc.OrchestratorAgentServiceStub(channel)

            resp = stub.SubmitTask(proto_req, timeout=timeout_ms / 1000)

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
            # Release slot via the same mechanism used for acquisition.
            # CRITICAL: we run in a gRPC thread-pool thread with no event loop
            # of its own. Must schedule onto the captured main loop — calling
            # get_running_loop() here raises RuntimeError, silently leaking
            # the Redis-backed slot forever.
            if instance is not None and self.instance_pool:
                import asyncio as _asyncio
                try:
                    main_loop = getattr(self, "_main_loop", None)
                    if main_loop is None:
                        raise RuntimeError("main loop not captured")
                    _asyncio.run_coroutine_threadsafe(
                        self.instance_pool.release_instance(instance.port, 0, True),
                        main_loop,
                    ).result(timeout=30)
                except Exception as e:
                    logger.warning(f"Failed to release instance slot {instance.port}: {e}")
            else:
                self.registry.adjust_slots_sync(agent.agent_id, delta=+1)

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

        # Use remote IP as hostname if agent sends "localhost" or "unknown"
        reported_host = data.get("hostname", "unknown")
        if reported_host in ("localhost", "127.0.0.1", "unknown", ""):
            remote_ip = request.remote
            if remote_ip:
                reported_host = remote_ip
                logger.info(f"Agent {agent_id} reported hostname={data.get('hostname')}, using remote IP {remote_ip}")

        agent_info = AgentInfo(
            agent_id=agent_id,
            hostname=reported_host,
            port=data.get("port", 8080),
            model=data.get("model", "unknown"),
            vram_available_mb=data.get("vram_available_mb", 0),
            slots_idle=data.get("slots_idle", 1),
            slots_total=data.get("slots_total", data.get("slots_idle", 1)),
            vision_enabled=data.get("vision_enabled", False),
            capabilities=data.get("capabilities", []),
            # transports defaults to ["http"] (universal); agents that can
            # serve gRPC or DDS must declare it in their register payload.
            transports=data.get("transports") or ["http"],
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
            slots_idle = data.get("slots_idle")
        except Exception as e:
            logger.warning(f"Failed to parse heartbeat JSON, using defaults: {e}")
            status = "idle"
            slots_idle = None

        success = await self.registry.update_heartbeat(agent_id, status=status, slots_idle=slots_idle)
        return web.json_response({"success": success})

    async def handle_unregister_agent(self, request: web.Request) -> web.Response:
        """Unregister an agent"""
        agent_id = request.match_info['agent_id']
        agent_info = await self.registry.get_agent(agent_id)
        grpc_url = getattr(agent_info, "grpc_address", "") if agent_info else ""

        success = await self.registry.unregister_agent(agent_id)
        if not success:
            return web.json_response({"error": "Agent not found"}, status=404)

        await self.selector.unregister_agent(agent_id)

        if self.grpc and self.grpc.is_available():
            await self.grpc.unregister_agent(agent_id, grpc_url=grpc_url or None)

        return web.json_response({"success": True})

    async def _execute_agent_request(self, agent: AgentInfo, task: Task, messages: list,
                                     all_messages: list, max_tokens: int, temperature: float,
                                     stream_requested: bool, protocol: str,
                                     fuzzy_qos: str, context_id: str, session_id: str) -> tuple:
        """Execute request to a single agent with an optional forced protocol.

        Returns: (response_data, transport_success, error_msg, streaming_handled)

        Protocol consistency:
        - When `protocol` is explicitly provided ("http"/"grpc"/"dds"), this method MUST
          not silently fall back to a different protocol.
        - The HTTP entrypoint passes protocol="http" to guarantee end-to-end HTTP.
        """
        response_data = {}
        transport_success = False
        error_msg = None
        _streaming_handled = False

        forced = (protocol or "").strip().lower() or None
        if forced not in (None, "http", "grpc", "dds"):
            forced = None  # legacy callers may still pass weird values; treat as auto

        try:
            # ============================================================
            # Forced HTTP (no gRPC/DDS attempts)
            # ============================================================
            if forced == "http":
                agent_url = f"http://{agent.hostname}:{agent.port}"
                try:
                    import aiohttp as _aiohttp
                    async with self._pool_session.post(
                        f"{agent_url}/chat",
                        json={
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                        timeout=_aiohttp.ClientTimeout(total=self.config.task_timeout_seconds)
                    ) as _resp:
                        http_result = await _resp.json()

                    content = http_result.get("content", "")
                    if not content and "choices" in http_result:
                        content = http_result["choices"][0].get("message", {}).get("content", "")
                    if not content and "response" in http_result:
                        content = http_result["response"]

                    response_data = {
                        "content": content,
                        "processing_time_ms": http_result.get("processing_time_ms", 0),
                        "success": True,
                    }
                    transport_success = True
                    for msg in messages:
                        await self.ctx.add_message(context_id, ChatMessage(role=msg.get("role"), content=msg.get("content")))
                    if content:
                        await self.ctx.add_message(context_id, ChatMessage(role="assistant", content=content))
                except Exception as e:
                    error_msg = f"HTTP request failed: {e}"
                    logger.error(error_msg)
                    response_data = {"content": "", "error": error_msg, "success": False}

                return response_data, transport_success, error_msg, _streaming_handled

            # === gRPC path ===
            if forced in (None, "grpc") and self.grpc and self.grpc.is_available() and getattr(agent, "grpc_address", ""):
                try:
                    from grpc_layer import AgentTaskRequest as GRPCAgentTaskRequest
                    grpc_request = GRPCAgentTaskRequest(
                        task_id=task.task_id,
                        requester_id="orchestrator",
                        task_type=task.task_type,
                        messages=all_messages,
                        priority=task.priority,
                        timeout_ms=task.timeout_ms,
                        requires_context=True,
                        context_id=context_id,
                        stream=stream_requested,
                    )

                    if stream_requested:
                        self.grpc.prepare_stream_waiter(task.task_id)
                        asyncio.create_task(self.grpc.publish_agent_request(
                            grpc_request,
                            agent_id=agent.agent_id,
                            agent_grpc_url=agent.grpc_address,
                        ))
                        # Streaming path returns early - handled by streaming logic
                        return None, False, None, True
                    else:
                        self.grpc.prepare_agent_response_waiter(task.task_id)
                        await self.grpc.publish_agent_request(
                            grpc_request,
                            agent_id=agent.agent_id,
                            agent_grpc_url=agent.grpc_address,
                        )
                        response_data = await self.grpc.wait_for_agent_response(task.task_id, timeout_ms=task.timeout_ms)
                        if isinstance(response_data, dict) and response_data.get("error"):
                            error_msg = response_data.get("error")
                        else:
                            transport_success = True
                            # Store context
                            for msg in messages:
                                await self.ctx.add_message(context_id, ChatMessage(role=msg.get("role"), content=msg.get("content")))
                            content = response_data.get("content", "") if isinstance(response_data, dict) else ""
                            if content:
                                await self.ctx.add_message(context_id, ChatMessage(role="assistant", content=content))
                except Exception as e:
                    error_msg = str(e)
                    if forced == "grpc":
                        # Forced protocol: do not fall back.
                        return {"content": "", "error": error_msg, "success": False}, False, error_msg, _streaming_handled
                    logger.warning(f"gRPC failed: {e}, trying DDS")

            # === DDS path ===
            if (forced in (None, "dds")) and not transport_success:
                try:
                    dds_request = AgentTaskRequest(
                        task_id=task.task_id,
                        requester_id="orchestrator",
                        task_type=task.task_type,
                        messages=all_messages,
                        priority=task.priority,
                        timeout_ms=task.timeout_ms,
                        requires_context=True,
                        context_id=context_id,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=stream_requested,
                        target_agent_id=agent.agent_id,
                    )

                    dds_priority = {1: 0, 2: 5, 5: 5, 10: 10, 20: 20}.get(task.priority, 0)

                    if stream_requested and self.dds.is_available():
                        self.dds.prepare_stream_waiter(task.task_id)
                        await self.dds.publish_agent_request(dds_request, priority=dds_priority,
                                                              qos_profile=fuzzy_qos)
                        # Streaming path returns early
                        return None, False, None, True
                    else:
                        self.dds.prepare_agent_response_waiter(task.task_id)
                        await self.dds.publish_agent_request(dds_request, priority=dds_priority,
                                                              qos_profile=fuzzy_qos)
                        response_data = await self.dds.wait_for_agent_response(task.task_id, timeout_ms=task.timeout_ms)

                        content = ""
                        if isinstance(response_data, dict):
                            content = response_data.get("content", "")
                        else:
                            content = getattr(response_data, "content", "")

                        if isinstance(response_data, dict) and response_data.get("error") == "DDS not available":
                            error_msg = "DDS not available"
                        else:
                            transport_success = True
                            # Store context
                            for msg in messages:
                                await self.ctx.add_message(context_id, ChatMessage(role=msg.get("role"), content=msg.get("content")))
                            if content:
                                await self.ctx.add_message(context_id, ChatMessage(role="assistant", content=content))

                except Exception as e:
                    error_msg = str(e)
                    if forced == "dds":
                        return {"content": "", "error": error_msg, "success": False}, False, error_msg, _streaming_handled
                    logger.warning(f"DDS failed: {e}, falling back to HTTP")
                    self.dds._pending_agent_responses.pop(task.task_id, None)
                    self.dds._pending_agent_responses.pop(f"stream_{task.task_id}", None)

            # === HTTP fallback ===
            if forced is None and not transport_success:
                logger.info(f"Using HTTP fallback for task {task.task_id}")
                agent_url = f"http://{agent.hostname}:{agent.port}"
                try:
                    import aiohttp as _aiohttp
                    async with self._pool_session.post(
                        f"{agent_url}/chat",
                        json={
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                        timeout=_aiohttp.ClientTimeout(total=self.config.task_timeout_seconds)
                    ) as _resp:
                        http_result = await _resp.json()

                    content = http_result.get("content", "")
                    if not content and "choices" in http_result:
                        content = http_result["choices"][0].get("message", {}).get("content", "")
                    if not content and "response" in http_result:
                        content = http_result["response"]

                    response_data = {
                        "content": content,
                        "processing_time_ms": http_result.get("processing_time_ms", 0),
                        "success": True,
                    }
                    transport_success = True
                    # Store context
                    for msg in messages:
                        await self.ctx.add_message(context_id, ChatMessage(role=msg.get("role"), content=msg.get("content")))
                    if content:
                        await self.ctx.add_message(context_id, ChatMessage(role="assistant", content=content))

                except Exception as e:
                    error_msg = f"HTTP fallback failed: {e}"
                    logger.error(error_msg)
                    response_data = {"content": "", "error": error_msg, "success": False}

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Unexpected error in agent execution: {e}")

        return response_data, transport_success, error_msg, _streaming_handled

    async def _execute_with_retry(self, primary_agent: AgentInfo, fallback_agents: list,
                                   task: Task, messages: list, all_messages: list,
                                   max_tokens: int, temperature: float, priority: int,
                                   fuzzy_qos: str, context_id: str, session_id: str,
                                   protocol: str) -> tuple:
        """Execute request with retry strategy: try primary, then fallback agents with exponential backoff.

        Returns: (response_content, response_data, successful_agent_id)
        """
        response_content = ""
        response_data = {}
        current_agent = primary_agent
        attempted_agents = [primary_agent.agent_id]
        max_retries = min(3, len([primary_agent] + fallback_agents))

        for retry_attempt in range(max_retries):
            if retry_attempt > 0:
                # Apply exponential backoff: 0.5s, 1s, 2s
                backoff = min(0.5 * (2 ** (retry_attempt - 1)), 5.0)
                logger.info(f"Retry {retry_attempt}/{max_retries}: waiting {backoff}s before trying agent {current_agent.agent_id}")
                await asyncio.sleep(backoff)

            # For retry attempts after first, update task to use next fallback agent
            if retry_attempt > 0:
                if retry_attempt <= len(fallback_agents):
                    current_agent = fallback_agents[retry_attempt - 1]
                    attempted_agents.append(current_agent.agent_id)

                    # Update slot allocation: release previous, acquire new
                    await self.registry.adjust_slots(attempted_agents[-2], delta=+1)
                    if not await self.registry.adjust_slots(current_agent.agent_id, delta=-1):
                        logger.warning(f"Could not acquire slot on retry agent {current_agent.agent_id}")
                        continue

                    task.assigned_agent_id = current_agent.agent_id
                    task.task_id = f"task-{str(uuid.uuid4())}-retry-{retry_attempt}"
                    await self.scheduler.track_task(task)

            # Execute agent request (protocol forced by entrypoint)
            try:
                response_data, transport_success, error_msg, _ = await self._execute_agent_request(
                    current_agent, task, messages, all_messages, max_tokens, temperature,
                    False, protocol, fuzzy_qos, context_id, session_id
                )

                if transport_success:
                    response_content = response_data.get("content", "") if isinstance(response_data, dict) else getattr(response_data, "content", "")
                    if response_content:
                        logger.info(f"Retry attempt {retry_attempt} successful with agent {current_agent.agent_id}")
                        return response_content, response_data, current_agent.agent_id
            except Exception as e:
                logger.warning(f"Retry attempt {retry_attempt} failed: {e}")

        return response_content, response_data, current_agent.agent_id

    async def _execute_fanout(self, agents: list, task: Task, messages: list, all_messages: list,
                              max_tokens: int, temperature: float, priority: int,
                              fuzzy_qos: str, context_id: str, session_id: str,
                              protocol: str, dds_priority: int = 0) -> tuple:
        """Execute request with fanout strategy: send to multiple agents in parallel, return first success.

        Returns: (response_content, response_data, successful_agent_id)
        """
        if not agents or len(agents) < 2:
            return "", {}, ""

        logger.info(f"Fanout strategy: sending to up to {min(3, len(agents))} agents in parallel")

        fanout_tasks = []
        fanout_agents = []
        fanout_task_ids = []

        for fanout_idx, fanout_agent in enumerate(agents[:3]):
            # Try to acquire slot for this agent
            if not await self.registry.adjust_slots(fanout_agent.agent_id, delta=-1):
                logger.warning(f"Fanout: could not acquire slot for agent {fanout_agent.agent_id}")
                continue

            fanout_agents.append(fanout_agent)
            fanout_task_id = f"task-{str(uuid.uuid4())}-fanout-{fanout_idx}"
            fanout_task_ids.append(fanout_task_id)

            # Create task for parallel execution
            async def _fanout_attempt(fa=fanout_agent, ti=fanout_idx, task_id=fanout_task_id):
                """Execute single agent in fanout context"""
                try:
                    # Track fanout task
                    fanout_task = Task(
                        task_id=task_id,
                        task_type=task.task_type,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        priority=task.priority,
                        assigned_agent_id=fa.agent_id,
                    )
                    await self.scheduler.track_task(fanout_task)

                    # Execute agent request
                    response_data, transport_success, error_msg, _ = await self._execute_agent_request(
                        fa, fanout_task, messages, all_messages, max_tokens, temperature,
                        False, protocol, fuzzy_qos, context_id, session_id
                    )

                    if transport_success:
                        fa_content = response_data.get("content", "") if isinstance(response_data, dict) else getattr(response_data, "content", "")
                        return (fa_content, response_data, fa.agent_id)

                    return ("", {}, fa.agent_id)
                except Exception as e:
                    logger.error(f"Fanout attempt on agent {fa.agent_id} failed: {e}")
                    return ("", {}, fa.agent_id)

            # Create explicit tasks (required for asyncio.wait in Python 3.10+)
            fanout_tasks.append(asyncio.create_task(_fanout_attempt()))

        # Run all tasks in parallel, get first successful response
        response_content = ""
        response_data = {}
        successful_agent_id = ""

        if fanout_tasks:
            try:
                # Fanout budget mirrors the request deadline (task.timeout_ms),
                # so slow agents don't block the caller past its own budget.
                _fanout_timeout_s = max(1.0, (task.timeout_ms or 120000) / 1000.0)
                done, pending = await asyncio.wait(
                    fanout_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=_fanout_timeout_s,
                )

                for task_coro in done:
                    try:
                        fa_content, fa_data, fa_agent_id = await task_coro
                        if fa_content:
                            response_content = fa_content
                            response_data = fa_data if fa_data else {"content": fa_content, "success": True}
                            successful_agent_id = fa_agent_id
                            logger.info(f"Fanout winner: {fa_agent_id}")
                            break
                    except Exception as e:
                        logger.warning(f"Fanout response processing failed: {e}")

                # Cancel pending tasks
                for task_coro in pending:
                    task_coro.cancel()
                    try:
                        await task_coro
                    except asyncio.CancelledError:
                        pass
            finally:
                # Release slots for all fanout agents
                for fa in fanout_agents:
                    await self.registry.adjust_slots(fa.agent_id, delta=+1)

        return response_content, response_data, successful_agent_id

    async def handle_chat(self, request: web.Request) -> web.Response:
        """Handle chat request"""
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)
        messages = data.get("messages", [])

        if not messages:
            return web.json_response({"error": "No messages provided"}, status=400)

        # Get parameters — BENCH_DEFAULT_* env vars FORCE-override client values
        # when set. Used to isolate transport latency during benchmarks.
        import os as _os_bench
        _bench_mt_env = _os_bench.environ.get("BENCH_DEFAULT_MAX_TOKENS")
        _bench_temp_env = _os_bench.environ.get("BENCH_DEFAULT_TEMPERATURE")
        if _bench_mt_env is not None:
            max_tokens = int(_bench_mt_env)  # force override
        else:
            max_tokens = data.get("max_tokens", self.config.default_max_tokens)
        if _bench_temp_env is not None:
            temperature = float(_bench_temp_env)
        else:
            temperature = data.get("temperature", self.config.default_temperature)
        priority = data.get("priority", TaskPriority.NORMAL.value)
        task_type = data.get("task_type", "chat")
        stream_requested = data.get("stream", False)
        protocol = data.get("protocol")  # optional: "http", "grpc", "dds"
        # Must be available before any fanout/retry call — fanout reads it
        # even when protocol=="http" (it decides on fallbacks). Previously
        # it was only set inside the DDS branch, so HTTP paths triggered
        # NameError on fallback.
        dds_priority = {1: 0, 2: 5, 5: 5, 10: 10, 20: 20}.get(priority, 0)

        # ============================================================
        # PROTOCOL CONSISTENCY: HTTP entrypoint forces HTTP end-to-end.
        # gRPC/DDS have their own native entrypoints.
        protocol = "http"

        # Deadline parity: derive a single per-request budget used for agent
        # acquisition AND response receipt. Matches the DDS handler's budget
        # policy (see _process_dds_client_request) so success rates aren't
        # biased by inconsistent timeouts across transports.
        _req_timeout_ms = int(data.get("timeout_ms", 120000) or 120000)
        _req_timeout_ms = max(
            1000,
            min(_req_timeout_ms, self.config.task_timeout_seconds * 1000),
        )
        _wait_s = _req_timeout_ms / 1000.0

        # === Context Management ===
        # Extract or create session_id and retrieve conversation history
        session_id = data.get("session_id") or str(uuid.uuid4())
        context_id = await self.ctx.get_or_create_for_user(session_id)

        # Get existing messages from context and prepend to incoming messages
        existing_messages = await self.ctx.get_messages(context_id)

        # Merge: existing messages + new incoming messages
        # Keep incoming messages as primary, but preserve context history
        all_messages = existing_messages + [ChatMessage(role=m.get("role"), content=m.get("content")) for m in messages]

        # Log context info
        logger.info(f"Session {session_id}: context_id={context_id}, history_size={len(existing_messages)}, total_messages={len(all_messages)}")

        # === Instance Pool Path (when Redis + InstancePool configured) ===
        if self.instance_pool:
            return await self._handle_chat_pool(
                request, data, messages, max_tokens, temperature,
                priority, stream_requested, protocol,
            )

        # === Legacy Path (original agent registry) ===
        # Wait for an available agent (queue instead of 503 rejection)
        # Fuzzy engine selects best agent + QoS profile + strategy
        fuzzy_urgency = data.get("urgency")
        fuzzy_complexity = data.get("complexity")
        agent, fuzzy_qos, fuzzy_strategy = await self._wait_for_available_agent(
            timeout_s=_wait_s, messages=messages, priority=priority,
            urgency=fuzzy_urgency, complexity=fuzzy_complexity,
        )
        if not agent:
            return web.json_response({
                "error": "No agents available after timeout",
                "code": "NO_AGENTS"
            }, status=503)
        logger.info(f"Selected agent: {agent.agent_id} (qos={fuzzy_qos}, strategy={fuzzy_strategy})")

        # === Strategy Dispatch ===
        # For retry strategy, get additional agents for fallback
        agent_list = [agent]
        if fuzzy_strategy == "retry" or fuzzy_strategy == "fanout":
            # Get available agents for retry/fanout dispatch
            all_agents = await self.registry.get_available_agents()
            # Filter to exclude the already-selected agent
            additional_agents = [a for a in all_agents if a.agent_id != agent.agent_id][:2]
            agent_list.extend(additional_agents)
            logger.info(f"Strategy {fuzzy_strategy}: using agent pool of {len(agent_list)} agents")

        # Create task — propagate the request's deadline so downstream
        # response waits use the same budget as the acquire phase.
        task = Task(
            task_id=f"task-{str(uuid.uuid4())}",
            task_type="chat",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=TaskScheduler._map_priority(priority),
            assigned_agent_id=agent.agent_id,
            timeout_ms=_req_timeout_ms,
        )

        # Update agent status (atomic slot decrement, status auto-derived)
        # Retry if another request grabbed the slot between wait and decrement
        for _retry in range(3):
            if await self.registry.adjust_slots(agent.agent_id, delta=-1):
                break
            agent, fuzzy_qos, fuzzy_strategy = await self._wait_for_available_agent(
                timeout_s=_wait_s, messages=messages, priority=priority)
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

        # === Strategy-based dispatch ===
        # Track attempted agents for logging and retry decisions
        attempted_agents = [agent.agent_id]

        try:
            # === gRPC path (ONLY when entry protocol is gRPC) ===
            stream_requested = data.get("stream", False)
            if (protocol == "grpc"
                    and not transport_success
                    and self.grpc and self.grpc.is_available()
                    and getattr(agent, "grpc_address", "")):
                try:
                    from grpc_layer import AgentTaskRequest as GRPCAgentTaskRequest
                    grpc_request = GRPCAgentTaskRequest(
                        task_id=task.task_id,
                        requester_id="orchestrator",
                        task_type=task.task_type,
                        messages=all_messages,
                        priority=priority,
                        timeout_ms=task.timeout_ms,
                        requires_context=True,
                        context_id=context_id,
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
                            async for chunk in self.grpc.stream_agent_response(task.task_id, timeout_ms=task.timeout_ms):
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
                            _streaming_handled = True
                            await self.registry.adjust_slots(agent.agent_id, delta=+1)
                            await self.scheduler.complete_task(task.task_id, response="streaming")

                        return sse_response

                    else:
                        # Non-streaming gRPC path
                        self.grpc.prepare_agent_response_waiter(task.task_id)
                        await self.grpc.publish_agent_request(
                            grpc_request,
                            agent_id=agent.agent_id,
                            agent_grpc_url=agent.grpc_address,
                        )
                        response_data = await self.grpc.wait_for_agent_response(task.task_id, timeout_ms=task.timeout_ms)
                        if isinstance(response_data, dict) and response_data.get("error"):
                            logger.warning(f"gRPC response error: {response_data.get('error')}")
                        else:
                            transport_success = True
                            # Store the user messages and assistant response in context
                            content = response_data.get("content", "") if isinstance(response_data, dict) else ""
                            for msg in messages:
                                await self.ctx.add_message(context_id, ChatMessage(role=msg.get("role"), content=msg.get("content")))
                            if content:
                                await self.ctx.add_message(context_id, ChatMessage(role="assistant", content=content))
                            logger.info(f"Context updated for {session_id}: added {len(messages)} user message(s) and assistant response")

                except Exception as e:
                    logger.warning(f"gRPC communication failed: {e}")

            # === DDS path (ONLY when entry protocol is DDS) ===
            if protocol == "dds" and not transport_success:
                try:
                    dds_request = AgentTaskRequest(
                        task_id=task.task_id,
                        requester_id="orchestrator",
                        task_type=task.task_type,
                        messages=all_messages,
                        priority=priority,
                        timeout_ms=task.timeout_ms,
                        requires_context=True,
                        context_id=context_id,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=stream_requested,
                    )

                    # dds_priority already computed at the top of the handler.
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
                            async for chunk in self.dds.stream_agent_response(task.task_id, timeout_ms=task.timeout_ms):
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
                            _streaming_handled = True
                            await self.registry.adjust_slots(agent.agent_id, delta=+1)
                            await self.scheduler.complete_task(task.task_id, response="streaming")

                        return sse_response

                    else:
                        # Non-streaming DDS path
                        self.dds.prepare_agent_response_waiter(task.task_id)
                        await self.dds.publish_agent_request(dds_request, priority=dds_priority,
                                                              qos_profile=fuzzy_qos)
                        response_data = await self.dds.wait_for_agent_response(task.task_id, timeout_ms=task.timeout_ms)

                        content = ""
                        if isinstance(response_data, dict):
                            content = response_data.get("content", "")
                        else:
                            content = getattr(response_data, "content", "")

                        if isinstance(response_data, dict) and response_data.get("error") == "DDS not available":
                            logger.info("DDS not available, will try HTTP fallback")
                        else:
                            transport_success = True
                            # Store the user messages and assistant response in context
                            for msg in messages:
                                await self.ctx.add_message(context_id, ChatMessage(role=msg.get("role"), content=msg.get("content")))
                            if content:
                                await self.ctx.add_message(context_id, ChatMessage(role="assistant", content=content))
                            logger.info(f"Context updated for {session_id}: added {len(messages)} user message(s) and assistant response")

                except Exception as e:
                    logger.warning(f"DDS communication failed: {e}")
                    self.dds._pending_agent_responses.pop(task.task_id, None)
                    self.dds._pending_agent_responses.pop(f"stream_{task.task_id}", None)

            # If transport failed, use HTTP fallback (ONLY when entry protocol is HTTP)
            if protocol == "http" and not transport_success:
                logger.info(f"Using HTTP fallback for task {task.task_id}")
                agent_url = f"http://{agent.hostname}:{agent.port}"
                logger.info(f"HTTP fallback calling agent at {agent_url}/chat")
                try:
                    import aiohttp as _aiohttp
                    async with self._pool_session.post(
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
                    # Store the user messages and assistant response in context
                    for msg in messages:
                        await self.ctx.add_message(context_id, ChatMessage(role=msg.get("role"), content=msg.get("content")))
                    if content:
                        await self.ctx.add_message(context_id, ChatMessage(role="assistant", content=content))
                    logger.info(f"Context updated for {session_id}: added {len(messages)} user message(s) and assistant response")
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

            # === Fanout Strategy (Parallel Execution) ===
            # If response failed and strategy is "fanout", try multiple agents in parallel
            if (not response_content and fuzzy_strategy == "fanout" and len(agent_list) > 1):
                fa_content, fa_data, fa_agent_id = await self._execute_fanout(
                    agent_list, task, messages, all_messages,
                    max_tokens, temperature, priority,
                    fuzzy_qos, context_id, session_id, protocol, dds_priority
                )
                if fa_content:
                    response_content = fa_content
                    response_data = fa_data if fa_data else {"content": fa_content, "success": True}
                    agent = next((a for a in agent_list if a.agent_id == fa_agent_id), agent)
                    attempted_agents.append(fa_agent_id)
                    logger.info(f"Fanout winner: {fa_agent_id}")

            # === Retry Logic for Strategy ===
            # If response failed and strategy is "retry", try next agent from pool
            if (not response_content and fuzzy_strategy == "retry" and len(agent_list) > 1):
                # Get fallback agents (all except primary)
                fallback_agents = agent_list[1:]

                retry_content, retry_data, retry_agent_id = await self._execute_with_retry(
                    agent, fallback_agents, task, messages, all_messages,
                    max_tokens, temperature, priority,
                    fuzzy_qos, context_id, session_id, protocol
                )

                if retry_content:
                    response_content = retry_content
                    response_data = retry_data if retry_data else {"content": retry_content, "success": True}
                    agent = next((a for a in agent_list if a.agent_id == retry_agent_id), agent)
                    attempted_agents.append(retry_agent_id)
                    # Store context after successful retry
                    for msg in messages:
                        await self.ctx.add_message(context_id, ChatMessage(role=msg.get("role"), content=msg.get("content")))
                    await self.ctx.add_message(context_id, ChatMessage(role="assistant", content=response_content))

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

    async def _handle_chat_pool(self, request, data, messages, max_tokens,
                                temperature, priority, stream_requested, protocol):
        """Handle chat using InstancePool routing (38 instances)."""
        import time as _time

        # This endpoint is HTTP-native; protocol must be HTTP by invariant.
        if protocol != "http":
            raise ValueError(f"HTTP entrypoint cannot run protocol={protocol!r}")

        # Backpressure check
        try:
            if self.backpressure and not await self.backpressure.allow_request():
                return web.json_response(
                    {"error": "Rate limit exceeded", "code": "RATE_LIMITED"},
                    status=429,
                )
        except Exception as _bp_err:
            logger.warning("backpressure check failed (ignored): %s", _bp_err)

        # Track active requests
        if self.redis_mgr:
            await self.redis_mgr.incr_active()

        t_start = _time.time()
        task_id = f"task-{str(uuid.uuid4())}"
        instance = None

        # Request deadline propagates to response waits so pool path behaves
        # consistently with legacy handle_chat path.
        _req_timeout_ms = int(data.get("timeout_ms", 120000) or 120000)
        _req_timeout_ms = max(
            1000,
            min(_req_timeout_ms, self.config.task_timeout_seconds * 1000),
        )

        try:
            # Select instance with retry + event-driven wait (Fix 4A).
            # When all slots are busy, wait for a slot:available BLPOP notification
            # instead of spinning with asyncio.sleep — reduces CPU and improves
            # responsiveness (wakes up as soon as any slot is released).
            queue_timeout = self.config.task_timeout_seconds
            loop = asyncio.get_running_loop()
            deadline = loop.time() + queue_timeout
            delay = 0.005  # 5ms initial (used only when redis_mgr is absent)
            while True:
                instance = await self.instance_pool.select_instance()
                if instance:
                    break
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return web.json_response(
                        {"error": "All instances at capacity", "code": "NO_CAPACITY"},
                        status=503,
                    )
                wait_time = min(delay, remaining)
                if self.redis_mgr and hasattr(self.redis_mgr, "wait_slot_available"):
                    await self.redis_mgr.wait_slot_available(wait_time)
                else:
                    await asyncio.sleep(wait_time)
                delay = min(delay * 2, 0.1)  # cap at 100ms

            # Circuit breaker disabled for benchmark fairness — all protocols
            # share the same instances and false-negative content detection
            # (reasoning_content vs content) would unfairly penalize some paths.
            # if self.backpressure and await self.backpressure.is_circuit_open(instance.port):
            #     await self.instance_pool.release_instance(instance.port, 0, False)
            #     instance = None
            #     return web.json_response(
            #         {"error": "Instance circuit breaker open", "code": "CIRCUIT_OPEN"},
            #         status=503,
            #     )

            # Dispatch to the instance's agent via DDS with target_agent_id.
            # Resolve agent_id from instance hostname so the DDS agent can
            # filter and only process requests meant for it (avoids broadcast
            # overhead where all agents deserialize every request).
            _target_agent_id = ""
            for _a in (await self.registry.get_all_agents()):
                if _a.hostname == instance.hostname:
                    _target_agent_id = _a.agent_id
                    break

            dds_request = AgentTaskRequest(
                task_id=task_id,
                requester_id="orchestrator",
                task_type="chat",
                messages=messages,
                priority=priority,
                timeout_ms=self.config.task_timeout_seconds * 1000,
                requires_context=False,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream_requested,
                target_agent_id=_target_agent_id,
            )

            # Map priority to DDS TRANSPORT_PRIORITY
            priority_map = {1: 0, 2: 5, 5: 5, 10: 10, 20: 20}
            dds_priority = priority_map.get(priority, 0)

            # Protocol consistency: HTTP entry always runs HTTP end-to-end.
            effective_protocol = "http"

            # ── DDS Path (DISABLED FOR HTTP ENTRYPOINT) ───────────────
            if effective_protocol == "dds" and self.dds.is_available():
                self.dds.prepare_agent_response_waiter(task_id)
                await self.dds.publish_agent_request(dds_request, priority=dds_priority)

                if stream_requested:
                    self.dds.prepare_stream_waiter(task_id)
                    sse_response = web.StreamResponse(
                        status=200, reason='OK',
                        headers={
                            'Content-Type': 'text/event-stream',
                            'Cache-Control': 'no-cache',
                            'Connection': 'keep-alive',
                        }
                    )
                    await sse_response.prepare(request)
                    stream_ok = True
                    try:
                        async for chunk in self.dds.stream_agent_response(task_id, timeout_ms=_req_timeout_ms):
                            content = chunk.get("content", "")
                            is_final = chunk.get("is_final", False)
                            if is_final and not content:
                                await sse_response.write(b"data: [DONE]\n\n")
                                break
                            sse_data = json.dumps({
                                "id": task_id,
                                "object": "chat.completion.chunk",
                                "choices": [{"index": 0, "delta": {"content": content},
                                             "finish_reason": "stop" if is_final else None}],
                            })
                            await sse_response.write(f"data: {sse_data}\n\n".encode())
                            if is_final:
                                await sse_response.write(b"data: [DONE]\n\n")
                                break
                    except Exception:
                        stream_ok = False
                        raise
                    finally:
                        latency_ms = (_time.time() - t_start) * 1000
                        inst_port = instance.port
                        inst_type = instance.inst_type
                        await self.instance_pool.release_instance(inst_port, latency_ms, stream_ok)
                        instance = None
                        if self.mongo_store:
                            asyncio.create_task(self.mongo_store.log_request({
                                "request_id": task_id, "instance_port": inst_port,
                                "instance_type": inst_type, "protocol": "dds",
                                "algorithm": self.instance_pool._algorithm.value,
                                "latency_ms": latency_ms, "success": stream_ok,
                                "scenario": data.get("scenario", ""),
                            }))
                    return sse_response
                else:
                    # Non-streaming DDS: publish request, wait for response
                    dds_resp = await self.dds.wait_for_agent_response(
                        task_id, timeout_ms=self.config.task_timeout_seconds * 1000
                    )
                    # dds_resp can be dict or AgentTaskResponse dataclass
                    if dds_resp is None:
                        content = ""
                    elif isinstance(dds_resp, dict):
                        content = dds_resp.get("content", "")
                    elif hasattr(dds_resp, "content"):
                        content = dds_resp.content or ""
                    else:
                        content = str(dds_resp)
                    success = bool(content)
                    latency_ms = (_time.time() - t_start) * 1000
                    inst_port = instance.port
                    inst_type = instance.inst_type
                    await self.instance_pool.release_instance(inst_port, latency_ms, success)
                    instance = None
                    if self.mongo_store:
                        asyncio.create_task(self.mongo_store.log_request({
                            "request_id": task_id, "instance_port": inst_port,
                            "instance_type": inst_type, "protocol": "dds",
                            "algorithm": self.instance_pool._algorithm.value,
                            "latency_ms": latency_ms, "success": success,
                            "scenario": data.get("scenario", ""),
                        }))
                    return web.json_response({
                        "id": task_id, "object": "chat.completion",
                        "choices": [{"index": 0,
                                     "message": {"role": "assistant", "content": content},
                                     "finish_reason": "stop" if content else "error"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "instance_port": inst_port, "instance_type": inst_type,
                        "protocol": "dds", "processing_time_ms": int(latency_ms),
                    })

            # ── gRPC Path (DISABLED FOR HTTP ENTRYPOINT) ──────────────
            elif effective_protocol == "grpc" and self.grpc:
                content = ""
                success = False
                try:
                    # Target agent's gRPC address: hostname:grpc_port
                    grpc_port = 59000 + (instance.port - 8000)  # convention: agent gRPC at 59082, 59088
                    agent_grpc_addr = f"{instance.hostname}:{grpc_port}"
                    grpc_resp = await self.grpc.forward_to_instance(
                        agent_grpc_addr, dds_request, task_id,
                        timeout_s=self.config.task_timeout_seconds,
                    )
                    content = grpc_resp.get("content", "") if grpc_resp else ""
                    success = bool(content)
                except Exception as e:
                    logger.error(f"gRPC to instance :{instance.port} failed: {e}")
                    content = ""
                    success = False

                latency_ms = (_time.time() - t_start) * 1000
                inst_port = instance.port
                inst_type = instance.inst_type
                await self.instance_pool.release_instance(inst_port, latency_ms, success)
                instance = None
                if self.mongo_store:
                    asyncio.create_task(self.mongo_store.log_request({
                        "request_id": task_id, "instance_port": inst_port,
                        "instance_type": inst_type, "protocol": "grpc",
                        "algorithm": self.instance_pool._algorithm.value,
                        "latency_ms": latency_ms, "success": success,
                        "scenario": data.get("scenario", ""),
                    }))
                return web.json_response({
                    "id": task_id, "object": "chat.completion",
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop" if content else "error"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "instance_port": inst_port, "instance_type": inst_type,
                    "protocol": "grpc", "processing_time_ms": int(latency_ms),
                })

            # ── HTTP Path (default) ───────────────────────────────────
            # Orchestrator → HTTP → llama-server directly
            else:
                content = ""
                success = False
                agent_url = f"http://{instance.hostname}:{instance.port}"
                try:
                    async with self._pool_session.post(
                        f"{agent_url}/v1/chat/completions",
                        json={"messages": messages, "max_tokens": max_tokens,
                              "temperature": temperature},
                    ) as _resp:
                        http_result = await _resp.json()
                    content = http_result.get("content", "")
                    if not content and "choices" in http_result:
                        msg = http_result["choices"][0].get("message", {})
                        content = msg.get("content", "") or msg.get("reasoning_content", "")
                    success = bool(content)
                except Exception as e:
                    content = ""
                    success = False
                    logger.error(f"HTTP to instance :{instance.port} failed: {e}")

                latency_ms = (_time.time() - t_start) * 1000
                inst_port = instance.port
                inst_type = instance.inst_type
                await self.instance_pool.release_instance(inst_port, latency_ms, success)
                instance = None

                if self.mongo_store:
                    asyncio.create_task(self.mongo_store.log_request({
                        "request_id": task_id, "instance_port": inst_port,
                        "instance_type": inst_type, "protocol": "http",
                        "algorithm": self.instance_pool._algorithm.value,
                        "latency_ms": latency_ms, "success": success,
                        "scenario": data.get("scenario", ""),
                    }))

                return web.json_response({
                    "id": task_id, "object": "chat.completion",
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop" if content else "error"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "instance_port": inst_port, "instance_type": inst_type,
                    "protocol": "http", "processing_time_ms": int(latency_ms),
                })
        except Exception as e:
            latency_ms = (_time.time() - t_start) * 1000
            if instance:
                await self.instance_pool.release_instance(instance.port, latency_ms, False)
            # Cleanup orphaned DDS waiters
            if self.dds.is_available():
                self.dds._pending_agent_responses.pop(task_id, None)
                self.dds._pending_agent_responses.pop(f"stream_{task_id}", None)
            logger.error(f"Pool chat error: {e}")
            return web.json_response({"error": str(e)}, status=500)
        finally:
            # ALWAYS decrement active counter (streaming and non-streaming)
            if self.redis_mgr:
                await self.redis_mgr.decr_active()

    async def handle_set_algorithm(self, request: web.Request) -> web.Response:
        """PUT /api/v1/routing/algorithm — change routing algorithm at runtime."""
        if not self.instance_pool:
            return web.json_response({"error": "Instance pool not configured"}, status=404)
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        algo_name = data.get("algorithm", "")
        try:
            from instance_pool import RoutingAlgorithm
            algo = RoutingAlgorithm(algo_name)
            self.instance_pool.set_algorithm(algo)
            return web.json_response({"success": True, "algorithm": algo.value})
        except ValueError:
            return web.json_response(
                {"error": f"Unknown algorithm: {algo_name}",
                 "valid": ["round_robin", "least_loaded", "weighted_score"]},
                status=400,
            )

    async def handle_pool_status(self, request: web.Request) -> web.Response:
        """GET /api/v1/pool/status — snapshot of instance pool."""
        if not self.instance_pool:
            return web.json_response({"error": "Instance pool not configured"}, status=404)
        status = await self.instance_pool.get_status()
        if self.backpressure:
            status["pressure_level"] = await self.backpressure.get_pressure_level()
            status["open_circuits"] = self.backpressure.get_open_circuits()
        return web.json_response(status)

    async def handle_metrics_summary(self, request: web.Request) -> web.Response:
        """GET /api/v1/metrics/summary — query MongoDB metrics."""
        if not self.mongo_store:
            return web.json_response({"error": "MongoDB not configured"}, status=404)
        scenario = request.query.get("scenario")
        summary = await self.mongo_store.get_metrics_summary(scenario)
        return web.json_response(summary)

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

            # Log detailed fuzzy decision with agent metrics
            logger.debug(
                f"Fuzzy selection - input: urgency={decision.inputs.get('urgency', '?'):.0f}, "
                f"complexity={decision.inputs.get('complexity', '?'):.0f} | "
                f"selected {decision.agent_id}: score={decision.agent_score:.1f}, "
                f"qos={decision.qos_profile}, strategy={decision.strategy} | "
                f"agent: load={(100-agent.slots_idle/max(agent.slots_total,1)*100):.0f}%, "
                f"latency={agent.avg_latency_ms:.0f}ms, profile={agent.agent_profile} | "
                f"all_scores={decision.all_scores} | inference={decision.inference_time_ms:.1f}ms"
            )
            return agent, decision.qos_profile, decision.strategy

        # Fallback: baseline selection
        agent = max(agents, key=lambda a: a.slots_idle)
        logger.debug(f"Using fallback selection (fuzzy disabled) - selected {agent.agent_id}")
        return agent, None, "single"

    async def _wait_for_available_agent(self, timeout_s=300, messages=None,
                                         priority=5, urgency=None, complexity=None,
                                         transport="http"):
        """Wait for an agent with idle slots using asyncio.Condition, then select using fuzzy or baseline.

        Args:
            transport: which transport the entrypoint requires — "http", "grpc",
                or "dds". Agents that do not declare this transport are filtered
                out. A heterogeneous cluster (e.g. AMD ROCm 6.3 without gRPC)
                otherwise routes to an agent that silently stalls.

        Returns: (agent, qos_profile, strategy) or (None, None, None)
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s

        async with self.registry.agent_available_condition:
            while loop.time() < deadline:
                agents = [
                    agent for agent in self.registry.agents.values()
                    if agent.status == "idle" and agent.slots_idle > 0
                    and transport in getattr(agent, "transports", ["http"])
                ]
                if agents:
                    # Found available agents, select outside the lock to avoid blocking
                    break

                # Wait for a notification or timeout
                wait_time = deadline - loop.time()
                if wait_time <= 0:
                    return None, None, None

                try:
                    await asyncio.wait_for(self.registry.agent_available_condition.wait(), timeout=wait_time)
                except asyncio.TimeoutError:
                    return None, None, None

        # Selection happens outside the lock to prevent blocking the registry
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

        # Aguardar agente disponível (paridade com handle_chat; evita fail-fast
        # quando um benchmark bate este endpoint sob carga — ver bug corrigido
        # no DDS handler em 2026-04-18).
        _req_timeout_ms = int(data.get("timeout_ms", 300000))
        _wait_s = min(_req_timeout_ms / 1000.0, float(self.config.task_timeout_seconds))
        agent, fuzzy_qos, fuzzy_strategy = await self._wait_for_available_agent(
            timeout_s=_wait_s, messages=messages, priority=data.get("priority", 5),
        )
        if not agent:
            return web.json_response({
                "error": "No agents available after timeout",
                "code": "NO_AGENTS",
            }, status=503)
        agent_url = f"http://{agent.hostname}:{agent.port}"

        if not await self.registry.adjust_slots(agent.agent_id, delta=-1):
            return web.json_response({"error": "Agent no longer available", "code": "AGENT_UNAVAILABLE"}, status=503)

        try:
            import aiohttp as _aiohttp
            async with self._pool_session.post(
                f"{agent_url}/generate",
                json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
                timeout=_aiohttp.ClientTimeout(total=120),
            ) as _resp:
                result = await _resp.json()
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

    # Whitelist of DDS topics an unauthenticated HTTP client may publish to.
    # Keep this minimal — the primary DDS writers are internal. Debug endpoint
    # MUST NOT let callers inject into agent/orchestrator control topics.
    _ALLOWED_PUBLISH_TOPICS = frozenset({"client/request", "client/ping"})

    async def handle_dds_publish(self, request: web.Request) -> web.Response:
        """Publish to a whitelisted DDS topic (debug endpoint)."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        topic = data.get("topic")
        message = data.get("message")

        if not topic or not message:
            return web.json_response({"error": "topic and message required"}, status=400)

        if topic not in self._ALLOWED_PUBLISH_TOPICS:
            logger.warning(f"Rejected /dds/publish to non-whitelisted topic: {topic!r}")
            return web.json_response(
                {"error": f"Topic '{topic}' not allowed"},
                status=403,
            )

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
