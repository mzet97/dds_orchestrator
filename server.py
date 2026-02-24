"""
Orchestrator HTTP Server
Provides REST API for clients to interact with the orchestration system
"""

import asyncio
import logging
import time
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
                 selector: AgentSelector = None):
        self.config = config
        self.registry = registry
        self.scheduler = scheduler
        self.dds = dds_layer
        self.selector = selector or AgentSelector()  # Selector para escolher agente especializado

        self.app = None
        self.runner = None
        self.site = None

        # Background tasks
        self._heartbeat_task = None
        self._cleanup_task = None
        self._cleanup_lock = asyncio.Lock()

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
        self.app.router.add_delete('/agents/{agent_id}', self.handle_unregister_agent)

        # API v1 compatibility
        self.app.router.add_get('/api/v1/agents', self.handle_list_agents)
        self.app.router.add_get('/api/v1/agents/{agent_id}', self.handle_get_agent)
        self.app.router.add_post('/api/v1/agents/register', self.handle_register_agent)
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

    async def stop(self):
        """Stop the orchestrator server"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

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
                await self.registry.remove_stale_agents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    async def _dds_client_loop(self):
        """Background task to process DDS client requests"""
        import sys
        import time
        print("DDS client loop started", flush=True, file=sys.stderr)
        logger.info("DDS client loop started")
        self._last_debug = time.time()
        while True:
            try:
                await asyncio.sleep(0.1)  # Poll every 100ms

                if not self.dds.is_available():
                    continue

                # Debug: print every few seconds that we're polling
                now = time.time()
                if now - self._last_debug > 10:
                    msg = f"[DDS] Polling for client requests... DDS available: {self.dds.is_available()}"
                    print(msg, flush=True)  # Goes to stdout which is captured by nohup
                    logger.info(msg)
                    self._last_debug = now

                # Read client requests from DDS
                client_requests = await self.dds.read_client_requests(timeout_ms=100)

                if client_requests:
                    # print(f"RECEIVED {len(client_requests)} CLIENT REQUESTS!", flush=True, file=sys.stderr)
                    logger.info(f"Received {len(client_requests)} client requests via DDS")
                else:
                    # Debug: check if there are any messages at all
                    # This helps debug if the reader is working
                    pass  # Debug: no requests found

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
                    # Process valid requests concurrently
                    task = asyncio.create_task(self._process_dds_client_request(req))
                    task.add_done_callback(lambda t, rid=request_id: self._handle_task_completion(t, rid))

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
        except:
            messages = [{"role": "user", "content": messages_json}]

        logger.info(f"Processing DDS client request: {request_id}")

        # Get available agent
        agents = await self.registry.get_available_agents()
        if not agents:
            # Send error response
            await self._send_dds_client_response(request_id, client_id, "", success=0, error="No agents available")
            return

        agent = agents[0]
        logger.info(f"Selected agent: {agent.agent_id}")

        # Send task to agent via DDS
        dds_request = AgentTaskRequest(
            task_id=request_id,
            requester_id="orchestrator",
            task_type="chat",
            messages=messages,
            priority=5,
            timeout_ms=60000,
            requires_context=0,
        )
        await self.dds.publish_agent_request(dds_request)

        # Wait for agent response
        agent_response = await self.dds.wait_for_agent_response(request_id, timeout_ms=120000)

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

    async def _send_dds_client_response(self, request_id, client_id, content, success=True, error="", prompt_tokens=0, completion_tokens=0, processing_time_ms=0):
        """Send response to client via DDS using IDL-generated ClientResponse type"""
        from orchestrator import ClientResponse

        response = ClientResponse(
            request_id=request_id,
            content=content,
            is_final=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            processing_time_ms=processing_time_ms,
            success=success,
            error_message=error,
        )

        await self.dds.publish_client_response(response)

    async def _cleanup_loop(self):
        """Background task to cleanup old tasks"""
        while True:
            try:
                await asyncio.sleep(60)

                # Clean up old completed tasks
                async with self._cleanup_lock:
                    # Keep only last 1000 tasks
                    if len(self.scheduler.tasks) > 1000:
                        tasks_to_remove = []
                        for task_id, task in self.scheduler.tasks.items():
                            if task.status in ["completed", "failed", "cancelled"]:
                                if time.time() - task.completed_at > 3600:  # 1 hour
                                    tasks_to_remove.append(task_id)

                        for task_id in tasks_to_remove:
                            del self.scheduler.tasks[task_id]

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
            vision_enabled=data.get("vision_enabled", False),
            capabilities=data.get("capabilities", []),
        )

        await self.registry.register_agent(agent_info)

        # Registrar também no selector para seleção inteligente
        try:
            await self.selector.register_agent(
                agent_id=agent_info.agent_id,
                specialization="generic",
                max_load=agent_info.slots_idle
            )
        except Exception as e:
            logger.warning(f"Failed to register in selector: {e}")

        return web.json_response({
            "success": True,
            "agent_id": agent_info.agent_id
        })

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
        data = await request.json()
        messages = data.get("messages", [])

        if not messages:
            return web.json_response({"error": "No messages provided"}, status=400)

        # Get parameters
        max_tokens = data.get("max_tokens", self.config.default_max_tokens)
        temperature = data.get("temperature", self.config.default_temperature)
        priority = data.get("priority", TaskPriority.NORMAL.value)
        task_type = data.get("task_type", "chat")

        # Determine task type and specialization
        criteria = SelectionCriteria(
            task_type=TaskType(task_type) if task_type in [t.value for t in TaskType] else TaskType.CHAT,
            requires_vision=data.get("requires_vision", False),
            requires_embedding=data.get("requires_embedding", False),
            priority=priority
        )

        # Get any available agent from registry (simpler approach)
        agents = await self.registry.get_available_agents()
        if not agents:
            return web.json_response({
                "error": "No agents available for this task type",
                "code": "NO_AGENTS"
            }, status=503)

        # Use first available agent (simplified - can be improved)
        selected_agent_id = agents[0].agent_id
        logger.info(f"Selected agent: {selected_agent_id}")

        # Get agent info from registry
        agent = await self.registry.get_agent(selected_agent_id)

        # Create task
        task = Task(
            task_id=f"task-{int(time.time() * 1000)}",
            task_type="chat",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=TaskPriority(priority),
            assigned_agent_id=agent.agent_id,
        )

        # Update agent status
        await self.registry.update_heartbeat(agent.agent_id, status="busy", slots_idle=agent.slots_idle - 1)

        # Submit to scheduler
        await self.scheduler.submit_task(task)

        # Try DDS first, fall back to HTTP if it fails
        dds_success = False
        response_data = {}

        try:
            # Send task to agent via DDS topic
            dds_request = AgentTaskRequest(
                task_id=task.task_id,
                requester_id="orchestrator",
                task_type=task.task_type,
                messages=messages,
                priority=priority,
                timeout_ms=task.timeout_ms,
                requires_context=task.requires_context,
            )
            await self.dds.publish_agent_request(dds_request)

            # Wait for response from agent via DDS topic
            # Agent publishes to agent/response topic
            response_data = await self.dds.wait_for_agent_response(task.task_id, timeout_ms=120000)  # 120s timeout for DDS
            # Only consider success if we got actual content (not just an error message)
            # Response can be either a dict or an IdlStruct object
            content = ""
            if isinstance(response_data, dict):
                content = response_data.get("content", "")
            else:
                content = getattr(response_data, "content", "")

            if content:
                dds_success = True
        except Exception as e:
            logger.warning(f"DDS communication failed: {e}, falling back to HTTP")

        # If DDS failed, use HTTP fallback
        logger.info(f"DDS result: dds_success={dds_success}, response_data={response_data}")
        if not dds_success:
            logger.info(f"Using HTTP fallback for task {task.task_id}")
            agent_url = f"http://{agent.hostname}:{agent.port}"
            logger.info(f"Calling agent at {agent_url}")
            try:
                # Convert messages to prompt for /generate endpoint
                prompt = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    prompt += f"{role}: {content}\n"

                http_result = await self.dds.send_request_via_http(
                    agent_url,
                    {"prompt": prompt.strip(), "max_tokens": max_tokens, "temperature": temperature},
                    timeout=task.timeout_ms // 1000
                )
                logger.info(f"HTTP result: {http_result}")
                response_data = {
                    "content": http_result.get("response", ""),
                    "processing_time_ms": http_result.get("processing_time_ms", 0),
                    "success": http_result.get("success", 1)
                }
            except Exception as e:
                logger.error(f"HTTP fallback also failed: {e}")
                response_data = {"content": "", "error": str(e), "success": 0}

        # Complete task with result - handle both dict and IdlStruct
        response_content = ""
        if isinstance(response_data, dict):
            response_content = response_data.get("content", "")
            processing_time = response_data.get("processing_time_ms", 0)
        else:
            response_content = getattr(response_data, "content", "")
            processing_time = getattr(response_data, "processing_time_ms", 0)

        await self.scheduler.complete_task(
            task.task_id,
            response=response_content,
            processing_time_ms=processing_time
        )

        # Update agent status back to idle
        await self.registry.update_heartbeat(agent.agent_id, status="idle", slots_idle=agent.slots_idle + 1)

        return web.json_response({
            "task_id": task.task_id,
            "agent_id": agent.agent_id,
            "status": "completed" if response_content else "failed",
            "response": response_content,
        })

    async def handle_generate(self, request: web.Request) -> web.Response:
        """Handle generate request"""
        data = await request.json()
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

        agent = available[0]
        agent_url = f"http://{agent.hostname}:{agent.port}"
        result = await self.dds.send_request_via_http(
            agent_url,
            {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        )

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
        data = await request.json()
        topic = data.get("topic")
        message = data.get("message")

        if not topic or not message:
            return web.json_response({"error": "topic and message required"}, status=400)

        await self.publish(topic, message)

        return web.json_response({"success": True})


# Helper for agent info dict
def asdict(obj):
    """Convert dataclass to dict"""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: asdict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [asdict(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: asdict(v) for k, v in obj.items()}
    else:
        return obj
