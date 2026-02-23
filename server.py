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
from dds import DDSLayer, AgentTaskRequest

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

        # Task management
        self.app.router.add_post('/chat', self.handle_chat)
        self.app.router.add_post('/generate', self.handle_generate)
        self.app.router.add_get('/tasks/{task_id}', self.handle_get_task)
        self.app.router.add_delete('/tasks/{task_id}', self.handle_cancel_task)
        self.app.router.add_get('/tasks', self.handle_list_tasks)

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
        data = await request.json()

        agent_info = AgentInfo(
            agent_id=data["agent_id"],
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
        await self.selector.register_agent(
            agent_id=agent_info.agent_id,
            specialization=agent_info.specialization if hasattr(agent_info, 'specialization') else "generic",
            max_load=agent_info.slots_idle
        )

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

        # Select best agent using selector (considera especialização)
        selected_agent_id = await self.selector.select_agent(criteria)

        if not selected_agent_id:
            # Fallback: usar registry diretamente
            agent = await self.registry.select_agent(
                requirements={"model": data.get("model")}
            )
            if agent:
                selected_agent_id = agent.agent_id

        if not selected_agent_id:
            return web.json_response({
                "error": "No agents available for this task type",
                "code": "NO_AGENTS"
            }, status=503)

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

        # If DDS available, send via DDS
        if self.dds.is_available():
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
        else:
            # HTTP fallback - send directly to agent
            agent_url = f"http://{agent.hostname}:{agent.port}"
            result = await self.dds.send_request_via_http(
                agent_url,
                {"prompt": messages[-1].get("content", ""), "max_tokens": max_tokens}
            )

            # Complete task with result
            await self.scheduler.complete_task(
                task.task_id,
                response=result.get("response", ""),
                processing_time_ms=result.get("processing_time_ms", 0)
            )

            # Update agent status
            await self.registry.update_heartbeat(agent.agent_id, status="idle", slots_idle=agent.slots_idle)

        return web.json_response({
            "task_id": task.task_id,
            "agent_id": agent.agent_id,
            "status": "queued"
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
