"""
Task Scheduler - manages task queuing and execution
"""

import asyncio
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """Represents a task in the scheduler"""
    task_id: str
    task_type: str  # chat, completion, embedding
    messages: List[dict]
    priority: TaskPriority = TaskPriority.NORMAL
    max_tokens: int = 256
    temperature: float = 0.7
    timeout_ms: int = 120000
    requires_context: bool = False

    # Execution metadata
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Result
    response: Optional[str] = None
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    processing_time_ms: int = 0


class TaskScheduler:
    """Scheduler for managing tasks"""

    def __init__(self, config):
        self.config = config
        self.tasks: Dict[str, Task] = {}
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running_tasks: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        self._task_handlers: Dict[str, Callable] = {}

        # Start background tasks
        self._cleanup_task = None

    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a task type"""
        self._task_handlers[task_type] = handler

    async def submit_task(self, task: Task) -> str:
        """Submit a new task"""
        async with self._lock:
            task.status = TaskStatus.QUEUED
            self.tasks[task.task_id] = task

        # Add to priority queue (lower number = higher priority)
        priority = task.priority.value
        await self.queue.put((priority, task.task_id))

        logger.info(f"Task {task.task_id} submitted with priority {task.priority.name}")
        return task.task_id

    async def submit_chat(self, messages: List[dict], **kwargs) -> str:
        """Submit a chat task"""
        task = Task(
            task_id=str(uuid.uuid4()),
            task_type="chat",
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.default_max_tokens),
            temperature=kwargs.get("temperature", self.config.default_temperature),
            priority=TaskPriority(kwargs.get("priority", TaskPriority.NORMAL.value)),
            timeout_ms=kwargs.get("timeout_ms", self.config.task_timeout_seconds * 1000),
        )
        return await self.submit_task(task)

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        async with self._lock:
            return self.tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False

            task.status = TaskStatus.CANCELLED
            logger.info(f"Task {task_id} cancelled")
            return True

    async def get_next_task(self) -> Optional[Task]:
        """Get next task from queue"""
        try:
            # Wait for task with timeout
            priority, task_id = await asyncio.wait_for(
                self.queue.get(),
                timeout=1.0
            )

            async with self._lock:
                task = self.tasks.get(task_id)
                if task and task.status == TaskStatus.CANCELLED:
                    return None
                if task:
                    task.status = TaskStatus.RUNNING
                    task.started_at = time.time()
                    self.running_tasks[task_id] = task

                return task

        except asyncio.TimeoutError:
            return None

    async def complete_task(self, task_id: str, response: str = None,
                           error: str = None, **metrics):
        """Mark task as completed"""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return

            task.completed_at = time.time()
            task.response = response
            task.error = error

            if error:
                task.status = TaskStatus.FAILED
            else:
                task.status = TaskStatus.COMPLETED

            # Update metrics
            if "prompt_tokens" in metrics:
                task.prompt_tokens = metrics["prompt_tokens"]
            if "completion_tokens" in metrics:
                task.completion_tokens = metrics["completion_tokens"]
            if "processing_time_ms" in metrics:
                task.processing_time_ms = metrics["processing_time_ms"]

            # Remove from running
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

            logger.info(f"Task {task_id} completed in {task.processing_time_ms}ms")

    async def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks"""
        async with self._lock:
            return [t for t in self.tasks.values()
                    if t.status in [TaskStatus.PENDING, TaskStatus.QUEUED]]

    async def get_running_tasks(self) -> List[Task]:
        """Get all running tasks"""
        async with self._lock:
            return list(self.running_tasks.values())

    async def get_completed_tasks(self, limit: int = 100) -> List[Task]:
        """Get completed tasks"""
        async with self._lock:
            completed = [t for t in self.tasks.values()
                        if t.status == TaskStatus.COMPLETED]
            return sorted(completed, key=lambda t: t.completed_at, reverse=True)[:limit]

    async def get_stats(self) -> dict:
        """Get scheduler statistics"""
        async with self._lock:
            total = len(self.tasks)
            pending = sum(1 for t in self.tasks.values() if t.status == TaskStatus.QUEUED)
            running = len(self.running_tasks)
            completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)

            return {
                "total_tasks": total,
                "pending_tasks": pending,
                "running_tasks": running,
                "completed_tasks": completed,
                "failed_tasks": failed,
                "queue_size": self.queue.qsize(),
            }
