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

        # TODO: Implement periodic cleanup of old completed/failed tasks
        # self._cleanup_task = None

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

    @staticmethod
    def _map_priority(value) -> "TaskPriority":
        """Map external priority (any int 0-10+) or TaskPriority to TaskPriority enum.
        External scale: higher number = higher priority (e.g., 10=CRITICAL, 8+=HIGH, 5+=NORMAL, <5=LOW).
        Internal scale: lower number = higher priority (heapq min-heap).
        """
        if isinstance(value, TaskPriority):
            return value
        v = int(value)
        if v >= 10:
            return TaskPriority.CRITICAL
        elif v >= 7:
            return TaskPriority.HIGH
        elif v >= 4:
            return TaskPriority.NORMAL
        else:
            return TaskPriority.LOW

    async def submit_chat(self, messages: List[dict], **kwargs) -> str:
        """Submit a chat task"""
        task = Task(
            task_id=str(uuid.uuid4()),
            task_type="chat",
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.default_max_tokens),
            temperature=kwargs.get("temperature", self.config.default_temperature),
            priority=self._map_priority(kwargs.get("priority", TaskPriority.NORMAL.value)),
            timeout_ms=kwargs.get("timeout_ms", self.config.task_timeout_seconds * 1000),
        )
        return await self.submit_task(task)

    async def track_task(self, task: Task) -> str:
        """Track a task that is processed inline (not queued).

        Unlike submit_task(), this does NOT put the task in the priority queue.
        Use this when the caller handles routing directly and just needs the
        task to appear in stats/history with correct RUNNING status.
        """
        async with self._lock:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            self.tasks[task.task_id] = task
            self.running_tasks[task.task_id] = task
        logger.debug(f"Tracking inline task {task.task_id}")
        return task.task_id

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
            task.completed_at = time.time()
            logger.info(f"Task {task_id} cancelled")
            return True

    async def get_next_task(self) -> Optional[Task]:
        """Get next task from queue, skipping cancelled tasks"""
        try:
            while True:
                # Wait for task with timeout
                priority, task_id = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                self.queue.task_done()

                async with self._lock:
                    task = self.tasks.get(task_id)
                    if not task or task.status == TaskStatus.CANCELLED:
                        # Skip missing or cancelled tasks and try the next one
                        continue
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
            return sorted(completed, key=lambda t: t.completed_at or 0, reverse=True)[:limit]

    async def cleanup_old_tasks(self, max_tasks: int = 500, max_age_seconds: int = 300):
        """Remove old completed/failed/cancelled tasks to prevent memory growth.

        Cleanup triggers when task count exceeds max_tasks. Removes completed
        tasks older than max_age_seconds (default 5 min). If still over limit
        after age-based cleanup, removes oldest completed tasks by completed_at.
        """
        async with self._lock:
            if len(self.tasks) <= max_tasks:
                return
            completed_statuses = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT}
            now = time.time()
            # Phase 1: remove tasks older than max_age_seconds
            tasks_to_remove = []
            for task_id, task in self.tasks.items():
                if task.status in completed_statuses:
                    if task.completed_at is not None and now - task.completed_at > max_age_seconds:
                        tasks_to_remove.append(task_id)
            # Phase 2: if still over limit, remove oldest completed tasks
            if len(self.tasks) - len(tasks_to_remove) > max_tasks:
                remaining = [(tid, t) for tid, t in self.tasks.items()
                             if tid not in set(tasks_to_remove) and t.status in completed_statuses
                             and t.completed_at is not None]
                remaining.sort(key=lambda x: x[1].completed_at)
                excess = len(self.tasks) - len(tasks_to_remove) - max_tasks
                tasks_to_remove.extend(tid for tid, _ in remaining[:excess])
            for task_id in tasks_to_remove:
                self.tasks.pop(task_id, None)
                self.running_tasks.pop(task_id, None)
            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks, {len(self.tasks)} remaining")

    async def get_stats(self) -> dict:
        """Get scheduler statistics"""
        async with self._lock:
            total = len(self.tasks)
            queued = sum(1 for t in self.tasks.values() if t.status == TaskStatus.QUEUED)
            running = len(self.running_tasks)
            completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)

            return {
                "total_tasks": total,
                "queued_tasks": queued,
                "running_tasks": running,
                "completed_tasks": completed,
                "failed_tasks": failed,
                "queue_size": self.queue.qsize(),
            }
