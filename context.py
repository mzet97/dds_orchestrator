#!/usr/bin/env python3
"""
Context Manager for DDS-LLM-Orchestrator
Manages conversation context and state across agents
"""
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import ChatMessage


@dataclass
class ConversationContext:
    """Conversation context"""
    context_id: str
    user_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))
    last_updated: int = field(default_factory=lambda: int(time.time()))
    metadata: Dict = field(default_factory=dict)
    max_messages: int = 20


class ContextManager:
    """
    Manages conversation contexts across agents
    """

    def __init__(self, max_contexts: int = 1000, max_messages_per_context: int = 20):
        self.max_contexts = max_contexts
        self.max_messages = max_messages_per_context

        # Context storage
        self._contexts: Dict[str, ConversationContext] = {}
        self._user_contexts: Dict[str, List[str]] = {}  # user_id -> [context_ids]

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def create_context(
        self,
        user_id: str,
        initial_message: Optional[ChatMessage] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Create a new conversation context"""
        async with self._lock:
            # Check if at max capacity
            if len(self._contexts) >= self.max_contexts:
                await self._cleanup_oldest()

            context_id = str(uuid.uuid4())
            context = ConversationContext(
                context_id=context_id,
                user_id=user_id,
                metadata=metadata or {},
            )

            if initial_message:
                context.messages.append(initial_message)

            self._contexts[context_id] = context

            # Track user contexts
            if user_id not in self._user_contexts:
                self._user_contexts[user_id] = []
            self._user_contexts[user_id].append(context_id)

            return context_id

    async def get_context(self, context_id: str) -> Optional[ConversationContext]:
        """Get context by ID"""
        async with self._lock:
            return self._contexts.get(context_id)

    async def add_message(self, context_id: str, message: ChatMessage) -> bool:
        """Add message to context"""
        async with self._lock:
            context = self._contexts.get(context_id)
            if not context:
                return False

            context.messages.append(message)
            context.last_updated = int(time.time())

            # Trim if too many messages
            if len(context.messages) > self.max_messages:
                # Keep system message if exists, trim from beginning
                if context.messages and context.messages[0].role == "system":
                    context.messages = [context.messages[0]] + context.messages[-(self.max_messages - 1):]
                else:
                    context.messages = context.messages[-self.max_messages:]

            return True

    async def get_messages(self, context_id: str) -> List[ChatMessage]:
        """Get all messages in context"""
        async with self._lock:
            context = self._contexts.get(context_id)
            if not context:
                return []
            return list(context.messages)

    async def clear_context(self, context_id: str) -> bool:
        """Clear context"""
        async with self._lock:
            if context_id in self._contexts:
                context = self._contexts[context_id]
                user_id = context.user_id

                del self._contexts[context_id]

                # Remove from user contexts
                if user_id in self._user_contexts:
                    if context_id in self._user_contexts[user_id]:
                        self._user_contexts[user_id].remove(context_id)

                return True
            return False

    async def get_user_contexts(self, user_id: str) -> List[str]:
        """Get all context IDs for a user"""
        async with self._lock:
            return list(self._user_contexts.get(user_id, []))

    async def _cleanup_oldest(self):
        """Remove oldest context"""
        if not self._contexts:
            return

        # Find oldest context
        oldest_id = min(self._contexts.keys(), key=lambda k: self._contexts[k].last_updated)
        await self.clear_context(oldest_id)

    async def cleanup_user(self, user_id: str):
        """Remove all contexts for a user"""
        async with self._lock:
            context_ids = self._user_contexts.get(user_id, [])
            for context_id in context_ids:
                if context_id in self._contexts:
                    del self._contexts[context_id]
            self._user_contexts[user_id] = []

    async def get_stats(self) -> Dict:
        """Get context manager statistics"""
        async with self._lock:
            total_messages = sum(len(c.messages) for c in self._contexts.values())
            return {
                "total_contexts": len(self._contexts),
                "total_users": len(self._user_contexts),
                "total_messages": total_messages,
                "max_contexts": self.max_contexts,
            }
