"""
DDS Communication Layer for the Orchestrator
Uses CycloneDDS for pub/sub communication with agents
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# DDS Topic Names
# Cliente <-> Orchestrator
TOPIC_CLIENT_REQUEST = "client/request"
TOPIC_CLIENT_RESPONSE = "client/response"

# Orchestrator <-> Agent
TOPIC_AGENT_REGISTER = "agent/register"
TOPIC_AGENT_REQUEST = "agent/request"
TOPIC_AGENT_RESPONSE = "agent/response"
TOPIC_AGENT_STATUS = "agent/status"
TOPIC_AGENT_CONTEXT = "agent/context"

# Orchestrator commands
TOPIC_ORCHESTRATOR_COMMAND = "orchestrator/command"

# Agent <-> LLM
TOPIC_LLM_REQUEST = "llm/request"
TOPIC_LLM_RESPONSE = "llm/response"
TOPIC_LLM_STATUS = "llm/status"


@dataclass
class DDSMessage:
    """Base DDS message structure"""
    message_id: str
    timestamp: float
    source: str
    data: dict


@dataclass
class ClientTaskRequest:
    """Client task request via DDS"""
    request_id: str
    client_id: str
    task_type: str
    messages: List[dict]
    priority: int
    timeout_ms: int
    requires_context: bool


@dataclass
class ClientTaskResponse:
    """Client task response via DDS"""
    request_id: str
    client_id: str
    content: str
    is_final: bool
    prompt_tokens: int
    completion_tokens: int
    processing_time_ms: int
    success: bool
    error_message: Optional[str]


@dataclass
class AgentRegistration:
    """Agent registration message"""
    agent_id: str
    hostname: str
    port: int
    model: str
    vram_available_mb: int
    slots_idle: int
    vision_enabled: bool
    capabilities: List[str]


@dataclass
class AgentTaskRequest:
    """Task request to agent"""
    task_id: str
    requester_id: str
    task_type: str
    messages: List[dict]
    priority: int
    timeout_ms: int
    requires_context: bool


@dataclass
class AgentTaskResponse:
    """Task response from agent"""
    task_id: str
    agent_id: str
    content: str
    is_final: bool
    prompt_tokens: int
    completion_tokens: int
    processing_time_ms: int
    success: bool
    error_message: Optional[str]


@dataclass
class AgentHeartbeat:
    """Agent heartbeat message"""
    agent_id: str
    state: str  # idle, busy, error
    current_slots: int
    memory_usage_mb: int
    timestamp: float


# === LLM Communication ===

@dataclass
class LLMRequest:
    """Request from Agent to LLM"""
    request_id: str
    agent_id: str
    prompt: str
    max_tokens: int
    temperature: float
    stream: bool = False
    stop: Optional[List[str]] = None


@dataclass
class LLMResponse:
    """Response from LLM to Agent"""
    request_id: str
    agent_id: str
    content: str
    is_final: bool
    prompt_tokens: int
    completion_tokens: int
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class LLMStatus:
    """Status message from LLM"""
    llm_id: str
    status: str  # ready, busy, error
    slots_available: int
    slots_total: int
    memory_used_mb: int
    memory_total_mb: int
    current_model: str
    timestamp: float


class DDSLayer:
    """DDS communication layer for orchestrator"""

    def __init__(self, config):
        self.config = config
        self.dds_available = False
        self.participant = None
        self.publishers: Dict[str, Any] = {}
        self.subscribers: Dict[str, Any] = {}
        self.handlers: Dict[str, Callable] = {}

        # Initialize DDS if available
        if config.dds_enabled:
            self._init_dds()

    def _init_dds(self):
        """Initialize DDS entities"""
        try:
            import cyclonedds
            from cyclonedds.domain import DomainParticipant
            from cyclonedds.topic import Topic
            from cyclonedds.pub import Publisher
            from cyclonedds.sub import Subscriber
            from cyclonedds.idl import IdlStruct
            from cyclonedds.idl.types import sequence

            # Create domain participant
            self.participant = DomainParticipant(self.config.dds_domain)

            # Create topics
            self._create_topics()

            # Create publishers and subscribers
            self._create_pubsub()

            self.dds_available = True
            logger.info(f"DDS initialized on domain {self.config.dds_domain}")

        except ImportError:
            logger.warning("CycloneDDS not available, using HTTP fallback")
            self.dds_available = False
        except Exception as e:
            logger.error(f"Failed to initialize DDS: {e}")
            self.dds_available = False

    def _create_topics(self):
        """Create DDS topics"""
        from cyclonedds.topic import Topic
        from cyclonedds.idl import IdlStruct
        # Using new cyclonedds API - bounded_str[N] instead of string, int32 instead of long
        # boolean doesn't exist in new API, use Python's bool
        from cyclonedds.idl.types import bounded_str, int32, sequence

        # Define IDL types dynamically for Orchestrator topics
        # Agent Registration
        class AgentRegistrationType(IdlStruct):
            agent_id: bounded_str[256]
            hostname: bounded_str[256]
            port: int32
            model: bounded_str[256]
            vram_available_mb: int32
            slots_idle: int32
            vision_enabled: bool
            reasoning_enabled: bool
            registered_at: int32

        # Agent Status
        class AgentStatusType(IdlStruct):
            agent_id: bounded_str[256]
            state: bounded_str[64]
            current_slots: int32
            idle_slots: int32
            memory_usage_mb: int32
            vram_usage_mb: int32
            current_model: bounded_str[256]
            last_heartbeat: int32

        # Task Request
        class TaskRequestType(IdlStruct):
            task_id: bounded_str[256]
            requester_id: bounded_str[256]
            task_type: bounded_str[64]
            messages_json: bounded_str[16384]
            priority: int32
            timeout_ms: int32
            requires_context: bool
            context_id: bounded_str[256]
            created_at: int32

        # Task Response
        class TaskResponseType(IdlStruct):
            task_id: bounded_str[256]
            agent_id: bounded_str[256]
            content: bounded_str[16384]
            is_final: bool
            prompt_tokens: int32
            completion_tokens: int32
            processing_time_ms: int32
            success: bool
            error_message: bounded_str[1024]
            created_at: int32

        # Store types for later use
        self._topic_types = {
            TOPIC_AGENT_REGISTER: AgentRegistrationType,
            TOPIC_AGENT_STATUS: AgentStatusType,
            TOPIC_AGENT_REQUEST: TaskRequestType,
            TOPIC_AGENT_RESPONSE: TaskResponseType,
        }

        # Create topics
        self.topics = {}
        for topic_name, topic_type in self._topic_types.items():
            self.topics[topic_name] = Topic(self.participant, topic_name, topic_type)
            logger.info(f"Created topic: {topic_name}")

    def _create_pubsub(self):
        """Create publishers and subscribers"""
        from cyclonedds.pub import DataWriter
        from cyclonedds.sub import DataReader
        from cyclonedds.core import Qos, Policy
        from cyclonedds.util import duration

        # QoS for reliable communication (requests, responses)
        self.qos_reliable = Qos(
            Policy.Reliability.Reliable(duration(seconds=10)),
            Policy.Durability.Volatile,
            Policy.History.KeepLast(1),
        )

        # QoS for best effort (status, heartbeat)
        self.qos_best_effort = Qos(
            Policy.Reliability.BestEffort,
            Policy.Durability.Volatile,
            Policy.History.KeepLast(5),
        )

        # Create publishers
        self.publishers = {
            TOPIC_AGENT_REQUEST: DataWriter(
                self.participant, self.topics[TOPIC_AGENT_REQUEST], self.qos_reliable
            ),
            TOPIC_AGENT_REGISTER: DataWriter(
                self.participant, self.topics[TOPIC_AGENT_REGISTER], self.qos_reliable
            ),
        }

        # Create subscribers with readers
        self.subscribers = {
            TOPIC_AGENT_STATUS: DataReader(
                self.participant, self.topics[TOPIC_AGENT_STATUS], self.qos_best_effort
            ),
            TOPIC_AGENT_RESPONSE: DataReader(
                self.participant, self.topics[TOPIC_AGENT_RESPONSE], self.qos_reliable
            ),
        }

        logger.info(f"Created {len(self.publishers)} publishers and {len(self.subscribers)} subscribers")

    async def publish(self, topic: str, data: dict):
        """Publish message to topic"""
        if not self.dds_available:
            logger.debug(f"DDS unavailable, skipping publish to {topic}")
            return

        # Use DDS writer if available
        if topic in self.publishers:
            try:
                topic_type = self._topic_types.get(topic)
                if topic_type:
                    # Create message instance and write
                    msg = topic_type(**data)
                    self.publishers[topic].write(msg)
                    logger.debug(f"Published to {topic}: {data}")
            except Exception as e:
                logger.error(f"Failed to publish to {topic}: {e}")
        else:
            logger.debug(f"No writer for topic {topic}, skipping")

    def read_messages(self, topic: str, timeout_ms: int = 100) -> list:
        """Read messages from a topic subscriber"""
        if not self.dds_available or topic not in self.subscribers:
            return []

        try:
            from cyclonedds.util import duration
            samples = self.subscribers[topic].take(timeout=duration(milliseconds=timeout_ms))
            return [s.data for s in samples if s.data]
        except Exception as e:
            logger.debug(f"No messages from {topic}: {e}")
            return []

    async def read_status_updates(self, timeout_ms: int = 100) -> list:
        """Read agent status updates"""
        return self.read_messages(TOPIC_AGENT_STATUS, timeout_ms)

    async def read_responses(self, timeout_ms: int = 100) -> list:
        """Read agent task responses"""
        return self.read_messages(TOPIC_AGENT_RESPONSE, timeout_ms)

    async def wait_for_agent_response(self, task_id: str, timeout_ms: int = 60000) -> dict:
        """Wait for a specific agent response by task_id"""
        import asyncio

        if not self.dds_available:
            logger.warning("DDS not available, cannot wait for response")
            return {"content": "", "error": "DDS not available"}

        start_time = time.time()
        timeout_seconds = timeout_ms / 1000.0

        while time.time() - start_time < timeout_seconds:
            responses = self.read_messages(TOPIC_AGENT_RESPONSE, timeout_ms=100)

            for response in responses:
                if response.get("task_id") == task_id:
                    logger.info(f"Received response for task {task_id}")
                    return response

            await asyncio.sleep(0.01)  # Small sleep to avoid CPU spinning

        logger.warning(f"Timeout waiting for response for task {task_id}")
        return {"content": "", "error": "Timeout waiting for response"}

    async def subscribe(self, topic: str, handler: Callable):
        """Subscribe to topic with handler"""
        self.handlers[topic] = handler

        if not self.dds_available:
            logger.warning(f"DDS unavailable, {topic} subscription will not receive messages")
            return

        # In production, this would create a DataReader and listener
        logger.info(f"Subscribed to {topic}")

    async def publish_agent_request(self, request: AgentTaskRequest):
        """Publish task request to agents"""
        data = asdict(request)
        await self.publish(TOPIC_AGENT_REQUEST, data)

    async def publish_orchestrator_command(self, command_id: str,
                                          target_agent_id: str,
                                          action: str, payload: str):
        """Publish command to agent(s)"""
        data = {
            "command_id": command_id,
            "target_agent_id": target_agent_id,
            "action": action,
            "payload": payload,
        }
        await self.publish(TOPIC_ORCHESTRATOR_COMMAND, data)

    # Client DDS communication methods
    async def publish_client_request(self, request: ClientTaskRequest):
        """Publish client task request to orchestrator"""
        data = asdict(request)
        await self.publish(TOPIC_CLIENT_REQUEST, data)

    async def subscribe_client_request(self, handler: Callable):
        """Subscribe to client requests"""
        await self.subscribe(TOPIC_CLIENT_REQUEST, handler)

    async def publish_client_response(self, response: ClientTaskResponse):
        """Publish response to client via DDS"""
        data = asdict(response)
        await self.publish(TOPIC_CLIENT_RESPONSE, data)

    async def subscribe_client_response(self, handler: Callable):
        """Subscribe to client responses"""
        await self.subscribe(TOPIC_CLIENT_RESPONSE, handler)

    # HTTP Fallback methods
    async def send_request_via_http(self, agent_url: str, request: dict,
                                     timeout: int = 120) -> dict:
        """Send request via HTTP fallback"""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{agent_url}/generate",
                    json=request,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    return await response.json()
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_agent_health_via_http(self, agent_url: str) -> dict:
        """Get agent health via HTTP"""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{agent_url}/health") as response:
                    return await response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"available": False, "error": str(e)}

    def close(self):
        """Close DDS connections and release resources"""
        if self.participant:
            try:
                self.participant.close()
                logger.info("DDS participant closed")
            except Exception as e:
                logger.error(f"Error closing DDS participant: {e}")

    def is_available(self) -> bool:
        """Check if DDS is available"""
        return self.dds_available


class DDSMessageSerializer:
    """Serialize/deserialize DDS messages"""

    @staticmethod
    def serialize(obj) -> bytes:
        """Serialize object to bytes"""
        return json.dumps(asdict(obj)).encode('utf-8')

    @staticmethod
    def deserialize(data: bytes, msg_type):
        """Deserialize bytes to object"""
        obj_dict = json.loads(data.decode('utf-8'))
        return msg_type(**obj_dict)
