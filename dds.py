"""
DDS Communication Layer for the Orchestrator
Uses CycloneDDS for pub/sub communication with agents
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict

try:
    import orjson
    def _json_dumps(obj): return orjson.dumps(obj).decode('utf-8')
    def _json_loads(s): return orjson.loads(s) if isinstance(s, (bytes, bytearray)) else orjson.loads(s.encode('utf-8'))
except ImportError:
    import json
    def _json_dumps(obj): return json.dumps(obj)
    def _json_loads(s): return json.loads(s)

# Ensure orchestrator package is importable
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

# IDL types are imported lazily inside _init_dds() to avoid crashing the
# entire module when cyclonedds is not installed (HTTP-only mode).
TaskRequest = None
TaskResponse = None
AgentRegistration = None
AgentStatus = None
ClientRequest = None
ClientResponse = None


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

# Agent <-> LLM (not used by orchestrator; listed for reference only)
# Actual topic names used by agent_llm_dds.py and C++ llama-server:
#   "llama_chat_completion_request"
#   "llama_chat_completion_response"
#   "llama_server_status"


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
    messages_json: str  # JSON string of messages
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
class AgentTaskRequest:
    """Task request to agent"""
    task_id: str
    requester_id: str
    task_type: str
    messages: List[dict]
    priority: int
    timeout_ms: int
    requires_context: bool
    stream: bool = False
    max_tokens: int = 50
    temperature: float = 0.7
    urgency: int = 5
    complexity: int = 5
    target_agent_id: str = ""
    context_id: str = ""


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
    is_final: int
    prompt_tokens: int
    completion_tokens: int
    processing_time_ms: int
    success: int
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
        self.topics: Dict[str, Any] = {}
        self._topic_types: Dict[str, Any] = {}
        # Per-task waiters for agent responses: task_id -> (asyncio.Event, [result])
        self._pending_agent_responses: Dict[str, Any] = {}
        # Per-request waiters for client responses: request_id -> (asyncio.Event, [result])
        self._pending_client_responses: Dict[str, Any] = {}
        # Event loop reference captured when DDS is initialized (for thread-safe callbacks)
        self._event_loop = None

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

            # Import IDL types now that we know cyclonedds is available.
            # These are assigned to module-level names so other code can
            # reference them without guarding against ImportError.
            global TaskRequest, TaskResponse, AgentRegistration, AgentStatus
            global ClientRequest, ClientResponse
            from orchestrator import (
                TaskRequest, TaskResponse,
                AgentRegistration, AgentStatus,
                ClientRequest, ClientResponse,
            )

            # Create domain participant
            self.participant = DomainParticipant(self.config.dds_domain)

            # Create topics
            self._create_topics()

            # Create publishers and subscribers
            self._create_pubsub()

            # Event loop will be captured lazily when subscribe() is called
            # (at init time the loop is not yet running)
            self._event_loop = None

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
        # Use IDL-generated types from orchestrator module (imported in _init_dds)

        # Store types for later use
        self._topic_types = {
            TOPIC_AGENT_REGISTER: AgentRegistration,
            TOPIC_AGENT_STATUS: AgentStatus,
            TOPIC_AGENT_REQUEST: TaskRequest,
            TOPIC_AGENT_RESPONSE: TaskResponse,
            TOPIC_CLIENT_REQUEST: ClientRequest,
            TOPIC_CLIENT_RESPONSE: ClientResponse,
        }

        # Create topics
        self.topics = {}
        for topic_name, topic_type in self._topic_types.items():
            self.topics[topic_name] = Topic(self.participant, topic_name, topic_type)
            logger.info(f"Created topic: {topic_name}")

    def _create_pubsub(self):
        """Create publishers and subscribers for orchestrator role.

        The orchestrator:
        - Reads from: client/request, agent/register, agent/response, agent/status
        - Writes to: agent/request, client/response
        """
        from cyclonedds.pub import DataWriter
        from cyclonedds.sub import DataReader
        from cyclonedds.core import Policy
        from cyclonedds.qos import Qos
        from cyclonedds.util import duration

        # QoS for reliable communication (requests, responses).
        # Reliable timeout 120s: long LLM inferences (S3 prompts ~15-20s)
        #   need much more than 2s for the reliable_ack window.
        # KeepLast(128): buffer up to 128 undelivered messages so that rapid
        #   back-to-back writes under multi-client load are not silently dropped.
        # TRANSPORT_PRIORITY(0): default priority; overridden per-request via priority writers.
        self.qos_reliable = Qos(
            Policy.Reliability.Reliable(duration(seconds=120)),
            Policy.Durability.Volatile,
            Policy.History.KeepLast(128),
            Policy.TransportPriority(0),
            Policy.LatencyBudget(duration(microseconds=0)),
        )

        # QoS for best effort (status, heartbeat)
        # DEADLINE(2s): triggers on_requested_deadline_missed when data stops
        # LIVELINESS AUTOMATIC(1s): middleware detects process failure via lease
        self.qos_best_effort = Qos(
            Policy.Reliability.BestEffort,
            Policy.Durability.Volatile,
            Policy.History.KeepLast(5),
            Policy.Deadline(duration(seconds=2)),
            Policy.Liveliness.Automatic(lease_duration=duration(seconds=1)),
        )

        # Priority writers: CycloneDDS doesn't support per-write QoS override,
        # so we create separate DataWriters with different TRANSPORT_PRIORITY values.
        self._priority_writers: Dict[int, Any] = {}
        # QoS profile writers: keyed by profile name (low_cost, balanced, critical)
        self._qos_profile_writers: Dict[str, Any] = {}
        # Fix 2A — per-agent partition writers for agent/request.
        # Each writer uses a DDS Partition matching the agent_id so that only
        # the targeted agent receives (and deserializes) the request.
        # Eliminates the broadcast overhead where all agents process all messages.
        self._partition_writers: Dict[str, Any] = {}
        self._partition_publishers: Dict[str, Any] = {}  # keep Publisher refs alive

        # Create publishers (orchestrator sends to agents and clients)
        self.publishers = {
            TOPIC_AGENT_REQUEST: DataWriter(
                self.participant, self.topics[TOPIC_AGENT_REQUEST], self.qos_reliable
            ),
            TOPIC_CLIENT_RESPONSE: DataWriter(
                self.participant, self.topics[TOPIC_CLIENT_RESPONSE], self.qos_reliable
            ),
        }

        # Create subscribers/readers (orchestrator receives from clients, agents)
        self.subscribers = {
            TOPIC_CLIENT_REQUEST: DataReader(
                self.participant, self.topics[TOPIC_CLIENT_REQUEST], self.qos_reliable
            ),
            TOPIC_AGENT_REGISTER: DataReader(
                self.participant, self.topics[TOPIC_AGENT_REGISTER], self.qos_reliable
            ),
            TOPIC_AGENT_RESPONSE: DataReader(
                self.participant, self.topics[TOPIC_AGENT_RESPONSE], self.qos_reliable
            ),
            TOPIC_AGENT_STATUS: DataReader(
                self.participant, self.topics[TOPIC_AGENT_STATUS], self.qos_best_effort
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
                    # Create message instance with all fields via constructor
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
            # take() without timeout — some CycloneDDS versions don't support timeout kwarg
            samples = self.subscribers[topic].take()
            result = []
            for s in samples:
                # Handle both DataSample objects and raw data
                if hasattr(s, 'data'):
                    if s.data:
                        result.append(s.data)
                elif s:
                    result.append(s)
            if result:
                logger.debug(f"Read {len(result)} messages from {topic}")
            return result
        except Exception as e:
            logger.debug(f"No messages from {topic}: {e}")
            return []

    async def read_registrations(self, timeout_ms: int = 100) -> list:
        """Read agent registration messages"""
        return self.read_messages(TOPIC_AGENT_REGISTER, timeout_ms)

    async def read_status_updates(self, timeout_ms: int = 100) -> list:
        """Read agent status updates"""
        return self.read_messages(TOPIC_AGENT_STATUS, timeout_ms)

    async def read_responses(self, timeout_ms: int = 100) -> list:
        """Read agent task responses"""
        return self.read_messages(TOPIC_AGENT_RESPONSE, timeout_ms)

    async def read_client_requests(self, timeout_ms: int = 100) -> list:
        """Read client task requests"""
        return self.read_messages(TOPIC_CLIENT_REQUEST, timeout_ms)

    async def wait_for_client_response(self, request_id: str, timeout_ms: int = 60000) -> dict:
        """Wait for a specific client response by request_id.

        Uses a per-request (event, result) pair stored in _pending_client_responses.
        The event is set by dispatch_client_responses() which runs as a background
        loop and is the only consumer of the TOPIC_CLIENT_RESPONSE reader.
        """
        if not self.dds_available:
            logger.warning("DDS not available, cannot wait for response")
            return {"content": "", "error": "DDS not available"}

        event = asyncio.Event()
        result_container = [None]
        self._pending_client_responses[request_id] = (event, result_container)

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_ms / 1000.0)
            logger.info(f"Received response for client request {request_id}")
            return result_container[0]
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for response for client request {request_id}")
            return {"content": "", "error": "Timeout waiting for response"}
        finally:
            self._pending_client_responses.pop(request_id, None)

    def prepare_agent_response_waiter(self, task_id: str):
        """Pre-register a waiter for the given task_id.

        Call this BEFORE publishing the agent request so that a response
        arriving between publish and wait is not discarded by the
        dispatch_agent_responses loop.
        """
        if task_id not in self._pending_agent_responses:
            event = asyncio.Event()
            result_container = [None]
            self._pending_agent_responses[task_id] = (event, result_container)

    async def wait_for_agent_response(self, task_id: str, timeout_ms: int = 60000) -> dict:
        """Wait for a specific agent response by task_id.

        Uses a per-task (event, result) pair stored in _pending_agent_responses.
        The event is set by dispatch_agent_responses() which runs as a background
        loop and is the only consumer of the TOPIC_AGENT_RESPONSE reader — this
        avoids race conditions caused by multiple concurrent callers each calling
        take() and stealing each other's samples.

        If prepare_agent_response_waiter() was called beforehand, reuses the
        existing registration; otherwise registers a new one.
        """
        if not self.dds_available:
            logger.warning("DDS not available, cannot wait for response")
            return {"content": "", "error": "DDS not available"}

        if task_id in self._pending_agent_responses:
            event, result_container = self._pending_agent_responses[task_id]
        else:
            event = asyncio.Event()
            result_container = [None]
            self._pending_agent_responses[task_id] = (event, result_container)

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_ms / 1000.0)
            logger.info(f"Received response for task {task_id}")
            return result_container[0]
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for response for task {task_id}")
            return {"content": "", "error": "Timeout waiting for response"}
        finally:
            self._pending_agent_responses.pop(task_id, None)

    def prepare_stream_waiter(self, task_id: str):
        """Pre-register a streaming waiter for the given task_id.

        Unlike prepare_agent_response_waiter (single response), this creates
        a waiter keyed as 'stream_{task_id}' that uses an asyncio.Queue to accumulate
        individual chunks without busy waiting.
        """
        key = f"stream_{task_id}"
        if key not in self._pending_agent_responses:
            queue = asyncio.Queue()
            self._pending_agent_responses[key] = queue

    async def stream_agent_response(self, task_id: str, timeout_ms: int = 120000):
        """Async generator that yields individual response chunks from agent.

        Each chunk is a dict with keys: content, is_final, prompt_tokens,
        completion_tokens, processing_time_ms.
        """
        key = f"stream_{task_id}"
        if key not in self._pending_agent_responses:
            self.prepare_stream_waiter(task_id)

        queue = self._pending_agent_responses[key]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (timeout_ms / 1000)

        while loop.time() < deadline:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=timeout)
                yield chunk
                if chunk.get("is_final", False):
                    self._pending_agent_responses.pop(key, None)
                    return
            except asyncio.TimeoutError:
                break

        # Timeout
        self._pending_agent_responses.pop(key, None)
        yield {"content": "", "is_final": True, "error": "timeout"}

    def start_dispatchers(self):
        """Start dedicated dispatch threads for DDS responses.

        Uses tight take() loops in daemon threads with call_soon_threadsafe
        to dispatch samples to the asyncio event loop without blocking it.
        """
        if not self.dds_available:
            return

        import threading

        # Capture event loop for cross-thread dispatch
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("No running event loop for dispatch threads")
            return

        # Try event-driven listeners first; fall back to dispatch threads if unavailable.
        try:
            from cyclonedds.core import Listener
            self._dds_listeners = []  # keep refs alive

            if TOPIC_AGENT_RESPONSE in self.subscribers:
                reader = self.subscribers[TOPIC_AGENT_RESPONSE]
                dispatch_fn = self._dispatch_agent_response_sample
                def _on_agent_resp(_reader, _df=dispatch_fn, _r=reader):
                    try:
                        for s in _r.take():
                            if s and self._event_loop and not self._event_loop.is_closed():
                                self._event_loop.call_soon_threadsafe(_df, s)
                    except Exception as e:
                        logger.exception(f"Error in _on_agent_resp: {e}")
                lst = Listener(on_data_available=_on_agent_resp)
                reader.set_listener(lst)
                self._dds_listeners.append(lst)
                logger.info("Attached event-driven listener for agent responses")

            if TOPIC_CLIENT_RESPONSE in self.subscribers:
                reader = self.subscribers[TOPIC_CLIENT_RESPONSE]
                dispatch_fn = self._dispatch_client_response_sample
                def _on_client_resp(_reader, _df=dispatch_fn, _r=reader):
                    try:
                        for s in _r.take():
                            if s and self._event_loop and not self._event_loop.is_closed():
                                self._event_loop.call_soon_threadsafe(_df, s)
                    except Exception as e:
                        logger.exception(f"Error in _on_client_resp: {e}")
                lst = Listener(on_data_available=_on_client_resp)
                reader.set_listener(lst)
                self._dds_listeners.append(lst)
                logger.info("Attached event-driven listener for client responses")

            return  # Listeners attached, skip thread fallback
        except Exception as e:
            logger.warning(f"Listener attach failed ({e}); falling back to dispatch threads")

        # Fallback: dispatch threads with tight take() loop
        if TOPIC_AGENT_RESPONSE in self.subscribers:
            reader = self.subscribers[TOPIC_AGENT_RESPONSE]
            t = threading.Thread(
                target=self._take_dispatch_loop,
                args=(reader, self._dispatch_agent_response_sample),
                daemon=True, name="dds-agent-resp-dispatch")
            t.start()
            logger.info("Started dispatch thread for agent responses")

        if TOPIC_CLIENT_RESPONSE in self.subscribers:
            reader = self.subscribers[TOPIC_CLIENT_RESPONSE]
            t = threading.Thread(
                target=self._take_dispatch_loop,
                args=(reader, self._dispatch_client_response_sample),
                daemon=True, name="dds-client-resp-dispatch")
            t.start()
            logger.info("Started dispatch thread for client responses")

    # Keep old name as alias for server.py compatibility
    start_waitset_dispatchers = start_dispatchers

    def _take_dispatch_loop(self, reader, dispatch_fn):
        """Tight take() loop in dedicated thread. 0.5ms sleep between polls."""
        import time as _time
        while self.dds_available:
            try:
                samples = reader.take()
                if samples and self._event_loop and not self._event_loop.is_closed():
                    for sample in samples:
                        if sample:
                            self._event_loop.call_soon_threadsafe(dispatch_fn, sample)
            except Exception as e:
                logger.exception(f"Error in _take_dispatch_loop: {e}")
            _time.sleep(0.0005)  # 0.5ms — balances latency vs CPU

    def _dispatch_agent_response_sample(self, sample):
        """Dispatch a single agent response sample (called from asyncio thread)."""
        task_id = getattr(sample, "task_id", None)
        if not task_id:
            return
        is_final = getattr(sample, "is_final", True)

        # Streaming path
        stream_key = f"stream_{task_id}"
        if stream_key in self._pending_agent_responses:
            queue = self._pending_agent_responses[stream_key]
            queue.put_nowait({
                "content": getattr(sample, "content", ""),
                "is_final": bool(is_final),
                "prompt_tokens": getattr(sample, "prompt_tokens", 0),
                "completion_tokens": getattr(sample, "completion_tokens", 0),
                "processing_time_ms": getattr(sample, "processing_time_ms", 0),
            })
            return

        # Non-streaming path
        if task_id in self._pending_agent_responses:
            if is_final:
                event, container = self._pending_agent_responses[task_id]
                container[0] = sample
                event.set()

    def _dispatch_client_response_sample(self, sample):
        """Dispatch a single client response sample (called from asyncio thread)."""
        request_id = getattr(sample, "request_id", None)
        if request_id and request_id in self._pending_client_responses:
            event, container = self._pending_client_responses[request_id]
            container[0] = sample
            event.set()

    async def dispatch_client_responses(self):
        """Legacy polling fallback — only used if WaitSet init fails."""
        while True:
            await asyncio.sleep(3600)  # Effectively idle (WaitSet handles dispatch)

    async def dispatch_agent_responses(self):
        """Legacy polling fallback — only used if WaitSet init fails."""
        while True:
            await asyncio.sleep(3600)  # Effectively idle (WaitSet handles dispatch)

    async def subscribe(self, topic: str, handler: Callable):
        """Subscribe to topic with handler - creates a DataReader with Listener"""
        self.handlers[topic] = handler

        if not self.dds_available:
            logger.warning(f"DDS unavailable, {topic} subscription will not receive messages")
            return

        topic_obj = self.topics.get(topic)
        if not topic_obj:
            logger.warning(f"Topic {topic} not found, cannot subscribe")
            return

        try:
            from cyclonedds.sub import DataReader, Subscriber
            from cyclonedds.core import Listener

            handler_ref = handler

            # Capture event loop lazily (now we're inside an async context)
            try:
                self._event_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            captured_loop = self._event_loop

            class TopicListener(Listener):
                def on_data_available(self, reader):
                    if captured_loop is None or captured_loop.is_closed():
                        return
                    samples = reader.take()
                    for sample in samples:
                        valid = (
                            (hasattr(sample, 'sample_info') and sample.sample_info.valid_data)
                            or (not hasattr(sample, 'sample_info') and sample)
                        )
                        if valid:
                            captured_loop.call_soon_threadsafe(
                                lambda s=sample: captured_loop.create_task(handler_ref(s))
                            )

            listener = TopicListener()
            qos = self.qos_reliable if topic not in (TOPIC_AGENT_STATUS,) else self.qos_best_effort
            reader = DataReader(Subscriber(self.participant), topic_obj, qos=qos, listener=listener)
            self.subscribers[topic] = reader
            logger.info(f"Subscribed to {topic} with listener")
        except Exception as e:
            logger.error(f"Failed to create subscription for {topic}: {e}")
            logger.info(f"Subscribed to {topic} (handler registered, no listener)")

    async def publish_agent_request(self, request: AgentTaskRequest, priority: int = 0,
                                     qos_profile: str = None):
        """Publish task request to agents with optional TRANSPORT_PRIORITY or QoS profile.

        Args:
            request: The task request to publish.
            priority: DDS TRANSPORT_PRIORITY value (0=LOW, 5=NORMAL, 10=HIGH, 20=CRITICAL).
            qos_profile: Named QoS profile ("low_cost", "balanced", "critical").
                         If specified, overrides the priority-based writer.
        """
        # Convert to DDS format - messages need to be JSON string
        # Normalize messages: Pydantic ChatMessage / dataclass / dict → dict
        def _msg_to_dict(m):
            if isinstance(m, dict):
                return m
            if hasattr(m, "model_dump"):
                return m.model_dump()
            return {"role": getattr(m, "role", ""), "content": getattr(m, "content", "")}
        _messages_serialized = [_msg_to_dict(m) for m in request.messages]
        data = {
            "task_id": request.task_id,
            "requester_id": request.requester_id,
            "task_type": request.task_type,
            "messages_json": _json_dumps(_messages_serialized),
            "priority": request.priority,
            "timeout_ms": request.timeout_ms,
            "requires_context": bool(request.requires_context),
            "context_id": getattr(request, "context_id", "") or "",
            "created_at": int(time.time() * 1000),
            "stream": request.stream,
            "max_tokens": getattr(request, "max_tokens", 50),
            "temperature": getattr(request, "temperature", 0.7),
            "target_agent_id": getattr(request, "target_agent_id", ""),
        }

        if not self.dds_available:
            logger.debug("DDS unavailable, skipping publish_agent_request")
            return

        # Fix 2A: use per-agent partition writer when target_agent_id is set.
        # The partitioned DataWriter delivers only to the matching agent's reader,
        # eliminating broadcast overhead (all agents deserializing all requests).
        target = data.get("target_agent_id", "")
        if target:
            pw = self._get_partition_writer(target)
            if pw is not None:
                try:
                    topic_type = self._topic_types.get(TOPIC_AGENT_REQUEST)
                    if topic_type:
                        msg = topic_type(**data)
                        pw.write(msg)
                        logger.debug(f"Published to {TOPIC_AGENT_REQUEST} via partition '{target}'")
                        return
                except Exception as e:
                    logger.warning(f"Partition write for '{target}' failed: {e}; falling back")

        # Use QoS profile writer if specified (from fuzzy decision)
        if qos_profile:
            writer = self._get_qos_profile_writer(qos_profile)
            if writer:
                try:
                    topic_type = self._topic_types.get(TOPIC_AGENT_REQUEST)
                    if topic_type:
                        msg = topic_type(**data)
                        writer.write(msg)
                        logger.debug(f"Published to {TOPIC_AGENT_REQUEST} with QoS profile={qos_profile}")
                        return
                except Exception as e:
                    logger.warning(f"QoS profile writer failed: {e}, falling back to default")

        # Use priority-specific writer if priority > 0
        if priority > 0:
            writer = self._get_priority_writer(priority)
            if writer:
                try:
                    topic_type = self._topic_types.get(TOPIC_AGENT_REQUEST)
                    if topic_type:
                        msg = topic_type(**data)
                        writer.write(msg)
                        logger.debug(f"Published to {TOPIC_AGENT_REQUEST} with TRANSPORT_PRIORITY={priority}")
                        return
                except Exception as e:
                    logger.error(f"Failed to publish with priority writer: {e}")

        # Fallback to default writer (priority=0)
        await self.publish(TOPIC_AGENT_REQUEST, data)

    def _get_priority_writer(self, priority: int):
        """Get or create a DataWriter with the specified TRANSPORT_PRIORITY."""
        if priority in self._priority_writers:
            return self._priority_writers[priority]

        try:
            from cyclonedds.pub import DataWriter
            from cyclonedds.core import Policy
            from cyclonedds.qos import Qos
            from cyclonedds.util import duration

            qos = Qos(
                Policy.Reliability.Reliable(duration(seconds=10)),
                Policy.Durability.Volatile,
                Policy.History.KeepLast(8),
                Policy.TransportPriority(priority),
            )
            writer = DataWriter(
                self.participant, self.topics[TOPIC_AGENT_REQUEST], qos
            )
            self._priority_writers[priority] = writer
            logger.info(f"Created priority writer with TRANSPORT_PRIORITY={priority}")
            return writer
        except Exception as e:
            logger.error(f"Failed to create priority writer: {e}")
            return None

    def _get_qos_profile_writer(self, profile_name: str):
        """Get or create a DataWriter for the named QoS profile.

        Uses qos_profiles.py factory to create CycloneDDS Qos objects.
        Writers are cached for reuse.
        """
        if profile_name in self._qos_profile_writers:
            return self._qos_profile_writers[profile_name]

        try:
            from cyclonedds.pub import DataWriter
            from qos_profiles import QoSProfile, create_qos

            profile = QoSProfile(profile_name)
            qos = create_qos(profile)
            if qos is None:
                return None

            writer = DataWriter(
                self.participant, self.topics[TOPIC_AGENT_REQUEST], qos
            )
            self._qos_profile_writers[profile_name] = writer
            logger.info(f"Created QoS profile writer: {profile_name}")
            return writer
        except Exception as e:
            logger.error(f"Failed to create QoS profile writer '{profile_name}': {e}")
            return None

    def _get_partition_writer(self, agent_id: str):
        """Get or create a DataWriter for agent/request restricted to agent_id partition.

        Fix 2A: with a partitioned writer the DDS middleware delivers the message
        only to the DataReader that subscribed with matching partition, eliminating
        the per-agent Python-level filtering and broadcast deserialization overhead.
        Falls back to None so the caller can use the default broadcast writer.
        """
        if agent_id in self._partition_writers:
            return self._partition_writers[agent_id]

        try:
            from cyclonedds.pub import Publisher, DataWriter
            from cyclonedds.core import Policy
            from cyclonedds.qos import Qos
            from cyclonedds.util import duration

            pub_qos = Qos(Policy.Partition([agent_id]))
            pub = Publisher(self.participant, pub_qos)
            writer = DataWriter(pub, self.topics[TOPIC_AGENT_REQUEST], self.qos_reliable)
            self._partition_publishers[agent_id] = pub   # keep Publisher alive
            self._partition_writers[agent_id] = writer
            logger.info(f"Created partition writer for agent '{agent_id}'")
            return writer
        except Exception as e:
            logger.warning(f"Failed to create partition writer for '{agent_id}': {e}; "
                           "falling back to broadcast")
            return None

    async def publish_orchestrator_command(self, command_id: str,
                                          target_agent_id: str,
                                          action: str, payload: str):
        """Publish command to agent(s)"""
        if not self.dds_available:
            logger.warning(
                f"[DDS] publish_orchestrator_command skipped (DDS unavailable): "
                f"command_id={command_id}, action={action}, target={target_agent_id}"
            )
            return
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
        # Convert to DDS format - add created_at
        data = {
            "request_id": request.request_id,
            "client_id": request.client_id,
            "task_type": request.task_type,
            "messages_json": request.messages_json,
            "priority": request.priority,
            "timeout_ms": request.timeout_ms,
            "requires_context": request.requires_context,
            "created_at": int(time.time()),
        }
        await self.publish(TOPIC_CLIENT_REQUEST, data)

    async def subscribe_client_request(self, handler: Callable):
        """Subscribe to client requests"""
        await self.subscribe(TOPIC_CLIENT_REQUEST, handler)

    async def publish_client_response(self, response):
        """Publish response to client via DDS (accepts IDL ClientResponse object directly)"""
        if not self.dds_available:
            return
        if TOPIC_CLIENT_RESPONSE in self.publishers:
            try:
                self.publishers[TOPIC_CLIENT_RESPONSE].write(response)
                logger.info(f"Published client response to {TOPIC_CLIENT_RESPONSE}")
            except Exception as e:
                logger.error(f"Failed to publish client response: {e}")

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
        """Close DDS connections and release resources.

        CycloneDDS Python bindings release C entities when the Python object
        is garbage-collected.  Clearing the dicts drops all references so
        the GC can finalize them.  The participant must be cleared last
        because deleting it recursively deletes all child entities.
        """
        self.publishers.clear()
        self.subscribers.clear()
        self.topics.clear()
        self._partition_writers.clear()
        self._partition_publishers.clear()
        self._priority_writers.clear()
        self._qos_profile_writers.clear()

        if self.participant:
            try:
                del self.participant
            except Exception as e:
                logger.error(f"Error closing DDS participant: {e}")
            self.participant = None
            logger.info("DDS participant released")

        self.handlers.clear()
        self.dds_available = False

    def is_available(self) -> bool:
        """Check if DDS is available"""
        return self.dds_available


class DDSMessageSerializer:
    """Serialize/deserialize DDS messages"""

    @staticmethod
    def serialize(obj) -> bytes:
        """Serialize object to bytes"""
        return _json_dumps(asdict(obj)).encode('utf-8')

    @staticmethod
    def deserialize(data: bytes, msg_type):
        """Deserialize bytes to object"""
        obj_dict = _json_loads(data.decode('utf-8'))
        return msg_type(**obj_dict)
