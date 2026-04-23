"""
Configuration for the Orchestrator
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator"""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080

    # DDS settings
    dds_domain: int = 0
    dds_enabled: bool = True

    # Agent settings
    max_agents: int = 1000
    agent_timeout_seconds: int = 30

    # Task settings
    max_concurrent_tasks: int = 2000
    task_timeout_seconds: int = 120
    # Cap for fair-wait slot acquisition (fails fast on cluster deadlock
    # instead of waiting the whole task budget).
    fair_wait_cap_seconds: int = 60
    # Timeout for sync bridge to release a slot (run_coroutine_threadsafe
    # from gRPC worker thread). Should be comfortably > expected event
    # loop queue latency, generous enough to survive GC pauses.
    slot_release_timeout_seconds: int = 30

    # gRPC settings
    grpc_enabled: bool = False
    grpc_port: int = 50052  # orchestrator gRPC listen port

    # Fuzzy logic settings
    fuzzy_enabled: bool = False
    fuzzy_default_urgency: int = 5
    fuzzy_default_complexity: int = 5

    # Logging
    log_level: str = "INFO"

    # Model defaults
    default_model: str = "glm-4.6v-flash"
    default_max_tokens: int = 256
    default_temperature: float = 0.7

    # Redis settings
    redis_url: str = ""
    redis_password: str = ""

    # MongoDB settings
    mongo_url: str = ""
    mongo_db: str = "dds_orchestrator"

    # Instance Pool / Routing
    routing_algorithm: str = "least_loaded"
    max_rps: int = 5000
    instance_ports_gpu: str = ""
    instance_ports_cpu: str = ""
    instance_host: str = "192.168.1.61"
    instance_host_map: str = ""  # "host:port,host:port,..." per-instance hostname
    slots_per_gpu: int = 15
    slots_per_cpu: int = 4
    weight_gpu: float = 1.0
    weight_cpu: float = 0.3

    # Expected types for each field, used for validation after loading
    _field_types = {
        "host": str,
        "port": int,
        "dds_domain": int,
        "dds_enabled": bool,
        "max_agents": int,
        "agent_timeout_seconds": int,
        "max_concurrent_tasks": int,
        "task_timeout_seconds": int,
        "fair_wait_cap_seconds": int,
        "slot_release_timeout_seconds": int,
        "log_level": str,
        "default_model": str,
        "default_max_tokens": int,
        "default_temperature": float,
        "grpc_enabled": bool,
        "grpc_port": int,
        "redis_url": str,
        "redis_password": str,
        "mongo_url": str,
        "mongo_db": str,
        "routing_algorithm": str,
        "max_rps": int,
        "instance_ports_gpu": str,
        "instance_ports_cpu": str,
        "instance_host": str,
        "instance_host_map": str,
        "slots_per_gpu": int,
        "slots_per_cpu": int,
        "weight_gpu": float,
        "weight_cpu": float,
    }

    def load_from_file(self, config_path: Path):
        """Load configuration from YAML file.

        Supports both flat keys (host: ...) and the nested structure used in
        config.yaml (server: {host: ..., port: ...}, dds: {domain: ...}, ...).
        """
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        if not data:
            return

        # Mapping from nested YAML paths to flat OrchestratorConfig attributes.
        # Format: ("section", "yaml_key"): "config_attr"
        nested_map = {
            ("server", "host"):               "host",
            ("server", "port"):               "port",
            ("dds", "domain"):                "dds_domain",
            ("dds", "enabled"):               "dds_enabled",
            ("agents", "timeout"):            "agent_timeout_seconds",
            ("agents", "heartbeat_interval"): None,  # not a config field, ignore
            ("scheduler", "max_queue_size"):  "max_concurrent_tasks",
            ("scheduler", "priority_levels"): None,  # not a config field, ignore
        }

        # Flatten nested sections first.
        flat: dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    attr = nested_map.get((key, sub_key))
                    if attr:
                        flat[attr] = sub_val
            else:
                flat[key] = value

        # Apply flat key/value pairs with type validation.
        for key, value in flat.items():
            if hasattr(self, key):
                expected_type = self._field_types.get(key)
                if expected_type is not None and not isinstance(value, expected_type):
                    if expected_type is bool:
                        if isinstance(value, str):
                            value = value.lower() in ("true", "1", "yes")
                        else:
                            value = bool(value)
                    else:
                        value = expected_type(value)
                setattr(self, key, value)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "host": self.host,
            "port": self.port,
            "dds_domain": self.dds_domain,
            "dds_enabled": self.dds_enabled,
            "max_agents": self.max_agents,
            "agent_timeout_seconds": self.agent_timeout_seconds,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "task_timeout_seconds": self.task_timeout_seconds,
            "log_level": self.log_level,
            "default_model": self.default_model,
            "default_max_tokens": self.default_max_tokens,
            "default_temperature": self.default_temperature,
            "redis_url": self.redis_url,
            "mongo_url": self.mongo_url,
            "routing_algorithm": self.routing_algorithm,
            "max_rps": self.max_rps,
            "instance_ports_gpu": self.instance_ports_gpu,
            "instance_ports_cpu": self.instance_ports_cpu,
            "instance_host": self.instance_host,
            "instance_host_map": self.instance_host_map,
            "slots_per_gpu": self.slots_per_gpu,
            "slots_per_cpu": self.slots_per_cpu,
            "weight_gpu": self.weight_gpu,
            "weight_cpu": self.weight_cpu,
        }


def load_config_from_env() -> OrchestratorConfig:
    """Load configuration from environment variables"""
    return OrchestratorConfig(
        host=os.environ.get("ORCH_HOST", "0.0.0.0"),
        port=int(os.environ.get("ORCH_PORT", "8080")),
        dds_domain=int(os.environ.get("DDS_DOMAIN", "0")),
        dds_enabled=os.environ.get("DDS_ENABLED", "true").lower() == "true",
        max_agents=int(os.environ.get("MAX_AGENTS", "1000")),
        agent_timeout_seconds=int(os.environ.get("AGENT_TIMEOUT_SECONDS", "300")),
        max_concurrent_tasks=int(os.environ.get("MAX_CONCURRENT_TASKS", "2000")),
        task_timeout_seconds=int(os.environ.get("TASK_TIMEOUT_SECONDS", "120")),
        default_max_tokens=int(os.environ.get("DEFAULT_MAX_TOKENS", "256")),
        default_temperature=float(os.environ.get("DEFAULT_TEMPERATURE", "0.7")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        redis_url=os.environ.get("REDIS_URL", ""),
        redis_password=os.environ.get("REDIS_PASSWORD", ""),
        mongo_url=os.environ.get("MONGO_URL", ""),
        mongo_db=os.environ.get("MONGO_DB", "dds_orchestrator"),
        routing_algorithm=os.environ.get("ROUTING_ALGORITHM", "least_loaded"),
        max_rps=int(os.environ.get("MAX_RPS", "5000")),
        instance_ports_gpu=os.environ.get("INSTANCE_PORTS_GPU", ""),
        instance_ports_cpu=os.environ.get("INSTANCE_PORTS_CPU", ""),
        instance_host=os.environ.get("INSTANCE_HOST", "192.168.1.61"),
        instance_host_map=os.environ.get("INSTANCE_HOST_MAP", ""),
        slots_per_gpu=int(os.environ.get("SLOTS_PER_GPU", "15")),
        slots_per_cpu=int(os.environ.get("SLOTS_PER_CPU", "4")),
        weight_gpu=float(os.environ.get("WEIGHT_GPU", "1.0")),
        weight_cpu=float(os.environ.get("WEIGHT_CPU", "0.3")),
    )
