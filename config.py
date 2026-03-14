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
    max_agents: int = 10
    agent_timeout_seconds: int = 300

    # Task settings
    max_concurrent_tasks: int = 50
    task_timeout_seconds: int = 120

    # gRPC settings
    grpc_enabled: bool = False
    grpc_port: int = 50052  # orchestrator gRPC listen port

    # Logging
    log_level: str = "INFO"

    # Model defaults
    default_model: str = "glm-4.6v-flash"
    default_max_tokens: int = 256
    default_temperature: float = 0.7

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
        "log_level": str,
        "default_model": str,
        "default_max_tokens": int,
        "default_temperature": float,
        "grpc_enabled": bool,
        "grpc_port": int,
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
        }


def load_config_from_env() -> OrchestratorConfig:
    """Load configuration from environment variables"""
    return OrchestratorConfig(
        host=os.environ.get("ORCH_HOST", "0.0.0.0"),
        port=int(os.environ.get("ORCH_PORT", "8080")),
        dds_domain=int(os.environ.get("DDS_DOMAIN", "0")),
        dds_enabled=os.environ.get("DDS_ENABLED", "true").lower() == "true",
        max_agents=int(os.environ.get("MAX_AGENTS", "10")),
        agent_timeout_seconds=int(os.environ.get("AGENT_TIMEOUT_SECONDS", "300")),
        max_concurrent_tasks=int(os.environ.get("MAX_CONCURRENT_TASKS", "50")),
        task_timeout_seconds=int(os.environ.get("TASK_TIMEOUT_SECONDS", "120")),
        default_max_tokens=int(os.environ.get("DEFAULT_MAX_TOKENS", "256")),
        default_temperature=float(os.environ.get("DEFAULT_TEMPERATURE", "0.7")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )
