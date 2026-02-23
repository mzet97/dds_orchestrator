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

    # Logging
    log_level: str = "INFO"

    # Model defaults
    default_model: str = "glm-4.6v-flash"
    default_max_tokens: int = 256
    default_temperature: float = 0.7

    def load_from_file(self, config_path: Path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        if not data:
            return

        # Update attributes from config
        for key, value in data.items():
            if hasattr(self, key):
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
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )
