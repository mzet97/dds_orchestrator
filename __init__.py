"""
DDS-LLM Orchestrator Module
Main entry point for the orchestration system
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.server import OrchestratorServer
from orchestrator.config import OrchestratorConfig
from orchestrator.registry import AgentRegistry
from orchestrator.scheduler import TaskScheduler
from orchestrator.dds import DDSLayer


async def main():
    """Main entry point for the orchestrator"""

    parser = argparse.ArgumentParser(description="DDS-LLM Orchestrator")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP server host")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--dds-domain", type=int, default=0, help="DDS domain ID")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")

    args = parser.parse_args()

    # Load configuration
    config = OrchestratorConfig(
        host=args.host,
        port=args.port,
        dds_domain=args.dds_domain,
        log_level=args.log_level
    )

    # Override from config file if exists
    config_file = Path(args.config)
    if config_file.exists():
        config.load_from_file(config_file)

    print("=" * 60)
    print("DDS-LLM Orchestrator")
    print("=" * 60)
    print(f"Host: {config.host}")
    print(f"Port: {config.port}")
    print(f"DDS Domain: {config.dds_domain}")
    print(f"Log Level: {config.log_level}")
    print("=" * 60)

    # Initialize components
    registry = AgentRegistry(config)
    scheduler = TaskScheduler(config)
    dds_layer = DDSLayer(config)

    # Create server
    server = OrchestratorServer(
        config=config,
        registry=registry,
        scheduler=scheduler,
        dds_layer=dds_layer
    )

    # Start server
    try:
        await server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
