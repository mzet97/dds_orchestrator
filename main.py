"""
Main entry point for the orchestrator module
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from server import OrchestratorServer
from config import OrchestratorConfig, load_config_from_env
from registry import AgentRegistry
from scheduler import TaskScheduler
from selector import AgentSelector
from dds import DDSLayer


def setup_logging(level: str):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DDS-LLM Orchestrator")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP server host")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--dds-domain", type=int, default=0, help="DDS domain ID")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load config
    if args.config:
        config = OrchestratorConfig()
        config.load_from_file(Path(args.config))
    else:
        config = load_config_from_env()
        config.port = args.port
        config.host = args.host
        config.dds_domain = args.dds_domain
        config.log_level = args.log_level

    logger.info("=" * 60)
    logger.info("DDS-LLM Orchestrator Starting")
    logger.info("=" * 60)
    logger.info(f"Host: {config.host}")
    logger.info(f"Port: {config.port}")
    logger.info(f"DDS Domain: {config.dds_domain}")
    logger.info(f"DDS Enabled: {config.dds_enabled}")
    logger.info("=" * 60)

    # Initialize components
    registry = AgentRegistry(config)
    scheduler = TaskScheduler(config)
    dds_layer = DDSLayer(config)
    selector = AgentSelector()  # Seletor de agentes especializados

    # Create server
    server = OrchestratorServer(
        config=config,
        registry=registry,
        scheduler=scheduler,
        dds_layer=dds_layer,
        selector=selector
    )

    # Start server
    try:
        await server.start()

        # Keep running
        logger.info("Orchestrator is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(3600)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await server.stop()
        logger.info("Orchestrator stopped.")


if __name__ == "__main__":
    asyncio.run(main())
