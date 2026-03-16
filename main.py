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
    parser.add_argument("--port", type=int, default=None, help="HTTP server port")
    parser.add_argument("--host", type=str, default=None, help="HTTP server host")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--dds-domain", type=int, default=None, help="DDS domain ID")
    parser.add_argument("--log-level", type=str, default=None,
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level")
    parser.add_argument("--grpc-enabled", action="store_true", default=None,
                       help="Enable gRPC transport")
    parser.add_argument("--grpc-port", type=int, default=None,
                       help="gRPC listen port (default: 50052)")
    parser.add_argument("--fuzzy", action="store_true", default=None,
                       help="Enable fuzzy logic agent selection")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level or "INFO")
    logger = logging.getLogger(__name__)

    # Load config
    if args.config:
        config = OrchestratorConfig()
        config.load_from_file(Path(args.config))
        # CLI args override config file values when explicitly specified
        if args.port is not None:
            config.port = args.port
        if args.host is not None:
            config.host = args.host
        if args.dds_domain is not None:
            config.dds_domain = args.dds_domain
        if args.log_level is not None:
            config.log_level = args.log_level
        if args.grpc_enabled is not None:
            config.grpc_enabled = args.grpc_enabled
        if args.grpc_port is not None:
            config.grpc_port = args.grpc_port
        if args.fuzzy is not None:
            config.fuzzy_enabled = args.fuzzy
    else:
        config = load_config_from_env()
        config.port = args.port if args.port is not None else config.port
        config.host = args.host if args.host is not None else config.host
        config.dds_domain = args.dds_domain if args.dds_domain is not None else config.dds_domain
        config.log_level = args.log_level if args.log_level is not None else config.log_level
        if args.grpc_enabled is not None:
            config.grpc_enabled = args.grpc_enabled
        if args.grpc_port is not None:
            config.grpc_port = args.grpc_port
        if args.fuzzy is not None:
            config.fuzzy_enabled = args.fuzzy

    logger.info("=" * 60)
    logger.info("DDS-LLM Orchestrator Starting")
    logger.info("=" * 60)
    logger.info(f"Host: {config.host}")
    logger.info(f"Port: {config.port}")
    logger.info(f"DDS Domain: {config.dds_domain}")
    logger.info(f"DDS Enabled: {config.dds_enabled}")
    logger.info(f"gRPC Enabled: {config.grpc_enabled}")
    if config.grpc_enabled:
        logger.info(f"gRPC Port: {config.grpc_port}")
    logger.info(f"Fuzzy Enabled: {config.fuzzy_enabled}")
    logger.info("=" * 60)

    # Initialize components
    registry = AgentRegistry(config)
    scheduler = TaskScheduler(config)
    dds_layer = DDSLayer(config)
    selector = AgentSelector()

    # Initialize gRPC layer if enabled
    grpc_layer = None
    if config.grpc_enabled:
        from grpc_layer import GRPCLayer
        grpc_layer = GRPCLayer(config)

    # Initialize fuzzy decision engine if enabled
    fuzzy_engine = None
    if config.fuzzy_enabled:
        try:
            from fuzzy_selector import FuzzyDecisionEngine
            fuzzy_engine = FuzzyDecisionEngine()
            logger.info("Fuzzy decision engine initialized")
        except ImportError as e:
            logger.warning(f"Fuzzy engine not available: {e}. Using baseline selector.")

    # Create server
    server = OrchestratorServer(
        config=config,
        registry=registry,
        scheduler=scheduler,
        dds_layer=dds_layer,
        selector=selector,
        grpc_layer=grpc_layer,
        fuzzy_engine=fuzzy_engine,
    )

    # Start server
    try:
        await server.start()

        # Keep running
        logger.info("Orchestrator is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(3600)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down...")
    finally:
        await server.stop()
        logger.info("Orchestrator stopped.")


if __name__ == "__main__":
    asyncio.run(main())
