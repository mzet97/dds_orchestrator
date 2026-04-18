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
    parser.add_argument("--redis-url", type=str, default=None,
                       help="Redis URL (e.g. redis://redis.home.arpa:6379)")
    parser.add_argument("--redis-password", type=str, default=None,
                       help="Redis password")
    parser.add_argument("--mongo-url", type=str, default=None,
                       help="MongoDB URL")
    parser.add_argument("--routing-algorithm", type=str, default=None,
                       choices=["round_robin", "least_loaded", "weighted_score"],
                       help="Routing algorithm for instance pool")
    parser.add_argument("--instance-ports-gpu", type=str, default=None,
                       help="Comma-separated GPU instance ports")
    parser.add_argument("--instance-ports-cpu", type=str, default=None,
                       help="Comma-separated CPU instance ports")

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
        if args.redis_url is not None:
            config.redis_url = args.redis_url
        if args.redis_password is not None:
            config.redis_password = args.redis_password
        if args.mongo_url is not None:
            config.mongo_url = args.mongo_url
        if args.routing_algorithm is not None:
            config.routing_algorithm = args.routing_algorithm
        if args.instance_ports_gpu is not None:
            config.instance_ports_gpu = args.instance_ports_gpu
        if args.instance_ports_cpu is not None:
            config.instance_ports_cpu = args.instance_ports_cpu
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
        if args.redis_url is not None:
            config.redis_url = args.redis_url
        if args.redis_password is not None:
            config.redis_password = args.redis_password
        if args.mongo_url is not None:
            config.mongo_url = args.mongo_url
        if args.routing_algorithm is not None:
            config.routing_algorithm = args.routing_algorithm
        if args.instance_ports_gpu is not None:
            config.instance_ports_gpu = args.instance_ports_gpu
        if args.instance_ports_cpu is not None:
            config.instance_ports_cpu = args.instance_ports_cpu

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
    if config.redis_url:
        logger.info(f"Redis: {config.redis_url}")
    if config.mongo_url:
        logger.info(f"MongoDB: {config.mongo_url}")
    if config.instance_ports_gpu or config.instance_ports_cpu:
        logger.info(f"Routing: {config.routing_algorithm}")
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

    # Initialize Redis + MongoDB + InstancePool (Phase 5)
    redis_mgr = None
    mongo_store = None
    instance_pool = None
    backpressure = None

    if config.redis_url:
        try:
            from redis_layer import RedisStateManager
            redis_mgr = RedisStateManager(config.redis_url, config.redis_password)
            await redis_mgr.connect()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            redis_mgr = None

    if config.mongo_url:
        try:
            from mongo_layer import MongoMetricsStore
            mongo_store = MongoMetricsStore(config.mongo_url, config.mongo_db)
            await mongo_store.connect()
            await mongo_store.ensure_indexes()
            logger.info("MongoDB connected")
        except Exception as e:
            logger.warning(f"MongoDB not available: {e}")
            mongo_store = None

    if redis_mgr and (config.instance_ports_gpu or config.instance_ports_cpu):
        from instance_pool import InstancePool, InstanceInfo, RoutingAlgorithm
        from backpressure import BackpressureManager

        instance_pool = InstancePool(
            redis_mgr, mongo_store,
            algorithm=RoutingAlgorithm(config.routing_algorithm),
        )

        # Parse per-port hostname mapping: "host:port,host:port,..."
        host_map: dict[int, str] = {}
        if config.instance_host_map:
            for entry in config.instance_host_map.split(","):
                entry = entry.strip()
                if ":" in entry:
                    h, p = entry.rsplit(":", 1)
                    host_map[int(p)] = h

        # Register GPU instances
        if config.instance_ports_gpu:
            for port_str in config.instance_ports_gpu.split(","):
                port_str = port_str.strip()
                if port_str:
                    port = int(port_str)
                    hostname = host_map.get(port, config.instance_host)
                    await instance_pool.register_instance(
                        InstanceInfo(port=port, hostname=hostname,
                                     inst_type="gpu", slots_total=config.slots_per_gpu,
                                     weight=config.weight_gpu))
        # Register CPU instances
        if config.instance_ports_cpu:
            for port_str in config.instance_ports_cpu.split(","):
                port_str = port_str.strip()
                if port_str:
                    port = int(port_str)
                    hostname = host_map.get(port, config.instance_host)
                    await instance_pool.register_instance(
                        InstanceInfo(port=port, hostname=hostname,
                                     inst_type="cpu", slots_total=config.slots_per_cpu,
                                     weight=config.weight_cpu))

        backpressure = BackpressureManager(redis_mgr, config.max_rps)
        logger.info(f"InstancePool ready: {len(instance_pool._instances)} instances, "
                     f"algorithm={config.routing_algorithm}")

    # Create server
    server = OrchestratorServer(
        config=config,
        registry=registry,
        scheduler=scheduler,
        dds_layer=dds_layer,
        selector=selector,
        grpc_layer=grpc_layer,
        fuzzy_engine=fuzzy_engine,
        instance_pool=instance_pool,
        redis_mgr=redis_mgr,
        mongo_store=mongo_store,
        backpressure=backpressure,
    )

    # Start server
    try:
        await server.start()
        # Start periodic scheduler cleanup so long benchmark runs don't leak
        # completed-task memory indefinitely.
        scheduler.start_cleanup_loop(interval_s=60, max_tasks=500, max_age_seconds=300)

        # Keep running
        logger.info("Orchestrator is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(3600)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down...")
    finally:
        await scheduler.stop_cleanup_loop()
        await server.stop()
        if redis_mgr:
            await redis_mgr.close()
        if mongo_store:
            await mongo_store.close()
        logger.info("Orchestrator stopped.")


if __name__ == "__main__":
    asyncio.run(main())
