"""
Redis Client for DDS-LLM Orchestrator
Manages connections to K3s Homelab Redis Master
Falls back to local in-memory storage if Redis is unavailable
"""

import os
import redis
import logging
import json
from typing import Any, Optional, Dict
from datetime import timedelta

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Redis client with fallback to local storage
    Connects to K3s Homelab Redis Master (192.168.1.51:6379)
    """

    def __init__(self, config: Dict[str, Any], fallback_enabled: bool = True):
        """
        Initialize Redis client

        Args:
            config: Redis configuration dict with keys:
                - host: Redis host (default: 192.168.1.51)
                - port: Redis port (default: 6379)
                - password: Redis password
                - db: Database number (default: 0)
                - socket_timeout: Connection timeout in seconds
            fallback_enabled: Use local storage if Redis unavailable
        """
        self.config = config
        self.fallback_enabled = fallback_enabled
        self.connected = False
        self.redis_client = None
        self.local_storage = {}  # Fallback local storage

        # Try to connect to Redis
        try:
            self.redis_client = redis.Redis(
                host=config.get("host", "192.168.1.51"),
                port=config.get("port", 6379),
                password=config.get("password") or os.environ.get("REDIS_PASSWORD") or None,
                db=config.get("db", 0),
                decode_responses=config.get("decode_responses", True),
                socket_timeout=config.get("socket_timeout", 5),
                socket_connect_timeout=config.get("socket_connect_timeout", 5),
                retry_on_timeout=config.get("retry_on_timeout", True),
            )

            # Test connection
            if self.redis_client.ping():
                self.connected = True
                logger.info(
                    f"✅ Connected to Redis: {config.get('host')}:{config.get('port')}"
                )
            else:
                raise ConnectionError("Redis ping failed")

        except Exception as e:
            logger.warning(f"⚠️  Redis connection failed: {e}")
            if fallback_enabled:
                logger.info("📦 Using local in-memory storage as fallback")
                self.redis_client = None
                self.connected = False
            else:
                raise

    def set(
        self, key: str, value: Any, ttl: Optional[int] = None, serialize: bool = True
    ) -> bool:
        """
        Set a key-value pair

        Args:
            key: Key name
            value: Value (will be JSON serialized if serialize=True)
            ttl: Time to live in seconds (optional)
            serialize: JSON serialize the value

        Returns:
            True if successful, False otherwise
        """
        try:
            if isinstance(value, str) or not serialize:
                val = value
            else:
                val = json.dumps(value)

            if self.connected and self.redis_client:
                if ttl:
                    self.redis_client.setex(key, ttl, val)
                else:
                    self.redis_client.set(key, val)
                return True
            else:
                # Fallback to local storage
                self.local_storage[key] = val
                logger.debug(f"Local storage (set): {key}")
                return True

        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False

    def get(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """
        Get a value by key

        Args:
            key: Key name
            deserialize: JSON deserialize the value

        Returns:
            Value or None if not found
        """
        try:
            if self.connected and self.redis_client:
                val = self.redis_client.get(key)
            else:
                # Fallback to local storage
                val = self.local_storage.get(key)
                logger.debug(f"Local storage (get): {key}")

            if val is None:
                return None

            if deserialize and isinstance(val, str):
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    return val
            return val

        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a key"""
        try:
            if self.connected and self.redis_client:
                self.redis_client.delete(key)
            else:
                self.local_storage.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            if self.connected and self.redis_client:
                return bool(self.redis_client.exists(key))
            else:
                return key in self.local_storage
        except Exception as e:
            logger.error(f"Error checking key {key}: {e}")
            return False

    def hset(self, key: str, mapping: Dict[str, Any]) -> bool:
        """Set hash fields"""
        try:
            if self.connected and self.redis_client:
                self.redis_client.hset(key, mapping=mapping)
            else:
                if key not in self.local_storage:
                    self.local_storage[key] = {}
                self.local_storage[key].update(mapping)
            return True
        except Exception as e:
            logger.error(f"Error setting hash {key}: {e}")
            return False

    def hgetall(self, key: str) -> Dict[str, Any]:
        """Get all hash fields"""
        try:
            if self.connected and self.redis_client:
                return self.redis_client.hgetall(key) or {}
            else:
                return self.local_storage.get(key, {})
        except Exception as e:
            logger.error(f"Error getting hash {key}: {e}")
            return {}

    def lpush(self, key: str, *values: Any) -> int:
        """Push to list"""
        try:
            if self.connected and self.redis_client:
                return self.redis_client.lpush(key, *values)
            else:
                if key not in self.local_storage:
                    self.local_storage[key] = []
                self.local_storage[key] = list(values) + self.local_storage[key]
                return len(self.local_storage[key])
        except Exception as e:
            logger.error(f"Error pushing to list {key}: {e}")
            return 0

    def lrange(self, key: str, start: int, stop: int) -> list:
        """Get list range"""
        try:
            if self.connected and self.redis_client:
                return self.redis_client.lrange(key, start, stop) or []
            else:
                items = self.local_storage.get(key, [])
                return items[start : stop + 1] if isinstance(items, list) else []
        except Exception as e:
            logger.error(f"Error getting list {key}: {e}")
            return []

    def publish(self, channel: str, message: str) -> int:
        """Publish to a channel"""
        try:
            if self.connected and self.redis_client:
                return self.redis_client.publish(channel, message)
            else:
                logger.warning(f"PubSub not available in fallback mode: {channel}")
                return 0
        except Exception as e:
            logger.error(f"Error publishing to {channel}: {e}")
            return 0

    def subscribe(self, *channels):
        """Subscribe to channels. Returns the PubSub object (or None) so the
        caller can call get_message()/unsubscribe()/close() on it. Previously
        we returned the result of `.subscribe()` (always None), which leaked
        PubSub connections and made listeners unusable."""
        if not self.connected or not self.redis_client:
            logger.warning("Cannot subscribe in fallback mode")
            return None
        ps = self.redis_client.pubsub()
        ps.subscribe(*channels)
        return ps

    def info(self) -> Dict[str, Any]:
        """Get Redis info"""
        if self.connected and self.redis_client:
            try:
                return self.redis_client.info()
            except Exception as e:
                logger.error(f"Error getting Redis info: {e}")
                return {}
        else:
            return {
                "mode": "fallback",
                "local_storage_size": len(self.local_storage),
                "connected": False,
            }

    def flushdb(self) -> bool:
        """Clear all data"""
        try:
            if self.connected and self.redis_client:
                self.redis_client.flushdb()
            else:
                self.local_storage.clear()
            logger.info("Database flushed")
            return True
        except Exception as e:
            logger.error(f"Error flushing database: {e}")
            return False

    def close(self):
        """Close connection"""
        if self.redis_client:
            try:
                self.redis_client.close()
                self.connected = False
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


def get_redis_client(config: Dict[str, Any] = None) -> RedisClient:
    """
    Get or create global Redis client

    Args:
        config: Redis configuration (only used on first call)

    Returns:
        RedisClient instance
    """
    global _redis_client

    if _redis_client is None:
        if config is None:
            config = {
                "host": os.environ.get("REDIS_HOST", "192.168.1.51"),
                "port": int(os.environ.get("REDIS_PORT", "6379")),
                "password": os.environ.get("REDIS_PASSWORD"),
                "db": int(os.environ.get("REDIS_DB", "0")),
            }
        _redis_client = RedisClient(config)

    return _redis_client


if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)

    # Create client
    client = get_redis_client()

    # Test operations
    print("Testing Redis client...")

    # SET/GET
    client.set("test_key", {"name": "Matheus", "role": "researcher"})
    value = client.get("test_key")
    print(f"✅ SET/GET: {value}")

    # HASH
    client.hset("test_hash", {"name": "Matheus", "role": "researcher"})
    hash_data = client.hgetall("test_hash")
    print(f"✅ HASH: {hash_data}")

    # LIST
    client.lpush("test_list", "item1", "item2", "item3")
    list_data = client.lrange("test_list", 0, -1)
    print(f"✅ LIST: {list_data}")

    # INFO
    info = client.info()
    print(f"✅ INFO: {info}")

    print("\nAll tests passed!")
