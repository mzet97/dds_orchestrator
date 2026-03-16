"""
QoS Profile Factory for CycloneDDS.

Defines named QoS profiles that the fuzzy decision engine can select
based on task urgency, complexity, and system state. Each profile maps
to a concrete CycloneDDS Qos object with specific RELIABILITY, DURABILITY,
DEADLINE, HISTORY, and TRANSPORT_PRIORITY settings.

Profiles:
  LOW_COST   - Best-effort, volatile, minimal overhead
  BALANCED   - Reliable with moderate deadlines
  CRITICAL   - Reliable, transient-local, strict deadlines, high priority
"""

from enum import Enum
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class QoSProfile(Enum):
    LOW_COST = "low_cost"
    BALANCED = "balanced"
    CRITICAL = "critical"


# Human-readable descriptions for logging
PROFILE_DESCRIPTIONS = {
    QoSProfile.LOW_COST: "Best-effort, minimal overhead (telemetry, non-critical tasks)",
    QoSProfile.BALANCED: "Reliable with moderate deadlines (normal requests)",
    QoSProfile.CRITICAL: "Reliable, persistent, strict deadlines (critical tasks)",
}


def create_qos(profile: QoSProfile) -> Any:
    """Create a CycloneDDS Qos object for the given profile.

    Returns None if CycloneDDS is not available.
    """
    try:
        from cyclonedds.core import Policy
        from cyclonedds.qos import Qos
        from cyclonedds.util import duration
    except ImportError:
        logger.warning("CycloneDDS not available, cannot create QoS profile")
        return None

    if profile == QoSProfile.LOW_COST:
        return Qos(
            Policy.Reliability.BestEffort,
            Policy.Durability.Volatile,
            Policy.History.KeepLast(1),
            Policy.TransportPriority(0),
        )

    elif profile == QoSProfile.BALANCED:
        return Qos(
            Policy.Reliability.Reliable(duration(seconds=5)),
            Policy.Durability.Volatile,
            Policy.Deadline(duration(seconds=10)),
            Policy.History.KeepLast(4),
            Policy.TransportPriority(5),
        )

    elif profile == QoSProfile.CRITICAL:
        return Qos(
            Policy.Reliability.Reliable(duration(seconds=10)),
            Policy.Durability.TransientLocal,
            Policy.Deadline(duration(seconds=30)),
            Policy.History.KeepLast(8),
            Policy.TransportPriority(10),
        )

    else:
        logger.warning(f"Unknown QoS profile: {profile}, using BALANCED")
        return create_qos(QoSProfile.BALANCED)


def profile_from_score(score: float) -> QoSProfile:
    """Map a fuzzy output score (0-10) to a QoSProfile.

    0-3: LOW_COST, 4-7: BALANCED, 8-10: CRITICAL
    """
    if score <= 3.0:
        return QoSProfile.LOW_COST
    elif score <= 7.0:
        return QoSProfile.BALANCED
    else:
        return QoSProfile.CRITICAL
