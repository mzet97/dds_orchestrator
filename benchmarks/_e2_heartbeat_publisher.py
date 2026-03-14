#!/usr/bin/env python3
"""
Helper: Publicador de heartbeats DDS para o benchmark E2.
Lançado como subprocesso pelo E2_failure_detection_dds.py.

Usage (interno):
    python _e2_heartbeat_publisher.py <periodo_ms> <domain_id>
"""
import sys
import time
from dataclasses import dataclass, field

periodo_ms = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
domain_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
interval_s = (periodo_ms / 10) / 1000.0  # publica 10x mais rápido que o deadline

try:
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.topic import Topic
    from cyclonedds.pub import Publisher, DataWriter
    from cyclonedds.qos import Qos, Policy
    from cyclonedds.util import duration
    from cyclonedds.idl import IdlStruct
    from cyclonedds.idl.annotations import key as idl_key
except ImportError as e:
    print(f"ERRO: cyclonedds não disponível: {e}", file=sys.stderr)
    sys.exit(1)


@dataclass
class BenchmarkHeartbeat(IdlStruct, typename="BenchmarkHeartbeat"):
    agent_id: str = ""
    seq: int = 0
    timestamp: float = 0.0


try:
    participant = DomainParticipant(domain_id)
    topic = Topic(participant, "benchmark/heartbeat", BenchmarkHeartbeat)

    # QoS with DEADLINE + LIVELINESS — must match reader's requested QoS
    lease_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    qos_policies = [
        Policy.Reliability.Reliable(duration(seconds=10)),
        Policy.Deadline(duration(milliseconds=periodo_ms)),
    ]
    if lease_ms > 0:
        qos_policies.append(
            Policy.Liveliness.Automatic(lease_duration=duration(milliseconds=lease_ms))
        )
    qos = Qos(*qos_policies)
    writer = DataWriter(Publisher(participant), topic, qos=qos)

    seq = 0
    while True:
        writer.write(BenchmarkHeartbeat(
            agent_id="e2-test-agent",
            seq=seq,
            timestamp=time.time()
        ))
        seq += 1
        time.sleep(interval_s)

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"Publisher error: {e}", file=sys.stderr)
    sys.exit(1)
