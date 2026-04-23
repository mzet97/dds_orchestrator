"""
dds_orchestrator.transport — camada de transporte abstrata.

Materializa a "Camada de Transporte" descrita na dissertação v3
(Seção sec:impl_servidor): uma interface única (`TransportAdapter`) com
três realizações intercambiáveis (HTTP, gRPC, DDS).

Uso recomendado:

    from dds_orchestrator.transport import (
        TransportAdapter, TransportResult,
        HttpTransportAdapter, DdsTransportAdapter, GrpcTransportAdapter,
        build_transport_registry,
    )

    transports = build_transport_registry(
        dds_layer=dds_layer,
        grpc_layer=grpc_layer,
        http_session=shared_session,
    )
    # transports["dds"].dispatch(task, agent, timeout_ms=...)
"""
from .base import TransportAdapter, TransportResult
from .dds_adapter import DdsTransportAdapter
from .grpc_adapter import GrpcTransportAdapter
from .http_adapter import HttpTransportAdapter


def build_transport_registry(*, dds_layer=None, grpc_layer=None,
                             http_session=None) -> dict[str, TransportAdapter]:
    """Monta o dicionário {nome → adapter} consumido pelo dispatcher.

    Chamar a partir de `main.py` logo após a construção de `dds_layer` e
    `grpc_layer`. O dispatcher de `server.py` pode ignorar por ora (o
    scaffolding é zero-impacto até ser ativado na migração incremental).
    """
    return {
        "http": HttpTransportAdapter(session=http_session),
        "dds": DdsTransportAdapter(dds_layer=dds_layer),
        "grpc": GrpcTransportAdapter(grpc_layer=grpc_layer),
    }


__all__ = [
    "TransportAdapter",
    "TransportResult",
    "HttpTransportAdapter",
    "DdsTransportAdapter",
    "GrpcTransportAdapter",
    "build_transport_registry",
]
