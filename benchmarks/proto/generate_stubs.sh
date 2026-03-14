#!/bin/bash
# Generate Python gRPC stubs from llama_service.proto
# Requires: pip install grpcio-tools
#
# Usage:
#   cd dds_orchestrator/benchmarks/proto
#   bash generate_stubs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m grpc_tools.protoc \
    -I"$SCRIPT_DIR" \
    --python_out="$SCRIPT_DIR" \
    --grpc_python_out="$SCRIPT_DIR" \
    "$SCRIPT_DIR/llama_service.proto"

# Fix import: protoc generates absolute import, but we need relative for package use
sed -i 's/^import llama_service_pb2/from . import llama_service_pb2/' \
    "$SCRIPT_DIR/llama_service_pb2_grpc.py"

echo "Generated stubs in $SCRIPT_DIR:"
ls -la "$SCRIPT_DIR"/llama_service_pb2*.py 2>/dev/null || echo "  (no files generated — check errors above)"
