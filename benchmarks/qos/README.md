# QoS Benchmarks B5-B7

This directory contains the benchmark tests for scenarios B5, B6, and B7 described in the thesis.

## Structure

```
benchmarks/qos/
├── benchmark_b51_failure_detection.py   # B5.1: Failure Detection
├── benchmark_b52_message_priority.py    # B5.2: Message Prioritization
├── benchmark_b53_load_balancing.py      # B5.3: Load Balancing
├── benchmark_b61_autogen_comparison.py  # B6: AutoGen vs DDS
├── benchmark_b71_dds_grpc.py           # B7: DDS vs gRPC
└── README.md                            # This file
```

## Quick Start

### B5.1 - Failure Detection
```bash
cd dds_orchestrator
python -m benchmarks.qos.benchmark_b51_failure_detection --mode all --iterations 10
```

### B5.2 - Message Prioritization
```bash
python -m benchmarks.qos.benchmark_b52_message_priority --mode all --high-priority 10 --low-priority 90
```

### B5.3 - Load Balancing
```bash
python -m benchmarks.qos.benchmark_b53_load_balancing --mode all --replicas 3 --requests 1000
```

### B6 - AutoGen Comparison
```bash
python -m benchmarks.qos.benchmark_b61_autogen_comparison --mode all --iterations 100
```

### B7 - DDS vs gRPC
```bash
python -m benchmarks.qos.benchmark_b71_dds_grpc --mode all --iterations 100
```

## Output

Results are saved to `dds_orchestrator/benchmark_results/` as JSON files with timestamps.

## Requirements

- Python 3.8+
- cyclonedds Python package
- For gRPC tests: grpcio, grpcio-tools
- For Redis tests: redis-py
