# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The DDS Orchestrator is the central component of the DDS-LLM system. It manages LLM agents using DDS (Data Distribution Service) as low-latency middleware, providing a REST API compatible with OpenAI.

## Common Commands

### Run Orchestrator
```bash
cd dds_orchestrator
pip install -r requirements.txt
python main.py --port 8080 --dds-domain 0
```

### Run with custom config
```bash
python main.py --config config.yaml --port 8080 --dds-domain 0
```

### Run with debug logging
```bash
python main.py --port 8080 --log-level DEBUG
```

### Test API
```bash
curl -X POST http://localhost:8080/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "phi-4-mini", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Run Tests
```bash
# DDS round-trip latency test
python test_dds_roundtrip.py

# End-to-end integration test
python test_e2e.py

# Core orchestrator components (registry, scheduler, context)
python test_orchestrator.py

# DDS communication test
python test_dds.py

# Benchmark DDS overhead
python benchmark_dds_overhead.py
```

## Architecture

```
Client HTTP/WebSocket
         ↓
   ┌─────────────┐
   │  Servidor   │ ← aiohttp server
   │  REST API   │
   └──────┬──────┘
          ↓
   ┌─────────────┐
   │  Agendador  │ ← TaskScheduler (priority queue)
   └──────┬──────┘
          ↓
   ┌─────────────┐
   │  Registro   │ ← AgentRegistry (heartbeats)
   └──────┬──────┘
          ↓
   ┌─────────────┐
   │    DDS      │ ← CycloneDDS
   │   Layer     │
   └──────┬──────┘
          ↓
   Agentes LLM
```

## Key Components

| File | Purpose |
|------|---------|
| `main.py` | Entry point - initializes all components |
| `server.py` | HTTP/WebSocket server using aiohttp |
| `config.py` | Configuration management |
| `registry.py` | Agent registration and heartbeat monitoring |
| `scheduler.py` | Priority-based task scheduling |
| `selector.py` | Agent selection logic |
| `dds.py` | DDS communication layer |
| `dds_client.py` | DDS client for pub/sub |
| `http_client.py` | HTTP fallback client |
| `models.py` | Pydantic data models |
| `context.py` | Request context management |
| `api/routes.py` | REST API routes |

## DDS Topics

| Topic | Type | Description |
|-------|------|-------------|
| `agent/register` | Pub/Sub | Agent registration |
| `agent/request` | Pub/Sub | Task requests |
| `agent/response` | Pub/Sub | Task responses |
| `agent/status` | Pub/Sub | Status heartbeats |

## Configuration

Configuration via `config.yaml`:
```yaml
server:
  host: "0.0.0.0"
  port: 8080

dds:
  domain: 0
  participant_name: "orchestrator"

agents:
  heartbeat_interval: 5
  timeout: 30

scheduler:
  max_queue_size: 100
  priority_levels: 3
```

## Main.py CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--port` | HTTP server port | 8080 |
| `--host` | HTTP server host | 0.0.0.0 |
| `--config` | Config file path | None |
| `--dds-domain` | DDS domain ID | 0 |
| `--log-level` | DEBUG, INFO, WARNING, ERROR | INFO |
