# DDS Orchestrator

Servidor de orquestração para gerenciar agentes LLM utilizando DDS (Data Distribution Service) como middleware de comunicação.

## Descrição

O DDS Orchestrator é o componente central do sistema DDS-LLM responsável por:
- Registrar e monitorar agentes disponíveis
- Agendar e distribuir tarefas aos agentes
- Selecionar o melhor agente para cada requisição
- Fornecer API REST/WebSocket compatível com OpenAI
- Suportar comunicação via DDS e HTTP

## Arquitetura

```
Cliente HTTP/WebSocket
         ↓
   ┌─────────────┐
   │  Servidor   │ ← aiohttp server
   │  REST API   │
   └──────┬──────┘
          ↓
   ┌─────────────┐
   │  Agendador  │ ← TaskScheduler (fila prioritária)
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

## Estrutura

```
dds_orchestrator/
├── main.py              # Entry point
├── server.py            # Servidor HTTP/WebSocket
├── config.py            # Configuração
├── registry.py         # Registro de agentes
├── scheduler.py        # Agendador de tarefas
├── selector.py         # Seleção de agente
├── dds.py              # Camada DDS
├── dds_client.py       # Cliente DDS
├── http_client.py      # Cliente HTTP fallback
├── models.py           # Modelos de dados Pydantic
├── context.py          # Gerenciamento de contexto
├── api/
│   └── routes.py       # Rotas REST
├── config.yaml         # Configuração YAML
├── requirements.txt    # Dependências Python
├── LICENSE
└── CLAUDE.md          # Instruções Claude Code
```

## Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-repo/dds-llm-orchestrator.git
cd dds_orchestrator

# Instale dependências
pip install -r requirements.txt
```

### Dependências

- Python 3.10+
- aiohttp
- pydantic
- cyclonedds (para comunicação DDS)
- pycyclonedds

## Como Executar

### Iniciar o Orquestrador

```bash
# Com DDS (padrão)
python main.py --port 8080 --dds-domain 0

# Com configuração customizada
python main.py --config config.yaml --port 8080 --dds-domain 0

# Com log detalhado
python main.py --port 8080 --log-level DEBUG
```

### Parâmetros

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--port` | Porta do servidor HTTP | 8080 |
| `--host` | Host do servidor | 0.0.0.0 |
| `--config` | Arquivo de configuração | None |
| `--dds-domain` | ID do domínio DDS | 0 |
| `--log-level` | Nível de log | INFO |

## API

### Chat Completions

```bash
curl -X POST http://localhost:8080/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4-mini",
    "messages": [{"role": "user", "content": "Olá!"}],
    "stream": false
  }'
```

### Streaming

```bash
curl -X POST http://localhost:8080/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4-mini",
    "messages": [{"role": "user", "content": "Conte uma história"}],
    "stream": true
  }'
```

### Listar Agentes

```bash
curl http://localhost:8080/api/v1/agents
```

## Registro de Agentes

Os agentes se registram automaticamente enviando heartbeat periódico:

```json
{
  "agent_id": "uuid",
  "specialization": "text|vision|embedding",
  "model": "phi-4-mini",
  "status": "idle|busy",
  "capabilities": ["streaming", "functions"]
}
```

## Tópicos DDS

| Tópico | Tipo | Descrição |
|--------|------|-----------|
| `agent/register` | Pub/Sub | Registro de agentes |
| `agent/request` | Pub/Sub | Requisições de tarefas |
| `agent/response` | Pub/Sub | Respostas de tarefas |
| `agent/status` | Pub/Sub | Heartbeat de status |

## Configuração

### config.yaml

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

## Testes

```bash
# Teste DDS round-trip
python test_dds_roundtrip.py

# Teste end-to-end
python test_e2e.py

# Teste do orchestrator
python test_orchestrator.py

# Benchmarks
python benchmark_dds_overhead.py
```

## Troubleshooting

### Agentes não aparecem registrados

1. Verifique se o domínio DDS está configurado corretamente
2. Check logs para erros de conexão
3. Confirme que os agentes estão enviando heartbeats

### Latência alta

1. Configure CycloneDDS para shared memory em `cyclonedds-shm.xml`
2. Ajuste o `heartbeat_interval` para menor valor

## Licença

MIT License
