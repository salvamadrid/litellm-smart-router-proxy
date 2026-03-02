# LiteLLM Smart Router Proxy

OpenAI-compatible LiteLLM proxy that routes prompts to different models by **complexity**, and emits **observable evidence**:

- `x-router-*` headers on responses
- per-request **tokens + cost** in `artifacts/router_metrics.jsonl`

## Quickstart

Requirements:

- Docker + Docker Compose
- An LLM provider key (example: `OPENAI_API_KEY`)

```bash
make setup-env
# edit .env and set OPENAI_API_KEY + LITELLM_MASTER_KEY
make up
make smoke
```

If scripts are not executable on your system:

```bash
chmod +x scripts/*.sh
```

## Demonstrate routing (headers)

```bash
make demo-routing
```

You should see headers like:

- `x-router-tier`
- `x-router-score`
- `x-router-model`

## Cost / tokens metrics

The proxy writes JSONL to:

- `artifacts/router_metrics.jsonl` (mounted from the container)

Each line is one request with fields like `execution_id`, `execution_mode`, `prompt_tokens`, `total_tokens`, `cost_usd`.

## Files

- `docker-compose.yml`: LiteLLM + Postgres stack
- `config.yaml`: models + callback registration
- `custom_callbacks.py`: routing + response headers + JSONL metrics logger
- `scripts/demo_routing.sh`: curl demo printing `x-router-*`

## Publish

See `PUBLISHING.md`.
