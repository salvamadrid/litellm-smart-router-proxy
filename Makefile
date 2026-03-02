.PHONY: help setup-env check-env up down logs restart recreate demo-routing smoke reset-metrics

LITELLM_URL ?= http://localhost:4000

help:
	@echo "Available targets:"
	@echo "  make setup-env      # Create .env from .env.example if missing"
	@echo "  make check-env      # Validate required env variables"
	@echo "  make up             # Start LiteLLM proxy + Postgres"
	@echo "  make down           # Stop stack"
	@echo "  make logs           # Follow proxy logs"
	@echo "  make restart        # Restart stack"
	@echo "  make recreate       # Recreate proxy container (apply compose changes)"
	@echo "  make smoke          # Basic /health + completion checks"
	@echo "  make demo-routing   # Show x-router-* evidence"
	@echo "  make reset-metrics  # Delete artifacts/router_metrics.jsonl"

setup-env:
	@if [ ! -f .env ]; then cp .env.example .env; echo ".env created from .env.example"; else echo ".env already exists"; fi

check-env:
	@./scripts/check_env.sh

up: check-env
	@mkdir -p artifacts
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f litellm postgres

restart: down up

recreate:
	@mkdir -p artifacts
	docker compose up -d --force-recreate litellm

smoke:
	@LITELLM_URL="$(LITELLM_URL)" ./scripts/smoke.sh

demo-routing:
	@./scripts/demo_routing.sh

reset-metrics:
	@rm -f artifacts/router_metrics.jsonl && echo "Deleted artifacts/router_metrics.jsonl"

