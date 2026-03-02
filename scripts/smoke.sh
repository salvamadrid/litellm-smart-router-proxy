#!/usr/bin/env bash
set -euo pipefail

LITELLM_URL="${LITELLM_URL:-http://localhost:4000}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-$(awk -F= '/^LITELLM_MASTER_KEY=/{print $2}' .env 2>/dev/null || true)}"

if [ -z "${LITELLM_MASTER_KEY}" ]; then
  echo "Missing LITELLM_MASTER_KEY (env var or .env file)" >&2
  exit 1
fi

echo "[smoke] /health"
curl -sS "${LITELLM_URL}/health" >/dev/null

echo "[smoke] chat completion (smart-router)"
curl -sS \
  -X POST "${LITELLM_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  -d '{"model":"smart-router","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0}' >/dev/null

echo "OK"

