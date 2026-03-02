#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

# Allow running via `make` without manually exporting variables.
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

req() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required env var: ${name}" >&2
    exit 1
  fi
}

req OPENAI_API_KEY
req LITELLM_MASTER_KEY
req POSTGRES_DB
req POSTGRES_USER
req POSTGRES_PASSWORD

echo "OK"

