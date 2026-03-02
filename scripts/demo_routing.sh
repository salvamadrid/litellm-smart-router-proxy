#!/usr/bin/env bash
set -euo pipefail

LITELLM_URL="${LITELLM_URL:-http://localhost:4000}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-$(awk -F= '/^LITELLM_MASTER_KEY=/{print $2}' .env 2>/dev/null || true)}"

if [ -z "${LITELLM_MASTER_KEY}" ]; then
  echo "Missing LITELLM_MASTER_KEY (env var or .env file)"
  exit 1
fi

call() {
  local label="$1"
  local user_msg="$2"
  local tmp_body
  tmp_body="$(mktemp)"

  echo
  echo "== ${label} =="

  local user_json
  user_json="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${user_msg}")"

  curl -sS -D - -o "${tmp_body}" \
    -X POST "${LITELLM_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
    -d "$(cat <<JSON
{
  "model": "smart-router",
  "messages": [
    {"role": "user", "content": ${user_json}}
  ]
}
JSON
)" | awk '
BEGIN{IGNORECASE=1}
/^x-router-/{print}
/^x-response-model:/{print}
'

  if command -v jq >/dev/null 2>&1; then
    echo "response.model: $(jq -r '.model // empty' "${tmp_body}")"
  fi

  rm -f "${tmp_body}"
}

call "SIMPLE (greeting)" "hi"
call "REPO-EDIT (file changes + tools)" "Please update a Makefile to add a new target and update make help."
call "REASONING (step-by-step)" "Think step by step and explain your reasoning: prove that the sum of two even numbers is even."

echo
