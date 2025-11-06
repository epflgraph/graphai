#!/usr/bin/env bash
# --------------------------------------------------------------------
# test_python_agent_access.sh
# --------------------------------------------------------------------
# Verifies that PYTHON_AGENT_TOKEN is valid and API access works.
# --------------------------------------------------------------------

set -euo pipefail

ENV_FILE=".env"
HOST=${HOST:-http://localhost:28800}

# Load variables
if [[ -f "$ENV_FILE" ]]; then
  export $(grep -v '^#' "$ENV_FILE" | xargs)
else
  echo "❌ .env file not found at $ENV_FILE"
  exit 1
fi

if [[ -z "${PYTHON_AGENT_TOKEN:-}" ]]; then
  echo "❌ PYTHON_AGENT_TOKEN not found in $ENV_FILE. Run get_python_agent_token.sh first."
  exit 1
fi

echo "🔎 Testing API access for token user..."
echo "------------------------------------------------------------"

# 1. Check /users/me/
status=$(curl -s -o /tmp/me.json -w "%{http_code}" \
  -H "Authorization: Bearer $PYTHON_AGENT_TOKEN" \
  "$HOST/users/me/")

if [[ "$status" != "200" ]]; then
  echo "❌ /users/me/ failed (HTTP $status)"
  cat /tmp/me.json
  exit 1
fi

username=$(jq -r '.username' /tmp/me.json)
echo "✅ Authenticated as: $username"

raw_text='GraphAI is awesome!'
echo "⚙️  Testing API call with text: \"$raw_text\""

# 2. Try a simple authenticated API call (e.g. /text/keywords)
status2=$(curl -s -o /tmp/test.json -w "%{http_code}" \
  -H "Authorization: Bearer $PYTHON_AGENT_TOKEN" \
  "$HOST/text/keywords" \
  -d "{\"raw_text\":\"$raw_text\"}" \
  -H "Content-Type: application/json" || true)

echo "GET /text/keywords → HTTP $status2"
cat /tmp/test.json | jq .
echo "------------------------------------------------------------"

if [[ "$status2" == "401" ]]; then
  echo "❌ Unauthorized. Token may be expired or missing proper scopes."
  exit 1
fi

echo "✅ API user $username can perform requests successfully."