#!/usr/bin/env bash
# --------------------------------------------------------------------
# get_python_agent_token.sh
# --------------------------------------------------------------------
# Fetches a JWT token for the API user "pythonagent" and stores it
# in the .env file as PYTHON_AGENT_TOKEN.
# --------------------------------------------------------------------

set -euo pipefail

# Location of your .env file (edit if needed)
ENV_FILE=".env"
HOST=${HOST:-http://localhost:28800}

# Load credentials
if [[ -f "$ENV_FILE" ]]; then
  export $(grep -v '^#' "$ENV_FILE" | xargs)
else
  echo "❌ .env file not found at $ENV_FILE"
  exit 1
fi

if [[ -z "${PYTHON_AGENT_USERNAME:-}" || -z "${PYTHON_AGENT_PASSWORD:-}" ]]; then
  echo "❌ PYTHON_AGENT_USERNAME or PYTHON_AGENT_PASSWORD missing in $ENV_FILE"
  exit 1
fi

echo "🔐 Requesting token for user: $PYTHON_AGENT_USERNAME"

# Request token
response=$(curl -s -X POST "$HOST/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=$PYTHON_AGENT_USERNAME&password=$PYTHON_AGENT_PASSWORD")

# Extract token
token=$(echo "$response" | jq -r '.access_token // empty')

if [[ -z "$token" ]]; then
  echo "❌ Failed to obtain token. Response was:"
  echo "$response"
  exit 1
fi

echo "➡️  Add the following to your .env file:"
echo "   PYTHON_AGENT_TOKEN=$token"
echo "✅ Token obtained successfully."