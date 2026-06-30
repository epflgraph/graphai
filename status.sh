#!/usr/bin/env bash
set -euo pipefail

echo "Restarting all services..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.env"
  set +a
fi

BROKER_HOST="${AI_RABBITMQ_DEV_HOST:-127.0.0.1}"
BROKER_PORT="${AI_RABBITMQ_DEV_PORT:-5672}"
REDIS_HOST="${AI_REDIS_DEV_HOST:-127.0.0.1}"
REDIS_PORT="${AI_REDIS_DEV_PORT:-6379}"

wait_for_tcp_port() {
  local host="$1"
  local port="$2"
  local name="$3"
  local timeout="${4:-60}"
  local elapsed=0
  while ! (echo >"/dev/tcp/$host/$port") >/dev/null 2>&1; do
    if [ "$elapsed" -ge "$timeout" ]; then
      echo "ERROR: $name is not reachable at $host:$port after ${timeout}s."
      return 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    echo "Waiting for $name on $host:$port (${elapsed}s/${timeout}s)..."
  done
  echo "$name is reachable at $host:$port."
}

if command -v docker >/dev/null 2>&1 && [ -f "$REPO_ROOT/docker-compose.yml" ]; then
  if ! docker compose ps --status running --services 2>/dev/null | rg -qx "rabbitmq"; then
    echo "Starting docker compose service: rabbitmq"
    docker compose up -d rabbitmq
  fi
  if ! docker compose ps --status running --services 2>/dev/null | rg -qx "redis"; then
    echo "Starting docker compose service: redis"
    docker compose up -d redis
  fi
fi

wait_for_tcp_port "$BROKER_HOST" "$BROKER_PORT" "RabbitMQ"
wait_for_tcp_port "$REDIS_HOST" "$REDIS_PORT" "Redis"
