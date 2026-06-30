#!/usr/bin/env bash
set -euo pipefail

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

SERVICES=(
  promtail.service
  celery-cpu-cache.service
  celery-cpu-embedding.service
  celery-cpu-image.service
  celery-cpu-ontl_scrp_celery.service
  celery-cpu-rag.service
  celery-cpu-text.service
  celery-cpu-translate.service
  celery-cpu-video.service
  celery-cpu-voice.service
  celery-gpu0-embedding.service
  celery-gpu1-voice.service
  celery-gpu2-voice.service
  celery-gpu3-translation.service
  uvicorn-api-graphai.service
)

check_tcp_port() {
  local host="$1"
  local port="$2"
  local name="$3"
  if (echo >"/dev/tcp/$host/$port") >/dev/null 2>&1; then
    echo "[OK] $name reachable at $host:$port"
  else
    echo "[FAIL] $name NOT reachable at $host:$port"
    return 1
  fi
}

echo "Checking dependency endpoints..."
FAILED=0

check_tcp_port "$BROKER_HOST" "$BROKER_PORT" "RabbitMQ" || FAILED=1
check_tcp_port "$REDIS_HOST" "$REDIS_PORT" "Redis" || FAILED=1

if command -v docker >/dev/null 2>&1 && [ -f "$REPO_ROOT/docker-compose.yml" ]; then
  if docker compose ps --status running --services 2>/dev/null | rg -qx "rabbitmq"; then
    echo "[OK] docker compose service rabbitmq is running"
  else
    echo "[FAIL] docker compose service rabbitmq is not running"
    FAILED=1
  fi

  if docker compose ps --status running --services 2>/dev/null | rg -qx "redis"; then
    echo "[OK] docker compose service redis is running"
  else
    echo "[FAIL] docker compose service redis is not running"
    FAILED=1
  fi
fi

echo
echo "Checking user-level services status..."
for SERVICE in "${SERVICES[@]}"; do
  if systemctl --user is-active --quiet "$SERVICE"; then
    echo "[OK] $SERVICE is running"
  else
    echo "[FAIL] $SERVICE is NOT running"
    systemctl --user status "$SERVICE" --no-pager || true
    echo
    FAILED=1
  fi
done

if [ "$FAILED" -eq 0 ]; then
  echo
  echo "All services and dependencies are healthy."
else
  echo
  echo "One or more dependencies/services are unhealthy."
  exit 1
fi
