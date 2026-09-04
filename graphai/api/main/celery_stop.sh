#!/usr/bin/env bash
set -euo pipefail

echo "=== [GraphAI] Deploying Celery GPU/CPU workers ==="

# ---------------------------------------------------------------------
# 1) Load .env from repo root to get AI_RABBITMQ_DEV_CELERY_USER/PASS
# ---------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ -f "$REPO_ROOT/.env" ]; then
  echo "Loading environment from $REPO_ROOT/.env"
  # auto-export vars while sourcing
  set -a
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.env"
  set +a
else
  echo "ERROR: .env not found in $REPO_ROOT"
  exit 1
fi

: "${AI_RABBITMQ_DEV_CELERY_USER:?AI_RABBITMQ_DEV_CELERY_USER not set in .env}"
: "${AI_RABBITMQ_DEV_CELERY_PASS:?AI_RABBITMQ_DEV_CELERY_PASS not set in .env}"

export CELERY_BROKER_URL="amqp://${AI_RABBITMQ_DEV_CELERY_USER}:${AI_RABBITMQ_DEV_CELERY_PASS}@localhost:5672//"
export CELERY_RESULT_BACKEND="redis://localhost:6379/1"

echo "Broker:  amqp://${AI_RABBITMQ_DEV_CELERY_USER}:***@localhost:5672//"
echo "Backend: $CELERY_RESULT_BACKEND"

# Run from repo root so `-A graphai.celery_app` resolves correctly
cd "$REPO_ROOT"

# ---------------------------------------------------------------------
# 2) Stop existing Celery workers
# ---------------------------------------------------------------------
echo "=== [GraphAI] Stopping existing Celery workers ==="
PIDS="$(pgrep -f 'celery.*graphai.celery_app' || true)"

if [ -n "$PIDS" ]; then
  echo "Killing PIDs: $PIDS"
  kill $PIDS || true
  sleep 3

  PIDS="$(pgrep -f 'celery.*graphai.celery_app' || true)"
  if [ -n "$PIDS" ]; then
    echo "Force killing remaining PIDs: $PIDS"
    kill -9 $PIDS || true
  fi
else
  echo "No existing Celery workers found."
fi
