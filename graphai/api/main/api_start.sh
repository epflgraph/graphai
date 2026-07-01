#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------
# Load .env from repo root to get AI_RABBITMQ_DEV_CELERY_USER/PASS
# ---------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ -f "$REPO_ROOT/.env" ]; then
  echo "Loading environment from $REPO_ROOT/.env"
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

# Point Celery at that user on the host-mapped port
export CELERY_BROKER_URL="amqp://${AI_RABBITMQ_DEV_CELERY_USER}:${AI_RABBITMQ_DEV_CELERY_PASS}@localhost:5672//"

# Match whatever Redis DB you want; keep consistent with workers
export CELERY_RESULT_BACKEND="redis://localhost:6379/1"

# Optional: helpful log (remove if it prints secrets in your terminal history)
echo "Broker:  amqp://${AI_RABBITMQ_DEV_CELERY_USER}:***@localhost:5672//"
echo "Backend: $CELERY_RESULT_BACKEND"

# Ensure uvicorn import path is correct (adjust if this script isn't in api/main)
cd "$SCRIPT_DIR"

uvicorn main:app --host 0.0.0.0 --port 28800 --workers "${API_WORKERS:-4}"