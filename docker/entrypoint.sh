#!/usr/bin/env bash
set -euo pipefail

cd /app

export PYTHONPATH="/app:${PYTHONPATH:-}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-amqp://guest:guest@rabbitmq:5672//}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-redis://redis:6379/1}"

if [[ ! -f /app/config.ini ]]; then
  echo "ERROR: /app/config.ini is required."
  exit 1
fi

mkdir -p \
  "${GRAPH_CACHE_ROOT:-/var/graphai/storage}" \
  "${GRAPH_LOG_ROOT:-/var/graphai/logs}"

cd /app/graphai/api/main

ROLE="${GRAPHAI_ROLE:-api}"

case "${ROLE}" in
  api)
    echo "[graphai] Starting API..."
    API_ARGS=(
      main:app
      --host "${API_HOST:-0.0.0.0}"
      --port "${API_PORT:-28800}"
      --workers "${API_WORKERS:-4}"
    )

    if [[ "${API_PROXY_HEADERS:-0}" == "1" ]]; then
      API_ARGS+=(--proxy-headers)
    fi

    if [[ -n "${API_FORWARDED_ALLOW_IPS:-}" ]]; then
      API_ARGS+=(--forwarded-allow-ips "${API_FORWARDED_ALLOW_IPS}")
    fi

    exec uvicorn "${API_ARGS[@]}"
    ;;

  worker)
    CELERY_QUEUES="${CELERY_QUEUES:?CELERY_QUEUES is required for GRAPHAI_ROLE=worker}"

    WORKER_ARGS=(
      --broker="${CELERY_BROKER_URL}"
      -A main.celery_instance
      worker
      -l "${CELERY_LOG_LEVEL:-info}"
      -Q "${CELERY_QUEUES}"
      --concurrency="${CELERY_CONCURRENCY:-1}"
      -n "${CELERY_WORKER_NAME:-graphai_worker@%h}"
    )

    if [[ -n "${CELERY_POOL:-}" ]]; then
      WORKER_ARGS+=(-P "${CELERY_POOL}")
    fi

    if [[ -n "${CELERY_PREFETCH_MULTIPLIER:-}" ]]; then
      WORKER_ARGS+=(--prefetch-multiplier "${CELERY_PREFETCH_MULTIPLIER}")
    fi

    if [[ -n "${CELERY_EXTRA_ARGS:-}" ]]; then
      # shellcheck disable=SC2206
      EXTRA_ARGS=( ${CELERY_EXTRA_ARGS} )
      WORKER_ARGS+=("${EXTRA_ARGS[@]}")
    fi

    echo "[graphai] Starting Celery worker..."
    echo "[graphai]   queues=${CELERY_QUEUES}"
    echo "[graphai]   concurrency=${CELERY_CONCURRENCY:-1}"
    echo "[graphai]   name=${CELERY_WORKER_NAME:-graphai_worker@%h}"
    [[ -n "${CELERY_POOL:-}" ]] && echo "[graphai]   pool=${CELERY_POOL}"
    [[ -n "${CELERY_PREFETCH_MULTIPLIER:-}" ]] && echo "[graphai]   prefetch=${CELERY_PREFETCH_MULTIPLIER}"
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "[graphai]   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

    exec celery "${WORKER_ARGS[@]}"
    ;;

  beat)
    echo "[graphai] Starting Celery beat..."
    exec celery \
      --broker="${CELERY_BROKER_URL}" \
      -A main.celery_instance \
      beat \
      -l "${CELERY_LOG_LEVEL:-info}"
    ;;

  flower)
    echo "[graphai] Starting Flower..."
    FLOWER_ARGS=(
      --broker="${CELERY_BROKER_URL}"
      -A main.celery_instance
      flower
      --port "${FLOWER_PORT:-5555}"
    )

    if [[ -n "${FLOWER_BASIC_AUTH:-}" ]]; then
      FLOWER_ARGS+=(--basic_auth="${FLOWER_BASIC_AUTH}")
    fi

    exec celery "${FLOWER_ARGS[@]}"
    ;;

  shell)
    exec /bin/bash
    ;;

  *)
    echo "ERROR: Unknown GRAPHAI_ROLE='${ROLE}'"
    echo "Valid values: api, worker, beat, flower, shell"
    exit 1
    ;;
esac
