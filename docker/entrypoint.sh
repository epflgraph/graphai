#!/usr/bin/env bash
set -euo pipefail

cd /app

export CELERY_BROKER_URL="${CELERY_BROKER_URL:-amqp://guest:guest@127.0.0.1:5672//}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-redis://127.0.0.1:6379/1}"
export PYTHONPATH="/app:${PYTHONPATH:-}"

if [[ ! -f /app/config.ini ]]; then
  echo "ERROR: /app/config.ini is required."
  exit 1
fi

mkdir -p "${GRAPH_CACHE_ROOT:-/var/graphai/storage}" "${GRAPH_LOG_ROOT:-/var/graphai/logs}"

echo "[graphai] Starting Redis..."
redis-server --daemonize yes --appendonly yes

echo "[graphai] Starting RabbitMQ..."
rabbitmq-server -detached
until rabbitmq-diagnostics -q ping >/dev/null 2>&1; do
  sleep 1
done

pids=()
start_bg() {
  "$@" &
  pids+=("$!")
}

cleanup() {
  set +e
  for pid in "${pids[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  redis-cli shutdown >/dev/null 2>&1 || true
  rabbitmqctl stop >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cd /app/graphai/api/main

echo "[graphai] Starting Celery workers..."
start_bg celery --broker="${CELERY_BROKER_URL}" -A main.celery_instance worker -l info \
  -Q embedding_gpu,voice_gpu,translation_gpu \
  --concurrency="${CELERY_GPU_CONCURRENCY:-1}" \
  -n graphai_gpu@%h

start_bg celery --broker="${CELERY_BROKER_URL}" -A main.celery_instance worker -l info -P prefork \
  -Q caching,rag,image,text,video,voice,translation,ontology,scraping,embedding,celery \
  --concurrency="${CELERY_CPU_CONCURRENCY:-8}" \
  -n graphai_cpu@%h

if [[ "${START_CELERY_BEAT:-1}" == "1" ]]; then
  start_bg celery --broker="${CELERY_BROKER_URL}" -A main.celery_instance beat -l info
fi

if [[ "${START_FLOWER:-0}" == "1" ]]; then
  start_bg celery --broker="${CELERY_BROKER_URL}" -A main.celery_instance flower \
    --port="${FLOWER_PORT:-5555}"
fi

echo "[graphai] Starting API..."
start_bg uvicorn main:app --host 0.0.0.0 --port "${API_PORT:-28800}" --workers 1

wait -n
