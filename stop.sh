#!/usr/bin/env bash
set -euo pipefail

echo "Stopping all services..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

systemctl --user stop promtail.service
systemctl --user stop celery-cpu-cache.service
systemctl --user stop celery-cpu-embedding.service
systemctl --user stop celery-cpu-image.service
systemctl --user stop celery-cpu-ontl_scrp_celery.service
systemctl --user stop celery-cpu-rag.service
systemctl --user stop celery-cpu-text.service
systemctl --user stop celery-cpu-translate.service
systemctl --user stop celery-cpu-video.service
systemctl --user stop celery-cpu-voice.service
systemctl --user stop celery-gpu0-embedding.service
systemctl --user stop celery-gpu1-voice.service
systemctl --user stop celery-gpu2-voice.service
systemctl --user stop celery-gpu3-translation.service
systemctl --user stop uvicorn-api-graphai.service
systemctl --user daemon-reload

echo "All user services stopped."

# Optional: also stop broker/backend containers used by local development.
# Enable with STOP_DEPENDENCIES=1 ./stop.sh
if [ "${STOP_DEPENDENCIES:-0}" = "1" ] && command -v docker >/dev/null 2>&1 && [ -f "$REPO_ROOT/docker-compose.yml" ]; then
  echo "Stopping docker compose dependencies: rabbitmq redis"
  docker compose stop rabbitmq redis
fi

echo "Stop sequence complete."
