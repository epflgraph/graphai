#!/usr/bin/env bash
set -euo pipefail

SERVICES=(
  promtail.service
  celery-cpu-cache.service
  celery-cpu-rag.service
  celery-cpu-img.service
  celery-cpu-ontl_scrp_celery.service
  celery-cpu-text.service
  celery-cpu-embedding.service
  celery-cpu-video_voice_transl.service
  celery-gpu0-embedding.service
  celery-gpu1-voice.service
  celery-gpu2-voice.service
  celery-gpu3-translation.service
  uvicorn-api-graphai.service
)

echo "Checking user-level services status..."
echo

FAILED=0

for SERVICE in "${SERVICES[@]}"; do
  if systemctl --user is-active --quiet "$SERVICE"; then
    echo "✅ $SERVICE is running"
  else
    echo "❌ $SERVICE is NOT running"
    systemctl --user status "$SERVICE" --no-pager || true
    echo
    FAILED=1
  fi
done

if [ "$FAILED" -eq 0 ]; then
  echo
  echo "🎉 All services are running successfully."
else
  echo
  echo "🚨 One or more services failed to start."
  exit 1
fi