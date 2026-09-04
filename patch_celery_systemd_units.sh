#!/usr/bin/env bash
set -euo pipefail

# Patch user systemd Celery unit files for the new graphai.celery_app entry point.
# Run this once on each machine where you run ./start.sh, ./stop.sh or ./restart.sh.

USER_SYSTEMD_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
BACKUP_DIR="$USER_SYSTEMD_DIR/pre-graphai-celery-app-backup-$(date +%Y%m%d-%H%M%S)"

if [ ! -d "$USER_SYSTEMD_DIR" ]; then
  echo "ERROR: systemd user directory not found: $USER_SYSTEMD_DIR"
  exit 1
fi

map_service_to_imports() {
  local service_name="$1"
  case "$service_name" in
    celery-cpu-cache.service)               echo "graphai.celery.caching.tasks" ;;
    celery-cpu-embedding.service)           echo "graphai.celery.embedding.tasks" ;;
    celery-cpu-image.service)               echo "graphai.celery.image.tasks" ;;
    celery-cpu-rag.service)                 echo "graphai.celery.retrieval.tasks" ;;
    celery-cpu-text.service)                echo "graphai.celery.text.tasks" ;;
    celery-cpu-translate.service)           echo "graphai.celery.translation.tasks" ;;
    celery-cpu-video.service)               echo "graphai.celery.video.tasks" ;;
    celery-cpu-voice.service)               echo "graphai.celery.voice.tasks" ;;
    celery-gpu0-embedding.service)          echo "graphai.celery.embedding.tasks" ;;
    celery-gpu1-voice.service)              echo "graphai.celery.voice.tasks" ;;
    celery-gpu2-voice.service)              echo "graphai.celery.voice.tasks" ;;
    celery-gpu3-translation.service)        echo "graphai.celery.translation.tasks" ;;
    celery-cpu-ontl_scrp_celery.service)    echo "graphai.celery.ontology.tasks,graphai.celery.scraping.tasks" ;;
    *)                                      echo "" ;;
  esac
}

mkdir -p "$BACKUP_DIR"

patched=0
for service_file in "$USER_SYSTEMD_DIR"/celery-*.service; do
  [ -f "$service_file" ] || continue

  service_name="$(basename "$service_file")"
  backup_path="$BACKUP_DIR/$service_name"
  cp "$service_file" "$backup_path"

  imports="$(map_service_to_imports "$service_name")"

  if grep -q 'main\.celery_instance' "$service_file"; then
    sed -i 's/main\.celery_instance/graphai.celery_app/g' "$service_file"
    echo "Patched entry point: $service_name"
    patched=$((patched + 1))
  fi

  # The old entry point required the worker to run from graphai/api/main.
  # The new entry point must run from the repo root so graphai.celery_app is importable.
  if grep -qE '^WorkingDirectory=.*/graphai/api/main$' "$service_file"; then
    sed -i 's|/graphai/api/main$||' "$service_file"
    echo "Patched WorkingDirectory: $service_name"
  fi

  if [ -n "$imports" ]; then
    if grep -q "GRAPHAI_CELERY_IMPORTS=$imports" "$service_file"; then
      : # already correct
    elif grep -q 'GRAPHAI_CELERY_IMPORTS' "$service_file"; then
      # Replace an existing (stale) value
      sed -i "s|Environment=\"GRAPHAI_CELERY_IMPORTS=.*\"|Environment=\"GRAPHAI_CELERY_IMPORTS=$imports\"|" "$service_file"
      echo "Updated GRAPHAI_CELERY_IMPORTS: $service_name -> $imports"
    else
      # Insert a new Environment= line just above the first ExecStart=
      sed -i "/^ExecStart=/i Environment=\"GRAPHAI_CELERY_IMPORTS=$imports\"" "$service_file"
      echo "Added GRAPHAI_CELERY_IMPORTS: $service_name -> $imports"
    fi
  else
    echo "WARNING: unknown service name, skipping GRAPHAI_CELERY_IMPORTS: $service_name"
  fi
done

if [ "$patched" -eq 0 ]; then
  echo "No 'main.celery_instance' references found in $USER_SYSTEMD_DIR/celery-*.service"
  echo "Either they are already patched or the unit files live elsewhere."
fi

echo "Backups saved to: $BACKUP_DIR"
echo "Reloading systemd user units..."
systemctl --user daemon-reload
echo "Done. You can now run ./restart.sh or ./start.sh as usual."
