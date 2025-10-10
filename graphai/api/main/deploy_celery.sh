#!/bin/bash
set -euo pipefail

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# deploy.sh lives in: <repo>/graphai/api/main/deploy.sh
# repo root is three levels up:
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# Load project env (if present)
if [ -f "$ENV_FILE" ]; then
  set -a
  . "$ENV_FILE"
  set +a
else
  echo "WARN: .env not found at $ENV_FILE"
  # Exit
  exit 1
fi

# (debug) show what we loaded
env | egrep '^CELERY_|^VIRTUAL_ENV' || true

# Force isolation to the venv
# (adjust the venv path if different)
VENV_DIR="$(cd "$(dirname "$0")/../.."; pwd)/.venv311"
export PATH="$VENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

# Debug: show which python/celery and NumPy actually load
echo "Using python: $(which python)"
echo "Using celery: $(which celery)"
python - <<'PY'
import sys, numpy
print("Python exe:", sys.executable)
print("NumPy:", numpy.__version__, "from", numpy.__file__)
PY

# Celery 5+: --without-gossip is a worker option; beat/flower don't accept it
nice -n 0  celery -A main.celery_instance beat --detach
nice -n 0  celery -A main.celery_instance worker --hostname workerHigh@%h  -l info -P threads  --prefetch-multiplier 20 -c 16 -Q text_10,retrieval_10             -D --without-gossip
nice -n 20 celery -A main.celery_instance worker --hostname workerLow@%h   -l info -P threads  --prefetch-multiplier  1 -c 16 -Q celery,video_2,ontology_6,text_6 -D --without-gossip
nice -n 20 celery -A main.celery_instance worker --hostname workerMid@%h   -l info -P prefork  --prefetch-multiplier  1 -c 20 -Q scraping_6                       -D --without-gossip
nice -n 10 celery -A main.celery_instance worker --hostname workerCache@%h -l info -P threads  --prefetch-multiplier 10 -c 16 -Q caching_6                        -D --without-gossip

# Flower (no --without-gossip flag)
FLOWER_UNAUTHENTICATED_API=true celery -A main.celery_instance flower --port=5555
