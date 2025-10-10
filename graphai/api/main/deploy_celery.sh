#!/bin/bash
set -euo pipefail

# ---------- paths & env ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
else
  echo "WARN: .env not found at $ENV_FILE"; exit 1
fi

# Force this repo’s venv
VENV_DIR="$REPO_ROOT/.venv311"
export PATH="$VENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

echo "Using python: $(command -v python)"
echo "Using celery: $(command -v celery)"
python - <<'PY'
import sys, numpy
print("Python exe:", sys.executable)
print("NumPy:", numpy.__version__, "from", numpy.__file__)
PY

# ---------- stop helpers ----------
stop_node() {
  local name="${1:-}"
  if [ -z "$name" ]; then
    echo "stop_node: missing name" >&2
    return 1
  fi
  local pidfile="/tmp/${name}.pid"

  if [ -f "$pidfile" ]; then
    local pid; pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (pid $pid)…"
      kill "$pid" 2>/dev/null || true
      for _ in {1..20}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
}

for n in celery-beat celery-workerHigh celery-workerLow celery-workerMid celery-workerCache celery-flower; do
  stop_node "$n"
done

# (belt & suspenders) kill any stragglers from this app
pkill -f "celery -A main.celery_instance" 2>/dev/null || true

# ---------- start daemons ----------
# Beat
nice -n 0 celery -A main.celery_instance beat \
  --pidfile /tmp/celery-beat.pid \
  --logfile /tmp/celery-beat.log \
  --detach

# Workers (use prefetch=1 everywhere unless you really want buffering)
nice -n 0 celery -A main.celery_instance worker \
  --hostname workerHigh@%h -l info -P threads --prefetch-multiplier 1 -c 1 \
  -Q text_10,retrieval_10 --without-gossip \
  --pidfile /tmp/celery-workerHigh.pid --logfile /tmp/celery-workerHigh.log -D

nice -n 20 celery -A main.celery_instance worker \
  --hostname workerLow@%h -l info -P threads --prefetch-multiplier 1 -c 1 \
  -Q celery,video_2,ontology_6,text_6 --without-gossip \
  --pidfile /tmp/celery-workerLow.pid --logfile /tmp/celery-workerLow.log -D

# Choose EXACTLY ONE of these for workerMid:
# A) prefork (will spawn parent+child even with -c 1)
nice -n 20 celery -A main.celery_instance worker \
  --hostname workerMid@%h -l info -P prefork --prefetch-multiplier 1 -c 1 \
  -Q scraping_6 --without-gossip \
  --pidfile /tmp/celery-workerMid.pid --logfile /tmp/celery-workerMid.log -D

# B) solo (uncomment to force single process, single task at a time)
# nice -n 20 celery -A main.celery_instance worker \
#   --hostname workerMid@%h -l info -P solo \
#   -Q scraping_6 --without-gossip \
#   --pidfile /tmp/celery-workerMid.pid --logfile /tmp/celery-workerMid.log -D

nice -n 10 celery -A main.celery_instance worker \
  --hostname workerCache@%h -l info -P threads --prefetch-multiplier 1 -c 1 \
  -Q caching_6 --without-gossip \
  --pidfile /tmp/celery-workerCache.pid --logfile /tmp/celery-workerCache.log -D

# Flower (no --without-gossip)
FLOWER_UNAUTHENTICATED_API=true celery -A main.celery_instance flower \
  --port=5555 \
  --pidfile=/tmp/celery-flower.pid \
  --logfile=/tmp/celery-flower.log \
  --detach





# #!/bin/bash
# set -euo pipefail

# # Resolve paths
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# # deploy.sh lives in: <repo>/graphai/api/main/deploy.sh
# # repo root is three levels up:
# REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# ENV_FILE="$REPO_ROOT/.env"

# # Load project env (if present)
# if [ -f "$ENV_FILE" ]; then
#   set -a
#   . "$ENV_FILE"
#   set +a
# else
#   echo "WARN: .env not found at $ENV_FILE"
#   # Exit
#   exit 1
# fi

# # (debug) show what we loaded
# env | egrep '^CELERY_|^VIRTUAL_ENV' || true

# # Force isolation to the venv
# # (adjust the venv path if different)
# VENV_DIR="$(cd "$(dirname "$0")/../.."; pwd)/.venv311"
# export PATH="$VENV_DIR/bin:$PATH"
# export PYTHONNOUSERSITE=1
# unset PYTHONPATH
# unset PYTHONHOME

# # Debug: show which python/celery and NumPy actually load
# echo "Using python: $(which python)"
# echo "Using celery: $(which celery)"
# python - <<'PY'
# import sys, numpy
# print("Python exe:", sys.executable)
# print("NumPy:", numpy.__version__, "from", numpy.__file__)
# PY




# # # Celery 5+: --without-gossip is a worker option; beat/flower don't accept it
# # nice -n 0  celery -A main.celery_instance beat --detach
# # nice -n 0  celery -A main.celery_instance worker --hostname workerHigh@%h  -l info -P threads  --prefetch-multiplier 20 -c 1 -Q text_10,retrieval_10             -D --without-gossip
# # nice -n 20 celery -A main.celery_instance worker --hostname workerLow@%h   -l info -P threads  --prefetch-multiplier  1 -c 1 -Q celery,video_2,ontology_6,text_6 -D --without-gossip
# # nice -n 20 celery -A main.celery_instance worker --hostname workerMid@%h   -l info -P prefork  --prefetch-multiplier  1 -c 1 -Q scraping_6                       -D --without-gossip
# # nice -n 10 celery -A main.celery_instance worker --hostname workerCache@%h -l info -P threads  --prefetch-multiplier 10 -c 1 -Q caching_6                        -D --without-gossip

# # Beat
# nice -n 0 celery -A main.celery_instance beat \
#   --pidfile /tmp/celery-beat.pid \
#   --logfile /tmp/celery-beat.log \
#   --detach

# # Workers (one proc each except prefork)
# nice -n 0  celery -A main.celery_instance worker \
#   --hostname workerHigh@%h -l info -P threads --prefetch-multiplier 20 -c 1 \
#   -Q text_10,retrieval_10 --without-gossip \
#   --pidfile /tmp/celery-workerHigh.pid --logfile /tmp/celery-workerHigh.log -D

# nice -n 20 celery -A main.celery_instance worker \
#   --hostname workerLow@%h -l info -P threads --prefetch-multiplier 1 -c 1 \
#   -Q celery,video_2,ontology_6,text_6 --without-gossip \
#   --pidfile /tmp/celery-workerLow.pid --logfile /tmp/celery-workerLow.log -D

# # If you want *one* process only, avoid prefork:
# # option A: keep prefork (will be 2 procs with -c 1)
# nice -n 20 celery -A main.celery_instance worker \
#   --hostname workerMid@%h -l info -P prefork --prefetch-multiplier 1 -c 1 \
#   -Q scraping_6 --without-gossip \
#   --pidfile /tmp/celery-workerMid.pid --logfile /tmp/celery-workerMid.log -D

# # option B: change to SOLO (strictly 1 process, 1 task at a time)
# # nice -n 20 celery -A main.celery_instance worker \
# #   --hostname workerMid@%h -l info -P solo -Q scraping_6 --without-gossip \
# #   --pidfile /tmp/celery-workerMid.pid --logfile /tmp/celery-workerMid.log -D

# nice -n 10 celery -A main.celery_instance worker \
#   --hostname workerCache@%h -l info -P threads --prefetch-multiplier 10 -c 1 \
#   -Q caching_6 --without-gossip \
#   --pidfile /tmp/celery-workerCache.pid --logfile /tmp/celery-workerCache.log -D

# # Flower (no --without-gossip)
# FLOWER_UNAUTHENTICATED_API=true celery -A main.celery_instance flower \
#   --port=5555 --pidfile=/tmp/celery-flower.pid --logfile=/tmp/celery-flower.log --detach


# # # Flower (no --without-gossip flag)
# # FLOWER_UNAUTHENTICATED_API=true celery -A main.celery_instance flower --port=5555



