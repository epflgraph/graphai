#!/bin/bash
set -euo pipefail

host="0.0.0.0"
port=28800
PID_FILE="/tmp/graphai-api.pid"
LOG_FILE="/tmp/graphai-api.log"

# ---------- resolve paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
VENV_DIR="$REPO_ROOT/.venv311"

# ---------- load .env if available ----------
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  . "$ENV_FILE"
  set +a
else
  echo "WARN: .env not found at $ENV_FILE" >&2
  exit 1
fi

# ---------- parse flags ----------
while getopts ":h:p:" opt; do
  case "$opt" in
    h) host="$OPTARG" ;;
    p) port="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG. Usage: $0 -h 0.0.0.0 -p 28800" >&2; exit 1 ;;
  esac
done

# ---------- use venv if it exists ----------
if [[ -d "$VENV_DIR" ]]; then
  export PATH="$VENV_DIR/bin:$PATH"
fi

# ---------- stop existing process if running ----------
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(cat "$PID_FILE" 2>/dev/null || true)
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Stopping existing API (pid $old_pid)..."
    kill "$old_pid" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "$old_pid" 2>/dev/null || break
      sleep 0.25
    done
    kill -9 "$old_pid" 2>/dev/null || true
    echo "✅ Stopped previous API."
  fi
  rm -f "$PID_FILE"
fi

# ---------- start uvicorn ----------
echo "Launching GraphAI API at http://$host:$port"
echo "Logs → $LOG_FILE"

cd "$SCRIPT_DIR"
uvicorn --host "$host" --port "$port" --workers 1 main:app
new_pid=$!
echo "$new_pid" > "$PID_FILE"

echo "✅ API started (pid $new_pid)"
echo "To follow logs: tail -f $LOG_FILE"

# #!/bin/bash
# host="0.0.0.0"
# port=28800
# TIMEOUT=240

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

# while getopts ":h:p:" opt; do
#   case $opt in
#     h) host="$OPTARG"
#     ;;
#     p) port="$OPTARG"
#     ;;
#     \?) echo "Invalid option -$OPTARG. Usage: deploy.sh -h 0.0.0.0 -p 28800" >&2
#     exit 1
#     ;;
#   esac

#   case $OPTARG in
#     -*) echo "Option $opt needs a valid argument"
#     exit 1
#     ;;
#   esac
# done

# uvicorn --host $host --port $port --workers 1 main:app
# #gunicorn main:app -b $host:$port -w 1 -k uvicorn.workers.UvicornWorker --timeout $TIMEOUT
