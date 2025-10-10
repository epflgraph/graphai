#!/bin/bash
host="0.0.0.0"
port=28800
TIMEOUT=240

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

while getopts ":h:p:" opt; do
  case $opt in
    h) host="$OPTARG"
    ;;
    p) port="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG. Usage: deploy.sh -h 0.0.0.0 -p 28800" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

uvicorn --host $host --port $port --workers 1 main:app
#gunicorn main:app -b $host:$port -w 1 -k uvicorn.workers.UvicornWorker --timeout $TIMEOUT
