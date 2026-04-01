#!/usr/bin/env bash
set -euo pipefail

COMPOSE="docker compose --env-file .env"

log() { printf "\n🟦 [%s] %s\n" "$(date '+%F %T')" "$*"; }
ok()  { echo "🟢 $*"; }
warn(){ echo "🟡 $*"; }
die() { echo "🔴 $*" >&2; exit 1; }

require_service_running() {
  local service="$1"
  local cid state

  cid="$($COMPOSE ps -q "$service")"
  [[ -n "$cid" ]] || die "Service '$service' not found"

  state="$(docker inspect -f '{{.State.Status}}' "$cid" 2>/dev/null || true)"
  [[ "$state" == "running" ]] || die "Service '$service' not running (state=$state)"

  ok "Service '$service' is running"
}

exec_py() {
  local service="$1"
  shift
  $COMPOSE exec -T "$service" python - "$@"
}

log "Checking services"
require_service_running api
require_service_running celery-cpu-cache

log "Testing DB connectivity (api container) 🗄️"
exec_py api <<'PY'
import os, socket, sys

host = os.environ.get("DB_HOST")
port = os.environ.get("DB_PORT")

print(f"🔎 DB_HOST={host}")
print(f"🔎 DB_PORT={port}")

if not host or not port:
    print("❌ Missing DB_HOST or DB_PORT", file=sys.stderr)
    raise SystemExit(1)

try:
    with socket.create_connection((host, int(port)), timeout=5):
        print("🟢 DB TCP connection OK")
except Exception as e:
    print(f"🔴 DB connection failed: {type(e).__name__}: {e}", file=sys.stderr)
    raise SystemExit(2)
PY

log "Testing Elasticsearch (celery-cpu-cache) 🔍"
exec_py celery-cpu-cache <<'PY'
import os, sys, json, requests

required = ["ES_HOST","ES_PORT","ES_USERNAME","ES_PASSWORD","ES_CAFILE"]
missing = [k for k in required if not os.environ.get(k)]

if missing:
    print(f"❌ Missing env vars: {', '.join(missing)}", file=sys.stderr)
    raise SystemExit(1)

host = os.environ["ES_HOST"]
port = os.environ["ES_PORT"]
cafile = os.environ["ES_CAFILE"]

print(f"🔎 ES_HOST={host}")
print(f"🔎 ES_PORT={port}")
print(f"🔎 CA file={cafile}")
print(f"📁 Exists: {os.path.exists(cafile)}")

if not os.path.exists(cafile):
    print("🔴 CA file missing inside container", file=sys.stderr)
    raise SystemExit(2)

url = f"https://{host}:{port}/_cluster/health"

try:
    r = requests.get(
        url,
        auth=(os.environ["ES_USERNAME"], os.environ["ES_PASSWORD"]),
        verify=cafile,
        timeout=10,
    )

    print(f"🌐 URL: {url}")
    print(f"📡 Status: {r.status_code}")
    r.raise_for_status()

    try:
        data = r.json()
        print("🟢 Cluster health:")
        print(json.dumps(data, indent=2)[:1000])
    except Exception:
        print("🟡 Raw response:")
        print(r.text[:1000])

except requests.exceptions.SSLError as e:
    print(f"🔴 SSL error: {e}", file=sys.stderr)
    raise SystemExit(3)
except requests.exceptions.RequestException as e:
    print(f"🔴 Request failed: {type(e).__name__}: {e}", file=sys.stderr)
    raise SystemExit(4)
PY

log "All checks passed 🎉"
ok "Everything looks healthy"
