#!/usr/bin/env bash
set -euo pipefail

# install_es_icu.sh — installs analysis-icu into a running Elasticsearch Docker container
# Reads connection details from config.ini ([elasticsearch] section)
# Usage:
#   ./install_es_icu.sh                 # auto-detect ES container by published port in config.ini
#   ./install_es_icu.sh <container>     # force a specific container id/name

CID="${1:-}"
CONFIG_PATH="config.ini"

# --- Parse config.ini under [elasticsearch] only ---
get_config() {
  local key="$1"
  awk -F ':' -v key="$key" '
    /^\[elasticsearch\]/ { in_es=1; next }
    /^\[/ { in_es=0 }              # stop when another section starts
    in_es && $1 ~ key {
      gsub(/^[ \t]+|[ \t]+$/, "", $2);  # trim spaces
      print $2;
      exit;
    }' "$CONFIG_PATH"
}

# --- Load config ---
HOST=$(get_config "host")
PORT=$(get_config "port")
USER=$(get_config "username")
PASSWORD=$(get_config "password")
CAFILE=$(get_config "cafile")
: "${HOST:?Missing host in config.ini}"
: "${PORT:?Missing port in config.ini}"
: "${USER:?Missing username in config.ini}"
: "${PASSWORD:?Missing password in config.ini}"
: "${CAFILE:?Missing cafile in config.ini}"

echo "🔧 Using config: $CONFIG_PATH"
echo "🌐 ES endpoint: https://${HOST}:${PORT}"
echo "🔐 CA file:     $CAFILE"
echo

# --- Helpers ---
exec_in() { docker exec -it "$CID" sh -lc "$*"; }
es_http() { curl -sS -u "${USER}:${PASSWORD}" --cacert "${CAFILE}" "$@"; }

pick_es_container() {
  # Prefer container publishing the configured $PORT (e.g., '0.0.0.0:9203->9200/tcp')
  local id
  id=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}} {{.Ports}}' \
      | awk -v p="$PORT" 'BEGIN{IGNORECASE=1}
             $0 ~ /kibana|logstash|opensearch/ {next}
             $0 ~ (":" p "->") {print $1; exit}')
  if [[ -n "$id" ]]; then echo "$id"; return; fi
  # Otherwise, any container whose image/name contains "elasticsearch" (excluding kibana/logstash/opensearch)
  id=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}} {{.Ports}}' \
      | awk 'BEGIN{IGNORECASE=1}
             $0 ~ /kibana|logstash|opensearch/ {next}
             $0 ~ /elasticsearch/ {print $1; exit}')
  if [[ -n "$id" ]]; then echo "$id"; return; fi
  # Fallback: anything exposing 9200–9299 (excluding others)
  id=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}} {{.Ports}}' \
      | awk 'BEGIN{IGNORECASE=1}
             $0 ~ /kibana|logstash|opensearch/ {next}
             $0 ~ /:92[0-9][0-9]->/ {print $1; exit}')
  if [[ -n "$id" ]]; then echo "$id"; return; fi
  echo ""
}

# --- Pick container if not provided ---
if [[ -z "$CID" ]]; then
  echo "🔎 Auto-detecting Elasticsearch container (preferring one publishing :$PORT)…"
  CID="$(pick_es_container)"
fi
if [[ -z "$CID" ]]; then
  echo "❌ Could not find a running Elasticsearch container automatically."
  echo "   Tip: docker ps --format 'table {{.ID}}\\t{{.Names}}\\t{{.Image}}\\t{{.Ports}}'"
  echo "   Then run: $0 <container-id-or-name>"
  exit 1
fi
echo "🆔 Using container: $CID"

# --- Guard: ensure not Kibana/Logstash/OpenSearch ---
INFO="$(docker inspect --format '{{.Config.Image}} {{.Name}} {{.Config.Cmd}}' "$CID" || true)"
if echo "$INFO" | grep -Eiq '(kibana|logstash|opensearch)'; then
  echo "❌ Selected container is not Elasticsearch: $INFO"
  exit 1
fi

# --- Confirm this container actually publishes HOST:PORT ---
PORTS_LINE="$(docker ps --format '{{.ID}} {{.Ports}}' | awk -v id="$CID" '$1==id { $1=""; sub(/^ /,""); print }')"
if ! echo "$PORTS_LINE" | grep -q ":${PORT}->"; then
  echo "⚠️  Selected container does not publish host port :${PORT}."
  echo "    Ports: $PORTS_LINE"
  echo "    Continuing (it may be reachable via a different network / compose service name)…"
fi

# --- Locate the elasticsearch-plugin tool inside the container ---
echo "🔎 Locating elasticsearch-plugin binary…"
PLUGIN_BIN="$(exec_in '
  for p in \
    /usr/share/elasticsearch/bin/elasticsearch-plugin \
    /opt/bitnami/elasticsearch/bin/elasticsearch-plugin \
    elasticsearch-plugin
  do
    if command -v "$p" >/dev/null 2>&1 || [ -x "$p" ]; then
      echo -n "$p"; exit 0
    fi
  done
  echo -n NO_PLUGIN_BIN
')"
if [[ "$PLUGIN_BIN" = "NO_PLUGIN_BIN" ]]; then
  echo "❌ Cannot find elasticsearch-plugin inside the container."
  echo "   Inspect: docker exec -it $CID sh -lc 'ls -R /usr/share /opt | head -n 200'"
  exit 1
fi
echo "🔧 Using plugin tool: $PLUGIN_BIN"

# --- Install if missing ---
echo "🔎 Checking installed plugins…"
if exec_in "$PLUGIN_BIN list | grep -Eqi '^analysis-icu$|(^| )analysis-icu( |$)'"; then
  echo "✅ analysis-icu is already installed."
else
  echo "📦 Installing analysis-icu (non-interactive)…"
  exec_in "$PLUGIN_BIN install --batch analysis-icu"
  echo "🔁 Restarting container…"
  docker restart "$CID" >/dev/null || true
fi

# --- Wait for ES to come back up ---
echo "⏳ Waiting for Elasticsearch at https://${HOST}:${PORT} …"
ATTEMPTS=90
SLEEP=2
OK=0
for i in $(seq 1 $ATTEMPTS); do
  if es_http "https://${HOST}:${PORT}" >/dev/null 2>&1; then OK=1; break; fi
  sleep "$SLEEP"
done
if [[ "$OK" -ne 1 ]]; then
  echo "❌ Elasticsearch did not come up after $((ATTEMPTS*SLEEP))s."
  echo "   Last container logs:"
  docker logs "$CID" | tail -n 200
  exit 1
fi
echo "✅ Elasticsearch is reachable."

# --- Verify plugin inside container ---
echo "🧪 Verifying plugin inside container…"
if exec_in "$PLUGIN_BIN list | grep -Eqi '^analysis-icu$|(^| )analysis-icu( |$)'"; then
  echo "✅ analysis-icu appears installed inside container."
else
  echo "❌ analysis-icu not listed after restart (inside container)."
  echo "   Logs: docker logs $CID | tail -n 200"
  exit 1
fi

# --- Verify via HTTP (node you connect to) ---
echo "🧪 Verifying via HTTP _cat/plugins on ${HOST}:${PORT}…"
HTTP_PLUGINS="$(es_http "https://${HOST}:${PORT}/_cat/plugins?v" || true)"
echo "$HTTP_PLUGINS"
if echo "$HTTP_PLUGINS" | grep -q 'analysis-icu'; then
  echo "🎉 analysis-icu is active from the client perspective."
else
  echo "⚠️  analysis-icu not visible via _cat/plugins at ${HOST}:${PORT}."
  echo "   • You may be querying a different ES node than the container we modified."
  echo "   • Or the node’s still warming up. Try again in a few seconds."
  echo "   • Ensure ${HOST}:${PORT} maps to container $CID:"
  docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Ports}}' | grep "$CID" || true
fi

# --- Optional extra: print cluster + node name for sanity ---
echo "ℹ️  Node identity:"
es_http "https://${HOST}:${PORT}" || true