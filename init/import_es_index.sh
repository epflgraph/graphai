#!/usr/bin/env bash
set -euo pipefail

# --- Usage ---
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <index_folder> [--replace]"
  exit 1
fi
INDEX_DIR=$1
REPLACE=${2:-""}

# --- Load configuration ---
CONFIG_PATH=config.ini

# --- Parse config.ini under [elasticsearch] section only ---
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

HOST=$(get_config "host")
PORT=$(get_config "port")
USER=$(get_config "username")
PASSWORD=$(get_config "password")
CAFILE=$(get_config "cafile")
INDEX_NAME=$(get_config "concept_detection_index")

# --- Require jq ---
if ! command -v jq >/dev/null 2>&1; then
  echo "❌ This script requires 'jq'. Please install it (brew install jq) and re-run."
  exit 1
fi

# --- Set environment variable for Node.js to use custom CA file ---
export NODE_EXTRA_CA_CERTS=$CAFILE

# --- Derived vars ---
SERVER="https://${USER}:${PASSWORD}@${HOST}:${PORT}"
MAPFILEPATH="${INDEX_DIR}/mappings.json"
DATAFILEPATH="${INDEX_DIR}/data.json"
ANALYZERPATH="${INDEX_DIR}/analyzer.json"

echo "📄 Using config: $CONFIG_PATH"
echo "🌐 Elasticsearch: $SERVER"
echo "🔐 CA file: $CAFILE"
echo "📂 Index folder: $INDEX_DIR"
echo "🧭 Target index: $INDEX_NAME"
echo

# --- Delete index if exists and --replace flag given ---
if [[ "$REPLACE" == "--replace" ]]; then
  echo "⚠️  --replace flag detected. Checking for existing index..."
  if curl -s -u "${USER}:${PASSWORD}" --cacert "$CAFILE" -I -X HEAD \
      "https://${HOST}:${PORT}/${INDEX_NAME}" | grep -q "200 OK"; then
    echo "🗑️  Index ${INDEX_NAME} exists. Deleting..."
    curl -s -u "${USER}:${PASSWORD}" --cacert "$CAFILE" \
      -X DELETE "https://${HOST}:${PORT}/${INDEX_NAME}" | jq .
    echo "✅ Index deleted."
  else
    echo "ℹ️  Index ${INDEX_NAME} does not exist — nothing to delete."
  fi
  echo
fi

# --- Check files ---
echo "🗂️ Checking files..."
ls -lh "$MAPFILEPATH" 2>/dev/null || { echo "❌ mappings.json not found at $MAPFILEPATH"; exit 1; }
ls -lh "$ANALYZERPATH" 2>/dev/null || echo "ℹ️ analyzer.json not found (will rely on settings in mappings.json if present)"
ls -lh "$DATAFILEPATH" 2>/dev/null || echo "ℹ️ data.json not found (you can import later)"
echo

# --- Verify connectivity ---
echo "🔎 Checking Elasticsearch connectivity..."
curl -s -u "${USER}:${PASSWORD}" --cacert "$CAFILE" "$SERVER" | jq .version.number
echo

# --- Create index with curl: combine settings+mappings properly ---
echo "🏗️ Creating index '${INDEX_NAME}' with settings + mappings via curl..."

TMP_CREATE_JSON="$(mktemp)"

if [[ -f "$ANALYZERPATH" ]]; then
  # Merge analyzer.settings and mapping.settings; use mapping.mappings or mapping root as mappings
  jq -n --slurpfile M "$MAPFILEPATH" --slurpfile A "$ANALYZERPATH" '
    {
      settings: (($A[0].settings // {}) * ($M[0].settings // {})),
      mappings: ($M[0].mappings // $M[0])
    }
  ' > "$TMP_CREATE_JSON"
else
  # Use mapping file alone; allow either {settings,mappings} or bare {mappings}
  jq -n --slurpfile M "$MAPFILEPATH" '
    {
      settings: ($M[0].settings // {}),
      mappings: ($M[0].mappings // $M[0])
    }
  ' > "$TMP_CREATE_JSON"
fi

# If index already exists (and no --replace), skip creation
if curl -s -u "${USER}:${PASSWORD}" --cacert "$CAFILE" -I -X HEAD \
      "https://${HOST}:${PORT}/${INDEX_NAME}" | grep -q "200 OK"; then
  echo "ℹ️  Index ${INDEX_NAME} already exists. Skipping create."
else
  # PUT /{index}
  HTTP_CODE=$(curl -sS -o /tmp/es_create_out.$$ -w "%{http_code}" \
    -u "${USER}:${PASSWORD}" --cacert "$CAFILE" \
    -H 'Content-Type: application/json' \
    -X PUT "https://${HOST}:${PORT}/${INDEX_NAME}" \
    --data-binary @"$TMP_CREATE_JSON")

  if [[ "$HTTP_CODE" != "200" ]]; then
    echo "❌ Failed to create index (HTTP $HTTP_CODE). Elasticsearch says:"
    if jq . >/dev/null 2>&1 < /tmp/es_create_out.$$; then
      jq '.' /tmp/es_create_out.$$
    else
      cat /tmp/es_create_out.$$
    fi
    rm -f "$TMP_CREATE_JSON" /tmp/es_create_out.$$
    exit 1
  fi
  rm -f /tmp/es_create_out.$$
  echo "✅ Index created."
fi

rm -f "$TMP_CREATE_JSON"
echo

# --- Simple sanity check: did the index get any field mappings?
echo "🧪 Sanity: listing top-level mapped fields…"
curl -sS -u "${USER}:${PASSWORD}" --cacert "$CAFILE" \
  "https://${HOST}:${PORT}/${INDEX_NAME}/_mapping?filter_path=**.mappings.properties" \
  | jq -r '.[].mappings.properties | keys[]?' || true
echo

# --- Import DATA with elasticdump (force destination index) ---
if [[ -f "$DATAFILEPATH" ]]; then
  echo "🚚 Importing data with elasticdump..."
  npx elasticdump \
    --input="$DATAFILEPATH" \
    --output="$SERVER" \
    --output-index="$INDEX_NAME" \
    --type=data \
    --tls-ca="$CAFILE" \
  || {
    echo "⚠️ TLS verify failed — retrying with --tls-reject-unauthorized=false"
    npx elasticdump \
      --input="$DATAFILEPATH" \
      --output="$SERVER" \
      --output-index="$INDEX_NAME" \
      --type=data \
      --tls-reject-unauthorized=false
  }
  echo
else
  echo "ℹ️ Skipping data import (no data.json found)."
fi

echo "✅ Done."