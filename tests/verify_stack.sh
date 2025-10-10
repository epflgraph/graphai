#!/usr/bin/env bash
set -euo pipefail

RED=$'\e[31m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; NC=$'\e[0m'
ok(){ echo "${GREEN}✔${NC} $*"; }
warn(){ echo "${YELLOW}!${NC} $*"; }
die(){ echo "${RED}✖${NC} $*"; exit 1; }

# --- Locate repo root & .env ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then :; else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
ENV_FILE="$REPO_ROOT/.env"
[ -f "$ENV_FILE" ] || warn ".env not found at $ENV_FILE (will use defaults)"

# export .env (if present)
set -a; [ -f "$ENV_FILE" ] && . "$ENV_FILE"; set +a

# --- Ensure venv (optional but recommended) ---
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d "$REPO_ROOT/.venv311" ]; then
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.venv311/bin/activate"
fi
command -v python >/dev/null || die "python not on PATH (activate your venv)"

ok "Using python: $(which python)"
ok "Python version: $(python -V 2>&1 | tr -d '\n')"
ok "Repo root: $REPO_ROOT"

# Detect Docker presence once; we’ll use this to skip local CLIs
if docker ps >/dev/null 2>&1; then
  SKIP_BREW_CLI=1
else
  SKIP_BREW_CLI=0
fi

# --- Helper to print versions (Docker-aware for rabbitmqctl/redis-cli) ---
vers () {
  local cmd="$1"; shift || true
  if [ "$SKIP_BREW_CLI" = 1 ] && [[ "$cmd" =~ ^(rabbitmqctl|redis-cli)$ ]]; then
    warn "$cmd not checked (Docker mode)"
    return 0
  fi
  if command -v "$cmd" >/dev/null; then
    echo -n "$cmd: "
    "$cmd" "$@" | head -n1 || true
  else
    warn "$cmd not found"
  fi
}

# --- Quick versions of the usual suspects (don’t fail if missing, just warn) ---
vers brew --version
vers ffmpeg -version
vers tesseract --version
vers pdftoppm -v || true
vers pdfinfo -v || true
vers fpcalc -version || true
vers convert -version || true
vers rabbitmqctl status || true
vers redis-cli --version || true
ok "Printed tool versions"

# --- Network ports ---
echo "Checking ports…"
nc -z localhost 5672 && ok "RabbitMQ port 5672 reachable" || die "RabbitMQ port 5672 not reachable"
nc -z localhost 6379 && ok "Redis port 6379 reachable" || die "Redis port 6379 not reachable"

# --- AMQP & Redis URLs from env (with safe defaults) ---
BROKER_URL="${CELERY_BROKER_URL:-amqp://graphai:graphai@127.0.0.1:5672/%2F}"
RESULT_URL="${CELERY_RESULT_BACKEND:-redis://127.0.0.1:6379/1}"
echo "Broker: $BROKER_URL"
echo "Backend: $RESULT_URL"

# --- Python module import smoke test ---
python - <<'PY' || exit 1
import sys, importlib, json
mods = [
  "kombu", "celery", "redis", "pydantic", "fastapi",
  "numpy", "pandas", "scipy", "sklearn",
  "pdf2image", "PIL", "fitz",  # PyMuPDF is 'fitz'
  "pytesseract"                # python binding; OK if missing
]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        failed.append((m, str(e)))
print("Imported", len(mods)-len(failed), "modules OK")
if failed:
    print("Missing/failed:", json.dumps(failed, indent=2))
    # Don't hard-fail for pytesseract; binaries matter more
    real_fail = [m for m,_ in failed if m not in ("pytesseract",)]
    if real_fail:
        sys.exit(2)
PY
ok "Core Python imports look fine"

# --- Check hashlib/blake2 + OpenSSL provider visibility ---
python - <<'PY' || exit 1
import hashlib, ssl
print("OpenSSL:", ssl.OPENSSL_VERSION)
print("blake2b?", 'blake2b' in hashlib.algorithms_available,
      "blake2s?", 'blake2s' in hashlib.algorithms_available)
PY
ok "Crypto/hashlib sanity ok"

# --- AMQP real connection test ---
python - <<PY || die "AMQP connection failed"
import os
from kombu import Connection
url=os.environ.get("CELERY_BROKER_URL","amqp://graphai:graphai@127.0.0.1:5672/%2F")
print("Testing AMQP:", url)
with Connection(url) as conn:
    conn.connect()
    print("AMQP OK, authenticated")
PY
ok "AMQP connect/auth OK"

# --- Redis real PING test ---
python - <<PY || die "Redis connection failed"
import os
import redis
url=os.environ.get("CELERY_RESULT_BACKEND","redis://127.0.0.1:6379/1")
print("Testing Redis:", url)
r = redis.Redis.from_url(url)
resp = r.ping()
print("Redis PING:", resp)
PY
ok "Redis PING OK"

# --- Poppler + pdf2image roundtrip ---
python - <<'PY' || { warn "pdf2image/poppler test failed (poppler missing?)"; exit 0; }
import tempfile, os
from pathlib import Path
try:
    import fitz  # PyMuPDF
    from pdf2image import convert_from_path
except Exception as e:
    raise SystemExit("Imports failed: "+str(e))
tmpdir = Path(tempfile.mkdtemp())
pdf = tmpdir/"hello.pdf"
img = tmpdir/"page-1.png"
doc = fitz.open()
page = doc.new_page()
page.insert_text((72,72), "hello pdf2image 👋", fontsize=24)
doc.save(pdf.open("wb")); doc.close()
pages = convert_from_path(str(pdf), fmt="png", dpi=72)
pages[0].save(img)
print("Rendered:", img, "exists?", img.exists())
assert img.exists()
PY
ok "pdf2image + poppler functional"

# --- ffmpeg trivial test (use mktemp; no overwrite prompt) ---
echo "Creating 1s silent wav via ffmpeg…"
if command -v ffmpeg >/dev/null; then
  TMPWAV="$(mktemp /tmp/silence.XXXXXX.wav)"
  ffmpeg -hide_banner -loglevel error -y -f lavfi -i anullsrc=r=16000:cl=mono -t 1 "$TMPWAV"
  [ -s "$TMPWAV" ] && ok "ffmpeg rendered $TMPWAV" || warn "ffmpeg test did not create file"
  rm -f "$TMPWAV"
else
  warn "ffmpeg not found; skipping render test"
fi

# --- tesseract presence test (binary) ---
if command -v tesseract >/dev/null; then
  ok "tesseract present"
else
  warn "tesseract not found (OCR features may be limited)"
fi

# --- Celery app sanity and (if running) ping workers ---
python - <<'PY'
try:
    from graphai.celery.common.celery_tools import celery_instance as app
    print("Celery app loaded.")
    print("broker_url:", app.conf.broker_url)
    print("result_backend:", app.conf.result_backend)
    insp = app.control.inspect(timeout=2)
    stats = insp.stats() if insp else None
    if stats:
        print("Workers seen:", list(stats.keys()))
        print("Ping:", insp.ping())
    else:
        print("No workers responding (ok if not started yet).")
except Exception as e:
    print("Celery app load failed:", e)
PY
ok "Celery config reachable (workers may or may not be up—this is just a check)"

echo
ok "All smoke tests finished."