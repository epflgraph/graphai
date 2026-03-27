#!/usr/bin/env bash
set -euo pipefail

RED=$'\e[31m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; NC=$'\e[0m'
ok(){ echo "${GREEN}✔${NC} $*"; }
warn(){ echo "${YELLOW}!${NC} $*"; }
die(){ echo "${RED}✖${NC} $*"; exit 1; }

AUTO_INSTALL=0
ASSUME_YES=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --install-missing) AUTO_INSTALL=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    *) die "Unknown option: $1 (supported: --install-missing, --yes)" ;;
  esac
  shift
done

MISSING_TOOLS=()
record_missing() {
  local cmd="$1"
  local it
  for it in "${MISSING_TOOLS[@]:-}"; do
    [ "$it" = "$cmd" ] && return 0
  done
  MISSING_TOOLS+=("$cmd")
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

# --- Locate repo root & .env ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
ENV_FILE="$REPO_ROOT/.env"

[ -f "$ENV_FILE" ] || warn ".env not found at $ENV_FILE (will use defaults)"

# --- Load .env safely enough for common KEY=VALUE files ---
# shellcheck disable=SC1090
if [ -f "$ENV_FILE" ]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

# --- Ensure venv (optional but recommended) ---
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d "$REPO_ROOT/.venv311" ]; then
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.venv311/bin/activate"
fi
command -v python >/dev/null 2>&1 || die "python not on PATH (activate your venv)"

ok "Using python: $(command -v python)"
ok "Python version: $(python -V 2>&1 | tr -d '\n')"
ok "Repo root: $REPO_ROOT"

cd "$REPO_ROOT"

# --- Detect Docker presence once; skip local CLIs if Docker is available ---
if have_cmd docker && docker ps >/dev/null 2>&1; then
  SKIP_LOCAL_BROKER_CLIS=1
else
  SKIP_LOCAL_BROKER_CLIS=0
fi

# --- Helper to print versions (Docker-aware for rabbitmqctl/redis-cli) ---
vers() {
  local cmd="$1"
  shift || true

  if [ "$SKIP_LOCAL_BROKER_CLIS" = 1 ] && [[ "$cmd" =~ ^(rabbitmqctl|redis-cli)$ ]]; then
    warn "$cmd not checked locally (Docker mode)"
    return 0
  fi

  if have_cmd "$cmd"; then
    echo -n "$cmd: "
    "$cmd" "$@" 2>/dev/null | head -n1 || true
  else
    warn "$cmd not found"
    record_missing "$cmd"
  fi
}

# --- TCP port test helper: use nc if available, else Python fallback ---
check_port() {
  local host="$1"
  local port="$2"
  local name="$3"

  if have_cmd nc; then
    if nc -z "$host" "$port" >/dev/null 2>&1; then
      ok "$name port $port reachable"
    else
      die "$name port $port not reachable"
    fi
  else
    python - "$host" "$port" "$name" <<'PY'
import socket, sys
host = sys.argv[1]
port = int(sys.argv[2])
name = sys.argv[3]
s = socket.socket()
s.settimeout(2)
try:
    s.connect((host, port))
except Exception as e:
    print(f"FAIL:{name}:{port}:{e}")
    sys.exit(1)
finally:
    s.close()
print(f"OK:{name}:{port}")
PY
    ok "$name port $port reachable"
  fi
}

print_missing_help() {
  local os distro pkgm
  os="$(uname -s)"
  distro="unknown"
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    distro="${ID:-unknown}"
  fi

  if have_cmd micromamba; then
    pkgm="micromamba"
  elif have_cmd apt-get; then
    pkgm="apt"
  elif have_cmd brew; then
    pkgm="brew"
  else
    pkgm="none"
  fi

  [ "${#MISSING_TOOLS[@]}" -gt 0 ] || return 0

  echo
  warn "Missing tools detected: ${MISSING_TOOLS[*]}"
  echo "Install guidance ($os/$distro):"

  case "$pkgm" in
    micromamba)
      echo "  micromamba install -y -c conda-forge tesseract poppler imagemagick"
      echo "  # optional, only if fpcalc is needed:"
      echo "  micromamba install -y -c conda-forge libchromaprint"
      ;;
    apt)
      echo "  sudo apt-get update"
      echo "  sudo apt-get install -y tesseract-ocr poppler-utils imagemagick"
      echo "  # optional, only if fpcalc is needed:"
      echo "  sudo apt-get install -y libchromaprint-tools || sudo apt-get install -y chromaprint-tools"
      ;;
    brew)
      echo "  brew install tesseract poppler imagemagick chromaprint"
      ;;
    *)
      echo "  Install manually:"
      echo "  - tesseract (OCR)"
      echo "  - poppler (provides pdftoppm/pdfinfo)"
      echo "  - imagemagick (provides magick/convert)"
      echo "  - chromaprint (provides fpcalc)"
      ;;
  esac
}

install_missing_tools() {
  local pkgm installer=() pkgs=() p

  if have_cmd micromamba; then
    pkgm="micromamba"
    installer=(micromamba)
  elif have_cmd apt-get; then
    pkgm="apt"
    if [ "$(id -u)" -eq 0 ]; then
      installer=(apt-get)
    elif have_cmd sudo; then
      installer=(sudo apt-get)
    else
      warn "apt-get available but no sudo and not root; cannot auto-install"
      return 0
    fi
  elif have_cmd brew; then
    pkgm="brew"
    installer=(brew)
  else
    warn "No supported package manager found for --install-missing (supports micromamba/apt-get/brew)"
    return 0
  fi

  for t in "${MISSING_TOOLS[@]}"; do
    case "$t" in
      tesseract) [ "$pkgm" = "apt" ] && pkgs+=("tesseract-ocr") || pkgs+=("tesseract") ;;
      pdftoppm|pdfinfo) [ "$pkgm" = "apt" ] && pkgs+=("poppler-utils") || pkgs+=("poppler") ;;
      convert|magick) pkgs+=("imagemagick") ;;
      fpcalc)
        case "$pkgm" in
          apt) pkgs+=("libchromaprint-tools") ;;
          micromamba) pkgs+=("libchromaprint") ;;
          *) pkgs+=("chromaprint") ;;
        esac
        ;;
      ffmpeg|ffprobe) pkgs+=("ffmpeg") ;;
      *) ;;
    esac
  done

  if [ "${#pkgs[@]}" -gt 0 ]; then
    mapfile -t pkgs < <(printf '%s\n' "${pkgs[@]}" | awk '!seen[$0]++')
  fi

  if [ "${#pkgs[@]}" -eq 0 ]; then
    ok "No auto-installable packages inferred from missing tools"
    return 0
  fi

  if [ "$ASSUME_YES" -ne 1 ]; then
    echo
    echo "Will install packages: ${pkgs[*]}"
    read -r -p "Proceed? [y/N] " ans
    case "${ans:-n}" in
      y|Y|yes|YES) ;;
      *) warn "Skipping installation"; return 0 ;;
    esac
  fi

  case "$pkgm" in
    micromamba)
      "${installer[@]}" install -y -c conda-forge "${pkgs[@]}"
      ;;
    apt)
      "${installer[@]}" update
      if ! "${installer[@]}" install -y "${pkgs[@]}"; then
        # fpcalc package name differs on some distros.
        if printf '%s\n' "${pkgs[@]}" | grep -qx 'libchromaprint-tools'; then
          warn "Retrying fpcalc package with chromaprint-tools"
          "${installer[@]}" install -y chromaprint-tools
        else
          return 1
        fi
      fi
      ;;
    brew)
      "${installer[@]}" install "${pkgs[@]}"
      ;;
  esac
}

# --- Quick versions of the usual suspects (don’t fail if missing, just warn) ---
if have_cmd brew; then
  vers brew --version
else
  warn "brew not found (ok on Linux if using apt/micromamba)"
fi
vers ffmpeg -version
vers ffprobe -version
vers tesseract --version
vers pdftoppm -v
vers pdfinfo -v
vers fpcalc -version
if have_cmd magick; then
  echo -n "magick: "
  magick -version 2>/dev/null | head -n1 || true
elif have_cmd convert; then
  vers convert -version
else
  warn "ImageMagick binary not found (checked: magick, convert)"
  record_missing "magick"
fi
vers rabbitmqctl status
vers redis-cli --version
ok "Printed tool versions"

# --- Network ports ---
echo "Checking ports…"
check_port localhost 5672 "RabbitMQ"
check_port localhost 6379 "Redis"

# --- AMQP & Redis URLs from env (with safe defaults) ---
BROKER_URL="${CELERY_BROKER_URL:-amqp://graphai:graphai@127.0.0.1:5672/%2F}"
RESULT_URL="${CELERY_RESULT_BACKEND:-redis://127.0.0.1:6379/1}"
echo "Broker: $BROKER_URL"
echo "Backend: $RESULT_URL"

# Export explicitly so child Python sees the resolved defaults too
export CELERY_BROKER_URL="$BROKER_URL"
export CELERY_RESULT_BACKEND="$RESULT_URL"

# --- Python module import smoke test ---
python - <<'PY'
import sys, importlib, json
mods = [
  "kombu", "celery", "redis", "pydantic", "fastapi",
  "numpy", "pandas", "scipy", "sklearn",
  "pdf2image", "PIL", "fitz",
  "pytesseract"
]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        failed.append((m, str(e)))

print("Imported", len(mods) - len(failed), "modules OK")
if failed:
    print("Missing/failed:", json.dumps(failed, indent=2))
    real_fail = [m for m, _ in failed if m not in ("pytesseract",)]
    if real_fail:
        sys.exit(2)
PY
ok "Core Python imports look fine"

# --- Check hashlib/blake2 + OpenSSL provider visibility ---
python - <<'PY'
import hashlib, ssl
print("OpenSSL:", ssl.OPENSSL_VERSION)
print("blake2b?", "blake2b" in hashlib.algorithms_available,
      "blake2s?", "blake2s" in hashlib.algorithms_available)
PY
ok "Crypto/hashlib sanity ok"

# --- AMQP real connection test ---
python - <<'PY'
import os
from kombu import Connection
url = os.environ["CELERY_BROKER_URL"]
print("Testing AMQP:", url)
with Connection(url) as conn:
    conn.connect()
    print("AMQP OK, authenticated")
PY
ok "AMQP connect/auth OK"

# --- Redis real PING test ---
python - <<'PY'
import os
import redis
url = os.environ["CELERY_RESULT_BACKEND"]
print("Testing Redis:", url)
r = redis.Redis.from_url(url)
resp = r.ping()
print("Redis PING:", resp)
PY
ok "Redis PING OK"

# --- Poppler + pdf2image roundtrip ---
if python - <<'PY'
import shutil, tempfile
from pathlib import Path

if shutil.which("pdftoppm") is None:
    raise SystemExit("pdftoppm not found")

import fitz
from pdf2image import convert_from_path

tmpdir = Path(tempfile.mkdtemp(prefix="graphai-smoke-"))
try:
    pdf = tmpdir / "hello.pdf"
    img = tmpdir / "page-1.png"

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello pdf2image", fontsize=24)
    doc.save(str(pdf))
    doc.close()

    pages = convert_from_path(str(pdf), fmt="png", dpi=72)
    pages[0].save(img)
    print("Rendered:", img, "exists?", img.exists())
    assert img.exists()
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
PY
then
  ok "pdf2image + poppler functional"
else
  warn "pdf2image/poppler test failed (poppler missing or not functional)"
fi

# --- ffmpeg / ffprobe trivial tests ---
echo "Creating 1s silent wav via ffmpeg…"
if have_cmd ffmpeg; then
  TMPWAV="$(mktemp /tmp/silence.XXXXXX.wav)"
  trap 'rm -f "${TMPWAV:-}"' EXIT

  if ffmpeg -hide_banner -loglevel error -y -f lavfi -i anullsrc=r=16000:cl=mono -t 1 "$TMPWAV"; then
    [ -s "$TMPWAV" ] && ok "ffmpeg rendered $TMPWAV" || warn "ffmpeg ran but output file is empty"
  else
    warn "ffmpeg render test failed"
  fi

  if have_cmd ffprobe; then
    if ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$TMPWAV" >/dev/null 2>&1; then
      ok "ffprobe can read generated audio"
    else
      warn "ffprobe could not read generated audio"
    fi
  else
    warn "ffprobe not found"
  fi

  rm -f "$TMPWAV"
  trap - EXIT
else
  warn "ffmpeg not found; skipping ffmpeg/ffprobe test"
fi

# --- tesseract presence test (binary) ---
if have_cmd tesseract; then
  if tesseract --list-langs >/dev/null 2>&1; then
    ok "tesseract present"
  else
    warn "tesseract present but language data may be missing"
  fi
else
  warn "tesseract not found (OCR features may be limited)"
fi

# --- Celery app sanity and worker ping ---
python - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())

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

print_missing_help
if [ "$AUTO_INSTALL" -eq 1 ]; then
  install_missing_tools
fi

echo
ok "All smoke tests finished."
