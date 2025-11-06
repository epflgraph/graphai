#!/usr/bin/env bash
set -euo pipefail

RED=$'\e[31m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; NC=$'\e[0m'
ok(){ echo "${GREEN}✔${NC} $*"; }
warn(){ echo "${YELLOW}!${NC} $*"; }
die(){ echo "${RED}✖${NC} $*"; exit 1; }

# Repo root + config.ini
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then :; else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
CONFIG_INI="$REPO_ROOT/config.ini"
[ -f "$CONFIG_INI" ] || die "config.ini not found at: $CONFIG_INI"
ok "Using config: $CONFIG_INI"

# Use the venv python if active, else system/python on PATH
VENV_PY="${VENV_PY:-python}"
command -v "$VENV_PY" >/dev/null || die "python not on PATH"

# Parse config.ini safely via Python; pass path via env (no argv)
read_cfg_sh_kv="$(
  CONFIG_INI_PATH="$CONFIG_INI" "$VENV_PY" - <<'PY'
import os, configparser, shlex
p = os.environ["CONFIG_INI_PATH"]
cfg = configparser.ConfigParser()
cfg.optionxform = str
cfg.read(p)

def get(sec, key, default=""):
    try:
        return cfg[sec][key]
    except KeyError:
        return default

out = {
    "DB_HOST": get("database", "host", "127.0.0.1"),
    "DB_PORT": get("database", "port", "3306"),
    "DB_USER": get("database", "user", "root"),
    "DB_PASS": get("database", "password", ""),
    "AUTH_SCHEMA": get("auth", "schema", "auth_graphai"),
    "CACHE_SCHEMA": get("cache", "schema", "graphai_cache"),
}
for k, v in out.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)"
# Load the shell-safe assignments
# shellcheck disable=SC2086
eval "$read_cfg_sh_kv"

ok "DB host: $DB_HOST  port: $DB_PORT  user: $DB_USER"
ok "AUTH schema: $AUTH_SCHEMA   CACHE schema: $CACHE_SCHEMA"
[ -n "${DB_PASS:-}" ] && warn "DB password: (set, hidden)" || warn "DB password: (empty)"

# Build mysql args & test
MYSQL_CMD="${MYSQL_CMD:-mysql}"
command -v "$MYSQL_CMD" >/dev/null 2>&1 || die "mysql client not found"
MYSQL_ARGS=( -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" --protocol=tcp --connect-timeout=5 )
[ -n "${DB_PASS:-}" ] && MYSQL_ARGS+=( -p"$DB_PASS" )
if "$MYSQL_CMD" "${MYSQL_ARGS[@]}" -e "SELECT 1;" >/dev/null 2>&1; then
  ok "MySQL connection OK"
else
  die "MySQL connection failed (host=$DB_HOST port=$DB_PORT user=$DB_USER)"
fi










# ---------- collect first user info ----------
echo
echo "== First user setup =="

# helper for reading with default
read_default () {
  local prompt="$1"; local def="${2:-}"; local ans=""
  if [ -n "$def" ]; then
    read -r -p "$prompt [$def]: " ans || true
    printf '%s\n' "${ans:-$def}"
  else
    read -r -p "$prompt: " ans || true
    printf '%s\n' "$ans"
  fi
}

FIRST_USER="$(read_default "Username" "admin")"
FIRST_EMAIL="$(read_default "Email" "admin@example.com")"
FIRST_FULLNAME="$(read_default "Full name" "$FIRST_USER")"

# password input — hidden and validated
FIRST_PASSWORD=""
while [ -z "${FIRST_PASSWORD:-}" ]; do
  read -r -s -p "Password: " FIRST_PASSWORD || true; echo
  if [ -z "${FIRST_PASSWORD:-}" ]; then
    echo "⚠️  Password cannot be empty."
  elif [ "${#FIRST_PASSWORD}" -lt 6 ]; then
    echo "⚠️  Password too short (min 6 characters)."
    FIRST_PASSWORD=""
  fi
done

FIRST_SCOPES_DEFAULT="global,user,voice,video,translation,text,scraping,ontology,image,completion,embedding,rag"
FIRST_SCOPES="$(read_default "Scopes (comma-separated)" "$FIRST_SCOPES_DEFAULT")"

echo
echo "✅ User info collected:"
echo "   Username : $FIRST_USER"
echo "   Email    : $FIRST_EMAIL"
echo "   Scopes   : $FIRST_SCOPES"
echo "   (password hidden)"









# ---------- hash password for SQL insert (sets $HASHED_PASSWORD) ----------
: "${FIRST_PASSWORD:?FIRST_PASSWORD not set}"
VENV_PY="${VENV_PY:-python}"
command -v "$VENV_PY" >/dev/null || { echo "python not on PATH"; exit 1; }

HASHED_PASSWORD=""

# Try project hasher first (mute passlib logging just in case it’s used inside)
if "$VENV_PY" -c "import logging; logging.getLogger('passlib').setLevel(50); from graphai.api.auth.auth_utils import get_password_hash" 2>/dev/null; then
  HASHED_PASSWORD="$(
    FIRST_PASSWORD="$FIRST_PASSWORD" "$VENV_PY" - <<'PY'
import os, logging
logging.getLogger("passlib").setLevel(50)  # CRITICAL
from graphai.api.auth.auth_utils import get_password_hash
print(get_password_hash(os.environ["FIRST_PASSWORD"]))
PY
  )"
  echo "✔ Password hashed via graphai.api.auth.auth_utils"
else
  echo "! Project hasher unavailable; using passlib (bcrypt_sha256)…"
  "$VENV_PY" -m pip install -q 'passlib>=1.7.4,<2' 'bcrypt>=4,<5' >/dev/null 2>&1 || true
  HASHED_PASSWORD="$(
    FIRST_PASSWORD="$FIRST_PASSWORD" "$VENV_PY" - <<'PY'
import os, logging
logging.getLogger("passlib").setLevel(50)  # CRITICAL
try:
    from passlib.hash import bcrypt_sha256 as algo  # avoids 72-byte limit
except Exception:
    from passlib.hash import bcrypt as algo
print(algo.hash(os.environ["FIRST_PASSWORD"]))
PY
  )"
  [ -n "$HASHED_PASSWORD" ] || { echo "✖ Failed to hash password"; exit 1; }
  echo "✔ Password hashed via passlib"
fi

# trim any trailing newline
HASHED_PASSWORD="${HASHED_PASSWORD%$'\n'}"






# # ---------- prepare SQL insert for first user (print only) ----------
# : "${AUTH_SCHEMA:?AUTH_SCHEMA not set}"
# : "${FIRST_USER:?FIRST_USER not set}"
# : "${FIRST_EMAIL:?FIRST_EMAIL not set}"
# : "${HASHED_PASSWORD:?HASHED_PASSWORD not set}"
# : "${FIRST_SCOPES:?FIRST_SCOPES not set}"

# # escape double quotes in scopes
# ESC_SCOPES="$(printf '%s' "$FIRST_SCOPES" | sed 's/"/\\"/g')"

# cat <<SQL
# -- SQL to create or update first user
# INSERT INTO \`$AUTH_SCHEMA\`.\`Users\`
# (\`username\`, \`full_name\`, \`email\`, \`hashed_password\`, \`disabled\`, \`scopes\`)
# VALUES
# ('$FIRST_USER', '$FIRST_FULLNAME', '$FIRST_EMAIL', '$HASHED_PASSWORD', 0, '$ESC_SCOPES')
# ON DUPLICATE KEY UPDATE
#   full_name=VALUES(full_name),
#   email=VALUES(email),
#   hashed_password=VALUES(hashed_password),
#   scopes=VALUES(scopes);
# SQL







# ---------- execute SQL insert ----------
: "${AUTH_SCHEMA:?AUTH_SCHEMA not set}"
: "${MYSQL_CMD:=mysql}"
: "${DB_HOST:?DB_HOST not set}"
: "${DB_PORT:?DB_PORT not set}"
: "${DB_USER:?DB_USER not set}"

# escape double quotes in scopes
ESC_SCOPES="$(printf '%s' "$FIRST_SCOPES" | sed 's/"/\\"/g')"

SQL_QUERY=$(cat <<SQL
INSERT INTO \`$AUTH_SCHEMA\`.\`Users\`
(\`username\`, \`full_name\`, \`email\`, \`hashed_password\`, \`disabled\`, \`scopes\`)
VALUES
('$FIRST_USER', '$FIRST_FULLNAME', '$FIRST_EMAIL', '$HASHED_PASSWORD', 0, '$ESC_SCOPES')
ON DUPLICATE KEY UPDATE
  full_name=VALUES(full_name),
  email=VALUES(email),
  hashed_password=VALUES(hashed_password),
  scopes=VALUES(scopes);
SQL
)

# Confirm before applying
echo
read -r -p "Apply SQL insert for user '$FIRST_USER'? (y/N): " APPLY
if [[ "$APPLY" =~ ^[Yy]$ ]]; then
  MYSQL_ARGS=(-h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER")
  if [ -n "${DB_PASS:-}" ]; then MYSQL_ARGS+=(-p"$DB_PASS"); fi

  echo "✔ Applying SQL to schema '$AUTH_SCHEMA'..."
  echo "$SQL_QUERY" | "$MYSQL_CMD" "${MYSQL_ARGS[@]}" || {
    echo "✖ MySQL command failed — check connection or privileges."
    exit 1
  }
  echo "✔ SQL executed successfully."
else
  echo "⚠ Skipped execution. Here’s the SQL:"
  echo "$SQL_QUERY"
fi
