#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------
# Remote MySQL. Default uses local mysql client. Can override MYSQL_CMD with
# a multi-word command, e.g.:
#   MYSQL_CMD='docker run --rm -i mysql:8 mysql' ./init/api_add_user.sh
#
# If using Docker and networking is restrictive, try:
#   DOCKER_NET_HOST=1 MYSQL_CMD='docker run --rm -i mysql:8 mysql' ./init/api_add_user.sh
# ---------------------------------------------------------------------
MYSQL_CMD="${MYSQL_CMD:-docker run --rm -i mysql:8 mysql}"
DOCKER_NET_HOST="${DOCKER_NET_HOST:-0}"

RED=$'\e[31m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; NC=$'\e[0m'
ok(){ echo "${GREEN}✔${NC} $*"; }
warn(){ echo "${YELLOW}!${NC} $*"; }
die(){ echo "${RED}✖${NC} $*"; exit 1; }

# Repo root + config.ini
CONFIG_INI="config.ini"
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
# shellcheck disable=SC2086
eval "$read_cfg_sh_kv"

ok "DB host: $DB_HOST  port: $DB_PORT  user: $DB_USER"
ok "AUTH schema: $AUTH_SCHEMA   CACHE schema: $CACHE_SCHEMA"
[ -n "${DB_PASS:-}" ] && warn "DB password: (set, hidden)" || warn "DB password: (empty)"

# --- Build mysql command array (supports multi-word MYSQL_CMD) ---
if [[ "$MYSQL_CMD" == *" "* ]]; then
  # shellcheck disable=SC2206
  MYSQL_CMD_ARR=($MYSQL_CMD)
  ok "Using complex MYSQL_CMD='${MYSQL_CMD_ARR[*]}' (skipping PATH check)"
else
  command -v "$MYSQL_CMD" >/dev/null 2>&1 || die "mysql client not found on PATH (MYSQL_CMD=$MYSQL_CMD)"
  MYSQL_CMD_ARR=("$MYSQL_CMD")
fi

# If using docker run and DOCKER_NET_HOST=1, inject --network host
# This is Linux-only; harmless to leave off otherwise.
if [[ "$DOCKER_NET_HOST" == "1" ]] && [[ "${MYSQL_CMD_ARR[*]}" == docker\ run* ]]; then
  # Insert right after "docker run"
  # Example: docker run --rm -i mysql:8 mysql  -> docker run --network host --rm -i mysql:8 mysql
  MYSQL_CMD_ARR=( "${MYSQL_CMD_ARR[0]}" "${MYSQL_CMD_ARR[1]}" --network host "${MYSQL_CMD_ARR[@]:2}" )
  warn "DOCKER_NET_HOST=1: using docker --network host"
fi

# If password is empty, ask once (hidden). You can still leave it empty if server allows.
if [ -z "${DB_PASS:-}" ]; then
  read -r -s -p "MySQL password for ${DB_USER}@${DB_HOST} (leave empty if none): " DB_PASS || true
  echo
fi

# --- Build mysql args & test connectivity ---
MYSQL_ARGS=( -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" --protocol=tcp --connect-timeout=5 )
[[ -n "${DB_PASS:-}" ]] && MYSQL_ARGS+=( -p"$DB_PASS" )

if "${MYSQL_CMD_ARR[@]}" "${MYSQL_ARGS[@]}" -e "SELECT 1;" >/dev/null 2>&1; then
  ok "MySQL connection OK"
else
  echo
  echo "✖ MySQL connection failed (host=$DB_HOST port=$DB_PORT user=$DB_USER)"
  echo "   Tried command: ${MYSQL_CMD_ARR[*]} ${MYSQL_ARGS[*]}"
  echo
  echo "   Quick debug you can run:"
  echo "     ${MYSQL_CMD_ARR[*]} ${MYSQL_ARGS[*]} -e 'SELECT VERSION();'"
  exit 1
fi

# ---------- collect first user info ----------
echo
echo "== First user setup =="

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

FIRST_PASSWORD=""
while [ -z "${FIRST_PASSWORD:-}" ]; do
  read -r -p "Password: " FIRST_PASSWORD || true; echo
  if [ -z "${FIRST_PASSWORD:-}" ]; then
    echo "⚠️  Password cannot be empty."
  elif [ "${#FIRST_PASSWORD}" -lt 6 ]; then
    echo "⚠️  Password too short (min 6 characters)."
    FIRST_PASSWORD=""
  fi
done

FIRST_SCOPES_DEFAULT="global,user,voice,video,translation,text,scraping,ontology,image,completion,embedding,rag,upload"
FIRST_SCOPES="$(read_default "Scopes (comma-separated)" "$FIRST_SCOPES_DEFAULT")"

echo
echo "✅ User info collected:"
echo "   Username : $FIRST_USER"
echo "   Email    : $FIRST_EMAIL"
echo "   Scopes   : $FIRST_SCOPES"
echo "   (password hidden)"

# ---------- hash password for SQL insert (sets $HASHED_PASSWORD) ----------
: "${FIRST_PASSWORD:?FIRST_PASSWORD not set}"
command -v "$VENV_PY" >/dev/null || die "python not on PATH"

HASHED_PASSWORD=""

if "$VENV_PY" -c "import logging; logging.getLogger('passlib').setLevel(50); from graphai.api.auth.auth_utils import get_password_hash" 2>/dev/null; then
  HASHED_PASSWORD="$(
    FIRST_PASSWORD="$FIRST_PASSWORD" "$VENV_PY" - <<'PY'
import os, logging
logging.getLogger("passlib").setLevel(50)
from graphai.api.auth.auth_utils import get_password_hash
print(get_password_hash(os.environ["FIRST_PASSWORD"]))
PY
  )"
  ok "Password hashed via graphai.api.auth.auth_utils"
else
  warn "Project hasher unavailable; using passlib bcrypt context to match API verifier"
  "$VENV_PY" -m pip install -q 'passlib>=1.7.4,<2' 'bcrypt<4' >/dev/null 2>&1 || true
  HASHED_PASSWORD="$(
    FIRST_PASSWORD="$FIRST_PASSWORD" "$VENV_PY" - <<'PY'
import os, logging
from passlib.context import CryptContext
logging.getLogger("passlib").setLevel(50)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
print(pwd_context.hash(os.environ["FIRST_PASSWORD"]))
PY
  )"
  [ -n "$HASHED_PASSWORD" ] || die "Failed to hash password"
  ok "Password hashed via passlib bcrypt context"
fi

HASHED_PASSWORD="${HASHED_PASSWORD%$'\n'}"

# ---------- execute SQL insert ----------
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

echo
read -r -p "Apply SQL insert for user '$FIRST_USER'? (y/N): " APPLY
if [[ "$APPLY" =~ ^[Yy]$ ]]; then
  echo "✔ Applying SQL to schema '$AUTH_SCHEMA'..."

  if ! echo "$SQL_QUERY" | "${MYSQL_CMD_ARR[@]}" "${MYSQL_ARGS[@]}"; then
    echo
    echo "✖ MySQL command failed — check connection or privileges."
    echo "⚠️  The attempted SQL was:"
    echo "------------------------------------------------------------"
    echo "$SQL_QUERY"
    echo "------------------------------------------------------------"
    exit 1
  fi

  ok "SQL executed successfully."
else
  warn "Skipped execution. Here’s the SQL:"
  echo "$SQL_QUERY"
fi
