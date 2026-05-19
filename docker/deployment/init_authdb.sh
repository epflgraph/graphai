#!/usr/bin/env bash
set -euo pipefail

CONFIG_INI="${CONFIG_INI:-config.ini}"
MYSQL_CMD="${MYSQL_CMD:-docker run --rm -i mysql:8 mysql}"

[ -f "$CONFIG_INI" ] || { echo "config.ini not found: $CONFIG_INI"; exit 1; }

eval "$(
  CONFIG_INI_PATH="$CONFIG_INI" python - <<'PY'
import os, configparser, shlex
cfg = configparser.ConfigParser()
cfg.read(os.environ["CONFIG_INI_PATH"])

def get(sec, key, default=""):
    return cfg[sec].get(key, default) if cfg.has_section(sec) else default

for k, v in {
    "DB_HOST": get("database", "host", "127.0.0.1"),
    "DB_PORT": get("database", "port", "3306"),
    "DB_USER": get("database", "user", "root"),
    "DB_PASS": get("database", "password", ""),
    "AUTH_SCHEMA": get("auth", "schema", "graphai_auth"),
}.items():
    print(f"{k}={shlex.quote(v)}")
PY
)"

MYSQL_ARGS=(-h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" --protocol=tcp)
[ -n "$DB_PASS" ] && MYSQL_ARGS+=(-p"$DB_PASS")

SQL=$(cat <<SQL
CREATE DATABASE IF NOT EXISTS \`$AUTH_SCHEMA\`;

USE \`$AUTH_SCHEMA\`;

CREATE TABLE IF NOT EXISTS \`Retrieve_Index_Aliases\` (
  \`index_name\` varchar(255) NOT NULL,
  \`alias_name\` varchar(255) NOT NULL,
  PRIMARY KEY (\`index_name\`,\`alias_name\`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS \`User_Rate_Limits\` (
  \`username\` varchar(255) NOT NULL,
  \`api_path\` varchar(255) NOT NULL,
  \`max_requests\` int(11) DEFAULT NULL,
  \`window_size\` int(11) DEFAULT NULL,
  PRIMARY KEY (\`username\`,\`api_path\`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS \`User_Retrieve_Access\` (
  \`username\` varchar(255) NOT NULL,
  \`index_name\` varchar(255) NOT NULL,
  PRIMARY KEY (\`username\`,\`index_name\`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS \`Users\` (
  \`username\` varchar(255) NOT NULL,
  \`full_name\` varchar(255) NOT NULL,
  \`email\` varchar(255) NOT NULL,
  \`hashed_password\` varchar(255) NOT NULL,
  \`disabled\` tinyint(1) NOT NULL DEFAULT 0,
  \`scopes\` varchar(255) DEFAULT 'user',
  PRIMARY KEY (\`username\`),
  KEY \`full_name\` (\`full_name\`),
  KEY \`email\` (\`email\`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
SQL
)

echo "$SQL" | $MYSQL_CMD "${MYSQL_ARGS[@]}"
echo "✔ Auth schema initialized: $AUTH_SCHEMA"
