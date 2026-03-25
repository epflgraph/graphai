#!/usr/bin/env python3
import configparser
import os
from pathlib import Path


CONFIG_PATH = Path(os.environ.get("GRAPH_CONFIG_PATH", "/app/config.ini"))

# env -> (section, key)
ENV_MAP = {
    "CELERY_BROKER_URL": ("celery", "broker_url"),
    "CELERY_RESULT_BACKEND": ("celery", "result_backend"),
    "DB_HOST": ("database", "host"),
    "DB_PORT": ("database", "port"),
    "DB_USER": ("database", "user"),
    "DB_PASSWORD": ("database", "password"),
    "DB_SCHEMA": ("cache", "schema"),
    "AUTH_SCHEMA": ("auth", "schema"),
    "AUTH_SECRET_KEY": ("auth", "secret_key"),
    "ES_HOST": ("elasticsearch", "host"),
    "ES_PORT": ("elasticsearch", "port"),
    "ES_USERNAME": ("elasticsearch", "username"),
    "ES_PASSWORD": ("elasticsearch", "password"),
    "ES_CAFILE": ("elasticsearch", "cafile"),
    "ES_CONCEPT_INDEX": ("elasticsearch", "concept_detection_index"),
    "CACHE_ROOT": ("cache", "root"),
    "WHISPER_MODEL_TYPE": ("whisper", "model_type"),
    "WHISPER_MODEL_PATH": ("whisper", "model_path"),
    "HUGGINGFACE_MODEL_PATH": ("huggingface", "model_path"),
    "FASTTEXT_MODEL_PATH": ("fasttext", "path"),
    "FASTTEXT_DIM": ("fasttext", "dim"),
    "PRELOAD_VIDEO": ("preload", "video"),
    "PRELOAD_TEXT": ("preload", "text"),
    "PRELOAD_ONTOLOGY": ("preload", "ontology"),
    "LOG_PATH": ("logging", "path"),
    "LOG_SERVER_NAME": ("logging", "server_name"),
}


def ensure_section(parser: configparser.ConfigParser, section: str) -> None:
    if not parser.has_section(section):
        parser.add_section(section)


def main() -> int:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    parser = configparser.ConfigParser()
    parser.read(CONFIG_PATH)

    modified = False
    for env_name, (section, key) in ENV_MAP.items():
        value = os.environ.get(env_name)
        if value is None or value == "":
            continue
        ensure_section(parser, section)
        parser.set(section, key, value)
        modified = True

    if modified:
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            parser.write(f)
        print(f"[runtime_config] Updated {CONFIG_PATH} from environment")
    else:
        print(f"[runtime_config] No env overrides applied to {CONFIG_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
