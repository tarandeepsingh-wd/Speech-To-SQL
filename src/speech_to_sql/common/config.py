from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"

_CONFIG_PATH = CONFIG_DIR / "recipe_db.json"
_SQL_SCHEMA_PATH = CONFIG_DIR / "sql_schema.json"
_ENV_LOADED = False


def load_project_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


@lru_cache(maxsize=4)
def load_recipe_db_config(config_path: str | None = None) -> dict[str, Any]:
    load_project_env()
    path = Path(
        config_path
        or os.getenv("RDB2_CONFIG_PATH", "")
        or _CONFIG_PATH
    )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=4)
def load_sql_schema_config(config_path: str | None = None) -> dict[str, Any]:
    load_project_env()
    path = Path(
        config_path
        or os.getenv("SPEECH_TO_SQL_SCHEMA_PATH", "")
        or _SQL_SCHEMA_PATH
    )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def endpoint_config(name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_recipe_db_config()
    return config["endpoints"][name]


def endpoint_path(name: str, config: dict[str, Any] | None = None, **template_values: Any) -> str:
    endpoint = endpoint_config(name, config)
    template = endpoint.get("path_template") or endpoint["path"]
    return template.format(**template_values)


def endpoint_params(
    name: str,
    values: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mapping = endpoint_config(name, config).get("params", {})
    params = {}
    for logical_name, value in values.items():
        api_name = mapping.get(logical_name)
        if not api_name or value in (None, "", []):
            continue
        params[api_name] = value
    return params


def endpoint_method(name: str, config: dict[str, Any] | None = None) -> str:
    return endpoint_config(name, config).get("method", "GET")


def endpoint_priority(name: str, config: dict[str, Any] | None = None) -> int:
    return int(endpoint_config(name, config).get("priority", 100))


def endpoint_reason(name: str, config: dict[str, Any] | None = None) -> str:
    return str(endpoint_config(name, config).get("reason", ""))
