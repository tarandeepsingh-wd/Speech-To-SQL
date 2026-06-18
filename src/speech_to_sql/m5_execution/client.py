from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

from speech_to_sql.common.config import load_recipe_db_config
from speech_to_sql.m5_execution.query_planner import ApiQueryPlan


_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


@dataclass
class ApiResult:
    plan: ApiQueryPlan
    ok: bool
    status_code: int | None
    data: dict[str, Any] | None
    recipes: list[dict[str, Any]]
    error: str | None = None
    from_cache: bool = False


class RecipeDBClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        config = load_recipe_db_config()
        request_config = config["request"]
        self.base_url = (
            base_url
            or os.getenv(config["base_url_env"])
            or config["default_base_url"]
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv(config["api_key_env"])
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else int(request_config["timeout_seconds"])
        )
        self.ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else int(request_config["ttl_seconds"])
        )
        self.min_interval_seconds = float(
            os.getenv(
                "RDB2_MIN_REQUEST_INTERVAL_SECONDS",
                request_config.get("min_interval_seconds", 0),
            )
        )
        self._cache: dict[tuple[str, str, tuple[tuple[str, str], ...]], tuple[float, dict[str, Any]]] = {}

    def execute(self, plan: ApiQueryPlan) -> ApiResult:
        cache_key = (
            plan.method.upper(),
            plan.url(self.base_url),
            tuple(sorted((str(k), str(v)) for k, v in plan.params.items())),
        )
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[0] <= self.ttl_seconds:
            data = cached[1]
            return ApiResult(
                plan=plan,
                ok=True,
                status_code=200,
                data=data,
                recipes=extract_recipes(data),
                from_cache=True,
            )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            self._throttle()
            if plan.method.upper() == "GET":
                response = requests.get(
                    plan.url(self.base_url),
                    params=plan.params,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            elif plan.method.upper() == "POST":
                response = requests.post(
                    plan.url(self.base_url),
                    params=plan.params,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            else:
                return ApiResult(
                    plan=plan,
                    ok=False,
                    status_code=None,
                    data=None,
                    recipes=[],
                    error=f"Unsupported method: {plan.method}",
                )

            data = response.json() if response.content else {}
            if response.ok:
                self._cache[cache_key] = (time.time(), data)
            return ApiResult(
                plan=plan,
                ok=response.ok,
                status_code=response.status_code,
                data=data,
                recipes=extract_recipes(data),
                error=None if response.ok else _response_error(data),
            )
        except requests.RequestException as exc:
            return ApiResult(
                plan=plan,
                ok=False,
                status_code=None,
                data=None,
                recipes=[],
                error=str(exc),
            )
        except ValueError as exc:
            return ApiResult(
                plan=plan,
                ok=False,
                status_code=None,
                data=None,
                recipes=[],
                error=f"Invalid JSON response: {exc}",
            )

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        global _LAST_REQUEST_AT
        with _REQUEST_LOCK:
            now = time.time()
            wait_seconds = self.min_interval_seconds - (now - _LAST_REQUEST_AT)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            _LAST_REQUEST_AT = time.time()


def extract_recipes(data: Any) -> list[dict[str, Any]]:
    config = load_recipe_db_config()
    if not isinstance(data, dict):
        return []
    for path in config["response"]["recipe_list_paths"]:
        candidate = _get_path(data, path)
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
    return []


def _response_error(data: Any) -> str | None:
    config = load_recipe_db_config()
    if not isinstance(data, dict):
        return None
    for key in config["response"]["error_fields"]:
        value = data.get(key)
        if value:
            return str(value)
    return None


def _get_path(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value
