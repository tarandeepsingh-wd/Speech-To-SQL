from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from speech_to_sql.common.config import load_project_env


class AthenaConfigError(RuntimeError):
    pass


class AthenaQueryError(RuntimeError):
    pass


def execute_athena_query(
    query: str,
    *,
    max_rows: int = 20,
    timeout_seconds: int = 90,
    poll_interval_seconds: float = 1.5,
) -> dict[str, Any]:
    """Execute a generated Athena query using local .env AWS settings."""
    query = query.strip()
    if not query:
        raise AthenaConfigError("Athena query cannot be empty.")

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as exc:
        raise AthenaConfigError("boto3 is not installed. Run: pip install boto3") from exc

    settings = _athena_settings()
    client_kwargs: dict[str, Any] = {
        "region_name": settings["region"],
        "aws_access_key_id": settings["access_key_id"],
        "aws_secret_access_key": settings["secret_access_key"],
    }
    if settings.get("session_token"):
        client_kwargs["aws_session_token"] = settings["session_token"]

    try:
        client = boto3.client("athena", **client_kwargs)
        start_kwargs: dict[str, Any] = {
            "QueryString": query,
            "QueryExecutionContext": {"Database": settings["database"]},
            "ResultConfiguration": {"OutputLocation": settings["output_location"]},
        }
        if settings.get("workgroup"):
            start_kwargs["WorkGroup"] = settings["workgroup"]

        execution_id = client.start_query_execution(**start_kwargs)["QueryExecutionId"]
        execution = _wait_for_query(
            client,
            execution_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        state = execution["Status"]["State"]
        if state != "SUCCEEDED":
            reason = execution["Status"].get("StateChangeReason") or "Athena query failed."
            raise AthenaQueryError(reason)

        columns, rows = _fetch_rows(client, execution_id, max_rows=max_rows)
    except (BotoCoreError, ClientError) as exc:
        raise AthenaQueryError(str(exc)) from exc

    stats = execution.get("Statistics") or {}
    return {
        "ok": True,
        "state": "SUCCEEDED",
        "query_execution_id": execution_id,
        "database": settings["database"],
        "workgroup": settings.get("workgroup") or "",
        "output_location": settings["output_location"],
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "bytes_scanned": stats.get("DataScannedInBytes"),
        "engine_ms": stats.get("EngineExecutionTimeInMillis"),
    }


def _athena_settings() -> dict[str, str]:
    load_project_env()
    file_values = _project_env_file_values()

    def value(name: str, default: str = "") -> str:
        return str(file_values.get(name, os.getenv(name, default)) or "").strip()

    settings = {
        "access_key_id": value("AWS_ACCESS_KEY_ID"),
        "secret_access_key": value("AWS_SECRET_ACCESS_KEY"),
        "session_token": value("AWS_SESSION_TOKEN"),
        "region": value("AWS_REGION") or value("AWS_DEFAULT_REGION"),
        "database": value("ATHENA_DATABASE"),
        "output_location": value("ATHENA_OUTPUT_LOCATION"),
        "workgroup": value("ATHENA_WORKGROUP", "primary"),
    }
    missing = [
        name
        for name, value in (
            ("AWS_ACCESS_KEY_ID", settings["access_key_id"]),
            ("AWS_SECRET_ACCESS_KEY", settings["secret_access_key"]),
            ("AWS_REGION", settings["region"]),
            ("ATHENA_DATABASE", settings["database"]),
            ("ATHENA_OUTPUT_LOCATION", settings["output_location"]),
        )
        if not value
    ]
    if missing:
        raise AthenaConfigError(
            "Missing required Athena .env value(s): " + ", ".join(missing)
        )
    return settings


def _project_env_file_values() -> dict[str, str]:
    env_path = Path(__file__).resolve().parents[3] / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        if not key:
            continue
        values[key] = raw_value.strip().strip('"').strip("'")
    return values


def _wait_for_query(
    client: Any,
    execution_id: str,
    *,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        execution = client.get_query_execution(QueryExecutionId=execution_id)["QueryExecution"]
        state = execution["Status"]["State"]
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return execution
        if time.monotonic() >= deadline:
            raise AthenaQueryError(
                f"Athena query timed out after {timeout_seconds} seconds: {execution_id}"
            )
        time.sleep(poll_interval_seconds)


def _fetch_rows(
    client: Any,
    execution_id: str,
    *,
    max_rows: int,
) -> tuple[list[str], list[list[str]]]:
    result = client.get_query_results(
        QueryExecutionId=execution_id,
        MaxResults=max(1, min(max_rows + 1, 1000)),
    )
    result_set = result.get("ResultSet") or {}
    metadata = result_set.get("ResultSetMetadata") or {}
    columns = [
        str(column.get("Name") or "")
        for column in metadata.get("ColumnInfo", [])
    ]
    rows: list[list[str]] = []
    for raw_row in result_set.get("Rows", []):
        values = [
            cell.get("VarCharValue", "")
            for cell in raw_row.get("Data", [])
        ]
        if columns and values == columns:
            continue
        rows.append(values)
        if len(rows) >= max_rows:
            break
    return columns, rows
