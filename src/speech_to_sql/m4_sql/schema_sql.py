from __future__ import annotations

from pathlib import Path


_SCHEMA_SQL_PATH = Path(__file__).resolve().parents[3] / "config" / "schema.sql"


def load_schema_sql(path: str | Path | None = None) -> str:
    schema_path = Path(path) if path else _SCHEMA_SQL_PATH
    return schema_path.read_text(encoding="utf-8")
