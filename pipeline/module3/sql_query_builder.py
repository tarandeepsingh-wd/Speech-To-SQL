from __future__ import annotations

from module3.config import load_sql_schema_config
from module3.intent_grounder import IntentGroundingError
from module3.sql_compiler import compile_query_ir
from module3.sql_intent_builder import build_query_ir
from module3.sql_ir import QueryIR
from module3.sql_validator import validate_query_ir


def build_sql_query(
    tag_result: dict,
    *,
    transcript: str | None = None,
    table_name: str | None = None,
    validate: bool = True,
    require_virtual_evidence: bool = False,
    query_ir: QueryIR | None = None,
) -> dict:
    """
    Build SQL by compiling a schema-aware QueryIR.

    `table_name` is kept for backward compatibility. The preferred way to
    change schema names is editing module3/config/sql_schema.json.
    """
    try:
        query_ir = query_ir or build_query_ir(tag_result, transcript=transcript)
    except IntentGroundingError as error:
        return {
            "query": None,
            "params": [],
            "ir": None,
            "grounding": error.grounding,
            "validation": {
                "ok": False,
                "errors": error.errors,
                "warnings": [],
            },
        }
    validation = validate_query_ir(
        query_ir,
        require_virtual_evidence=require_virtual_evidence,
    ) if validate else None
    if validation and not validation.ok:
        return {
            "query": None,
            "params": [],
            "ir": query_ir.as_dict(),
            "validation": validation.as_dict(),
        }

    compiled = compile_query_ir(query_ir)
    if validation:
        compiled["validation"] = validation.as_dict()
    if table_name:
        schema = load_sql_schema_config()
        default_table = schema["base_table"]["name"]
        compiled["query"] = compiled["query"].replace(
            f" FROM {default_table} AS ",
            f" FROM {table_name} AS ",
            1,
        )
    return compiled
