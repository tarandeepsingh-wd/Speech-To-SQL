from __future__ import annotations

from typing import Any

from speech_to_sql.m3_query_ir.sql_ir import QueryIR
from speech_to_sql.m4_sql.sql_plan_builder import build_sql_plan
from speech_to_sql.m4_sql.sql_renderer import render_sql_plan


def compile_query_ir(query_ir: QueryIR) -> dict[str, Any]:
    plan = build_sql_plan(query_ir)
    compiled = render_sql_plan(plan)
    compiled["ir"] = query_ir.as_dict()
    return compiled
