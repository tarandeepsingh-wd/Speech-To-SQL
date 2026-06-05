from __future__ import annotations

from module3.sql_plan import SqlPlan


def render_sql_plan(plan: SqlPlan) -> dict:
    distinct = "DISTINCT " if plan.distinct else ""
    sql = f"SELECT {distinct}{', '.join(plan.select)} FROM {plan.from_table} AS {plan.from_alias}"

    if plan.joins:
        sql += " " + " ".join(join.as_sql() for join in plan.joins)

    if plan.where:
        sql += " WHERE " + " AND ".join(
            f"({condition.sql})"
            for condition in plan.where
        )

    if plan.order_by:
        sql += " ORDER BY " + ", ".join(
            order.as_sql()
            for order in plan.order_by
        )

    if plan.limit:
        sql += " LIMIT ?"

    return {
        "query": sql,
        "params": plan.params(),
        "plan": plan.as_dict(),
        "join_assumptions": plan.join_assumptions,
    }
