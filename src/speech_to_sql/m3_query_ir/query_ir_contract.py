from __future__ import annotations

import re
from typing import Any

from speech_to_sql.common.config import load_sql_schema_config
from speech_to_sql.m3_query_ir.sql_ir import OrderBy, Predicate, QueryIR, SelectExpr
from speech_to_sql.m4_sql.sql_validator import validate_query_ir


class QueryIRContractError(ValueError):
    pass


_ALIAS_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_TOP_LEVEL_KEYS = {"root", "select", "predicates", "order_by", "limit"}
_ALLOWED_SELECT_KEYS = {"expression", "alias"}
_ALLOWED_PREDICATE_KEYS = {"source", "field", "operator", "values", "quantifier"}
_ALLOWED_ORDER_KEYS = {"field", "direction"}
_FORBIDDEN_SQL_KEYS = {
    "sql",
    "query",
    "raw_sql",
    "sql_query",
    "from",
    "join",
    "joins",
    "where",
    "where_clause",
}
_SCALAR_VALUE_TYPES = (str, int, float, bool)


def query_ir_from_payload(
    payload: dict[str, Any],
    *,
    validate: bool = True,
) -> QueryIR:
    """
    Convert a constrained QueryIR JSON payload into QueryIR.

    This is the guard boundary for any future LLM/RAG layer. Raw SQL is not an
    accepted field. The payload must describe intent only, then the schema
    validator and SQL planner decide whether it can become SQL.
    """
    if not isinstance(payload, dict):
        raise QueryIRContractError("QueryIR payload must be an object")

    forbidden = _find_forbidden_sql_keys(payload)
    if forbidden:
        raise QueryIRContractError(
            "Raw SQL is not accepted in QueryIR payload: "
            + ", ".join(sorted(forbidden))
        )

    unknown = set(payload) - _ALLOWED_TOP_LEVEL_KEYS
    if unknown:
        raise QueryIRContractError(f"Unknown QueryIR payload keys: {sorted(unknown)}")

    schema = load_sql_schema_config()
    root = _parse_root(payload.get("root"), schema)
    query_ir = QueryIR(
        root=root,
        select=_parse_select(payload.get("select"), schema, root),
        predicates=_parse_predicates(payload.get("predicates")),
        order_by=_parse_order_by(payload.get("order_by")),
        limit=_parse_limit(payload.get("limit")),
    )

    if validate:
        result = validate_query_ir(query_ir)
        if not result.ok:
            raise QueryIRContractError("; ".join(result.errors))
    return query_ir


def query_ir_from_contract(
    payload: dict[str, Any],
    *,
    validate: bool = True,
) -> QueryIR:
    return query_ir_from_payload(payload, validate=validate)


def query_ir_to_payload(query_ir: QueryIR) -> dict[str, Any]:
    return query_ir.as_dict()


def _parse_select(value: Any, schema: dict[str, Any], root: str) -> list[SelectExpr]:
    root_select = schema.get("roots", {}).get(root, {}).get("select")
    items = value if value is not None else root_select or schema.get("default_select", [])
    if not isinstance(items, list):
        raise QueryIRContractError("select must be a list")
    configured_select_aliases = _configured_select_aliases(schema, root)
    aggregate_aliases = _aggregate_aliases(schema)
    allowed_expressions = set(configured_select_aliases) | set(aggregate_aliases)
    result: list[SelectExpr] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise QueryIRContractError(f"select[{index}] must be an object")
        _reject_unknown_keys(item, _ALLOWED_SELECT_KEYS, f"select[{index}]")
        expression = item.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise QueryIRContractError(f"select[{index}].expression must be a non-empty string")
        expression = expression.strip()
        if expression not in allowed_expressions:
            raise QueryIRContractError(
                f"select[{index}].expression is not allowed by sql_schema"
            )
        alias = item.get("alias")
        expected_alias = aggregate_aliases.get(
            expression,
            configured_select_aliases.get(expression),
        )
        if alias is None and expected_alias:
            alias = expected_alias
        if alias is not None and (
            not isinstance(alias, str) or not _ALIAS_PATTERN.fullmatch(alias)
        ):
            raise QueryIRContractError(f"select[{index}].alias must be a string")
        if alias != expected_alias:
            raise QueryIRContractError(f"select[{index}].alias is not allowed by sql_schema")
        result.append(SelectExpr(expression, alias))
    return result


def _parse_root(value: Any, schema: dict[str, Any]) -> str:
    root = value if value is not None else schema.get("default_root", "recipe")
    if not isinstance(root, str) or not root.strip():
        raise QueryIRContractError("root must be a non-empty string")
    root = root.strip()
    if root not in schema.get("roots", {}):
        raise QueryIRContractError(f"root {root!r} is not allowed by sql_schema")
    return root


def _parse_predicates(value: Any) -> list[Predicate]:
    items = value if value is not None else []
    if not isinstance(items, list):
        raise QueryIRContractError("predicates must be a list")
    result: list[Predicate] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise QueryIRContractError(f"predicates[{index}] must be an object")
        _reject_unknown_keys(item, _ALLOWED_PREDICATE_KEYS, f"predicates[{index}]")
        source = _required_string(item, "source", f"predicates[{index}]")
        field = _required_string(item, "field", f"predicates[{index}]")
        operator = _required_string(item, "operator", f"predicates[{index}]")
        values = _parse_values(item.get("values"), f"predicates[{index}].values")
        quantifier = item.get("quantifier", "exists")
        if not isinstance(quantifier, str):
            raise QueryIRContractError(f"predicates[{index}].quantifier must be a string")
        result.append(Predicate(source, field, operator, values, quantifier))
    return result


def _parse_order_by(value: Any) -> list[OrderBy]:
    items = value if value is not None else []
    if not isinstance(items, list):
        raise QueryIRContractError("order_by must be a list")
    result: list[OrderBy] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise QueryIRContractError(f"order_by[{index}] must be an object")
        _reject_unknown_keys(item, _ALLOWED_ORDER_KEYS, f"order_by[{index}]")
        field = _required_string(item, "field", f"order_by[{index}]")
        direction = item.get("direction", "ASC")
        if not isinstance(direction, str):
            raise QueryIRContractError(f"order_by[{index}].direction must be a string")
        result.append(OrderBy(field, direction))
    return result


def _parse_limit(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise QueryIRContractError("limit must be an integer")
    return value


def _parse_values(value: Any, context: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise QueryIRContractError(f"{context} must be a list")
    values: list[Any] = []
    for index, item in enumerate(value):
        if not isinstance(item, _SCALAR_VALUE_TYPES):
            raise QueryIRContractError(f"{context}[{index}] must be a scalar value")
        values.append(item)
    return tuple(values)


def _required_string(item: dict[str, Any], key: str, context: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise QueryIRContractError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _reject_unknown_keys(item: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(item) - allowed
    if unknown:
        raise QueryIRContractError(f"Unknown {context} keys: {sorted(unknown)}")


def _find_forbidden_sql_keys(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            if key_text.lower() in _FORBIDDEN_SQL_KEYS:
                found.add(f"{path}.{key_text}".strip("."))
            found.update(_find_forbidden_sql_keys(nested, f"{path}.{key_text}".strip(".")))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.update(_find_forbidden_sql_keys(nested, f"{path}[{index}]"))
    return found


def _configured_select_aliases(schema: dict[str, Any], root: str) -> dict[str, str | None]:
    aliases: dict[str, str | None] = {}
    select_sources = []
    if root == schema.get("default_root", "recipe"):
        select_sources.append(schema.get("default_select", []))
    root_config = schema.get("roots", {}).get(root, {})
    if isinstance(root_config, dict):
        select_sources.append(root_config.get("select", []))
    select_sources.append(schema.get("selectable_expressions", {}).get(root, []))
    for items in select_sources:
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("expression"), str):
                continue
            alias = item.get("alias")
            aliases[item["expression"].strip()] = alias if isinstance(alias, str) else None
    return aliases


def _aggregate_aliases(schema: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for aggregate in schema.get("aggregates", {}).values():
        if not isinstance(aggregate, dict):
            continue
        expression = aggregate.get("expression")
        alias = aggregate.get("alias")
        if isinstance(expression, str) and isinstance(alias, str):
            aliases[expression.strip()] = alias
    return aliases
