from __future__ import annotations

from dataclasses import replace
from typing import Any

from module3.config import load_sql_schema_config
from module3.sql_ir import Predicate, QueryIR


def relax_query_ir(
    query_ir: QueryIR,
    *,
    schema: dict[str, Any] | None = None,
) -> QueryIR | None:
    schema = schema or load_sql_schema_config()
    for rule in _relaxation_rules(schema):
        relaxed = _apply_relaxation_rule(query_ir, rule)
        if relaxed is not None and relaxed.predicates != query_ir.predicates:
            return relaxed
    return None


def relaxation_chain(
    query_ir: QueryIR,
    *,
    schema: dict[str, Any] | None = None,
    max_steps: int | None = None,
) -> list[QueryIR]:
    schema = schema or load_sql_schema_config()
    chain: list[QueryIR] = []
    current = query_ir
    if max_steps is not None:
        steps = max_steps
    else:
        settings = schema.get("relaxation_settings") or {}
        steps = int(settings.get("max_chain_steps", len(_relaxation_rules(schema))))
    for _ in range(max(0, steps)):
        relaxed = relax_query_ir(current, schema=schema)
        if relaxed is None:
            break
        chain.append(relaxed)
        current = relaxed
    return chain


def _relaxation_rules(schema: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (dict(rule) for rule in schema.get("relaxation_policy", [])),
        key=lambda rule: int(rule.get("priority", 1000)),
    )


def _apply_relaxation_rule(query_ir: QueryIR, rule: dict[str, Any]) -> QueryIR | None:
    if "drop_quantifier" in rule:
        return _drop_quantifier(query_ir, str(rule["drop_quantifier"]))
    if "drop_predicates" in rule:
        return _drop_predicates(query_ir, rule["drop_predicates"])
    if "drop_source" in rule:
        return _drop_source(query_ir, str(rule["drop_source"]))
    if {"source", "from_operator", "to_operator"} <= set(rule):
        return _collapse_operator(query_ir, rule)
    return None


def _drop_quantifier(query_ir: QueryIR, quantifier: str) -> QueryIR | None:
    predicates = [
        predicate for predicate in query_ir.predicates
        if predicate.quantifier != quantifier
    ]
    if len(predicates) == len(query_ir.predicates):
        return None
    return replace(query_ir, predicates=predicates)


def _drop_source(query_ir: QueryIR, source: str) -> QueryIR | None:
    predicates = [
        predicate for predicate in query_ir.predicates
        if predicate.source != source
    ]
    if len(predicates) == len(query_ir.predicates):
        return None
    return replace(query_ir, predicates=predicates)


def _drop_predicates(query_ir: QueryIR, rule: dict[str, Any]) -> QueryIR | None:
    source = str(rule.get("source") or "")
    fields = {str(field) for field in rule.get("fields") or []}
    operators = {str(operator).upper() for operator in rule.get("operators") or []}
    quantifiers = {str(quantifier) for quantifier in rule.get("quantifiers") or []}

    predicates = []
    for predicate in query_ir.predicates:
        if source and predicate.source != source:
            predicates.append(predicate)
            continue
        if fields and predicate.field not in fields:
            predicates.append(predicate)
            continue
        if operators and predicate.operator.upper() not in operators:
            predicates.append(predicate)
            continue
        if quantifiers and predicate.quantifier not in quantifiers:
            predicates.append(predicate)
            continue

    if len(predicates) == len(query_ir.predicates):
        return None
    return replace(query_ir, predicates=predicates)


def _collapse_operator(query_ir: QueryIR, rule: dict[str, Any]) -> QueryIR | None:
    source = str(rule["source"])
    from_operator = str(rule["from_operator"]).upper()
    to_operator = str(rule["to_operator"]).upper()
    quantifier = str(rule.get("quantifier", "exists"))

    matching = [
        predicate for predicate in query_ir.predicates
        if predicate.source == source
        and predicate.operator.upper() == from_operator
        and predicate.quantifier == quantifier
        and len(predicate.values) == 1
    ]
    if len(matching) < 2:
        return None

    matching_ids = {id(predicate) for predicate in matching}
    other = [
        predicate for predicate in query_ir.predicates
        if id(predicate) not in matching_ids
    ]
    collapsed = Predicate(
        source=source,
        field=matching[0].field,
        operator=to_operator,
        values=tuple(predicate.values[0] for predicate in matching),
        quantifier=quantifier,
    )
    return replace(query_ir, predicates=other + [collapsed])
