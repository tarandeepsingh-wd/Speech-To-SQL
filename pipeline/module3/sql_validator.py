from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from module3.config import load_sql_schema_config
from module3.schema_discovery import load_virtual_schema
from module3.schema_registry import SchemaRegistry, load_schema_registry
from module3.sql_ir import Predicate, QueryIR


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class QueryIRValidationError(ValueError):
    def __init__(self, result: ValidationResult) -> None:
        super().__init__("Invalid QueryIR: " + "; ".join(result.errors))
        self.result = result


def validate_query_ir(
    query_ir: QueryIR,
    *,
    schema: dict[str, Any] | None = None,
    virtual_schema: dict[str, Any] | None = None,
    virtual_schema_path: str | Path | None = None,
    require_virtual_evidence: bool = False,
) -> ValidationResult:
    schema = schema or load_sql_schema_config()
    registry = load_schema_registry(schema)
    virtual_schema = virtual_schema if virtual_schema is not None else _load_virtual_schema_optional(virtual_schema_path)

    errors: list[str] = []
    warnings: list[str] = []

    if not query_ir.select:
        errors.append("select must contain at least one expression")

    if not registry.root_exists(query_ir.root):
        errors.append(f"unknown query root {query_ir.root!r}")

    root_source = registry.root_source(query_ir.root) if registry.root_exists(query_ir.root) else "base"

    for index, predicate in enumerate(query_ir.predicates):
        if root_source != "base" and predicate.source != root_source:
            errors.append(
                f"predicates[{index}] source {predicate.source!r} cannot be rendered "
                f"from root {query_ir.root!r} yet"
            )
            continue
        _validate_predicate(
            predicate,
            index,
            schema,
            registry,
            virtual_schema,
            require_virtual_evidence,
            errors,
            warnings,
        )

    for index, order in enumerate(query_ir.order_by):
        if order.field not in schema.get("order_fields", {}):
            errors.append(f"order_by[{index}] uses unknown field {order.field!r}")
        if order.direction.upper() not in {"ASC", "DESC"}:
            errors.append(f"order_by[{index}] has invalid direction {order.direction!r}")

    if query_ir.limit is not None:
        if not isinstance(query_ir.limit, int):
            errors.append("limit must be an integer")
        elif query_ir.limit <= 0:
            errors.append("limit must be positive")
        elif query_ir.limit > int(schema.get("max_limit", query_ir.limit)):
            errors.append(f"limit exceeds max_limit {schema['max_limit']}")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def assert_valid_query_ir(query_ir: QueryIR, **kwargs) -> ValidationResult:
    result = validate_query_ir(query_ir, **kwargs)
    if not result.ok:
        raise QueryIRValidationError(result)
    return result


def _validate_predicate(
    predicate: Predicate,
    index: int,
    schema: dict[str, Any],
    registry: SchemaRegistry,
    virtual_schema: dict[str, Any] | None,
    require_virtual_evidence: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    if predicate.source != "base" and not registry.relation(predicate.source):
        errors.append(f"predicates[{index}] uses unknown source {predicate.source!r}")
        return

    if predicate.quantifier not in schema.get("allowed_quantifiers", ["exists"]):
        errors.append(f"predicates[{index}] uses invalid quantifier {predicate.quantifier!r}")

    operator = predicate.operator.upper()
    if operator not in schema.get("allowed_operators", []):
        errors.append(f"predicates[{index}] uses unsupported operator {predicate.operator!r}")
    else:
        _validate_value_count(predicate, index, schema, errors)

    if not registry.predicate_field_exists(predicate):
        errors.append(
            f"predicates[{index}] field {predicate.source}.{predicate.field} is not in sql_schema"
        )
        return

    relation = registry.relation(predicate.source)
    if relation and not registry.relation_is_base(predicate.source) and not registry.relationship_is_direct_to_base(predicate.source):
        errors.append(
            f"predicates[{index}] source {predicate.source!r} needs a multi-hop join path "
            "that the SQL planner cannot render yet"
        )
        return

    if virtual_schema:
        if not _has_virtual_evidence(predicate, virtual_schema):
            message = (
                f"predicates[{index}] field {predicate.source}.{predicate.field} "
                "has no virtual-schema evidence yet"
            )
            if require_virtual_evidence:
                errors.append(message)
            else:
                warnings.append(message)


def _validate_value_count(
    predicate: Predicate,
    index: int,
    schema: dict[str, Any],
    errors: list[str],
) -> None:
    rule = schema["operator_value_counts"][predicate.operator.upper()]
    count = len(predicate.values)
    if isinstance(rule, int) and count != rule:
        errors.append(
            f"predicates[{index}] operator {predicate.operator!r} expects {rule} value(s), got {count}"
        )
    elif isinstance(rule, dict) and count < int(rule.get("min", 0)):
        errors.append(
            f"predicates[{index}] operator {predicate.operator!r} expects at least {rule['min']} value(s), got {count}"
        )


def _has_virtual_evidence(predicate: Predicate, virtual_schema: dict[str, Any]) -> bool:
    tables = virtual_schema.get("virtual_tables", {})
    if predicate.source == "base":
        return _observed_field(predicate.field, tables.get("recipes", {}))
    if predicate.source == "nutrition":
        return predicate.field in tables.get("recipe_nutrition", {}).get("fields", {})
    if predicate.source == "dietary":
        values = {str(value) for value in predicate.values}
        fields = tables.get("recipe_dietary", {}).get("fields", {})
        return bool(values & set(fields))
    if predicate.source == "ingredients":
        recipes = tables.get("recipes", {}).get("fields", {})
        return "ingredients" in recipes or bool(tables.get("recipe_ingredients", {}).get("fields"))
    return False


def _observed_field(field: str, recipe_table: dict[str, Any]) -> bool:
    fields = recipe_table.get("fields", {})
    if field in fields:
        return True
    for summary in fields.values():
        if summary.get("observed_as") == field:
            return True
    return False


def _load_virtual_schema_optional(path: str | Path | None) -> dict[str, Any] | None:
    try:
        return load_virtual_schema(path)
    except FileNotFoundError:
        return None
