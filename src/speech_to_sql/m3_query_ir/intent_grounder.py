from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from speech_to_sql.common.config import load_sql_schema_config
from speech_to_sql.m3_query_ir.nutrition_scope import nutrition_filter_source
from speech_to_sql.m3_query_ir.schema_registry import SchemaRegistry, load_schema_registry
from speech_to_sql.m3_query_ir.sql_ir import Predicate


@dataclass(frozen=True)
class GroundingEvidence:
    source: str
    field: str
    table: str
    column: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "field": self.field,
            "table": self.table,
            "column": self.column,
            "reason": self.reason,
        }


class IntentGroundingError(ValueError):
    def __init__(
        self,
        errors: list[str],
        grounding: list[dict[str, str]] | None = None,
        error_details: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__("Intent grounding failed: " + "; ".join(errors))
        self.errors = errors
        self.grounding = grounding or []
        self.error_details = error_details or []


def ground_intent(
    tag_result: dict[str, Any],
    *,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if tag_result.get("_schema_grounded"):
        return tag_result

    schema = schema or load_sql_schema_config()
    registry = load_schema_registry(schema)
    grounded = copy.deepcopy(tag_result)
    evidence: list[GroundingEvidence] = []
    errors: list[str] = []

    for item in grounded.get("_unresolved_schema_terms", []) or []:
        if not isinstance(item, dict):
            continue
        errors.append(
            f"Unresolved {item.get('source', 'schema')} term {item.get('original')!r}"
        )

    _ground_ingredients(grounded, registry, evidence, errors)
    _ground_nutrition_filters(grounded, schema, registry, evidence, errors)
    _ground_dietary_styles(grounded, registry, evidence, errors)
    _ground_time_filters(grounded, schema, registry, evidence, errors)
    _ground_category_filters(grounded, schema, registry, evidence, errors)
    _ground_sql_operators(grounded, schema, errors)

    grounded["_schema_grounded"] = True
    grounded["_grounding"] = [item.as_dict() for item in evidence]
    grounded["_grounding_errors"] = _unique_errors(errors)
    grounded["_grounding_error_details"] = [
        _classify_grounding_error(error)
        for error in grounded["_grounding_errors"]
    ]
    return grounded


def assert_grounded_intent(tag_result: dict[str, Any]) -> None:
    errors = list(tag_result.get("_grounding_errors") or [])
    if errors:
        raise IntentGroundingError(
            errors,
            tag_result.get("_grounding") or [],
            tag_result.get("_grounding_error_details") or [],
        )


def _ground_ingredients(
    tag_result: dict[str, Any],
    registry: SchemaRegistry,
    evidence: list[GroundingEvidence],
    errors: list[str],
) -> None:
    relation = registry.relation("ingredients")
    if not relation:
        errors.append("Schema has no ingredients relation")
        return
    column = relation.get("value_column")
    if not column:
        errors.append("Ingredients relation has no value column")
        return

    logic = tag_result.get("ingredient_logic") or {}
    for key in ("include", "exclude"):
        for value in logic.get(key) or []:
            if not isinstance(value, str) or not value.strip():
                errors.append(f"Invalid ingredient value {value!r}")
                continue
            evidence.append(GroundingEvidence(
                "ingredients",
                "ingredient",
                relation["table"],
                column,
                f"ingredient_logic.{key}",
            ))


def _ground_nutrition_filters(
    tag_result: dict[str, Any],
    schema: dict[str, Any],
    registry: SchemaRegistry,
    evidence: list[GroundingEvidence],
    errors: list[str],
) -> None:
    allowed_operators = set(schema.get("allowed_operators", []))
    for index, item in enumerate(tag_result.get("nutrition_filters") or []):
        if not isinstance(item, dict):
            errors.append(f"nutrition_filters[{index}] must be an object")
            continue
        source = nutrition_filter_source(item, schema=schema)
        relation = registry.relation(source) or {}
        if not relation:
            errors.append(f"nutrition_filters[{index}] uses unknown source {source!r}")
            continue
        group = str(item.get("nutrient_group") or "").strip()
        if group:
            relation = registry.relation("nutrition") or relation
            group_config = registry.nutrition_group(group)
            if not group_config:
                errors.append(f"Unknown nutrition group {group!r}")
                continue
            evidence.append(GroundingEvidence(
                "nutrition",
                group,
                relation.get("table", registry.base_table["name"]),
                ",".join(str(field) for field in group_config.get("fields", [])),
                "sql_schema.nutrition_groups",
            ))
            continue
        nutrient = str(item.get("nutrient") or "").strip()
        operator = str(item.get("operator") or "").strip().upper()
        if not nutrient:
            errors.append(f"nutrition_filters[{index}] is missing nutrient")
            continue
        if operator == "~":
            operator = "BETWEEN"
        if operator not in allowed_operators:
            errors.append(f"nutrition_filters[{index}] uses unsupported operator {operator!r}")
        if operator == "BETWEEN":
            values = item.get("values")
            if values is None:
                values = (item.get("min"), item.get("max"))
            try:
                low, high = (float(values[0]), float(values[1]))
                if low > high:
                    errors.append(f"nutrition_filters[{index}] BETWEEN lower bound must be <= upper bound")
            except (TypeError, ValueError, IndexError):
                errors.append(f"nutrition_filters[{index}] has non-numeric BETWEEN values")
        else:
            try:
                float(item["value"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"nutrition_filters[{index}] has non-numeric value")
        field = nutrient
        table = relation.get("table", registry.base_table["name"])
        reason = f"sql_schema.relations.{source}"
        if source == "nutrition":
            base_field = registry.base_field_for_relation_column("nutrition", nutrient)
            if base_field:
                source = "base"
                field = base_field
                table = registry.base_table["name"]
                reason = "sql_schema.base_columns"

        predicate = Predicate(source, field, "=", (0,))
        column = registry.predicate_column(predicate)
        if not column:
            errors.append(f"Unknown {source} field {nutrient!r}")
            continue
        evidence.append(GroundingEvidence(
            source,
            field,
            table,
            column,
            reason,
        ))


def _ground_dietary_styles(
    tag_result: dict[str, Any],
    registry: SchemaRegistry,
    evidence: list[GroundingEvidence],
    errors: list[str],
) -> None:
    relation = registry.relation("dietary") or {}
    columns = relation.get("columns", {})
    for field_name in ("dietary_styles", "excluded_dietary_styles"):
        for style in tag_result.get(field_name) or []:
            style_text = str(style).strip()
            column = columns.get(style_text)
            if not column:
                errors.append(f"Unknown dietary style {style_text!r}")
                continue
            evidence.append(GroundingEvidence(
                "dietary",
                style_text,
                relation.get("table", registry.base_table["name"]),
                column,
                f"{field_name}: sql_schema.relations.dietary",
            ))


def _ground_time_filters(
    tag_result: dict[str, Any],
    schema: dict[str, Any],
    registry: SchemaRegistry,
    evidence: list[GroundingEvidence],
    errors: list[str],
) -> None:
    allowed_operators = set(schema.get("allowed_operators", []))
    for index, item in enumerate(tag_result.get("time_filters") or []):
        if not isinstance(item, dict):
            errors.append(f"time_filters[{index}] must be an object")
            continue
        field = str(item.get("field") or "").strip()
        if not field:
            errors.append(f"time_filters[{index}] is missing field")
            continue
        column = registry.base_column(field)
        if not column:
            errors.append(f"Unknown base time field {field!r}")
            continue
        operator = str(item.get("operator") or "").strip().upper()
        if operator and operator not in allowed_operators:
            errors.append(f"time_filters[{index}] uses unsupported operator {operator!r}")
        evidence.append(GroundingEvidence(
            "base",
            field,
            registry.base_table["name"],
            column,
            "sql_schema.base_columns",
        ))


def _ground_category_filters(
    tag_result: dict[str, Any],
    schema: dict[str, Any],
    registry: SchemaRegistry,
    evidence: list[GroundingEvidence],
    errors: list[str],
) -> None:
    allowed_operators = set(schema.get("allowed_operators", []))
    for index, item in enumerate(tag_result.get("category_filters") or []):
        if not isinstance(item, dict):
            errors.append(f"category_filters[{index}] must be an object")
            continue
        field = str(item.get("field") or "").strip()
        operator = str(item.get("operator") or "=").strip().upper()
        value = item.get("value")
        if operator not in allowed_operators:
            errors.append(f"category_filters[{index}] uses unsupported operator {operator!r}")
        if not isinstance(value, (str, int, float, bool)) or str(value).strip() == "":
            errors.append(f"category_filters[{index}] has invalid value")
        source = str(item.get("source") or "base").strip() or "base"
        relation = registry.relation(source) if source != "base" else None
        column = registry.base_column(field) if source == "base" else registry.predicate_column(Predicate(source, field, "=", (value,)))
        if not column:
            errors.append(f"Unknown {source} category field {field!r}")
            continue
        evidence.append(GroundingEvidence(
            source if source != "base" else "category",
            field,
            (relation or {}).get("table", registry.base_table["name"]),
            column,
            f"sql_schema.relations.{source}" if source != "base" else "recipe_db.cuisine_aliases",
        ))


def _ground_sql_operators(
    tag_result: dict[str, Any],
    schema: dict[str, Any],
    errors: list[str],
) -> None:
    aggregates = set(schema.get("aggregates", {}))
    for index, item in enumerate(tag_result.get("sql_operators") or []):
        if not isinstance(item, dict):
            errors.append(f"sql_operators[{index}] must be an object")
            continue
        canonical = str(item.get("canonical") or "").strip().upper()
        op_type = str(item.get("type") or "").strip().upper()
        if op_type == "AGG" and canonical not in aggregates:
            errors.append(f"Unsupported aggregate {canonical!r}")


def _unique_errors(errors: list[str]) -> list[str]:
    result = []
    seen = set()
    for error in errors:
        if error in seen:
            continue
        seen.add(error)
        result.append(error)
    return result


def _classify_grounding_error(error: str) -> dict[str, str]:
    lowered = error.lower()
    if lowered.startswith("unresolved "):
        code = "unresolved_term"
    elif "unsupported operator" in lowered or "unsupported aggregate" in lowered:
        code = "unsupported_operator"
    elif "non-numeric" in lowered or "invalid value" in lowered or "must be an object" in lowered:
        code = "invalid_value"
    elif "missing" in lowered:
        code = "missing_required_field"
    elif lowered.startswith("unknown nutrition field"):
        code = "unknown_nutrition_field"
    elif lowered.startswith("unknown dietary style"):
        code = "unknown_dietary_style"
    elif lowered.startswith("unknown base time field"):
        code = "unknown_base_field"
    elif lowered.startswith("unknown category field"):
        code = "unknown_category_field"
    else:
        code = "grounding_error"

    source = "schema"
    for candidate in ("nutrition", "dietary", "category", "ingredient", "time", "sql"):
        if candidate in lowered:
            source = candidate
            break
    return {"code": code, "source": source, "message": error}
