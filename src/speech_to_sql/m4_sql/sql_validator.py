from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from speech_to_sql.common.config import load_sql_schema_config
from speech_to_sql.m3_query_ir.schema_discovery import load_virtual_schema
from speech_to_sql.m3_query_ir.schema_registry import SchemaRegistry, load_schema_registry
from speech_to_sql.m3_query_ir.sql_ir import Predicate, QueryIR


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

    _validate_schema_relationships(schema, errors)
    _validate_predicate_limits(query_ir, schema, errors)
    _validate_semantic_guards(query_ir, schema, errors, warnings)

    if not query_ir.select:
        errors.append("select must contain at least one expression")
    else:
        _validate_select_expressions(query_ir, schema, errors)

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
        elif not _order_column_exists(order.field, schema, registry):
            errors.append(
                f"order_by[{index}] field {order.field!r} maps to a column missing from sql_schema.collections"
            )
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

    _validate_dietary_values(predicate, index, registry, errors)

    operator = predicate.operator.upper()
    if operator not in schema.get("allowed_operators", []):
        errors.append(f"predicates[{index}] uses unsupported operator {predicate.operator!r}")
    else:
        _validate_operator_field_compatibility(predicate, index, registry, errors)
        _validate_value_count(predicate, index, schema, errors)
        _validate_value_constraints(predicate, index, schema, errors)

    if _validate_nutrition_group_predicate(predicate, index, registry, errors):
        return

    if not registry.predicate_field_exists(predicate):
        errors.append(
            f"predicates[{index}] field {predicate.source}.{predicate.field} is not in sql_schema"
        )
        return

    table_column = _predicate_table_column(predicate, registry)
    if table_column and not _schema_has_column(schema, *table_column):
        table, column = table_column
        errors.append(
            f"predicates[{index}] field {predicate.source}.{predicate.field} maps to "
            f"missing column {table}.{column}"
        )
        return

    relation = registry.relation(predicate.source)
    if relation and not _relation_has_renderable_path(predicate.source, registry):
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


def _validate_select_expressions(
    query_ir: QueryIR,
    schema: dict[str, Any],
    errors: list[str],
) -> None:
    allowed = _allowed_select_expressions(schema, query_ir.root)
    for index, select in enumerate(query_ir.select):
        expression = select.expression.strip()
        if expression not in allowed:
            errors.append(
                f"select[{index}] expression {expression!r} is not allowed by sql_schema"
            )


def _allowed_select_expressions(schema: dict[str, Any], root: str) -> set[str]:
    allowed: set[str] = set()

    def add_items(items: Any) -> None:
        for item in items or []:
            if isinstance(item, dict) and isinstance(item.get("expression"), str):
                allowed.add(item["expression"].strip())

    if root == schema.get("default_root", "recipe"):
        add_items(schema.get("default_select", []))
    root_config = schema.get("roots", {}).get(root, {})
    if isinstance(root_config, dict):
        add_items(root_config.get("select", []))
    add_items(schema.get("selectable_expressions", {}).get(root, []))
    for aggregate in schema.get("aggregates", {}).values():
        if isinstance(aggregate, dict) and isinstance(aggregate.get("expression"), str):
            allowed.add(aggregate["expression"].strip())
    return allowed


def _validate_operator_field_compatibility(
    predicate: Predicate,
    index: int,
    registry: SchemaRegistry,
    errors: list[str],
) -> None:
    catalog = registry.predicate_catalog_item(predicate)
    if not catalog:
        return
    operators = {str(item).upper() for item in catalog.get("operators", [])}
    if not operators:
        return
    operator = predicate.operator.upper()
    if operator not in operators:
        errors.append(
            f"predicates[{index}] operator {predicate.operator!r} is not valid for "
            f"{predicate.source}.{predicate.field} ({catalog.get('operator_profile', 'unknown')} field)"
        )


def _validate_nutrition_group_predicate(
    predicate: Predicate,
    index: int,
    registry: SchemaRegistry,
    errors: list[str],
) -> bool:
    if predicate.source != "nutrition":
        return False
    group = registry.nutrition_group(predicate.field)
    if not group:
        return False

    relation = registry.relation("nutrition") or {}
    columns = relation.get("columns", {})
    missing = [
        str(field)
        for field in group.get("fields", [])
        if str(field) not in columns
    ]
    if missing:
        errors.append(
            f"predicates[{index}] nutrition group {predicate.field!r} references "
            f"unknown fields {missing}"
        )
    if not group.get("fields"):
        errors.append(f"predicates[{index}] nutrition group {predicate.field!r} has no fields")
    return True


def _validate_predicate_limits(
    query_ir: QueryIR,
    schema: dict[str, Any],
    errors: list[str],
) -> None:
    limits = schema.get("predicate_limits") or {}
    max_total = limits.get("max_total")
    if isinstance(max_total, int) and len(query_ir.predicates) > max_total:
        errors.append(f"predicate count {len(query_ir.predicates)} exceeds max_total {max_total}")

    max_values = limits.get("max_values_per_predicate")
    if isinstance(max_values, int):
        for index, predicate in enumerate(query_ir.predicates):
            if len(predicate.values) > max_values:
                errors.append(
                    f"predicates[{index}] has {len(predicate.values)} values, exceeds max_values_per_predicate {max_values}"
                )

    per_source = limits.get("max_per_source") or {}
    counts: dict[str, int] = {}
    for predicate in query_ir.predicates:
        counts[predicate.source] = counts.get(predicate.source, 0) + 1
    for source, count in counts.items():
        max_count = per_source.get(source)
        if isinstance(max_count, int) and count > max_count:
            errors.append(f"source {source!r} has {count} predicates, exceeds max_per_source {max_count}")


def _validate_dietary_values(
    predicate: Predicate,
    index: int,
    registry: SchemaRegistry,
    errors: list[str],
) -> None:
    if predicate.source != "dietary":
        return
    relation = registry.relation("dietary") or {}
    columns = relation.get("columns", {})
    for value in predicate.values:
        if str(value) not in columns:
            errors.append(f"predicates[{index}] uses unknown dietary style {value!r}")


def _validate_value_constraints(
    predicate: Predicate,
    index: int,
    schema: dict[str, Any],
    errors: list[str],
) -> None:
    constraints = schema.get("value_constraints", {}).get(predicate.source, {})
    constraint = constraints.get(predicate.field)
    if not constraint:
        return

    numeric_values: list[float] = []
    for value_index, value in enumerate(predicate.values):
        if isinstance(value, bool):
            errors.append(
                f"predicates[{index}].values[{value_index}] must be numeric for constrained field {predicate.source}.{predicate.field}"
            )
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(
                f"predicates[{index}].values[{value_index}] must be numeric for constrained field {predicate.source}.{predicate.field}"
            )
            continue
        numeric_values.append(numeric)
        min_value = constraint.get("min")
        max_value = constraint.get("max")
        if isinstance(min_value, (int, float)) and numeric < float(min_value):
            errors.append(
                f"predicates[{index}] value {numeric:g} is below min {float(min_value):g} for {predicate.source}.{predicate.field}"
            )
        if isinstance(max_value, (int, float)) and numeric > float(max_value):
            errors.append(
                f"predicates[{index}] value {numeric:g} exceeds max {float(max_value):g} for {predicate.source}.{predicate.field}"
            )

    if predicate.operator.upper() == "BETWEEN" and len(numeric_values) == 2:
        if numeric_values[0] > numeric_values[1]:
            errors.append(
                f"predicates[{index}] BETWEEN lower bound must be <= upper bound"
            )


def _validate_semantic_guards(
    query_ir: QueryIR,
    schema: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    guards = schema.get("semantic_guards") or {}
    _validate_dietary_ingredient_category_conflicts(
        query_ir,
        guards.get("dietary_ingredient_category_conflicts") or {},
        errors,
        warnings,
    )


def _validate_dietary_ingredient_category_conflicts(
    query_ir: QueryIR,
    rules: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not rules:
        return

    dietary_styles = _dietary_styles_from_ir(query_ir)
    ingredient_values = _included_ingredients_from_ir(query_ir)
    if not dietary_styles or not ingredient_values:
        return

    emitted: set[str] = set()
    for style in dietary_styles:
        rule = rules.get(style)
        if not isinstance(rule, dict):
            continue
        blocked = {
            str(category).casefold()
            for category in rule.get("blocked_categories", [])
        }
        if not blocked:
            continue
        severity = str(rule.get("severity") or "warning").lower()
        for ingredient in ingredient_values:
            category = _ingredient_category(ingredient)
            if not category or category.casefold() not in blocked:
                continue
            message = (
                f"semantic conflict: dietary style {style!r} usually excludes "
                f"ingredient {ingredient!r} from category {category!r}"
            )
            if message in emitted:
                continue
            emitted.add(message)
            if severity == "error":
                errors.append(message)
            else:
                warnings.append(message)


def _dietary_styles_from_ir(query_ir: QueryIR) -> set[str]:
    styles: set[str] = set()
    for predicate in query_ir.predicates:
        if predicate.source != "dietary" or predicate.quantifier == "not_exists":
            continue
        styles.update(str(value) for value in predicate.values)
    return styles


def _included_ingredients_from_ir(query_ir: QueryIR) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for predicate in query_ir.predicates:
        if predicate.source != "ingredients" or predicate.quantifier == "not_exists":
            continue
        if predicate.operator.upper() not in {"=", "IN", "LIKE"}:
            continue
        for value in predicate.values:
            text = str(value).strip().lower()
            if not text or text in seen:
                continue
            seen.add(text)
            values.append(text)
    return values


def _ingredient_category(ingredient: str) -> str | None:
    try:
        from speech_to_sql.m2_nlu.data.db_loader import DB_INGREDIENTS
        from speech_to_sql.m2_nlu.data.ingredient_synonyms import canonicalize_ingredient
    except Exception:
        return None

    key = str(ingredient).strip().lower()
    canonical = canonicalize_ingredient(key)
    payload = DB_INGREDIENTS.get(canonical) or DB_INGREDIENTS.get(key)
    if not isinstance(payload, dict):
        return None
    category = payload.get("category")
    return str(category).strip() if category else None


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


def _relation_has_renderable_path(source: str, registry: SchemaRegistry) -> bool:
    if source == "base":
        return True
    if registry.relation_is_base(source) or registry.relationship_is_direct_to_base(source):
        return True

    relation = registry.relation(source) or {}
    parent_source = relation.get("requires")
    if not parent_source:
        return False
    parent_relation = registry.relation(str(parent_source))
    if not parent_relation or not registry.relationship_is_direct_to_base(str(parent_source)):
        return False

    child_join = registry.relationship_for_relation(source)
    if not child_join:
        return False
    return {
        child_join.from_table,
        child_join.to_table,
    } == {
        str(parent_relation.get("table")),
        str(relation.get("table")),
    }


def _validate_schema_relationships(schema: dict[str, Any], errors: list[str]) -> None:
    seen: set[str] = set()
    for relation in schema.get("relationships", []):
        name = str(relation.get("name") or "<unnamed>")
        checks: list[tuple[str, str]] = []
        from_table = str(relation.get("from_table") or "")
        to_table = str(relation.get("to_table") or "")
        from_column = str(relation.get("from_column") or "")
        to_column = str(relation.get("to_column") or "")
        if from_table and from_column:
            checks.append((from_table, from_column))
        if to_table and to_column:
            checks.append((to_table, to_column))
        for pair in relation.get("column_pairs") or []:
            checks.append((from_table, str(pair.get("from_column") or "")))
            checks.append((to_table, str(pair.get("to_column") or "")))
        for table, column in checks:
            if not table or not column or _schema_has_column(schema, table, column):
                continue
            message = f"relationship {name!r} references missing column {table}.{column}"
            if message not in seen:
                errors.append(message)
                seen.add(message)


def _predicate_table_column(
    predicate: Predicate,
    registry: SchemaRegistry,
) -> tuple[str, str] | None:
    if predicate.source == "base":
        column = registry.base_column(predicate.field)
        if not column:
            return None
        return registry.base_table["name"], column

    relation = registry.relation(predicate.source)
    if not relation:
        return None
    column = registry.predicate_column(predicate)
    if not column:
        return None
    return str(relation.get("table") or registry.base_table["name"]), column


def _order_column_exists(
    field: str,
    schema: dict[str, Any],
    registry: SchemaRegistry,
) -> bool:
    order_field = schema.get("order_fields", {}).get(field)
    if not order_field:
        return False
    source = order_field.get("source")
    logical_field = order_field.get("field")
    if source == "base":
        column = registry.base_column(logical_field)
        return bool(column and _schema_has_column(schema, registry.base_table["name"], column))

    relation = registry.relation(source)
    if not relation:
        return False
    column = relation.get("columns", {}).get(logical_field)
    table = str(relation.get("table") or registry.base_table["name"])
    return bool(column and _schema_has_column(schema, table, column))


def _schema_has_column(schema: dict[str, Any], table: str, column: str) -> bool:
    return column in schema.get("collections", {}).get(table, {}).get("fields", {})


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
