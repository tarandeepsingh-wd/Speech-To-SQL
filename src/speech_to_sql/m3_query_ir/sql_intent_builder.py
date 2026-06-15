from __future__ import annotations

import re
from typing import Any

from speech_to_sql.common.config import load_recipe_db_config, load_sql_schema_config
from speech_to_sql.m3_query_ir.intent_grounder import assert_grounded_intent
from speech_to_sql.m3_query_ir.intent_preparer import prepare_intent
from speech_to_sql.m3_query_ir.sort_resolver import resolve_sort_intent
from speech_to_sql.m3_query_ir.sql_ir import OrderBy, Predicate, QueryIR, SelectExpr
from speech_to_sql.m3_query_ir.time_resolver import resolve_time_filter


def build_query_ir(
    tag_result: dict,
    *,
    transcript: str | None = None,
) -> QueryIR:
    recipe_config = load_recipe_db_config()
    schema = load_sql_schema_config()
    transcript = transcript or tag_result.get("transcript", "") or tag_result.get("query", "")
    tag_result = prepare_intent(
        tag_result,
        transcript=transcript,
        recipe_config=recipe_config,
        schema=schema,
    )
    assert_grounded_intent(tag_result)
    text = transcript.lower()

    root = _root_name(text, schema)
    query = QueryIR(root=root, select=_select_exprs(tag_result, schema, root))
    query.predicates.extend(_ingredient_predicates(tag_result))
    query.predicates.extend(_nutrition_predicates(tag_result))
    query.predicates.extend(_dietary_predicates(tag_result, text, recipe_config))
    query.predicates.extend(_time_predicates(tag_result, text, recipe_config))
    query.predicates.extend(_semantic_filter_predicates(tag_result, schema))
    query.predicates.extend(_category_predicates(tag_result))

    order = _order_by(tag_result, text, recipe_config)
    if order:
        query.order_by.append(order)

    query.limit = _limit_value(tag_result, transcript)
    return query


def _select_exprs(tag_result: dict, schema: dict[str, Any], root: str) -> list[SelectExpr]:
    aggregate_ops = [
        op for op in tag_result.get("sql_operators", []) or []
        if op.get("canonical") in schema.get("aggregates", {})
    ]
    if aggregate_ops:
        aggregate = schema["aggregates"][aggregate_ops[0]["canonical"]]
        return [SelectExpr(aggregate["expression"], aggregate.get("alias"))]
    root_config = schema.get("roots", {}).get(root, {})
    select_items = root_config.get("select") or schema["default_select"]
    return [SelectExpr(item["expression"], item.get("alias")) for item in select_items]


def _root_name(text: str, schema: dict[str, Any]) -> str:
    default = schema.get("default_root", "recipe")
    roots = schema.get("roots", {})
    matched = []
    for name, root in roots.items():
        terms = root.get("terms", []) if isinstance(root, dict) else []
        if _contains_config_term(text, terms):
            matched.append((int(root.get("priority", 0)), name))
    if not matched:
        return default
    matched.sort()
    best = matched[0][1]
    if best != default and any(
        name == default for _, name in matched
    ):
        return default
    return best


def _ingredient_predicates(tag_result: dict) -> list[Predicate]:
    logic = tag_result.get("ingredient_logic") or {}
    include_source = (
        logic.get("include")
        if "include" in logic
        else tag_result.get("ingredients")
    )
    include = _unique_strings(include_source or [])
    exclude = _unique_strings(logic.get("exclude") or [])
    include_operator = logic.get("include_operator") or "AND"
    predicates: list[Predicate] = []

    if include:
        if include_operator == "OR":
            predicates.append(Predicate(
                source="ingredients",
                field="ingredient",
                operator="IN",
                values=tuple(include),
            ))
        else:
            predicates.extend(
                Predicate(
                    source="ingredients",
                    field="ingredient",
                    operator="=",
                    values=(ingredient,),
                )
                for ingredient in include
            )

    if exclude:
        predicates.append(Predicate(
            source="ingredients",
            field="ingredient",
            operator="IN",
            values=tuple(exclude),
            quantifier="not_exists",
        ))
    return predicates


def _nutrition_predicates(tag_result: dict) -> list[Predicate]:
    predicates: list[Predicate] = []
    for nf in tag_result.get("nutrition_filters", []) or []:
        nutrient_group = nf.get("nutrient_group")
        if nutrient_group:
            op = nf.get("operator") or "ANY_GT"
            try:
                value = float(nf.get("value", 0))
            except (TypeError, ValueError):
                value = 0.0
            predicates.append(Predicate(
                source="nutrition",
                field=str(nutrient_group),
                operator=str(op),
                values=(value,),
            ))
            continue
        nutrient = nf.get("nutrient")
        op = nf.get("operator")
        if not nutrient or not op:
            continue
        try:
            value = float(nf["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if op == "~":
            predicates.append(Predicate(
                source="nutrition",
                field=nutrient,
                operator="BETWEEN",
                values=(round(value * 0.9, 1), round(value * 1.1, 1)),
            ))
        else:
            predicates.append(Predicate(
                source="nutrition",
                field=nutrient,
                operator=op,
                values=(value,),
            ))
    return predicates


def _dietary_predicates(
    tag_result: dict,
    text: str,
    recipe_config: dict[str, Any],
) -> list[Predicate]:
    aliases = recipe_config["diet_aliases"]
    styles = []
    for style in _unique_strings(tag_result.get("dietary_styles") or []):
        api_style = aliases.get(style, style if style in aliases.values() else None)
        if not api_style:
            continue
        styles.append(api_style)

    if len(styles) > 1 and _contains_config_term(text, recipe_config.get("logic_terms", {}).get("or", [])):
        return [Predicate(
            source="dietary",
            field="dietary_tag",
            operator="IN",
            values=tuple(styles),
            quantifier="or_exists",
        )]

    predicates = []
    for api_style in styles:
        predicates.append(Predicate(
            source="dietary",
            field="dietary_tag",
            operator="=",
            values=(api_style,),
        ))
    return predicates


def _time_predicates(
    tag_result: dict,
    text: str,
    recipe_config: dict[str, Any],
) -> list[Predicate]:
    time_filter = resolve_time_filter(tag_result, text, recipe_config)
    if not time_filter:
        return []
    field, min_value, max_value = time_filter
    if min_value is not None and max_value is not None:
        return [Predicate("base", field, "BETWEEN", (float(min_value), float(max_value)))]
    if min_value is not None:
        return [Predicate("base", field, ">=", (float(min_value),))]
    if max_value is not None:
        return [Predicate("base", field, "<=", (float(max_value),))]
    return []


def _category_predicates(tag_result: dict) -> list[Predicate]:
    predicates: list[Predicate] = []
    for item in tag_result.get("category_filters") or []:
        if not isinstance(item, dict):
            continue
        field = item.get("field")
        value = item.get("value")
        operator = item.get("operator") or "="
        if not field or value is None:
            continue
        predicates.append(Predicate(
            source="base",
            field=str(field),
            operator=str(operator),
            values=(value,),
        ))
    return predicates


def _semantic_filter_predicates(
    tag_result: dict,
    schema: dict[str, Any],
) -> list[Predicate]:
    filters = schema.get("semantic_filters", {})
    predicates: list[Predicate] = []

    utensil_config = filters.get("utensils") or {}
    predicates.extend(_text_filter_predicates(
        tag_result.get("utensils") or [],
        utensil_config,
    ))

    action_config = filters.get("cooking_actions") or {}
    allowed_action_types = {
        str(item)
        for item in action_config.get("allowed_action_types", [])
    }
    actions = [
        item.get("action")
        for item in tag_result.get("actions") or []
        if isinstance(item, dict)
        and item.get("action")
        and (
            not allowed_action_types
            or str(item.get("type")) in allowed_action_types
        )
    ]
    predicates.extend(_text_filter_predicates(actions, action_config))
    return predicates


def _text_filter_predicates(
    values: list[Any],
    config: dict[str, Any],
) -> list[Predicate]:
    source = config.get("source")
    field = config.get("field")
    operator = str(config.get("operator") or "LIKE").upper()
    if not source or not field:
        return []

    predicates = []
    for value in _unique_strings(values):
        predicates.append(Predicate(
            source=str(source),
            field=str(field),
            operator=operator,
            values=(_transform_filter_value(value, operator, config),),
        ))
    return predicates


def _transform_filter_value(
    value: Any,
    operator: str,
    config: dict[str, Any],
) -> str:
    text = str(value).strip().lower()
    transform = config.get("value_transform")
    if transform == "contains" or (operator == "LIKE" and transform is None):
        return f"%{text}%"
    if transform == "prefix":
        return f"{text}%"
    if transform == "suffix":
        return f"%{text}"
    return text


def _order_by(
    tag_result: dict,
    text: str,
    recipe_config: dict[str, Any],
) -> OrderBy | None:
    order_ops = [
        op for op in tag_result.get("sql_operators", []) or []
        if str(op.get("canonical", "")).startswith("ORDER BY")
    ]
    if not order_ops and not _contains_config_term(text, recipe_config["sort_cues"]):
        return None

    sort_field, direction = resolve_sort_intent(
        text,
        recipe_config,
        order_ops=order_ops,
        default_direction="ASC",
    )
    if not sort_field:
        return None

    return OrderBy(sort_field, direction)


def _limit_value(tag_result: dict, transcript: str) -> int | None:
    for op in tag_result.get("sql_operators", []) or []:
        if op.get("canonical") != "LIMIT":
            continue
        match = re.search(r"\d+", str(op.get("text") or ""))
        if match:
            return int(match.group(0))
    match = re.search(r"\b(?:top|first|limit)\s+(\d+)\b", transcript, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _unique_strings(items: list[Any]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _contains_config_term(text: str, terms: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)
