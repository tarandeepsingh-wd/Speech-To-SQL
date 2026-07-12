from __future__ import annotations

import re
from typing import Any

from speech_to_sql.common.config import load_recipe_db_config, load_sql_schema_config
from speech_to_sql.m3_query_ir.intent_grounder import assert_grounded_intent
from speech_to_sql.m3_query_ir.intent_preparer import prepare_intent
from speech_to_sql.m3_query_ir.nutrition_scope import nutrition_filter_source
from speech_to_sql.m3_query_ir.schema_registry import SchemaRegistry, load_schema_registry
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
    query.predicates.extend(_dietary_predicates(tag_result, text, recipe_config, schema))
    query.predicates.extend(_time_predicates(tag_result, text, recipe_config))
    query.predicates.extend(_ingredient_quantity_predicates(text))
    query.predicates.extend(_ingredient_count_predicates(text))
    query.predicates.extend(_instruction_text_predicates(tag_result, text))
    query.predicates.extend(_semantic_filter_predicates(tag_result, schema, text))
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
    if _recipe_retrieval_context(text):
        return default
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


def _recipe_retrieval_context(text: str) -> bool:
    return bool(re.search(
        r"\b(?:recipe|recipes|recipe entries|dish|dishes|meal|meals|"
        r"food item|food items|cooking record|cooking records|"
        r"recipedb item|recipedb items|menu recipe|menu recipes)\b",
        text,
        re.IGNORECASE,
    ))


def _ingredient_predicates(tag_result: dict) -> list[Predicate]:
    logic = tag_result.get("ingredient_logic") or {}
    include_source = (
        logic.get("include")
        if "include" in logic
        else tag_result.get("ingredients")
    )
    include = _unique_strings(include_source or [])
    exclude = _unique_strings(logic.get("exclude") or [])
    include_groups = _ingredient_groups(logic, include)
    include_operator = logic.get("include_operator") or "AND"
    predicates: list[Predicate] = []

    if include_groups:
        for group in include_groups:
            values = _unique_strings(group.get("ingredients") or [])
            if not values:
                continue
            if len(values) > 1 and group.get("operator") == "OR":
                predicates.append(Predicate(
                    source="ingredients",
                    field="ingredient",
                    operator="IN",
                    values=tuple(values),
                ))
            else:
                predicates.extend(
                    Predicate(
                        source="ingredients",
                        field="ingredient",
                        operator="=",
                        values=(ingredient,),
                    )
                    for ingredient in values
                )
    elif include:
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


def _ingredient_groups(logic: dict[str, Any], include: list[str]) -> list[dict[str, Any]]:
    groups = logic.get("include_groups")
    if isinstance(groups, list) and groups:
        normalized = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            values = _unique_strings(group.get("ingredients") or [])
            if not values:
                continue
            normalized.append({
                "ingredients": values,
                "operator": "OR" if str(group.get("operator") or "").upper() == "OR" else "AND",
            })
        if normalized:
            return normalized

    include_operator = str(logic.get("include_operator") or "AND").upper()
    if include_operator == "OR" and len(include) > 1:
        return [{"ingredients": include, "operator": "OR"}]
    return [{"ingredients": [item], "operator": "AND"} for item in include]


def _nutrition_predicates(tag_result: dict) -> list[Predicate]:
    predicates: list[Predicate] = []
    schema = load_sql_schema_config()
    registry = load_schema_registry(schema)
    time_fields = {
        str(item.get("field") or "").strip().lower()
        for item in tag_result.get("time_filters", []) or []
        if isinstance(item, dict) and item.get("field")
    }
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
        if str(nutrient).strip().lower() in time_fields:
            continue
        source = nutrition_filter_source(nf, schema=schema)
        source, field = _nutrition_predicate_target(source, str(nutrient), registry)
        if str(op).upper() == "BETWEEN":
            values = nf.get("values")
            if values is None:
                values = (nf.get("min"), nf.get("max"))
            try:
                low, high = (float(values[0]), float(values[1]))
            except (TypeError, ValueError, IndexError):
                continue
            if low > high:
                low, high = high, low
            predicates.append(Predicate(
                source=source,
                field=field,
                operator="BETWEEN",
                values=(low, high),
            ))
            continue
        try:
            value = float(nf["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if op == "~":
            predicates.append(Predicate(
                source=source,
                field=field,
                operator="BETWEEN",
                values=(round(value * 0.9, 1), round(value * 1.1, 1)),
            ))
        else:
            predicates.append(Predicate(
                source=source,
                field=field,
                operator=op,
                values=(value,),
            ))
    return _simplify_numeric_predicates(predicates)


def _nutrition_predicate_target(
    source: str,
    nutrient: str,
    registry: SchemaRegistry,
) -> tuple[str, str]:
    if source != "nutrition":
        return source, nutrient
    base_field = registry.base_field_for_relation_column("nutrition", nutrient)
    if not base_field:
        return source, nutrient
    return "base", base_field


def _simplify_numeric_predicates(predicates: list[Predicate]) -> list[Predicate]:
    strongest: dict[tuple[str, str, str], Predicate] = {}
    passthrough: list[Predicate] = []

    for predicate in predicates:
        operator = predicate.operator
        if (
            predicate.source not in {"nutrition", "base"}
            or predicate.quantifier != "exists"
            or operator not in {"<", "<=", ">", ">="}
            or len(predicate.values) != 1
        ):
            passthrough.append(predicate)
            continue
        try:
            value = float(predicate.values[0])
        except (TypeError, ValueError):
            passthrough.append(predicate)
            continue
        direction = "lower" if operator in {">", ">="} else "upper"
        key = (predicate.source, predicate.field, direction)
        current = strongest.get(key)
        if not current:
            strongest[key] = predicate
            continue
        current_value = float(current.values[0])
        if _numeric_predicate_is_stronger(operator, value, current.operator, current_value):
            strongest[key] = predicate

    merged = [*passthrough, *strongest.values()]
    return sorted(merged, key=lambda predicate: _predicate_sort_key(predicates, predicate))


def _numeric_predicate_is_stronger(
    operator: str,
    value: float,
    current_operator: str,
    current_value: float,
) -> bool:
    if operator in {">", ">="}:
        if value != current_value:
            return value > current_value
        return operator == ">" and current_operator == ">="
    if value != current_value:
        return value < current_value
    return operator == "<" and current_operator == "<="


def _predicate_sort_key(original: list[Predicate], predicate: Predicate) -> int:
    try:
        return original.index(predicate)
    except ValueError:
        return len(original)


def _dietary_predicates(
    tag_result: dict,
    text: str,
    recipe_config: dict[str, Any],
    schema: dict[str, Any],
) -> list[Predicate]:
    aliases = recipe_config["diet_aliases"]
    styles = _resolved_dietary_styles(tag_result.get("dietary_styles") or [], aliases)
    excluded_styles = _resolved_dietary_styles(
        tag_result.get("excluded_dietary_styles") or [],
        aliases,
    )

    predicates = []
    for api_style in excluded_styles:
        predicates.append(Predicate(
            source="dietary",
            field="dietary_tag",
            operator="=",
            values=(api_style,),
            quantifier="not_exists",
        ))

    if len(styles) > 1 and (
        _contains_config_term(text, recipe_config.get("logic_terms", {}).get("or", []))
        or _is_dietary_classification_group(styles, schema)
    ):
        predicates.append(Predicate(
            source="dietary",
            field="dietary_tag",
            operator="IN",
            values=tuple(styles),
            quantifier="or_exists",
        ))
        return predicates

    for api_style in styles:
        predicates.append(Predicate(
            source="dietary",
            field="dietary_tag",
            operator="=",
            values=(api_style,),
        ))
    return predicates


def _is_dietary_classification_group(styles: list[str], schema: dict[str, Any]) -> bool:
    dietary_flag_columns = {
        str(field)
        for field in (
            schema.get("relations", {})
            .get("dietary", {})
            .get("columns", {})
        )
        if str(field) in {"vegan", "pescetarian"} or str(field).endswith("_vegetarian")
    }
    return bool(dietary_flag_columns and set(styles).issubset(dietary_flag_columns))


def _resolved_dietary_styles(styles: list[Any], aliases: dict[str, str]) -> list[str]:
    resolved = []
    for style in _unique_strings(styles):
        api_style = aliases.get(style, style if style in aliases.values() else None)
        if not api_style:
            continue
        resolved.append(api_style)
    return resolved


def _time_predicates(
    tag_result: dict,
    text: str,
    recipe_config: dict[str, Any],
) -> list[Predicate]:
    explicit = [
        item for item in tag_result.get("time_filters") or []
        if isinstance(item, dict)
    ]
    if explicit:
        predicates = []
        for item in explicit:
            predicate = _time_predicate_from_values(
                item.get("field") or recipe_config["time_fields"]["default"],
                item.get("min"),
                item.get("max"),
            )
            if predicate:
                predicates.append(predicate)
        return _simplify_numeric_predicates(predicates)

    time_filter = resolve_time_filter(tag_result, text, recipe_config)
    if not time_filter:
        return []
    field, min_value, max_value = time_filter
    predicate = _time_predicate_from_values(field, min_value, max_value)
    return [predicate] if predicate else []


def _time_predicate_from_values(
    field: str,
    min_value: Any,
    max_value: Any,
) -> Predicate | None:
    if min_value is not None and max_value is not None:
        if float(min_value) == 0:
            return Predicate("base", field, "<=", (float(max_value),))
        return Predicate("base", field, "BETWEEN", (float(min_value), float(max_value)))
    if min_value is not None:
        return Predicate("base", field, ">=", (float(min_value),))
    if max_value is not None:
        return Predicate("base", field, "<=", (float(max_value),))
    return None


def _category_predicates(tag_result: dict) -> list[Predicate]:
    predicates: list[Predicate] = []
    for item in tag_result.get("category_filters") or []:
        if not isinstance(item, dict):
            continue
        field = item.get("field")
        value = item.get("value")
        operator = item.get("operator") or "="
        source = item.get("source") or "base"
        if not field or value is None:
            continue
        predicates.append(Predicate(
            source=str(source),
            field=str(field),
            operator=str(operator),
            values=(value,),
        ))
    return predicates


def _ingredient_quantity_predicates(text: str) -> list[Predicate]:
    if not (
        re.search(r"\bingredient\s+quantity\b", text)
        or re.search(r"\bquantity\s+(?:available|present|listed)\b", text)
    ):
        return []
    return [Predicate(
        source="ingredients",
        field="quantity",
        operator="LIKE",
        values=("%",),
    )]


def _ingredient_count_predicates(text: str) -> list[Predicate]:
    patterns = (
        (r"\bbetween\s+(\d+)\s+(?:and|to)\s+(\d+)\s+ingredients?\b", "BETWEEN"),
        (r"\b(?:at\s+least|minimum|min)\s+(\d+)\s+ingredients?\b", ">="),
        (r"\b(?:more\s+than|greater\s+than|over|above)\s+(\d+)\s+ingredients?\b", ">"),
        (r"\b(?:fewer\s+than|less\s+than|under|below)\s+(\d+)\s+ingredients?\b", "<"),
        (r"\b(?:at\s+most|maximum|max)\s+(\d+)\s+ingredients?\b", "<="),
        (r"\b(?:exactly|equal\s+to|equals?)\s+(\d+)\s+ingredients?\b", "="),
    )
    predicates: list[Predicate] = []
    for pattern, operator in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                if operator == "BETWEEN":
                    low = float(match.group(1))
                    high = float(match.group(2))
                    if low > high:
                        low, high = high, low
                    predicates.append(Predicate(
                        source="ingredient_count",
                        field="count",
                        operator=operator,
                        values=(low, high),
                    ))
                else:
                    predicates.append(Predicate(
                        source="ingredient_count",
                        field="count",
                        operator=operator,
                        values=(float(match.group(1)),),
                    ))
            except (TypeError, ValueError):
                continue
        if predicates:
            break
    return predicates


def _instruction_text_predicates(tag_result: dict, text: str) -> list[Predicate]:
    if not _instruction_scope(text):
        return []
    scoped_terms = _instruction_scoped_terms(tag_result, text)
    action_values = list(scoped_terms["actions"])
    utensil_values = list(scoped_terms["utensils"])

    if not action_values and not utensil_values:
        action_values = [
            item.get("action")
            for item in tag_result.get("actions") or []
            if isinstance(item, dict) and item.get("action")
        ]
        utensil_values = list(tag_result.get("utensils") or [])

    if len(action_values) > 1:
        action_values = [
            value for value in action_values
            if str(value).strip().lower() not in {"cook"}
        ]
    values: list[Any] = [*action_values, *utensil_values]
    return [
        Predicate(
            source="instructions",
            field="steps",
            operator="LIKE",
            values=(_transform_filter_value(value, "LIKE", {"value_transform": "contains"}),),
        )
        for value in _unique_strings(values)
    ]


def _instruction_scope(text: str) -> bool:
    return bool(re.search(
        r"\b(?:instruction|instructions|step|steps|cooking\s+steps?)\b",
        text,
    ))


def _semantic_filter_predicates(
    tag_result: dict,
    schema: dict[str, Any],
    text: str,
) -> list[Predicate]:
    filters = schema.get("semantic_filters", {})
    predicates: list[Predicate] = []
    scoped_terms = _instruction_scoped_terms(tag_result, text) if _instruction_scope(text) else {
        "actions": set(),
        "utensils": set(),
    }

    utensil_config = filters.get("utensils") or {}
    predicates.extend(_text_filter_predicates(
        [
            utensil for utensil in tag_result.get("utensils") or []
            if str(utensil).strip().lower() not in scoped_terms["utensils"]
        ],
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
        and str(item.get("action")).strip().lower() not in scoped_terms["actions"]
        and (
            not allowed_action_types
            or str(item.get("type")) in allowed_action_types
        )
    ]
    predicates.extend(_text_filter_predicates(actions, action_config))
    return predicates


def _instruction_scoped_terms(tag_result: dict, text: str) -> dict[str, set[str]]:
    scoped = {"actions": set(), "utensils": set()}

    for item in tag_result.get("actions") or []:
        if not isinstance(item, dict) or not item.get("action"):
            continue
        if _span_is_instruction_scoped(text, item.get("span")):
            scoped["actions"].add(str(item["action"]).strip().lower())

    utensil_values = {
        str(value).strip().lower()
        for value in tag_result.get("utensils") or []
        if str(value).strip()
    }
    for item in tag_result.get("raw_spans") or []:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical") or "").strip().lower()
        if canonical not in utensil_values:
            continue
        if _span_is_instruction_scoped(text, item.get("span")):
            scoped["utensils"].add(canonical)

    return scoped


def _span_is_instruction_scoped(text: str, span: Any) -> bool:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return False
    try:
        start = int(span[0])
        end = int(span[1]) + 1
    except (TypeError, ValueError):
        return False

    text_lower = text.lower()
    prefix = text_lower[max(0, start - 96):start]
    suffix = text_lower[end:min(len(text_lower), end + 72)]
    instruction_pattern = r"\b(?:instruction|instructions|step|steps|cooking\s+steps?)\b"
    mention_pattern = r"\b(?:mention|mentions|mentioned|contain|contains|containing|include|includes|including|with|in)\b"

    instruction_matches = list(re.finditer(instruction_pattern, prefix))
    if instruction_matches:
        after_instruction = prefix[instruction_matches[-1].end():]
        local_tokens = re.findall(r"[a-z0-9]+", after_instruction)
        if len(local_tokens) <= 10 and re.search(mention_pattern, after_instruction):
            return True
    if re.search(
        rf"{instruction_pattern}\s*$",
        prefix,
    ):
        return True
    if re.match(
        rf"^\s*(?:in|inside|within|from|of)\s+(?:the\s+)?(?:{instruction_pattern})\b",
        suffix,
    ):
        return True
    return False


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
    if (
        not order_ops
        and not _contains_config_term(text, recipe_config["sort_cues"])
        and not re.search(r"\b(?:quickest|fastest|shortest)\b", text)
    ):
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
    match = re.search(
        r"\b(?:the\s+)?(\d+)\s+"
        r"(?:highest|lowest|quickest|fastest|slowest|most|least|best|top)\b",
        transcript,
        re.IGNORECASE,
    )
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
