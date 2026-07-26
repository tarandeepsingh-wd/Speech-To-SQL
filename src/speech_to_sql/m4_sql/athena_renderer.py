from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any, Mapping

from speech_to_sql.common.config import load_sql_schema_config


_BACKTICK_IDENTIFIER_RE = re.compile(r"`([^`]+)`")
_ALIAS_RE = re.compile(
    r'\b(?:FROM|JOIN)\s+"(?P<table>[^"]+)"\s+AS\s+(?P<alias>[A-Za-z_][A-Za-z0-9_]*)',
    re.IGNORECASE,
)
_QUALIFIED_IDENTIFIER_RE = re.compile(
    r'(?P<alias>\b[A-Za-z_][A-Za-z0-9_]*)\."(?P<column>[^"]+)"'
)
_KEY_FIELD_NAMES = frozenset({"_id", "id", "recipe_id", "recipe_no", "ing_id", "ingid", "ndb_id"})
_KEY_JOIN_RE = re.compile(
    r'(?P<left>\b[A-Za-z_][A-Za-z0-9_]*\."(?:_id|id|recipe_id|recipe_no|ing_id|ingid|ndb_id)")'
    r"\s*=\s*"
    r'(?P<right>\b[A-Za-z_][A-Za-z0-9_]*\."(?:_id|id|recipe_id|recipe_no|ing_id|ingid|ndb_id)")'
)
_LEGACY_TEXT_DOUBLE_CAST_RE = re.compile(
    r"CAST\s*\(\s*NULLIF\s*\(\s*(?P<expr>\b[A-Za-z_][A-Za-z0-9_]*\.\"[^\"]+\")\s*,\s*''\s*\)"
    r"\s+AS\s+DOUBLE\s*\)",
    re.IGNORECASE,
)
_NUMERIC_LITERAL = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
_SIMPLE_NUMERIC_COMPARISON_RE = re.compile(
    rf'(?P<expr>\b[A-Za-z_][A-Za-z0-9_]*\."[^"]+")'
    rf"\s*(?P<op><=|>=|<>|!=|<|>|=)\s*(?P<value>{_NUMERIC_LITERAL})(?![\w.])"
)
_SIMPLE_NUMERIC_BETWEEN_RE = re.compile(
    rf'(?P<expr>\b[A-Za-z_][A-Za-z0-9_]*\."[^"]+")'
    rf"\s+BETWEEN\s+(?P<low>{_NUMERIC_LITERAL})\s+AND\s+(?P<high>{_NUMERIC_LITERAL})(?![\w.])",
    re.IGNORECASE,
)
_ORDER_SUBQUERY_SELECT_RE = re.compile(
    r'\(SELECT\s+(?P<expr>\b[A-Za-z_][A-Za-z0-9_]*\."[^"]+")\s+FROM\s+',
    re.IGNORECASE,
)
_ORDER_BY_SIMPLE_RE = re.compile(
    r'ORDER\s+BY\s+(?P<expr>\b[A-Za-z_][A-Za-z0-9_]*\."[^"]+")\s+(?P<direction>ASC|DESC)\b',
    re.IGNORECASE,
)
_TRUTHY_FLAG_IN_RE = re.compile(
    r'(?P<expr>\b[A-Za-z_][A-Za-z0-9_]*\."[^"]+")\s+IN\s*\((?P<values>[^()]*)\)',
    re.IGNORECASE,
)
_TRUTHY_FLAG_NOT_IN_RE = re.compile(
    r'\((?P<expr>\b[A-Za-z_][A-Za-z0-9_]*\."[^"]+")\s+NOT\s+IN\s*\((?P<values>[^()]*)\)'
    r'\s+OR\s+(?P=expr)\s+IS\s+NULL\)',
    re.IGNORECASE,
)
_NUMERIC_TYPE_NAMES = frozenset(
    {
        "bigint",
        "decimal",
        "double",
        "float",
        "int",
        "integer",
        "numeric",
        "real",
        "smallint",
        "tinyint",
    }
)
_DEFAULT_TRUTHY_FLAG_COLUMNS = frozenset(
    {
        "lacto_vegetarian",
        "ovo_lacto_vegetarian",
        "ovo_vegetarian",
        "pescetarian",
        "vegan",
    }
)
_NUMERIC_COLUMN_HINTS = (
    "calorie",
    "carbohydrate",
    "cook_time",
    "energy",
    "fat",
    "fiber",
    "fibre",
    "gram",
    "kcal",
    "lipid",
    "mg",
    "prep_time",
    "protein",
    "sodium",
    "sugar",
    "time",
    "total_time",
    "vitamin",
)


def render_athena_sql(sql_result: Mapping[str, Any]) -> dict[str, Any]:
    """Return an Athena-ready full SQL query without changing normal app SQL."""
    query = sql_result.get("query")
    if not query:
        return {
            "query": None,
            "dialect": "athena",
            "params": list(sql_result.get("params") or []),
            "parameterized_query": query,
        }
    schema = load_sql_schema_config()
    rendered_query = render_athena_query(
        str(query),
        list(sql_result.get("params") or []),
        schema=schema,
    )
    rendered_query = _apply_dish_title_boost(
        rendered_query,
        sql_result.get("dish_title_boost"),
        schema,
    )
    return {
        "query": rendered_query,
        "dialect": "athena",
        "params": list(sql_result.get("params") or []),
        "parameterized_query": query,
    }


def render_athena_query(
    parameterized_query: str,
    params: list[Any] | tuple[Any, ...],
    *,
    schema: Mapping[str, Any] | None = None,
) -> str:
    """Convert qmark SQL into paste-ready AWS Athena SQL."""
    schema = schema or load_sql_schema_config()
    sql = _quote_backtick_identifiers(parameterized_query)
    aliases = _alias_to_table(sql)
    sql = _rewrite_known_column_aliases(sql, aliases, schema)
    sql = _rewrite_text_numeric_casts(sql)
    sql = _cast_key_joins(sql)
    sql = _inline_params(sql, list(params))
    sql = _cast_numeric_between(sql, aliases, schema)
    sql = _cast_numeric_comparisons(sql, aliases, schema)
    sql = _cast_numeric_order_subqueries(sql, aliases, schema)
    sql = _cast_simple_numeric_order_by(sql, aliases, schema)
    sql = _rewrite_truthy_flag_predicates(sql, schema)
    sql = _project_distinct_order_by(sql)
    return sql


def _quote_backtick_identifiers(sql: str) -> str:
    return _BACKTICK_IDENTIFIER_RE.sub(
        lambda match: '"' + match.group(1).replace('"', '""') + '"',
        sql,
    )


def _alias_to_table(sql: str) -> dict[str, str]:
    return {
        match.group("alias"): match.group("table")
        for match in _ALIAS_RE.finditer(sql)
    }


def _rewrite_known_column_aliases(
    sql: str,
    aliases: Mapping[str, str],
    schema: Mapping[str, Any],
) -> str:
    table_maps = {
        alias: _column_alias_map(table, schema)
        for alias, table in aliases.items()
    }

    def replace(match: re.Match[str]) -> str:
        alias = match.group("alias")
        column = match.group("column")
        replacement = table_maps.get(alias, {}).get(_column_key(column))
        if not replacement or replacement == column:
            return match.group(0)
        return f'{alias}."{replacement}"'

    return _QUALIFIED_IDENTIFIER_RE.sub(replace, sql)


def _column_alias_map(table: str, schema: Mapping[str, Any]) -> dict[str, str]:
    fields = (
        schema.get("collections", {})
        .get(table, {})
        .get("fields", {})
    )
    if not isinstance(fields, Mapping):
        fields = {}
    field_lookup = {_column_key(field): str(field) for field in fields}
    aliases = dict(field_lookup)

    def add_alias(alias: Any, target: Any) -> None:
        alias_text = str(alias or "").strip()
        target_text = str(target or "").strip()
        if not alias_text or not target_text:
            return
        actual = field_lookup.get(_column_key(target_text))
        if actual:
            aliases.setdefault(_column_key(alias_text), actual)

    if table == str((schema.get("base_table") or {}).get("name") or ""):
        base_columns = schema.get("base_columns") or {}
        for logical, column in base_columns.items():
            add_alias(logical, column)
        for alias, logical in ((schema.get("semantic_aliases") or {}).get("base") or {}).items():
            add_alias(alias, base_columns.get(str(logical), logical))

    for relation in (schema.get("relations") or {}).values():
        if not isinstance(relation, Mapping) or relation.get("table") != table:
            continue
        relation_columns = relation.get("columns") or {}
        for logical, column in relation_columns.items():
            add_alias(logical, column)

        source = str(_relation_source_for_table(schema, relation) or "")
        semantic_aliases = (schema.get("semantic_aliases") or {}).get(source) or {}
        for alias, logical in semantic_aliases.items():
            add_alias(alias, relation_columns.get(str(logical), logical))

    if "recipe_title" in fields:
        add_alias("title", "recipe_title")
        add_alias("name", "recipe_title")
        add_alias("recipe name", "recipe_title")

    return aliases


def _relation_source_for_table(
    schema: Mapping[str, Any],
    relation: Mapping[str, Any],
) -> str | None:
    for source, candidate in (schema.get("relations") or {}).items():
        if candidate is relation:
            return str(source)
    table = relation.get("table")
    for source, candidate in (schema.get("relations") or {}).items():
        if isinstance(candidate, Mapping) and candidate.get("table") == table:
            return str(source)
    return None


def _column_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _rewrite_text_numeric_casts(sql: str) -> str:
    return _LEGACY_TEXT_DOUBLE_CAST_RE.sub(
        lambda match: _safe_numeric_expression(match.group("expr")),
        sql,
    )


def _cast_key_joins(sql: str) -> str:
    return _KEY_JOIN_RE.sub(
        lambda match: (
            f"CAST({match.group('left')} AS VARCHAR) = "
            f"CAST({match.group('right')} AS VARCHAR)"
        ),
        sql,
    )


def _inline_params(sql: str, params: list[Any]) -> str:
    parts = sql.split("?")
    expected = len(parts) - 1
    if expected != len(params):
        raise ValueError(
            f"SQL placeholder count ({expected}) does not match params count ({len(params)})."
        )
    rendered = [parts[0]]
    for value, tail in zip(params, parts[1:]):
        rendered.append(_literal(value))
        rendered.append(tail)
    return "".join(rendered)


def _literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Cannot render non-finite numeric literal: {value!r}")
        if numeric.is_integer():
            return str(int(numeric))
        return repr(numeric)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _cast_numeric_between(
    sql: str,
    aliases: Mapping[str, str],
    schema: Mapping[str, Any],
) -> str:
    def replace(match: re.Match[str]) -> str:
        expr = match.group("expr")
        if not _should_cast_numeric_expression(expr, aliases, schema):
            return match.group(0)
        return (
            f"{_safe_numeric_expression(expr)} BETWEEN "
            f"{match.group('low')} AND {match.group('high')}"
        )

    return _SIMPLE_NUMERIC_BETWEEN_RE.sub(replace, sql)


def _cast_numeric_comparisons(
    sql: str,
    aliases: Mapping[str, str],
    schema: Mapping[str, Any],
) -> str:
    def replace(match: re.Match[str]) -> str:
        expr = match.group("expr")
        if not _should_cast_numeric_expression(expr, aliases, schema):
            return match.group(0)
        return f"{_safe_numeric_expression(expr)} {match.group('op')} {match.group('value')}"

    return _SIMPLE_NUMERIC_COMPARISON_RE.sub(replace, sql)


def _cast_numeric_order_subqueries(
    sql: str,
    aliases: Mapping[str, str],
    schema: Mapping[str, Any],
) -> str:
    def replace(match: re.Match[str]) -> str:
        expr = match.group("expr")
        if not _should_cast_numeric_expression(expr, aliases, schema):
            return match.group(0)
        return f"(SELECT {_safe_numeric_expression(expr)} FROM "

    return _ORDER_SUBQUERY_SELECT_RE.sub(replace, sql)


def _cast_simple_numeric_order_by(
    sql: str,
    aliases: Mapping[str, str],
    schema: Mapping[str, Any],
) -> str:
    def replace(match: re.Match[str]) -> str:
        expr = match.group("expr")
        if not _should_cast_numeric_expression(expr, aliases, schema):
            return match.group(0)
        return f"ORDER BY {_safe_numeric_expression(expr)} {match.group('direction')}"

    return _ORDER_BY_SIMPLE_RE.sub(replace, sql)


def _rewrite_truthy_flag_predicates(sql: str, schema: Mapping[str, Any]) -> str:
    flag_columns = _truthy_flag_columns(schema)

    def replace_not_in(match: re.Match[str]) -> str:
        expr = match.group("expr")
        if not _is_truthy_flag_expression(expr, flag_columns):
            return match.group(0)
        return _safe_not_truthy_expression(expr)

    def replace_in(match: re.Match[str]) -> str:
        expr = match.group("expr")
        if not _is_truthy_flag_expression(expr, flag_columns):
            return match.group(0)
        return _safe_truthy_expression(expr)

    sql = _TRUTHY_FLAG_NOT_IN_RE.sub(replace_not_in, sql)
    return _TRUTHY_FLAG_IN_RE.sub(replace_in, sql)


def _project_distinct_order_by(sql: str) -> str:
    stripped = sql.lstrip()
    leading = len(sql) - len(stripped)
    if not stripped.upper().startswith("SELECT DISTINCT "):
        return sql

    order_index = _find_top_level_keyword(sql, "ORDER BY")
    if order_index < 0:
        return sql

    from_index = _find_top_level_keyword(sql, "FROM", start=leading + len("SELECT DISTINCT "))
    if from_index < 0 or from_index > order_index:
        return sql

    suffix_index = _find_first_top_level_keyword(sql, ("LIMIT", "OFFSET", "FETCH"), start=order_index)
    if suffix_index < 0:
        suffix_index = len(sql)

    order_clause = sql[order_index + len("ORDER BY"):suffix_index].strip()
    order_items = _split_top_level_commas(order_clause)
    if not order_items:
        return sql

    projection_items: list[str] = []
    rewritten_order_items: list[str] = []
    for index, item in enumerate(order_items, start=1):
        expression, direction = _split_order_direction(item)
        projected_expression = _aggregate_scalar_order_expression(expression, direction)
        alias = f"__athena_order_{index}"
        projection_items.append(f'{projected_expression} AS "{alias}"')
        direction_sql = f" {direction}" if direction else ""
        rewritten_order_items.append(f'"{alias}"{direction_sql}')

    select_prefix = sql[:from_index].rstrip()
    before_order = sql[from_index:order_index].rstrip()
    suffix = sql[suffix_index:]
    suffix_sql = f" {suffix.lstrip()}" if suffix else ""
    return (
        f"{select_prefix}, {', '.join(projection_items)} "
        f"{before_order} ORDER BY {', '.join(rewritten_order_items)}{suffix_sql}"
    )


def _apply_dish_title_boost(
    sql: str,
    boost: Any,
    schema: Mapping[str, Any],
) -> str:
    if not isinstance(boost, Mapping):
        return sql
    phrase = _normalize_boost_text(boost.get("phrase"))
    tokens = _boost_tokens(boost, phrase)
    if len(tokens) < 2 or not phrase:
        return sql
    if _find_top_level_keyword(sql, "ORDER BY") >= 0:
        return sql
    if sql.lstrip().upper().startswith("WITH "):
        return sql

    base_table = str((schema.get("base_table") or {}).get("name") or "recipes")
    title_column = _base_title_column(schema)
    if not title_column:
        return sql
    if not re.search(
        rf'\bFROM\s+"{re.escape(base_table)}"\s+AS\s+r\b',
        sql,
        re.IGNORECASE,
    ):
        return sql

    query_body, suffix = _split_top_level_suffix(sql, ("LIMIT", "OFFSET", "FETCH"))
    title_expr = (
        f'TRIM(REGEXP_REPLACE(LOWER(COALESCE(CAST(r."{title_column}" AS VARCHAR), \'\')), '
        f"'[^a-z0-9]+', ' '))"
    )
    token_checks = " AND ".join(
        f"{title_expr} LIKE {_literal('%' + token + '%')}"
        for token in tokens
    )
    order_sql = (
        "CASE "
        f"WHEN {title_expr} = {_literal(phrase)} THEN 0 "
        f"WHEN {title_expr} LIKE {_literal('%' + phrase + '%')} THEN 1 "
        f"WHEN {token_checks} THEN 2 "
        "ELSE 3 END ASC, "
        f"{title_expr} ASC"
    )
    suffix_sql = f" {suffix.strip()}" if suffix else ""
    return (
        f"WITH candidate_recipes AS ({query_body.strip()}) "
        f"SELECT * FROM candidate_recipes AS r "
        f"ORDER BY {order_sql}{suffix_sql}"
    )


def _base_title_column(schema: Mapping[str, Any]) -> str | None:
    base_table = str((schema.get("base_table") or {}).get("name") or "")
    fields = (
        schema.get("collections", {})
        .get(base_table, {})
        .get("fields", {})
    )
    if not isinstance(fields, Mapping):
        return None
    for candidate in ("recipe_title", "title"):
        if candidate in fields:
            return candidate
    return None


def _boost_tokens(boost: Mapping[str, Any], phrase: str) -> list[str]:
    source = boost.get("tokens")
    tokens = source if isinstance(source, (list, tuple)) else phrase.split()
    result: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalized = _normalize_boost_text(token)
        if not normalized:
            continue
        for part in normalized.split():
            if part and part not in seen:
                seen.add(part)
                result.append(part)
    return result


def _normalize_boost_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _split_top_level_suffix(sql: str, keywords: tuple[str, ...]) -> tuple[str, str]:
    suffix_index = _find_first_top_level_keyword(sql, keywords)
    if suffix_index < 0:
        return sql, ""
    return sql[:suffix_index].rstrip(), sql[suffix_index:].strip()


def _aggregate_scalar_order_expression(expression: str, direction: str) -> str:
    inner = _strip_balanced_outer_parentheses(expression.strip())
    if not inner.upper().startswith("SELECT "):
        return expression.strip()

    from_index = _find_top_level_keyword(inner, "FROM", start=len("SELECT "))
    if from_index < 0:
        return expression.strip()

    select_expression = inner[len("SELECT "):from_index].strip()
    if re.match(r"^(?:AVG|COUNT|MAX|MIN|SUM)\s*\(", select_expression, re.IGNORECASE):
        return f"({inner})"

    aggregate = "MAX" if direction.upper() == "DESC" else "MIN"
    rest = inner[from_index:].strip()
    return f"(SELECT {aggregate}({select_expression}) {rest})"


def _split_order_direction(order_item: str) -> tuple[str, str]:
    item = order_item.strip()
    match = re.search(r"\s+(ASC|DESC)\s*$", item, re.IGNORECASE)
    if not match:
        return item, ""
    return item[:match.start()].strip(), match.group(1).upper()


def _split_top_level_commas(text: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth = 0
    in_single = False
    in_double = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_single:
            if char == "'" and index + 1 < len(text) and text[index + 1] == "'":
                index += 2
                continue
            if char == "'":
                in_single = False
            index += 1
            continue
        if in_double:
            if char == '"' and index + 1 < len(text) and text[index + 1] == '"':
                index += 2
                continue
            if char == '"':
                in_double = False
            index += 1
            continue
        if char == "'":
            in_single = True
        elif char == '"':
            in_double = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            item = text[start:index].strip()
            if item:
                items.append(item)
            start = index + 1
        index += 1

    tail = text[start:].strip()
    if tail:
        items.append(tail)
    return items


def _strip_balanced_outer_parentheses(text: str) -> str:
    stripped = text.strip()
    while stripped.startswith("(") and stripped.endswith(")"):
        close_index = _matching_outer_close_index(stripped)
        if close_index != len(stripped) - 1:
            break
        stripped = stripped[1:-1].strip()
    return stripped


def _matching_outer_close_index(text: str) -> int:
    depth = 0
    in_single = False
    in_double = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_single:
            if char == "'" and index + 1 < len(text) and text[index + 1] == "'":
                index += 2
                continue
            if char == "'":
                in_single = False
            index += 1
            continue
        if in_double:
            if char == '"' and index + 1 < len(text) and text[index + 1] == '"':
                index += 2
                continue
            if char == '"':
                in_double = False
            index += 1
            continue
        if char == "'":
            in_single = True
        elif char == '"':
            in_double = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _find_first_top_level_keyword(sql: str, keywords: tuple[str, ...], *, start: int = 0) -> int:
    matches = [
        index
        for keyword in keywords
        if (index := _find_top_level_keyword(sql, keyword, start=start)) >= 0
    ]
    return min(matches) if matches else -1


def _find_top_level_keyword(sql: str, keyword: str, *, start: int = 0) -> int:
    lower_sql = sql.lower()
    lower_keyword = keyword.lower()
    depth = 0
    in_single = False
    in_double = False
    index = max(0, start)
    while index < len(sql):
        char = sql[index]
        if in_single:
            if char == "'" and index + 1 < len(sql) and sql[index + 1] == "'":
                index += 2
                continue
            if char == "'":
                in_single = False
            index += 1
            continue
        if in_double:
            if char == '"' and index + 1 < len(sql) and sql[index + 1] == '"':
                index += 2
                continue
            if char == '"':
                in_double = False
            index += 1
            continue
        if char == "'":
            in_single = True
        elif char == '"':
            in_double = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif (
            depth == 0
            and lower_sql.startswith(lower_keyword, index)
            and _keyword_boundary(sql, index - 1)
            and _keyword_boundary(sql, index + len(keyword))
        ):
            return index
        index += 1
    return -1


def _keyword_boundary(sql: str, index: int) -> bool:
    if index < 0 or index >= len(sql):
        return True
    return not (sql[index].isalnum() or sql[index] == "_")


def _truthy_flag_columns(schema: Mapping[str, Any]) -> frozenset[str]:
    columns = set(_DEFAULT_TRUTHY_FLAG_COLUMNS)
    relations = schema.get("relations", {})
    for relation_name, relation in relations.items():
        if relation_name != "dietary" and relation.get("predicate_mode") != "truthy_flag":
            continue
        columns.update(str(column) for column in relation.get("columns", {}).values())
    return frozenset(columns)


def _is_truthy_flag_expression(expr: str, flag_columns: frozenset[str]) -> bool:
    parsed = _parse_qualified_identifier(expr)
    if not parsed:
        return False
    return parsed[1] in flag_columns


def _safe_truthy_expression(expr: str) -> str:
    text_expr = f"LOWER(TRIM(CAST({expr} AS VARCHAR)))"
    return f"({_safe_numeric_expression(expr)} = 1.0 OR {text_expr} IN ('true', 'yes', 'y'))"


def _safe_not_truthy_expression(expr: str) -> str:
    text_expr = f"LOWER(TRIM(CAST({expr} AS VARCHAR)))"
    numeric_not_true = f"COALESCE({_safe_numeric_expression(expr)} <> 1.0, TRUE)"
    text_not_true = f"COALESCE({text_expr} NOT IN ('true', 'yes', 'y'), TRUE)"
    return f"({expr} IS NULL OR ({numeric_not_true} AND {text_not_true}))"


def _safe_numeric_expression(expr: str) -> str:
    return f"TRY_CAST(NULLIF(TRIM(CAST({expr} AS VARCHAR)), '') AS DOUBLE)"


def _should_cast_numeric_expression(
    expr: str,
    aliases: Mapping[str, str],
    schema: Mapping[str, Any],
) -> bool:
    parsed = _parse_qualified_identifier(expr)
    if not parsed:
        return False
    alias, column = parsed
    if column.lower() in _KEY_FIELD_NAMES:
        return False

    table = aliases.get(alias)
    field_type = (
        schema.get("collections", {})
        .get(table or "", {})
        .get("fields", {})
        .get(column)
    )
    if str(field_type).lower() in _NUMERIC_TYPE_NAMES:
        return True

    lowered_column = column.lower()
    return any(hint in lowered_column for hint in _NUMERIC_COLUMN_HINTS)


def _parse_qualified_identifier(expr: str) -> tuple[str, str] | None:
    match = re.fullmatch(r'(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\."(?P<column>[^"]+)"', expr)
    if not match:
        return None
    return match.group("alias"), match.group("column")
