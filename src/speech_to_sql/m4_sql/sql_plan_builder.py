from __future__ import annotations

from typing import Any

from speech_to_sql.m3_query_ir.schema_registry import SchemaRegistry, load_schema_registry
from speech_to_sql.m3_query_ir.sql_ir import Predicate, QueryIR, SelectExpr
from speech_to_sql.m4_sql.sql_plan import SqlCondition, SqlJoin, SqlOrder, SqlPlan


def build_sql_plan(
    query_ir: QueryIR,
    *,
    registry: SchemaRegistry | None = None,
) -> SqlPlan:
    registry = registry or load_schema_registry()
    root = registry.root(query_ir.root)
    plan = SqlPlan(
        select=[_compile_select(item) for item in query_ir.select],
        from_table=registry.table_sql(root["table"]),
        from_alias=root["alias"],
        root=query_ir.root,
        root_source=registry.root_source(query_ir.root),
        limit=query_ir.limit,
    )

    relation_counts: dict[str, int] = {}
    planned_predicates = _add_scoped_ingredient_child_strategies(query_ir, registry, plan)
    planned_predicates.update(_add_multi_join_strategies(
        query_ir,
        registry,
        plan,
        planned_predicates,
    ))
    join_strategy_sources = _join_strategy_sources(query_ir, registry, planned_predicates)
    for predicate in query_ir.predicates:
        if id(predicate) in planned_predicates:
            continue
        condition, assumption = _predicate_condition(
            predicate,
            registry,
            relation_counts,
            plan,
            join_strategy_sources,
        )
        if condition:
            plan.where.append(condition)
        if assumption and assumption not in plan.join_assumptions:
            plan.join_assumptions.append(assumption)

    for order in query_ir.order_by:
        expression = _order_expression(order.field, registry)
        if not expression:
            continue
        direction = "DESC" if order.direction.upper() == "DESC" else "ASC"
        plan.order_by.append(SqlOrder(expression, direction))

    return plan


def _compile_select(select: SelectExpr) -> str:
    if select.alias:
        return f"{select.expression} AS {select.alias}"
    return select.expression


def _predicate_condition(
    predicate: Predicate,
    registry: SchemaRegistry,
    relation_counts: dict[str, int],
    plan: SqlPlan,
    join_strategy_sources: set[str],
) -> tuple[SqlCondition | None, dict[str, Any] | None]:
    if predicate.source == plan.root_source:
        return _root_source_condition(predicate, registry, plan), None
    if predicate.source == "base":
        return _base_condition(predicate, registry), None
    if not registry.relation(predicate.source):
        return None, None
    if registry.relation_is_base(predicate.source):
        return _same_table_relation_condition(predicate, registry), None
    if predicate.source in join_strategy_sources:
        return _join_relation_condition(predicate, registry, plan)
    return _exists_relation_condition(predicate, registry, relation_counts)


def _base_condition(
    predicate: Predicate,
    registry: SchemaRegistry,
) -> SqlCondition | None:
    column = registry.base_column(predicate.field)
    if not column:
        return None
    return _column_condition(
        registry.column_sql(registry.base_alias, column),
        predicate,
        registry,
    )


def _root_source_condition(
    predicate: Predicate,
    registry: SchemaRegistry,
    plan: SqlPlan,
) -> SqlCondition | None:
    if predicate.source == "base":
        column = registry.base_column(predicate.field)
    else:
        column = registry.predicate_column(predicate)
    if not column:
        return None
    relation = registry.relation(predicate.source) or {}
    if _uses_relation_match_strategy(predicate, registry, relation):
        return _relation_match_condition(plan.from_alias, predicate, registry, relation)
    filter_predicate = _normalized_predicate(predicate, relation)
    return _column_condition(
        registry.column_sql(plan.from_alias, column),
        filter_predicate,
        registry,
    )


def _same_table_relation_condition(
    predicate: Predicate,
    registry: SchemaRegistry,
) -> SqlCondition | None:
    relation = registry.relation(predicate.source) or {}
    if predicate.source == "nutrition" and registry.nutrition_group(predicate.field):
        return _nutrition_group_condition(predicate, registry, relation)
    if predicate.source == "dietary" and predicate.quantifier == "or_exists":
        return _dietary_or_condition(predicate, registry, relation)

    column = registry.predicate_column(predicate)
    if not column:
        return None
    column_sql = registry.column_sql(registry.base_alias, column)
    if relation.get("predicate_mode") == "truthy_flag" or predicate.source == "dietary":
        values = tuple(registry.truthy_values(predicate.source))
        placeholders = ", ".join("?" for _ in values)
        if predicate.quantifier == "not_exists":
            return SqlCondition(
                f"({column_sql} NOT IN ({placeholders}) OR {column_sql} IS NULL)",
                values,
                predicate.source,
            )
        return SqlCondition(
            f"{column_sql} IN ({placeholders})",
            values,
            predicate.source,
        )
    return _column_condition(column_sql, predicate, registry)


def _nutrition_group_condition(
    predicate: Predicate,
    registry: SchemaRegistry,
    relation: dict[str, Any],
    alias: str | None = None,
) -> SqlCondition | None:
    group = registry.nutrition_group(predicate.field)
    if not group:
        return None
    operator = predicate.operator.upper()
    values = tuple(predicate.values)
    if operator != "ANY_GT" or len(values) != 1:
        return None

    columns = relation.get("columns", {})
    parts: list[str] = []
    params: list[Any] = []
    for field in group.get("fields", []):
        column = columns.get(str(field))
        if not column:
            return None
        parts.append(f"{registry.column_sql(alias or registry.base_alias, column)} > ?")
        params.append(values[0])
    if not parts:
        return None
    return SqlCondition("(" + " OR ".join(parts) + ")", tuple(params), predicate.source)


def _dietary_or_condition(
    predicate: Predicate,
    registry: SchemaRegistry,
    relation: dict[str, Any],
) -> SqlCondition | None:
    columns = relation.get("columns", {})
    truthy_values = tuple(registry.truthy_values(predicate.source))
    if not truthy_values:
        return None

    parts: list[str] = []
    params: list[Any] = []
    placeholders = ", ".join("?" for _ in truthy_values)
    for value in predicate.values:
        column = columns.get(str(value))
        if not column:
            return None
        parts.append(f"{registry.column_sql(registry.base_alias, column)} IN ({placeholders})")
        params.extend(truthy_values)

    if not parts:
        return None
    return SqlCondition("(" + " OR ".join(parts) + ")", tuple(params), predicate.source)


def _exists_relation_condition(
    predicate: Predicate,
    registry: SchemaRegistry,
    relation_counts: dict[str, int],
) -> tuple[SqlCondition | None, dict[str, Any] | None]:
    relation = registry.relation(predicate.source)
    if not relation:
        return None, None

    relation_counts[predicate.source] = relation_counts.get(predicate.source, 0) + 1
    alias_prefix = (
        f"{relation['alias']}_exclude"
        if predicate.quantifier == "not_exists"
        else relation["alias"]
    )
    alias = f"{alias_prefix}{relation_counts[predicate.source]}"
    if predicate.source == "nutrition" and registry.nutrition_group(predicate.field):
        filter_condition = _nutrition_group_condition(
            predicate,
            registry,
            relation,
            alias=alias,
        )
    else:
        column = registry.predicate_column(predicate)
        if not column:
            return None, None
        filter_condition = _relation_match_condition(alias, predicate, registry, relation)
    if not filter_condition:
        return None, None

    join_condition = registry.join_condition(predicate.source, alias)
    join = registry.relationship_for_relation(predicate.source)
    join_assumption = join.as_dict() if join else None
    if not join_condition:
        chained = _chained_exists_relation_condition(
            predicate,
            registry,
            relation,
            alias,
            filter_condition,
        )
        if chained:
            return chained
        return None, join_assumption

    exists_sql = (
        f"SELECT 1 FROM {registry.table_sql(relation['table'])} AS {alias} "
        f"WHERE {join_condition} AND {filter_condition.sql}"
    )
    condition_sql = (
        f"NOT EXISTS ({exists_sql})"
        if predicate.quantifier == "not_exists"
        else f"EXISTS ({exists_sql})"
    )
    return SqlCondition(condition_sql, filter_condition.params, predicate.source), join_assumption


def _chained_exists_relation_condition(
    predicate: Predicate,
    registry: SchemaRegistry,
    relation: dict[str, Any],
    alias: str,
    filter_condition: SqlCondition,
) -> tuple[SqlCondition, dict[str, Any]] | None:
    parent_source = relation.get("requires")
    if not parent_source:
        return None
    parent_relation = registry.relation(str(parent_source))
    parent_join = registry.relationship_for_relation(str(parent_source))
    child_join = registry.relationship_for_relation(predicate.source)
    if not parent_relation or not parent_join or not child_join:
        return None
    if not registry.relationship_is_direct_to_base(str(parent_source)):
        return None

    parent_alias = f"{parent_relation['alias']}_chain"
    chain_condition = _relationship_condition_between_aliases(
        child_join,
        parent_relation["table"],
        parent_alias,
        relation["table"],
        alias,
        registry,
    )
    if not chain_condition:
        return None

    parent_base_condition = (
        f"{parent_alias}.{registry.quote_identifier(parent_join.from_column)} = "
        f"{registry.base_alias}.{registry.quote_identifier(parent_join.to_column)}"
    )
    exists_sql = (
        f"SELECT 1 FROM {registry.table_sql(parent_relation['table'])} AS {parent_alias} "
        f"JOIN {registry.table_sql(relation['table'])} AS {alias} "
        f"ON {chain_condition} "
        f"WHERE {parent_base_condition} AND {filter_condition.sql}"
    )
    condition_sql = (
        f"NOT EXISTS ({exists_sql})"
        if predicate.quantifier == "not_exists"
        else f"EXISTS ({exists_sql})"
    )
    return SqlCondition(condition_sql, filter_condition.params, predicate.source), {
        "name": f"{parent_join.name}->{child_join.name}",
        "path": [parent_join.as_dict(), child_join.as_dict()],
        "confidence": _path_confidence(parent_join.confidence, child_join.confidence),
    }


def _relationship_condition_between_aliases(
    join: Any,
    left_table: str,
    left_alias: str,
    right_table: str,
    right_alias: str,
    registry: SchemaRegistry,
) -> str | None:
    if join.from_table == left_table and join.to_table == right_table:
        return (
            f"{left_alias}.{registry.quote_identifier(join.from_column)} = "
            f"{right_alias}.{registry.quote_identifier(join.to_column)}"
        )
    if join.from_table == right_table and join.to_table == left_table:
        return (
            f"{right_alias}.{registry.quote_identifier(join.from_column)} = "
            f"{left_alias}.{registry.quote_identifier(join.to_column)}"
        )
    return None


def _path_confidence(*values: str) -> str:
    if any(value == "inferred" for value in values):
        return "inferred"
    if any(value == "index_supported" for value in values):
        return "index_supported"
    return values[0] if values else "unknown"


def _join_relation_condition(
    predicate: Predicate,
    registry: SchemaRegistry,
    plan: SqlPlan,
) -> tuple[SqlCondition | None, dict[str, Any] | None]:
    relation = registry.relation(predicate.source)
    if not relation:
        return None, None

    alias = relation["alias"]
    column = registry.predicate_column(predicate)
    if not column:
        return None, None

    filter_condition = _relation_match_condition(alias, predicate, registry, relation)
    if not filter_condition:
        return None, None

    join_condition = registry.join_condition(predicate.source, alias)
    join = registry.relationship_for_relation(predicate.source)
    join_assumption = join.as_dict() if join else None
    if not join_condition:
        return None, join_assumption

    if not any(existing.source == predicate.source for existing in plan.joins):
        plan.joins.append(SqlJoin(
            table=registry.table_sql(relation["table"]),
            alias=alias,
            on=join_condition,
            join_type=str(registry.relation_strategy(predicate.source).get("join_type") or "JOIN"),
            source=predicate.source,
        ))
        plan.distinct = bool(registry.relation_strategy(predicate.source).get("distinct", True))
    return filter_condition, join_assumption


def _join_strategy_sources(
    query_ir: QueryIR,
    registry: SchemaRegistry,
    skipped_predicates: set[int],
) -> set[str]:
    sources: set[str] = set()
    by_source: dict[str, list[Predicate]] = {}
    for predicate in query_ir.predicates:
        if id(predicate) in skipped_predicates:
            continue
        by_source.setdefault(predicate.source, []).append(predicate)

    for source, predicates in by_source.items():
        strategy = registry.relation_strategy(source)
        if strategy.get("simple_include") != "join":
            continue
        if len(predicates) != int(strategy.get("max_join_predicates", 1)):
            continue
        predicate = predicates[0]
        if predicate.quantifier != strategy.get("quantifier", "exists"):
            continue
        if predicate.operator.upper() != strategy.get("operator", "="):
            continue
        if len(predicate.values) != int(strategy.get("value_count", 1)):
            continue
        relation = registry.relation(source) or {}
        if not _uses_relation_match_strategy(predicate, registry, relation):
            continue
        sources.add(source)
    return sources


def _add_multi_join_strategies(
    query_ir: QueryIR,
    registry: SchemaRegistry,
    plan: SqlPlan,
    skipped_predicates: set[int] | None = None,
) -> set[int]:
    skipped: set[int] = set()
    skipped_predicates = skipped_predicates or set()
    if plan.root_source != "base":
        return skipped
    by_source: dict[str, list[Predicate]] = {}
    for predicate in query_ir.predicates:
        if id(predicate) in skipped_predicates:
            continue
        by_source.setdefault(predicate.source, []).append(predicate)

    for source, predicates in by_source.items():
        strategy = registry.relation_strategy(source)
        relation = registry.relation(source) or {}
        include_predicates = [
            predicate for predicate in predicates
            if predicate.quantifier == "exists"
            and predicate.operator.upper() == strategy.get("operator", "=")
            and len(predicate.values) == int(strategy.get("value_count", 1))
            and _uses_relation_match_strategy(predicate, registry, relation)
        ]
        if len(include_predicates) < 2:
            continue
        strategy_name = str(strategy.get("multi_include") or "")
        max_and_joins = int(strategy.get("max_and_join_predicates", 0) or 0)
        if strategy_name == "and_joins" and len(include_predicates) <= max_and_joins:
            join_assumption = _add_and_joins(source, include_predicates, registry, plan)
        elif strategy.get("fallback_multi_include") == "grouped_join" or strategy_name == "grouped_join":
            join_assumption = _add_grouped_join(source, include_predicates, registry, plan)
        else:
            continue
        if join_assumption and join_assumption not in plan.join_assumptions:
            plan.join_assumptions.append(join_assumption)
        skipped.update(id(predicate) for predicate in include_predicates)
    return skipped


def _add_scoped_ingredient_child_strategies(
    query_ir: QueryIR,
    registry: SchemaRegistry,
    plan: SqlPlan,
) -> set[int]:
    if plan.root_source != "base":
        return set()

    ingredient_relation = registry.relation("ingredients")
    if not ingredient_relation:
        return set()

    ingredient_predicates = [
        predicate for predicate in query_ir.predicates
        if predicate.source == "ingredients"
        and predicate.quantifier == "exists"
        and predicate.operator.upper() == "="
        and len(predicate.values) == 1
        and _uses_relation_match_strategy(predicate, registry, ingredient_relation)
    ]
    if len(ingredient_predicates) != 1:
        return set()

    parent_join = registry.relationship_for_relation("ingredients")
    if not parent_join or not registry.relationship_is_direct_to_base("ingredients"):
        return set()

    skipped: set[int] = set()
    ingredient_predicate = ingredient_predicates[0]
    for source, relation in registry.schema.get("relations", {}).items():
        if relation.get("requires") != "ingredients":
            continue
        child_predicates = [
            predicate for predicate in query_ir.predicates
            if predicate.source == source
            and predicate.quantifier == "exists"
        ]
        if not child_predicates:
            continue
        if _add_scoped_ingredient_child_condition(
            source,
            ingredient_predicate,
            child_predicates,
            registry,
            plan,
        ):
            skipped.add(id(ingredient_predicate))
            skipped.update(id(predicate) for predicate in child_predicates)
    return skipped


def _add_scoped_ingredient_child_condition(
    child_source: str,
    ingredient_predicate: Predicate,
    child_predicates: list[Predicate],
    registry: SchemaRegistry,
    plan: SqlPlan,
) -> bool:
    ingredient_relation = registry.relation("ingredients")
    child_relation = registry.relation(child_source)
    parent_join = registry.relationship_for_relation("ingredients")
    child_join = registry.relationship_for_relation(child_source)
    if not ingredient_relation or not child_relation or not parent_join or not child_join:
        return False

    ingredient_alias = f"{ingredient_relation['alias']}_{child_source}_scope"
    child_alias = f"{child_relation['alias']}_scope"
    chain_condition = _relationship_condition_between_aliases(
        child_join,
        ingredient_relation["table"],
        ingredient_alias,
        child_relation["table"],
        child_alias,
        registry,
    )
    if not chain_condition:
        return False

    filter_conditions: list[SqlCondition] = []
    ingredient_condition = _relation_match_condition(
        ingredient_alias,
        ingredient_predicate,
        registry,
        ingredient_relation,
    )
    if not ingredient_condition:
        return False
    filter_conditions.append(ingredient_condition)

    for predicate in child_predicates:
        condition = _relation_match_condition(
            child_alias,
            predicate,
            registry,
            child_relation,
        )
        if not condition:
            return False
        filter_conditions.append(condition)

    parent_base_condition = (
        f"{ingredient_alias}.{registry.quote_identifier(parent_join.from_column)} = "
        f"{registry.base_alias}.{registry.quote_identifier(parent_join.to_column)}"
    )
    params: list[Any] = []
    where_parts = [parent_base_condition]
    for condition in filter_conditions:
        where_parts.append(condition.sql)
        params.extend(condition.params)

    exists_sql = (
        f"SELECT 1 FROM {registry.table_sql(ingredient_relation['table'])} AS {ingredient_alias} "
        f"JOIN {registry.table_sql(child_relation['table'])} AS {child_alias} "
        f"ON {chain_condition} "
        f"WHERE {' AND '.join(where_parts)}"
    )
    plan.where.append(SqlCondition(
        f"EXISTS ({exists_sql})",
        tuple(params),
        child_source,
    ))

    join_assumption = {
        "name": f"{parent_join.name}->{child_join.name}",
        "path": [parent_join.as_dict(), child_join.as_dict()],
        "confidence": _path_confidence(parent_join.confidence, child_join.confidence),
    }
    if join_assumption not in plan.join_assumptions:
        plan.join_assumptions.append(join_assumption)
    return True


def _add_and_joins(
    source: str,
    predicates: list[Predicate],
    registry: SchemaRegistry,
    plan: SqlPlan,
) -> dict[str, Any] | None:
    relation = registry.relation(source)
    join = registry.relationship_for_relation(source)
    if not relation or not join:
        return None
    column = registry.predicate_column(predicates[0])
    if not column:
        return None

    from_column = registry.quote_identifier(join.from_column)
    to_column = registry.quote_identifier(join.to_column)
    value_column = registry.quote_identifier(column)
    strategy = registry.relation_strategy(source)
    table = registry.table_sql(relation["table"])
    use_filtered_subquery = strategy.get("and_join_mode") == "filtered_distinct_subquery"
    for index, predicate in enumerate(predicates, start=1):
        value = (
            str(predicate.values[0]).lower()
            if relation.get("normalizes_values")
            else predicate.values[0]
        )
        alias = f"{relation['alias']}{index}"
        if use_filtered_subquery:
            source_alias = f"{relation['alias']}_src{index}"
            join_table = (
                f"(SELECT DISTINCT {source_alias}.{from_column} "
                f"FROM {table} AS {source_alias} "
                f"WHERE {source_alias}.{value_column} = ?)"
            )
            on = f"{alias}.{from_column} = {registry.base_alias}.{to_column}"
        else:
            join_table = table
            on = (
                f"{alias}.{from_column} = "
                f"{registry.base_alias}.{to_column} "
                f"AND {alias}.{value_column} = ?"
            )
        plan.joins.append(SqlJoin(
            table=join_table,
            alias=alias,
            on=on,
            join_type=str(strategy.get("join_type") or "JOIN"),
            source=source,
            params=(value,),
        ))
    plan.distinct = bool(strategy.get("distinct", True))
    return join.as_dict()


def _add_grouped_join(
    source: str,
    predicates: list[Predicate],
    registry: SchemaRegistry,
    plan: SqlPlan,
) -> dict[str, Any] | None:
    relation = registry.relation(source)
    join = registry.relationship_for_relation(source)
    if not relation or not join:
        return None
    column = registry.predicate_column(predicates[0])
    if not column:
        return None
    strategy = registry.relation_strategy(source)
    if (
        strategy.get("grouped_match_mode") == "union_all"
        or relation.get("fallback_value_column")
        or strategy.get("match_operator")
    ):
        return _add_union_grouped_match_join(source, predicates, registry, plan)

    values = tuple(
        str(predicate.values[0]).lower()
        if relation.get("normalizes_values")
        else predicate.values[0]
        for predicate in predicates
    )
    placeholders = ", ".join("?" for _ in values)
    table_alias = f"{relation['alias']}_match"
    source_alias = f"{relation['alias']}_g"
    from_column = registry.quote_identifier(join.from_column)
    value_column = registry.quote_identifier(column)
    subquery = (
        f"(SELECT {source_alias}.{from_column} "
        f"FROM {registry.table_sql(relation['table'])} AS {source_alias} "
        f"WHERE {source_alias}.{value_column} IN ({placeholders}) "
        f"GROUP BY {source_alias}.{from_column} "
        f"HAVING COUNT(DISTINCT {source_alias}.{value_column}) = ?)"
    )
    plan.joins.append(SqlJoin(
        table=subquery,
        alias=table_alias,
        on=(
            f"{table_alias}.{from_column} = "
            f"{registry.base_alias}.{registry.quote_identifier(join.to_column)}"
        ),
        join_type=str(registry.relation_strategy(source).get("join_type") or "JOIN"),
        source=source,
        params=values + (len(values),),
    ))
    plan.distinct = bool(registry.relation_strategy(source).get("distinct", True))
    return join.as_dict()


def _add_union_grouped_match_join(
    source: str,
    predicates: list[Predicate],
    registry: SchemaRegistry,
    plan: SqlPlan,
) -> dict[str, Any] | None:
    relation = registry.relation(source)
    join = registry.relationship_for_relation(source)
    if not relation or not join:
        return None
    column = registry.predicate_column(predicates[0])
    if not column:
        return None

    from_column = registry.quote_identifier(join.from_column)
    table = registry.table_sql(relation["table"])
    branch_sql: list[str] = []
    params: list[Any] = []
    for index, predicate in enumerate(predicates, start=1):
        value = predicate.values[0]
        alias = f"{relation['alias']}_u{index}"
        match_condition = _relation_match_condition(alias, predicate, registry, relation)
        if not match_condition:
            return None
        branch_sql.append(
            f"SELECT DISTINCT {alias}.{from_column} "
            f"FROM {table} AS {alias} "
            f"WHERE {match_condition.sql}"
        )
        params.extend(match_condition.params)

    table_alias = f"{relation['alias']}_match"
    source_alias = f"{relation['alias']}_union"
    subquery = (
        f"(SELECT {source_alias}.{from_column} "
        f"FROM ({' UNION ALL '.join(branch_sql)}) AS {source_alias} "
        f"GROUP BY {source_alias}.{from_column} "
        f"HAVING COUNT(*) = ?)"
    )
    plan.joins.append(SqlJoin(
        table=subquery,
        alias=table_alias,
        on=(
            f"{table_alias}.{from_column} = "
            f"{registry.base_alias}.{registry.quote_identifier(join.to_column)}"
        ),
        join_type=str(registry.relation_strategy(source).get("join_type") or "JOIN"),
        source=source,
        params=tuple(params) + (len(predicates),),
    ))
    plan.distinct = bool(registry.relation_strategy(source).get("distinct", True))
    return join.as_dict()


def _relation_match_condition(
    alias: str,
    predicate: Predicate,
    registry: SchemaRegistry,
    relation: dict[str, Any],
) -> SqlCondition | None:
    if relation.get("predicate_mode") == "truthy_flag" or predicate.source == "dietary":
        return _truthy_relation_condition(alias, predicate, registry, relation)

    if not _uses_relation_match_strategy(predicate, registry, relation):
        filter_predicate = _normalized_predicate(predicate, relation)
        column = registry.predicate_column(filter_predicate)
        if not column:
            return None
        return _column_condition(
            registry.column_sql(alias, column),
            filter_predicate,
            registry,
        )

    operator = predicate.operator.upper()
    if operator not in {"=", "IN", "LIKE"}:
        filter_predicate = _normalized_predicate(predicate, relation)
        column = registry.predicate_column(filter_predicate)
        if not column:
            return None
        return _column_condition(
            registry.column_sql(alias, column),
            filter_predicate,
            registry,
        )

    values = tuple(predicate.values)
    if operator in {"=", "LIKE"} and len(values) != 1:
        return None
    if operator == "IN" and not values:
        return None

    parts: list[str] = []
    params: list[Any] = []
    for value in values:
        value_parts, value_params = _relation_match_value_conditions(
            alias,
            value,
            registry,
            relation,
            predicate_operator=operator,
        )
        if not value_parts:
            return None
        parts.append("(" + " OR ".join(value_parts) + ")")
        params.extend(value_params)

    if len(parts) == 1:
        return SqlCondition(parts[0], tuple(params), predicate.source)
    return SqlCondition("(" + " OR ".join(parts) + ")", tuple(params), predicate.source)


def _truthy_relation_condition(
    alias: str,
    predicate: Predicate,
    registry: SchemaRegistry,
    relation: dict[str, Any],
) -> SqlCondition | None:
    columns = relation.get("columns", {})
    truthy_values = tuple(registry.truthy_values(predicate.source))
    if not truthy_values:
        return None

    placeholders = ", ".join("?" for _ in truthy_values)
    parts: list[str] = []
    params: list[Any] = []
    for value in predicate.values:
        column = columns.get(str(value))
        if not column:
            return None
        parts.append(f"{registry.column_sql(alias, column)} IN ({placeholders})")
        params.extend(truthy_values)

    if not parts:
        return None
    joiner = " OR " if predicate.operator.upper() == "IN" or predicate.quantifier == "or_exists" else " AND "
    if len(parts) == 1:
        return SqlCondition(parts[0], tuple(params), predicate.source)
    return SqlCondition("(" + joiner.join(parts) + ")", tuple(params), predicate.source)


def _relation_match_value_conditions(
    alias: str,
    value: Any,
    registry: SchemaRegistry,
    relation: dict[str, Any],
    *,
    predicate_operator: str,
) -> tuple[list[str], list[Any]]:
    strategy = relation.get("strategy") or {}
    primary_column = relation.get("value_column")
    primary_operator = str(strategy.get("match_operator") or predicate_operator).upper()
    primary_transform = strategy.get("value_transform")
    normalizes = bool(relation.get("normalizes_values"))

    parts: list[str] = []
    params: list[Any] = []
    if primary_column:
        parts.append(f"{registry.column_sql(alias, primary_column)} {primary_operator} ?")
        params.append(_match_value(value, primary_operator, primary_transform, normalizes))

    fallback_column = relation.get("fallback_value_column")
    fallback_operator = str(strategy.get("fallback_match_operator") or "").upper()
    if fallback_column and fallback_operator:
        parts.append(f"{registry.column_sql(alias, fallback_column)} {fallback_operator} ?")
        params.append(_match_value(
            value,
            fallback_operator,
            strategy.get("fallback_value_transform"),
            normalizes,
        ))
    return parts, params


def _match_value(
    value: Any,
    operator: str,
    transform: Any,
    normalizes: bool,
) -> Any:
    text = str(value).lower() if normalizes else str(value)
    if transform == "contains" or (operator == "LIKE" and transform is None):
        return f"%{text}%"
    if transform == "prefix":
        return f"{text}%"
    if transform == "suffix":
        return f"%{text}"
    return text


def _uses_relation_match_strategy(
    predicate: Predicate,
    registry: SchemaRegistry,
    relation: dict[str, Any],
) -> bool:
    return (
        registry.is_relation_match_predicate(predicate, relation)
        and bool(
            relation.get("fallback_value_column")
            or relation.get("strategy", {}).get("match_operator")
        )
    )


def _normalized_predicate(predicate: Predicate, relation: dict[str, Any]) -> Predicate:
    if not relation.get("normalizes_values"):
        return predicate
    return Predicate(
        predicate.source,
        predicate.field,
        predicate.operator,
        tuple(str(value).lower() for value in predicate.values),
        predicate.quantifier,
    )


def _column_condition(
    column_sql: str,
    predicate: Predicate,
    registry: SchemaRegistry | None = None,
) -> SqlCondition | None:
    operator = predicate.operator.upper()
    values = tuple(predicate.values)
    expression = _predicate_expression(column_sql, predicate, registry)
    if operator == "BETWEEN" and len(values) == 2:
        return SqlCondition(f"{expression} BETWEEN ? AND ?", values, predicate.source)
    if operator == "IN" and values:
        placeholders = ", ".join("?" for _ in values)
        return SqlCondition(f"{expression} IN ({placeholders})", values, predicate.source)
    if operator == "LIKE" and len(values) == 1:
        return SqlCondition(f"{expression} LIKE ?", values, predicate.source)
    if operator in {"<", "<=", "=", ">=", ">"} and len(values) == 1:
        return SqlCondition(f"{expression} {operator} ?", values, predicate.source)
    return None


def _predicate_expression(
    column_sql: str,
    predicate: Predicate,
    registry: SchemaRegistry | None,
) -> str:
    if not registry:
        return column_sql
    item = registry.predicate_catalog_item(predicate)
    if not item or item.get("operator_profile") != "numeric":
        return column_sql
    table = str(item.get("table") or "")
    column = str(item.get("column") or "")
    column_type = (
        registry.schema.get("collections", {})
        .get(table, {})
        .get("fields", {})
        .get(column)
    )
    if str(column_type).lower() not in {"string", "text"}:
        return column_sql
    return f"CAST(NULLIF({column_sql}, '') AS DOUBLE)"


def _order_expression(field: str, registry: SchemaRegistry) -> str | None:
    order_field = registry.order_field(field)
    if not order_field:
        return None
    source = order_field["source"]
    if source == "base":
        column = registry.base_column(order_field["field"])
        if not column:
            return None
        return registry.column_sql(registry.base_alias, column)

    relation = registry.relation(source)
    if not relation:
        return None
    column = relation.get("columns", {}).get(order_field["field"])
    if not column:
        return None
    if registry.relation_is_base(source):
        return registry.column_sql(registry.base_alias, column)

    alias = f"{relation['alias']}_order"
    join_condition = registry.join_condition(source, alias)
    if not join_condition:
        return None
    return (
        f"(SELECT {registry.column_sql(alias, column)} "
        f"FROM {registry.table_sql(relation['table'])} AS {alias} "
        f"WHERE {join_condition})"
    )
