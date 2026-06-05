from __future__ import annotations

from typing import Any

from module3.schema_registry import SchemaRegistry, load_schema_registry
from module3.sql_ir import Predicate, QueryIR, SelectExpr
from module3.sql_plan import SqlCondition, SqlJoin, SqlOrder, SqlPlan


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
    planned_predicates = _add_multi_join_strategies(query_ir, registry, plan)
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
    filter_predicate = _normalized_predicate(predicate, relation)
    return _column_condition(
        registry.column_sql(plan.from_alias, column),
        filter_predicate,
    )


def _same_table_relation_condition(
    predicate: Predicate,
    registry: SchemaRegistry,
) -> SqlCondition | None:
    column = registry.predicate_column(predicate)
    if not column:
        return None
    column_sql = registry.column_sql(registry.base_alias, column)
    relation = registry.relation(predicate.source) or {}
    if relation.get("predicate_mode") == "truthy_flag" or predicate.source == "dietary":
        values = tuple(registry.truthy_values(predicate.source))
        placeholders = ", ".join("?" for _ in values)
        return SqlCondition(
            f"{column_sql} IN ({placeholders})",
            values,
            predicate.source,
        )
    return _column_condition(column_sql, predicate)


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
    column = registry.predicate_column(predicate)
    if not column:
        return None, None

    filter_predicate = _normalized_predicate(predicate, relation)
    filter_condition = _column_condition(
        registry.column_sql(alias, column),
        filter_predicate,
    )
    if not filter_condition:
        return None, None

    join_condition = registry.join_condition(predicate.source, alias)
    join = registry.relationship_for_relation(predicate.source)
    join_assumption = join.as_dict() if join else None
    if not join_condition:
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

    filter_predicate = _normalized_predicate(predicate, relation)
    filter_condition = _column_condition(
        registry.column_sql(alias, column),
        filter_predicate,
    )
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
        sources.add(source)
    return sources


def _add_multi_join_strategies(
    query_ir: QueryIR,
    registry: SchemaRegistry,
    plan: SqlPlan,
) -> set[int]:
    skipped: set[int] = set()
    if plan.root_source != "base":
        return skipped
    by_source: dict[str, list[Predicate]] = {}
    for predicate in query_ir.predicates:
        by_source.setdefault(predicate.source, []).append(predicate)

    for source, predicates in by_source.items():
        strategy = registry.relation_strategy(source)
        include_predicates = [
            predicate for predicate in predicates
            if predicate.quantifier == "exists"
            and predicate.operator.upper() == strategy.get("operator", "=")
            and len(predicate.values) == int(strategy.get("value_count", 1))
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


def _column_condition(column_sql: str, predicate: Predicate) -> SqlCondition | None:
    operator = predicate.operator.upper()
    values = tuple(predicate.values)
    if operator == "BETWEEN" and len(values) == 2:
        return SqlCondition(f"{column_sql} BETWEEN ? AND ?", values, predicate.source)
    if operator == "IN" and values:
        placeholders = ", ".join("?" for _ in values)
        return SqlCondition(f"{column_sql} IN ({placeholders})", values, predicate.source)
    if operator in {"<", "<=", "=", ">=", ">"} and len(values) == 1:
        return SqlCondition(f"{column_sql} {operator} ?", values, predicate.source)
    return None


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
