from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SqlCondition:
    sql: str
    params: tuple[Any, ...] = ()
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "sql": self.sql,
            "params": list(self.params),
            "source": self.source,
        }


@dataclass(frozen=True)
class SqlOrder:
    expression: str
    direction: str

    def as_sql(self) -> str:
        return f"{self.expression} {self.direction}"

    def as_dict(self) -> dict[str, str]:
        return {
            "expression": self.expression,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class SqlJoin:
    table: str
    alias: str
    on: str
    join_type: str = "JOIN"
    source: str = ""
    params: tuple[Any, ...] = ()

    def as_sql(self) -> str:
        return f"{self.join_type} {self.table} AS {self.alias} ON {self.on}"

    def as_dict(self) -> dict[str, str]:
        return {
            "table": self.table,
            "alias": self.alias,
            "on": self.on,
            "join_type": self.join_type,
            "source": self.source,
            "params": list(self.params),
        }


@dataclass
class SqlPlan:
    select: list[str]
    from_table: str
    from_alias: str
    root: str = "recipe"
    root_source: str = "base"
    distinct: bool = False
    joins: list[SqlJoin] = field(default_factory=list)
    where: list[SqlCondition] = field(default_factory=list)
    order_by: list[SqlOrder] = field(default_factory=list)
    limit: int | None = None
    join_assumptions: list[dict[str, Any]] = field(default_factory=list)

    def params(self) -> list[Any]:
        values: list[Any] = []
        for join in self.joins:
            values.extend(join.params)
        for condition in self.where:
            values.extend(condition.params)
        if self.limit:
            values.append(self.limit)
        return values

    def as_dict(self) -> dict[str, Any]:
        return {
            "select": self.select,
            "from_table": self.from_table,
            "from_alias": self.from_alias,
            "root": self.root,
            "root_source": self.root_source,
            "distinct": self.distinct,
            "joins": [join.as_dict() for join in self.joins],
            "where": [condition.as_dict() for condition in self.where],
            "order_by": [order.as_dict() for order in self.order_by],
            "limit": self.limit,
            "join_assumptions": self.join_assumptions,
        }
