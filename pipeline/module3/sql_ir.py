from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SelectExpr:
    expression: str
    alias: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {"expression": self.expression}
        if self.alias:
            result["alias"] = self.alias
        return result


@dataclass(frozen=True)
class Predicate:
    source: str
    field: str
    operator: str
    values: tuple[Any, ...]
    quantifier: str = "exists"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "field": self.field,
            "operator": self.operator,
            "values": list(self.values),
            "quantifier": self.quantifier,
        }


@dataclass(frozen=True)
class OrderBy:
    field: str
    direction: str = "ASC"

    def as_dict(self) -> dict[str, str]:
        return {"field": self.field, "direction": self.direction}


@dataclass
class QueryIR:
    root: str = "recipe"
    select: list[SelectExpr] = field(default_factory=list)
    predicates: list[Predicate] = field(default_factory=list)
    order_by: list[OrderBy] = field(default_factory=list)
    limit: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "select": [item.as_dict() for item in self.select],
            "predicates": [item.as_dict() for item in self.predicates],
            "order_by": [item.as_dict() for item in self.order_by],
            "limit": self.limit,
        }
