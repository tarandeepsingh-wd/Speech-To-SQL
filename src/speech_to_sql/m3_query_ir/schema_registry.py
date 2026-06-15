from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from speech_to_sql.common.config import load_sql_schema_config
from speech_to_sql.m3_query_ir.sql_ir import Predicate


@dataclass(frozen=True)
class JoinSpec:
    name: str
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    confidence: str
    evidence: tuple[str, ...]
    preferred: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "preferred": self.preferred,
        }


class SchemaRegistry:
    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        self.schema = schema or load_sql_schema_config()
        self._relationships = {
            item["name"]: item
            for item in self.schema.get("relationships", [])
        }

    @property
    def base_table(self) -> dict[str, Any]:
        return self.schema["base_table"]

    @property
    def default_root(self) -> str:
        return self.schema.get("default_root", "recipe")

    @property
    def base_alias(self) -> str:
        return self.base_table["alias"]

    def root(self, name: str | None = None) -> dict[str, Any]:
        root_name = name or self.default_root
        roots = self.schema.get("roots", {})
        return roots.get(root_name) or roots[self.default_root]

    def root_source(self, name: str | None = None) -> str:
        return self.root(name).get("source", "base")

    def root_exists(self, name: str) -> bool:
        return name in self.schema.get("roots", {})

    def table_sql(self, table_name: str) -> str:
        return self.quote_identifier(table_name)

    def column_sql(self, alias: str, column_name: str) -> str:
        return f"{alias}.{self.quote_identifier(column_name)}"

    def quote_identifier(self, identifier: str) -> str:
        quote = self.schema.get("identifier_quote", "")
        if not quote:
            return identifier
        escaped = str(identifier).replace(quote, quote + quote)
        return f"{quote}{escaped}{quote}"

    def base_column(self, logical_field: str) -> str | None:
        return self.schema.get("base_columns", {}).get(logical_field)

    def relation(self, source: str) -> dict[str, Any] | None:
        return self.schema.get("relations", {}).get(source)

    def relation_strategy(self, source: str) -> dict[str, Any]:
        relation = self.relation(source) or {}
        return dict(relation.get("strategy") or {})

    def nutrition_group(self, field: str) -> dict[str, Any] | None:
        group = self.schema.get("nutrition_groups", {}).get(field)
        return group if isinstance(group, dict) else None

    def order_field(self, field: str) -> dict[str, str] | None:
        return self.schema.get("order_fields", {}).get(field)

    def field_catalog_item(self, source: str, field: str) -> dict[str, Any] | None:
        return self.schema.get("field_catalog", {}).get(f"{source}.{field}")

    def predicate_catalog_item(self, predicate: Predicate) -> dict[str, Any] | None:
        if predicate.source == "dietary" and predicate.field == "dietary_tag":
            return self.field_catalog_item("dietary", "dietary_tag")
        return self.field_catalog_item(predicate.source, predicate.field)

    def predicate_column(self, predicate: Predicate) -> str | None:
        if predicate.source == "base":
            return self.base_column(predicate.field)

        relation = self.relation(predicate.source)
        if not relation:
            return None
        if predicate.source == "ingredients" and self.is_relation_match_predicate(predicate, relation):
            return relation.get("value_column")
        if predicate.source == "dietary":
            value = str(predicate.values[0]) if predicate.values else ""
            return relation.get("columns", {}).get(value)
        if predicate.source == "nutrition":
            return relation.get("columns", {}).get(predicate.field)
        return relation.get("columns", {}).get(predicate.field)

    def predicate_field_exists(self, predicate: Predicate) -> bool:
        if predicate.source == "nutrition" and self.nutrition_group(predicate.field):
            return True
        return self.predicate_column(predicate) is not None

    def is_relation_match_predicate(
        self,
        predicate: Predicate,
        relation: dict[str, Any] | None = None,
    ) -> bool:
        relation = relation or self.relation(predicate.source) or {}
        match_field = relation.get("match_field")
        return bool(match_field and predicate.field == match_field)

    def relation_is_base(self, source: str) -> bool:
        relation = self.relation(source)
        return bool(relation and relation.get("same_as_base"))

    def relationship_for_relation(self, source: str) -> JoinSpec | None:
        relation = self.relation(source)
        if not relation or not relation.get("join"):
            return None
        return self.relationship(str(relation["join"]))

    def relationship(self, name: str) -> JoinSpec | None:
        item = self._relationships.get(name)
        if not item:
            return None
        return JoinSpec(
            name=item["name"],
            from_table=item["from_table"],
            from_column=item["from_column"],
            to_table=item["to_table"],
            to_column=item["to_column"],
            confidence=item["confidence"],
            evidence=tuple(item.get("evidence", [])),
            preferred=bool(item.get("preferred")),
        )

    def join_condition(self, source: str, relation_alias: str) -> str | None:
        join = self.relationship_for_relation(source)
        if not join:
            return None
        if not self.relationship_is_direct_to_base(source):
            return None
        return (
            f"{relation_alias}.{self.quote_identifier(join.from_column)} = "
            f"{self.base_alias}.{self.quote_identifier(join.to_column)}"
        )

    def relationship_is_direct_to_base(self, source: str) -> bool:
        join = self.relationship_for_relation(source)
        if not join:
            return False
        if join.to_table != self.base_table["name"]:
            return False
        fields = self.schema.get("collections", {}).get(join.to_table, {}).get("fields", {})
        return join.to_column in fields

    def truthy_values(self, source: str) -> list[Any]:
        relation = self.relation(source) or {}
        return list(relation.get("truthy_values", ["1", "1.0", "true", "True"]))


def load_schema_registry(schema: dict[str, Any] | None = None) -> SchemaRegistry:
    return SchemaRegistry(schema)
