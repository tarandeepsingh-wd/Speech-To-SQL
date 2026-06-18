from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_JSON = PROJECT_ROOT / "config" / "sql_schema.json"
DEFAULT_SCHEMA_SQL = PROJECT_ROOT / "config" / "schema.sql"
DEFAULT_SCHEMA_PDF = PROJECT_ROOT / "docs" / "Recipe_db2.0 schema.pdf"

PDF_TABLE_MAP = {
    "ingredient_nutrition": "ingredient_nutrition",
    "micronutrients": "micronutrients",
}
PDF_SECTION_ORDER = [
    "ingredient_flavor",
    "ingredient_nutrition",
    "ingredient_phrase",
    "instructions",
    "micronutrients",
    "recipes",
]
SQL_TYPE_MAP = {
    "TEXT": "string",
    "BIGINT": "bigint",
    "DOUBLE": "double",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit sql_schema.json coverage against schema.sql and the shared PDF."
    )
    parser.add_argument("--schema-json", type=Path, default=DEFAULT_SCHEMA_JSON)
    parser.add_argument("--schema-sql", type=Path, default=DEFAULT_SCHEMA_SQL)
    parser.add_argument("--schema-pdf", type=Path, default=DEFAULT_SCHEMA_PDF)
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args()

    schema = _load_json(args.schema_json)
    sql_tables = _parse_schema_sql(args.schema_sql)

    errors: list[str] = []
    warnings: list[str] = []
    _audit_schema_sql_syntax(args.schema_sql, errors)
    _audit_collection_sql_coverage(schema, sql_tables, errors)
    _audit_relationships(schema, errors)
    _audit_relation_columns(schema, errors)
    _audit_base_column_coverage(schema, errors)
    _audit_order_fields(schema, errors)
    _audit_field_catalog(schema, errors)
    _audit_selectable_expressions(schema, errors)
    _audit_nutrition_groups(schema, errors)

    if not args.skip_pdf:
        _audit_pdf_coverage(schema, args.schema_pdf, errors, warnings)

    print("Schema audit")
    print(f"- JSON collections: {len(schema.get('collections', {}))}")
    print(f"- SQL tables: {len(sql_tables)}")
    print(f"- Relationships: {len(schema.get('relationships', []))}")
    print(f"- Base columns mapped: {len(schema.get('base_columns', {}))}")
    print(f"- Order fields mapped: {len(schema.get('order_fields', {}))}")

    if warnings:
        print("\nWarnings")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nErrors")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nPASS: sql_schema.json covers the configured schema contract.")
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_schema_sql(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    tables: dict[str, dict[str, str]] = {}
    table_pattern = re.compile(r"CREATE TABLE\s+`([^`]+)`\s*\((.*?)\);", re.DOTALL)
    column_pattern = re.compile(r"^\s*`([^`]+)`\s+([A-Z]+)", re.MULTILINE)
    for table_match in table_pattern.finditer(text):
        table_name = table_match.group(1)
        columns: dict[str, str] = {}
        for column_match in column_pattern.finditer(table_match.group(2)):
            sql_type = column_match.group(2).upper()
            columns[column_match.group(1)] = SQL_TYPE_MAP.get(sql_type, sql_type.lower())
        tables[table_name] = columns
    return tables


def _audit_schema_sql_syntax(path: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    table_pattern = re.compile(r"CREATE TABLE\s+`([^`]+)`\s*\((.*?)\);", re.DOTALL)
    for table_match in table_pattern.finditer(text):
        table_name = table_match.group(1)
        column_lines = [
            line.rstrip()
            for line in table_match.group(2).splitlines()
            if line.strip().startswith("`")
        ]
        for line in column_lines[:-1]:
            if not line.endswith(","):
                errors.append(f"{table_name}: column line is missing trailing comma: {line.strip()}")


def _audit_collection_sql_coverage(
    schema: dict[str, Any],
    sql_tables: dict[str, dict[str, str]],
    errors: list[str],
) -> None:
    collections = schema.get("collections", {})
    for table, collection in collections.items():
        fields = collection.get("fields", {})
        if table not in sql_tables:
            errors.append(f"{table}: exists in sql_schema.json but not schema.sql")
            continue
        sql_fields = sql_tables[table]
        missing_in_sql = sorted(set(fields) - set(sql_fields))
        missing_in_json = sorted(set(sql_fields) - set(fields))
        if missing_in_sql:
            errors.append(f"{table}: fields missing from schema.sql: {missing_in_sql[:12]}")
        if missing_in_json:
            errors.append(f"{table}: schema.sql fields missing from sql_schema.json: {missing_in_json[:12]}")

        for field, field_type in fields.items():
            sql_type = sql_fields.get(field)
            if sql_type and sql_type != field_type:
                errors.append(f"{table}.{field}: type mismatch JSON={field_type} SQL={sql_type}")


def _audit_relationships(schema: dict[str, Any], errors: list[str]) -> None:
    collections = schema.get("collections", {})
    for relationship in schema.get("relationships", []):
        name = relationship.get("name", "<unnamed>")
        for side, table_key, column_key in [
            ("from", "from_table", "from_column"),
            ("to", "to_table", "to_column"),
        ]:
            table = relationship.get(table_key)
            column = relationship.get(column_key)
            fields = collections.get(table, {}).get("fields", {})
            if not fields:
                errors.append(f"{name}: {side}_table {table!r} does not exist")
            elif column not in fields:
                errors.append(f"{name}: {side}_column {table}.{column} does not exist")
        for pair in relationship.get("column_pairs", []):
            from_fields = collections.get(relationship.get("from_table"), {}).get("fields", {})
            to_fields = collections.get(relationship.get("to_table"), {}).get("fields", {})
            if pair.get("from_column") not in from_fields:
                errors.append(f"{name}: column_pair from_column {pair.get('from_column')} does not exist")
            if pair.get("to_column") not in to_fields:
                errors.append(f"{name}: column_pair to_column {pair.get('to_column')} does not exist")


def _audit_relation_columns(schema: dict[str, Any], errors: list[str]) -> None:
    collections = schema.get("collections", {})
    base_table = schema.get("base_table", {}).get("name")
    base_fields = collections.get(base_table, {}).get("fields", {})
    for source, relation in schema.get("relations", {}).items():
        table = relation.get("table")
        fields = collections.get(table, {}).get("fields", {})
        if not fields:
            errors.append(f"relations.{source}: table {table!r} does not exist")
            continue
        columns = relation.get("columns", {})
        invalid = sorted({column for column in columns.values() if column not in fields})
        if invalid:
            errors.append(f"relations.{source}: mapped columns do not exist: {invalid[:12]}")

        expected = _expected_relation_columns(source, relation, fields, base_fields)
        missing = sorted(expected - set(columns.values()))
        if missing:
            errors.append(f"relations.{source}: missing column mappings: {missing[:12]}")

        join_name = relation.get("join")
        if join_name and not any(item.get("name") == join_name for item in schema.get("relationships", [])):
            errors.append(f"relations.{source}: join {join_name!r} is not declared")


def _expected_relation_columns(
    source: str,
    relation: dict[str, Any],
    fields: dict[str, str],
    base_fields: dict[str, str],
) -> set[str]:
    if source == "nutrition":
        return set(fields)
    if source == "dietary":
        return {
            column for column in base_fields
            if column.lower() in {"vegan", "pescetarian"} or column.lower().endswith("_vegetarian")
        }
    return set(fields)


def _audit_base_column_coverage(schema: dict[str, Any], errors: list[str]) -> None:
    collections = schema.get("collections", {})
    base_table = schema.get("base_table", {}).get("name")
    base_fields = collections.get(base_table, {}).get("fields", {})
    base_columns = schema.get("base_columns", {})
    invalid = sorted({column for column in base_columns.values() if column not in base_fields})
    if invalid:
        errors.append(f"base_columns: mapped columns do not exist: {invalid[:12]}")
    missing = sorted(set(base_fields) - set(base_columns.values()))
    if missing:
        errors.append(f"base_columns: missing base table fields: {missing[:12]}")


def _audit_order_fields(schema: dict[str, Any], errors: list[str]) -> None:
    for name, item in schema.get("order_fields", {}).items():
        source = item.get("source")
        field = item.get("field")
        if source == "base":
            if field not in schema.get("base_columns", {}):
                errors.append(f"order_fields.{name}: base field {field!r} is not mapped")
            continue
        relation = schema.get("relations", {}).get(source)
        if not relation:
            errors.append(f"order_fields.{name}: source {source!r} does not exist")
            continue
        if field not in relation.get("columns", {}):
            errors.append(f"order_fields.{name}: relation field {field!r} is not mapped")


def _audit_field_catalog(schema: dict[str, Any], errors: list[str]) -> None:
    collections = schema.get("collections", {})
    relations = schema.get("relations", {})
    operator_profiles = schema.get("operator_profiles", {})
    catalog = schema.get("field_catalog", {})
    if not catalog:
        errors.append("field_catalog is missing")
        return

    required: set[str] = set()
    required.update(f"base.{field}" for field in schema.get("base_columns", {}))
    for source, relation in relations.items():
        required.update(f"{source}.{field}" for field in relation.get("columns", {}))
        if source == "dietary":
            required.add("dietary.dietary_tag")

    missing = sorted(required - set(catalog))
    if missing:
        errors.append(f"field_catalog: missing fields: {missing[:12]}")

    for key, item in catalog.items():
        source = item.get("source")
        table = item.get("table")
        column = item.get("column")
        profile = item.get("operator_profile")
        if source == "base" and item.get("field") not in schema.get("base_columns", {}):
            errors.append(f"field_catalog.{key}: base field is not mapped")
        if source != "base" and source not in relations:
            errors.append(f"field_catalog.{key}: source {source!r} is not declared")
        if column is not None:
            fields = collections.get(table, {}).get("fields", {})
            if column not in fields:
                errors.append(f"field_catalog.{key}: column {table}.{column} does not exist")
        if profile not in operator_profiles:
            errors.append(f"field_catalog.{key}: operator_profile {profile!r} is not declared")
        operators = item.get("operators", [])
        if profile in operator_profiles and sorted(operators) != sorted(operator_profiles[profile]):
            errors.append(f"field_catalog.{key}: operators do not match operator_profile {profile!r}")


def _audit_selectable_expressions(schema: dict[str, Any], errors: list[str]) -> None:
    collections = schema.get("collections", {})
    for root, items in schema.get("selectable_expressions", {}).items():
        root_config = schema.get("roots", {}).get(root)
        if not isinstance(root_config, dict):
            errors.append(f"selectable_expressions.{root}: root is not declared")
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"selectable_expressions.{root}[{index}] must be an object")
                continue
            if item.get("kind") == "aggregate":
                continue
            table = item.get("table")
            column = item.get("column")
            fields = collections.get(table, {}).get("fields", {})
            if column not in fields:
                errors.append(
                    f"selectable_expressions.{root}[{index}] maps to missing column {table}.{column}"
                )


def _audit_nutrition_groups(schema: dict[str, Any], errors: list[str]) -> None:
    nutrition_columns = schema.get("relations", {}).get("nutrition", {}).get("columns", {})
    catalog = schema.get("field_catalog", {})
    for group_name, group in schema.get("nutrition_groups", {}).items():
        if not isinstance(group, dict):
            errors.append(f"nutrition_groups.{group_name}: must be an object")
            continue
        fields = [str(field) for field in group.get("fields", [])]
        if not fields:
            errors.append(f"nutrition_groups.{group_name}: has no fields")
        missing = sorted(field for field in fields if field not in nutrition_columns)
        if missing:
            errors.append(f"nutrition_groups.{group_name}: unknown fields {missing[:12]}")
        catalog_item = catalog.get(f"nutrition.{group_name}")
        if not catalog_item:
            errors.append(f"nutrition_groups.{group_name}: missing field_catalog entry")
        elif sorted(catalog_item.get("group_fields", [])) != sorted(fields):
            errors.append(f"nutrition_groups.{group_name}: field_catalog group_fields mismatch")


def _audit_pdf_coverage(
    schema: dict[str, Any],
    pdf_path: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not pdf_path.exists():
        warnings.append(f"PDF not found, skipped PDF coverage: {pdf_path}")
        return
    pdf_fields = _extract_pdf_fields(pdf_path, warnings)
    if not pdf_fields:
        warnings.append("No PDF fields could be extracted; JSON/schema.sql checks still ran.")
        return

    collections = schema.get("collections", {})
    for pdf_table, json_table in PDF_TABLE_MAP.items():
        extracted = pdf_fields.get(pdf_table, set())
        if not extracted:
            warnings.append(f"PDF section {pdf_table!r} had no extractable fields.")
            continue
        json_fields = {_canonical_field(field) for field in collections.get(json_table, {}).get("fields", {})}
        missing = sorted(extracted - json_fields)
        if missing:
            errors.append(f"PDF {pdf_table}: fields missing from sql_schema.json: {missing[:12]}")


def _extract_pdf_fields(pdf_path: Path, warnings: list[str]) -> dict[str, set[str]]:
    try:
        from pypdf import PdfReader
    except ImportError:
        warnings.append("pypdf is not installed, skipped PDF coverage.")
        return {}

    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    normalized_text = re.sub(r"\s+", " ", text)
    sections = _pdf_sections(normalized_text)
    result: dict[str, set[str]] = {}
    for table_name, section_text in sections.items():
        result[table_name] = _parse_pdf_section_fields(table_name, section_text)
    return result


def _pdf_sections(text: str) -> dict[str, str]:
    markers: list[tuple[str, int, int]] = []
    for table in PDF_SECTION_ORDER:
        match = re.search(rf"\b\d+\.\s+{re.escape(table)}\b", text)
        if match:
            markers.append((table, match.start(), match.end()))
    markers.sort(key=lambda item: item[1])

    sections: dict[str, str] = {}
    for index, (table, _start, end) in enumerate(markers):
        next_start = markers[index + 1][1] if index + 1 < len(markers) else len(text)
        sections[table] = text[end:next_start]
    return sections


def _parse_pdf_section_fields(table_name: str, text: str) -> set[str]:
    del table_name
    tokens = text.split()
    fields: set[str] = set()
    current: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in {"string", "bigint", "double"}:
            if current:
                fields.add(_canonical_field(" ".join(current)))
            current = []
            continue
        if lowered in {"from", "deserializer"}:
            continue
        current.append(token)
    return {field for field in fields if field}


def _canonical_field(field: str) -> str:
    text = re.sub(r"\s+", " ", field).strip().lower()
    text = text.replace(" ,", ",")
    text = text.replace("( ", "(").replace(" )", ")")
    return text


if __name__ == "__main__":
    sys.exit(main())
