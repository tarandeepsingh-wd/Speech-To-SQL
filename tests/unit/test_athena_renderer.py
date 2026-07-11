from speech_to_sql.m2_nlu.entity_tagger import tag_entities
from speech_to_sql.m4_sql.athena_renderer import render_athena_query, render_athena_sql
from speech_to_sql.m4_sql.sql_query_builder import build_sql_query


_SCHEMA = {
    "base_table": {"name": "recipes"},
    "base_columns": {
        "recipe_title": "recipe_title",
        "total_time": "total_time",
        "continent": "continent",
        "vegan": "vegan",
        "ovo_lacto_vegetarian": "ovo_lacto_vegetarian",
    },
    "semantic_aliases": {
        "base": {
            "title": "recipe_title",
            "name": "recipe_title",
            "recipe name": "recipe_title",
        },
        "nutrition": {
            "protein": "protein",
            "vitamin c": "vitamin_c",
        },
    },
    "relations": {
        "nutrition": {
            "table": "micronutrients",
            "columns": {
                "calories": "calories",
                "protein": "protein (g)",
                "vitamin_c": "vitamin c, total ascorbic acid (mg)",
            },
        },
        "dietary": {
            "table": "recipes",
            "predicate_mode": "truthy_flag",
            "columns": {
                "vegan": "vegan",
                "ovo_lacto_vegetarian": "ovo_lacto_vegetarian",
            },
        },
    },
    "collections": {
        "recipes": {
            "fields": {
                "recipe_id": "string",
                "recipe_title": "string",
                "total_time": "string",
                "continent": "string",
                "vegan": "string",
                "ovo_lacto_vegetarian": "string",
            }
        },
        "ingredient_phrase": {
            "fields": {
                "recipe_no": "string",
                "ingredient": "string",
            }
        },
        "micronutrients": {
            "fields": {
                "recipe_id": "bigint",
                "calories": "double",
                "protein (g)": "double",
                "vitamin c, total ascorbic acid (mg)": "double",
            }
        },
    }
}


def test_athena_renderer_inlines_params_and_uses_athena_identifiers():
    query = (
        "SELECT r.* FROM `recipes` AS r WHERE "
        "(CAST(NULLIF(r.`total_time`, '') AS DOUBLE) <= ?) "
        "AND (r.`continent` = ?) LIMIT ?"
    )

    rendered = render_athena_query(query, [30.0, "O'Reilly", 5], schema=_SCHEMA)

    assert "`" not in rendered
    assert "?" not in rendered
    assert 'FROM "recipes" AS r' in rendered
    assert (
        'TRY_CAST(NULLIF(TRIM(CAST(r."total_time" AS VARCHAR)), \'\') AS DOUBLE) <= 30'
        in rendered
    )
    assert 'r."continent" = \'O\'\'Reilly\'' in rendered
    assert rendered.endswith("LIMIT 5")


def test_athena_renderer_casts_schema_key_joins_and_numeric_columns():
    query = (
        "SELECT DISTINCT r.* FROM `recipes` AS r "
        "JOIN `ingredient_phrase` AS ip ON ip.`recipe_no` = r.`recipe_id` "
        "WHERE (EXISTS (SELECT 1 FROM `micronutrients` AS n1 "
        "WHERE n1.`recipe_id` = r.`recipe_id` AND n1.`calories` > ?))"
    )

    rendered = render_athena_query(query, [300.0], schema=_SCHEMA)

    assert 'CAST(ip."recipe_no" AS VARCHAR) = CAST(r."recipe_id" AS VARCHAR)' in rendered
    assert 'CAST(n1."recipe_id" AS VARCHAR) = CAST(r."recipe_id" AS VARCHAR)' in rendered
    assert (
        'TRY_CAST(NULLIF(TRIM(CAST(n1."calories" AS VARCHAR)), \'\') AS DOUBLE) > 300'
        in rendered
    )


def test_build_sql_query_can_emit_paste_ready_athena_sql():
    text = "top 5 chicken recipes with more than 40 grams protein under 30 minutes"

    sql = build_sql_query(tag_entities(text), transcript=text)
    athena = render_athena_sql(sql)["query"]

    assert sql["query"]
    assert athena
    assert "?" not in athena
    assert "`" not in athena
    assert "LIMIT 5" in athena
    assert "TRY_CAST" in athena


def test_athena_renderer_projects_distinct_order_expression():
    query = (
        "SELECT DISTINCT r.* FROM `recipes` AS r "
        "WHERE (r.`continent` = ?) "
        "ORDER BY (SELECT n_order.`calories` FROM `micronutrients` AS n_order "
        "WHERE n_order.`recipe_id` = r.`recipe_id`) ASC LIMIT ?"
    )

    rendered = render_athena_query(query, ["Asian", 10], schema=_SCHEMA)

    assert 'SELECT DISTINCT r.*, (SELECT MIN(' in rendered
    assert 'AS "__athena_order_1" FROM "recipes" AS r' in rendered
    assert 'ORDER BY "__athena_order_1" ASC LIMIT 10' in rendered
    assert 'ORDER BY (SELECT' not in rendered
    assert (
        'TRY_CAST(NULLIF(TRIM(CAST(n_order."calories" AS VARCHAR)), \'\') AS DOUBLE)'
        in rendered
    )


def test_athena_renderer_rewrites_truthy_flags_for_numeric_or_text_storage():
    query = (
        "SELECT r.* FROM `recipes` AS r WHERE "
        "(r.`vegan` IN (?, ?, ?, ?, ?, ?)) AND "
        "((r.`ovo_lacto_vegetarian` NOT IN (?, ?, ?, ?, ?, ?) "
        "OR r.`ovo_lacto_vegetarian` IS NULL))"
    )
    truthy = ["1", "1.0", "true", "True", "yes", "Y"]

    rendered = render_athena_query(query, truthy + truthy, schema=_SCHEMA)

    assert 'r."vegan" IN' not in rendered
    assert 'r."ovo_lacto_vegetarian" NOT IN' not in rendered
    assert (
        'TRY_CAST(NULLIF(TRIM(CAST(r."vegan" AS VARCHAR)), \'\') AS DOUBLE) = 1.0'
        in rendered
    )
    assert 'LOWER(TRIM(CAST(r."vegan" AS VARCHAR))) IN (\'true\', \'yes\', \'y\')' in rendered
    assert 'COALESCE(TRY_CAST(NULLIF(TRIM(CAST(r."ovo_lacto_vegetarian" AS VARCHAR)), \'\') AS DOUBLE) <> 1.0, TRUE)' in rendered


def test_athena_renderer_rewrites_known_schema_column_aliases():
    query = (
        "SELECT r.`title`, n.`protein` FROM `recipes` AS r "
        "JOIN `micronutrients` AS n ON n.`recipe_id` = r.`recipe_id` "
        "WHERE (r.`name` LIKE ?) AND (n.`vitamin c` > ?)"
    )

    rendered = render_athena_query(query, ["%curry%", 10], schema=_SCHEMA)

    assert 'r."title"' not in rendered
    assert 'r."name"' not in rendered
    assert 'n."protein"' not in rendered
    assert 'r."recipe_title"' in rendered
    assert 'n."protein (g)"' in rendered
    assert (
        'TRY_CAST(NULLIF(TRIM(CAST(n."vitamin c, total ascorbic acid (mg)" AS VARCHAR)), \'\') AS DOUBLE) > 10'
        in rendered
    )


def test_athena_renderer_preserves_nutrition_relation_semantics():
    query = (
        "SELECT r.* FROM `recipes` AS r WHERE "
        "(EXISTS (SELECT 1 FROM `micronutrients` AS n1 "
        "WHERE n1.`recipe_id` = r.`recipe_id` AND n1.`calories` < ?)) AND "
        "(EXISTS (SELECT 1 FROM `micronutrients` AS n2 "
        "WHERE n2.`recipe_id` = r.`recipe_id` AND n2.`protein (g)` >= ?))"
    )

    rendered = render_athena_query(query, [500, 20], schema=_SCHEMA)

    assert 'FROM "micronutrients" AS n1' in rendered
    assert 'FROM "micronutrients" AS n2' in rendered
    assert 'r."calories"' not in rendered
    assert 'r."protein (g)"' not in rendered
    assert (
        'TRY_CAST(NULLIF(TRIM(CAST(n1."calories" AS VARCHAR)), \'\') AS DOUBLE) < 500'
        in rendered
    )
    assert (
        'TRY_CAST(NULLIF(TRIM(CAST(n2."protein (g)" AS VARCHAR)), \'\') AS DOUBLE) >= 20'
        in rendered
    )
