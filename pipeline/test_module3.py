import sys
from pathlib import Path

sys.path.insert(0, ".")

from module3.query_planner import plan_queries
from module3.api_verifier import verify_recipes_against_ir
from module3.config import endpoint_path, load_recipe_db_config
from module3.query_relaxer import relax_query_ir, relaxation_chain
from module3.result_merger import dedupe_recipes, merge_api_results
from module3.client import ApiResult
from module3.query_planner import ApiQueryPlan
from module3.post_filter import post_filter
from module3.ranker import rank_and_limit
from module3.intent_enricher import enrich_tag_result
from module3.intent_grounder import ground_intent
from module3.intent_preparer import prepare_intent
from module3.query_ir_contract import (
    QueryIRContractError,
    query_ir_from_contract,
    query_ir_to_payload,
)
from module3.schema_discovery import _build_virtual_tables, _should_write_cache, _summarize_endpoint
from module3.schema_registry import load_schema_registry
from module3.schema_sql import load_schema_sql
from module3.schema_linker import link_tag_result
from module3.sql_ir import Predicate, QueryIR, SelectExpr
from module3.sql_plan_builder import build_sql_plan
from module3.sql_query_builder import build_sql_query
from module3.sql_validator import validate_query_ir
from module1_asr.entity_tagger import tag_entities


def _plan(label, recipes):
    return ApiResult(
        plan=ApiQueryPlan(label=label, method="GET", path="/x"),
        ok=True,
        status_code=200,
        data={"data": recipes},
        recipes=recipes,
    )


def test_planner_prefers_ingredient_endpoint_and_uses_real_params():
    config = load_recipe_db_config()
    tag_result = {
        "ingredient_logic": {
            "include": ["spinach"],
            "exclude": ["onion"],
            "include_operator": "AND",
        },
        "nutrition_filters": [{"nutrient": "calories", "operator": "<", "value": 300}],
        "dietary_styles": ["vegan"],
        "categories": [],
        "sql_operators": [],
    }

    plans = plan_queries(tag_result, transcript="vegan spinach without onion under 300 calories")

    assert plans[0].label == "ingredient_category_title"
    assert plans[0].path == endpoint_path("ingredient_category_title", config)
    ingredient_params = config["endpoints"]["ingredient_category_title"]["params"]
    assert plans[0].params[ingredient_params["include_ingredients"]] == "spinach"
    assert plans[0].params[ingredient_params["exclude_ingredients"]] == "onion"
    assert any(plan.path == endpoint_path("calories", config) for plan in plans)
    assert any(plan.path == endpoint_path("diet", config) for plan in plans)


def test_planner_uses_protein_endpoint():
    config = load_recipe_db_config()
    tag_result = {
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [{"nutrient": "protein", "operator": ">=", "value": 30}],
        "dietary_styles": [],
        "categories": [],
        "sql_operators": [],
    }

    plans = plan_queries(tag_result, transcript="recipes with at least 30g protein")

    protein_params = config["endpoints"]["protein"]["params"]
    assert plans[0].path == endpoint_path("protein", config)
    assert plans[0].params[protein_params["min"]] == 30.0


def test_planner_combines_region_and_diet_when_available():
    config = load_recipe_db_config()
    tag_result = {
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [],
        "dietary_styles": ["vegan"],
        "categories": [{"term": "Indian", "category": "region"}],
        "sql_operators": [],
    }

    plans = plan_queries(tag_result, transcript="vegan Indian recipes")

    region_diet_params = config["endpoints"]["region_diet"]["params"]
    assert plans[0].path == endpoint_path("region_diet", config)
    assert plans[0].params[region_diet_params["region"]] == "Indian"
    assert plans[0].params[region_diet_params["diet"]] == "vegan"


def test_merger_intersects_and_dedupes_by_recipe_id():
    first = _plan("a", [
        {"Recipe_id": "1", "Recipe_title": "A", "Calories": "100"},
        {"Recipe_id": "2", "Recipe_title": "B", "Calories": "200"},
    ])
    second = _plan("b", [
        {"Recipe_id": "2", "Protein (g)": "30"},
        {"Recipe_id": "3", "Protein (g)": "40"},
    ])

    merged = merge_api_results([first, second])

    assert len(merged) == 1
    assert merged[0]["Recipe_id"] == "2"
    assert merged[0]["Protein (g)"] == "30"


def test_post_filter_uses_ingredients_diet_and_nutrition():
    recipes = [
        {
            "Recipe_id": "1",
            "Recipe_title": "Spinach Garlic Soup",
            "Calories": "250",
            "vegan": "1.0",
            "ingredients": [{"ingredient": "spinach"}, {"ingredient": "garlic"}],
        },
        {
            "Recipe_id": "2",
            "Recipe_title": "Spinach Onion Soup",
            "Calories": "250",
            "vegan": "1.0",
            "ingredients": [{"ingredient": "spinach"}, {"ingredient": "onion"}],
        },
        {
            "Recipe_id": "3",
            "Recipe_title": "Spinach Cream Soup",
            "Calories": "400",
            "vegan": "0.0",
            "ingredients": [{"ingredient": "spinach"}],
        },
    ]
    tag_result = {
        "ingredient_logic": {
            "include": ["spinach"],
            "exclude": ["onion"],
            "include_operator": "AND",
        },
        "nutrition_filters": [{"nutrient": "calories", "operator": "<", "value": 300}],
        "dietary_styles": ["vegan"],
    }

    filtered = post_filter(recipes, tag_result)

    assert [recipe["Recipe_id"] for recipe in filtered] == ["1"]


def test_ranker_sorts_and_limits():
    recipes = [
        {"Recipe_id": "1", "Protein (g)": "10"},
        {"Recipe_id": "2", "Protein (g)": "50"},
        {"Recipe_id": "3", "Protein (g)": "30"},
    ]
    tag_result = {"sql_operators": [{"canonical": "LIMIT", "text": "top 2"}]}

    ranked = rank_and_limit(recipes, tag_result, transcript="top 2 highest protein recipes")

    assert [recipe["Recipe_id"] for recipe in ranked] == ["2", "3"]


def test_sql_query_builder_adds_where_order_and_limit():
    tag_result = {
        "dietary_styles": ["vegan"],
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [],
        "sql_operators": [{"canonical": "LIMIT", "text": "top 5"}],
    }

    sql = build_sql_query(tag_result, transcript="top 5 vegan recipes highest protein")

    assert sql["query"].startswith("SELECT r.* FROM `RecipeDB_micronutrients_v1` AS r WHERE")
    assert "r.`vegan` IN" in sql["query"]
    assert "ORDER BY r.`protein (g)` DESC" in sql["query"]
    assert sql["query"].endswith("LIMIT ?")
    assert sql["params"][-1] == 5
    assert sql["ir"]["predicates"][0]["source"] == "dietary"
    assert sql["validation"]["ok"] is True


def test_sql_query_builder_warns_on_dietary_ingredient_conflict():
    tag_result = {
        "ingredients": ["chicken", "milk"],
        "ingredient_logic": {
            "include": ["chicken", "milk"],
            "exclude": [],
            "include_operator": "AND",
        },
        "dietary_styles": ["vegan"],
        "nutrition_filters": [],
        "sql_operators": [],
    }

    sql = build_sql_query(tag_result, transcript="vegan recipes with chicken and milk")

    assert sql["validation"]["ok"] is True
    warnings = " ".join(sql["validation"]["warnings"])
    assert "vegan" in warnings
    assert "chicken" in warnings
    assert "milk" in warnings


def test_sql_query_builder_uses_schema_driven_utensil_and_action_filters():
    tag_result = {
        "ingredients": [],
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "utensils": ["skillet"],
        "actions": [{"action": "fry", "type": "state_change"}],
        "dietary_styles": [],
        "nutrition_filters": [],
        "sql_operators": [],
    }

    sql = build_sql_query(tag_result, transcript="fried recipes made in skillet")

    assert sql["validation"]["ok"] is True
    assert "r.`utensils` LIKE ?" in sql["query"]
    assert "r.`processes` LIKE ?" in sql["query"]
    assert "%skillet%" in sql["params"]
    assert "%fry%" in sql["params"]


def test_text_to_sql_uses_extracted_utensil_and_action_filters():
    tags = tag_entities("fried recipes made in skillet")
    sql = build_sql_query(tags, transcript="fried recipes made in skillet")

    assert {"action": "fry", "type": "state_change", "span": (0, 4)} in tags["actions"]
    assert tags["utensils"] == ["skillet"]
    assert "r.`utensils` LIKE ?" in sql["query"]
    assert "r.`processes` LIKE ?" in sql["query"]
    assert "%skillet%" in sql["params"]
    assert "%fry%" in sql["params"]


def test_text_to_sql_uses_pdf_enriched_nutrition_columns():
    tags = tag_entities("recipes with folate over 50")
    sql = build_sql_query(tags, transcript="recipes with folate over 50")

    assert tags["nutrition_filters"] == [
        {"nutrient": "folate_total", "operator": ">", "value": 50.0}
    ]
    assert "r.`folate, total (g)` > ?" in sql["query"]
    assert sql["params"] == [50.0]
    assert sql["validation"]["ok"] is True


def test_text_to_sql_uses_low_sugar_enrichment_without_category_conflict():
    tags = tag_entities("low sugar recipes")
    sql = build_sql_query(tags, transcript="low sugar recipes")

    assert tags["categories"]
    assert "r.`sugars, total (g)` <= ?" in sql["query"]
    assert sql["params"] == [10]
    assert sql["validation"]["ok"] is True


def test_schema_linker_resolves_aliases_without_code_field_names():
    config = load_recipe_db_config()
    tag_result = {
        "nutrition_filters": [
            {"nutrient": "kcal", "operator": "<", "value": 300},
            {"nutrient": "carbohydrates", "operator": "<", "value": 50},
        ],
        "dietary_styles": ["vegetarian"],
        "time_filters": [{"field": "prepare", "max": 20}],
    }

    linked = link_tag_result(tag_result)

    assert [item["nutrient"] for item in linked["nutrition_filters"]] == ["calories", "carbs"]
    assert linked["dietary_styles"] == [config["diet_aliases"]["vegetarian"]]
    assert linked["time_filters"][0]["field"] == config["time_fields"]["terms"]["prepare"]
    assert any(item["original"] == "kcal" and item["resolved"] == "calories" for item in linked["_schema_links"])


def test_intent_enricher_adds_configured_low_carb_filter():
    tag_result = {
        "ingredients": ["paneer"],
        "ingredient_logic": {"include": ["paneer"], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [],
    }

    enriched = enrich_tag_result(tag_result, transcript="low carb dinner with paneer")

    assert enriched["nutrition_filters"][0]["nutrient"] == "carbs"
    assert enriched["nutrition_filters"][0]["operator"] == "<="
    assert enriched["nutrition_filters"][0]["source"] == "intent_enrichment"
    assert enriched["_intent_enrichments"][0]["target"] == "nutrition_filters"


def test_intent_enricher_adds_numeric_minutes_filter_without_time_word():
    tag_result = {
        "ingredients": ["garlic"],
        "ingredient_logic": {"include": ["garlic"], "exclude": [], "include_operator": "AND"},
        "time_filters": None,
    }

    enriched = enrich_tag_result(
        tag_result,
        transcript="recipes between 20 and 40 minutes with garlic",
    )

    assert enriched["time_filters"][0]["field"] == "total_time"
    assert enriched["time_filters"][0]["min"] == 20.0
    assert enriched["time_filters"][0]["max"] == 40.0


def test_prepare_intent_is_single_enrich_then_link_layer():
    prepared = prepare_intent(
        {
            "nutrition_filters": [],
            "dietary_styles": ["vegetarian"],
            "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        },
        transcript="low carb vegetarian recipes",
    )

    assert prepared["_intent_prepared"] is True
    assert prepared["nutrition_filters"][0]["nutrient"] == "carbs"
    assert prepared["dietary_styles"] == ["ovo_lacto_vegetarian"]
    assert prepared["_intent_enrichments"][0]["target"] == "nutrition_filters"
    assert prepared["_schema_links"][0]["source"] == "dietary"
    assert prepared["_schema_grounded"] is True
    assert prepared["_grounding_errors"] == []


def test_grounder_records_schema_evidence_for_known_terms():
    grounded = prepare_intent(
        {
            "nutrition_filters": [{"nutrient": "kcal", "operator": "<", "value": 300}],
            "dietary_styles": ["vegetarian"],
            "categories": [{"term": "Indian", "category": "region"}],
            "ingredient_logic": {"include": ["paneer"], "exclude": [], "include_operator": "AND"},
        },
        transcript="vegetarian Indian paneer recipes under 300 kcal",
    )

    sources = {item["source"] for item in grounded["_grounding"]}

    assert grounded["_grounding_errors"] == []
    assert {"nutrition", "dietary", "category", "ingredients"} <= sources
    assert any(item["field"] == "calories" for item in grounded["_grounding"])
    assert any(item["field"] == "region" for item in grounded["_grounding"])


def test_sql_builder_rejects_unresolved_dietary_style_instead_of_dropping_it():
    sql = build_sql_query(
        {
            "dietary_styles": ["low-fodmap"],
            "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
            "nutrition_filters": [],
            "sql_operators": [],
        },
        transcript="low fodmap recipes",
    )

    assert sql["query"] is None
    assert sql["validation"]["ok"] is False
    assert any("Unresolved dietary term" in error for error in sql["validation"]["errors"])


def test_sql_builder_rejects_unsupported_aggregate_instead_of_ignoring_it():
    sql = build_sql_query(
        {
            "dietary_styles": [],
            "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
            "nutrition_filters": [],
            "sql_operators": [{"canonical": "AVG", "type": "AGG", "text": "average"}],
        },
        transcript="average calories",
    )

    assert sql["query"] is None
    assert sql["validation"]["ok"] is False
    assert any("Unsupported aggregate 'AVG'" in error for error in sql["validation"]["errors"])


def test_sql_builder_adds_schema_grounded_cuisine_predicates():
    sql = build_sql_query(
        {
            "categories": [{"term": "Indian", "category": "region"}],
            "dietary_styles": [],
            "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
            "nutrition_filters": [],
            "sql_operators": [],
        },
        transcript="Indian recipes",
    )

    assert sql["validation"]["ok"] is True
    assert "r.`region` = ?" in sql["query"]
    assert "r.`sub_region` = ?" in sql["query"]
    assert "r.`continent` = ?" in sql["query"]
    assert "Indian Subcontinent" in sql["params"]


def test_module1_tags_configured_cuisine_terms():
    result = tag_entities("Indian paneer recipes")

    assert any(
        item["term"] == "indian" and item["category"] == "cuisine"
        for item in result["categories"]
    )


def test_sql_intent_builder_does_not_import_planner_internals():
    source = Path("module3/sql_intent_builder.py").read_text(encoding="utf-8")

    assert "query_planner" not in source


def test_schema_registry_uses_real_index_supported_recipe_join():
    registry = load_schema_registry()
    join = registry.relationship_for_relation("ingredients")

    assert registry.base_table["name"] == "RecipeDB_micronutrients_v1"
    assert join is not None
    assert join.from_table == "RecipeDB_ingredient_phrase"
    assert join.from_column == "recipe_no"
    assert join.to_column == "Recipe_id"
    assert join.confidence == "index_supported"


def test_schema_sql_documents_real_tables_indexes_and_join_assumptions():
    schema_sql = load_schema_sql()

    assert "CREATE TABLE `RecipeDB_micronutrients_v1`" in schema_sql
    assert "CREATE TABLE `RecipeDB_ingredient_phrase`" in schema_sql
    assert "CREATE UNIQUE INDEX `idx_micronutrients_recipe_id`" in schema_sql
    assert "RecipeDB_ingredient_phrase.recipe_no -> RecipeDB_micronutrients_v1.Recipe_id" in schema_sql
    assert "MongoDB does not declare foreign keys" in schema_sql


def test_architecture_doc_locks_model_boundary_to_query_ir():
    source = Path("module3/ARCHITECTURE.md").read_text(encoding="utf-8")
    lowered = source.lower()

    assert "raw sql" in lowered
    assert "model or retriever" in lowered
    assert "query_ir_contract.py" in source
    assert "sql_validator.py" in source


def test_query_ir_contract_accepts_constrained_payload_and_compiles():
    payload = {
        "predicates": [
            {"source": "nutrition", "field": "calories", "operator": "<", "values": [300]},
            {"source": "dietary", "field": "dietary_tag", "operator": "=", "values": ["vegan"]},
        ],
        "order_by": [{"field": "protein", "direction": "DESC"}],
        "limit": 3,
    }

    query_ir = query_ir_from_contract(payload)
    sql = build_sql_query({}, query_ir=query_ir)

    assert query_ir_to_payload(query_ir)["select"][0]["expression"] == "r.*"
    assert sql["validation"]["ok"] is True
    assert "r.`calories` < ?" in sql["query"]
    assert "r.`vegan` IN" in sql["query"]
    assert "ORDER BY r.`protein (g)` DESC" in sql["query"]
    assert sql["params"][-1] == 3


def test_query_ir_contract_accepts_configured_aggregate_select():
    query_ir = query_ir_from_contract({"select": [{"expression": "COUNT(*)"}]})
    sql = build_sql_query({}, query_ir=query_ir)

    assert query_ir.select[0].alias == "count"
    assert sql["query"].startswith("SELECT COUNT(*) AS count FROM")


def test_query_ir_contract_rejects_raw_sql_payloads():
    payloads = [
        {"query": "SELECT * FROM RecipeDB_micronutrients_v1"},
        {"select": [{"expression": "r.*", "where": "1=1"}]},
    ]

    for payload in payloads:
        try:
            query_ir_from_contract(payload)
            assert False, "expected QueryIRContractError"
        except QueryIRContractError as error:
            assert "Raw SQL is not accepted" in str(error)


def test_query_ir_contract_rejects_hallucinated_fields_and_selects():
    bad_predicate = {
        "predicates": [
            {"source": "nutrition", "field": "not_real", "operator": "<", "values": [10]},
        ],
    }
    bad_select = {"select": [{"expression": "r.`Recipe_id`"}]}
    bad_alias = {"select": [{"expression": "r.*", "alias": "all_rows"}]}

    for payload in [bad_predicate, bad_select, bad_alias]:
        try:
            query_ir_from_contract(payload)
            assert False, "expected QueryIRContractError"
        except QueryIRContractError:
            pass


def test_query_ir_contract_rejects_out_of_range_values_and_predicate_spam():
    payloads = [
        {
            "predicates": [
                {"source": "nutrition", "field": "calories", "operator": "<", "values": [-1]},
            ],
        },
        {
            "predicates": [
                {"source": "nutrition", "field": "calories", "operator": "BETWEEN", "values": [500, 100]},
            ],
        },
        {
            "predicates": [
                {"source": "ingredients", "field": "ingredient", "operator": "=", "values": [str(index)]}
                for index in range(11)
            ],
        },
    ]

    for payload in payloads:
        try:
            query_ir_from_contract(payload)
            assert False, "expected QueryIRContractError"
        except QueryIRContractError:
            pass


def test_sql_compiler_is_plan_orchestrator_not_query_template():
    source = Path("module3/sql_compiler.py").read_text(encoding="utf-8")

    assert "build_sql_plan" in source
    assert "render_sql_plan" in source
    assert " WHERE " not in source
    assert "EXISTS" not in source


def test_sql_plan_shape_changes_with_query_ir():
    ingredient_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[
            Predicate("ingredients", "ingredient", "=", ("paneer",)),
            Predicate("nutrition", "carbs", "<=", (50,)),
        ],
    )
    nutrition_only_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[Predicate("nutrition", "calories", "<", (300,))],
    )

    ingredient_plan = build_sql_plan(ingredient_ir)
    nutrition_plan = build_sql_plan(nutrition_only_ir)

    assert len(ingredient_plan.where) == 2
    assert ingredient_plan.join_assumptions
    assert len(nutrition_plan.where) == 1
    assert nutrition_plan.join_assumptions == []


def test_single_include_ingredient_uses_configured_join_strategy():
    sql = build_sql_query(
        {
            "ingredient_logic": {"include": ["paneer"], "exclude": [], "include_operator": "AND"},
            "nutrition_filters": [],
            "dietary_styles": [],
            "sql_operators": [],
        },
        transcript="paneer recipes",
    )

    assert sql["validation"]["ok"] is True
    assert sql["query"].startswith("SELECT DISTINCT r.*")
    assert " JOIN `RecipeDB_ingredient_phrase` AS ip ON " in sql["query"]
    assert "ip.`matching_values` = ?" in sql["query"]
    assert "ip.`ingredient_lower` LIKE ?" in sql["query"]
    assert "EXISTS" not in sql["query"]
    assert sql["params"] == ["paneer", "%paneer%"]
    assert sql["plan"]["joins"][0]["source"] == "ingredients"


def test_complex_ingredient_filters_keep_exists_strategy():
    sql = build_sql_query(
        {
            "ingredient_logic": {
                "include": ["paneer", "spinach"],
                "exclude": ["cream"],
                "include_operator": "AND",
            },
            "nutrition_filters": [],
            "dietary_styles": [],
            "sql_operators": [],
        },
        transcript="paneer and spinach recipes without cream",
    )

    assert sql["validation"]["ok"] is True
    assert sql["query"].startswith("SELECT DISTINCT")
    assert "UNION ALL" in sql["query"]
    assert "ip_u1.`matching_values` = ?" in sql["query"]
    assert "ip_u1.`ingredient_lower` LIKE ?" in sql["query"]
    assert "GROUP BY ip_union.`recipe_no`" in sql["query"]
    assert "HAVING COUNT(*) = ?" in sql["query"]
    assert "NOT EXISTS" in sql["query"]
    assert sql["params"] == [
        "paneer",
        "%paneer%",
        "spinach",
        "%spinach%",
        2,
        "cream",
        "%cream%",
    ]


def test_multi_include_ingredients_use_grouped_join_strategy():
    sql = build_sql_query(
        {
            "ingredient_logic": {
                "include": ["chicken", "tomato", "potato"],
                "exclude": [],
                "include_operator": "AND",
            },
            "nutrition_filters": [],
            "dietary_styles": [],
            "sql_operators": [],
        },
        transcript="chicken tomato potato recipes",
    )

    assert sql["validation"]["ok"] is True
    assert "UNION ALL" in sql["query"]
    assert "ip_u1.`matching_values` = ?" in sql["query"]
    assert "ip_u1.`ingredient_lower` LIKE ?" in sql["query"]
    assert "GROUP BY ip_union.`recipe_no`" in sql["query"]
    assert "HAVING COUNT(*) = ?" in sql["query"]
    assert "JOIN `RecipeDB_ingredient_phrase` AS ip1" not in sql["query"]
    assert sql["params"] == [
        "chicken",
        "%chicken%",
        "tomato",
        "%tomato%",
        "potato",
        "%potato%",
        3,
    ]


def test_dynamic_root_can_start_from_ingredient_table():
    sql = build_sql_query(
        {
            "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
            "nutrition_filters": [],
            "dietary_styles": [],
            "sql_operators": [],
        },
        transcript="show ingredients",
    )

    assert sql["validation"]["ok"] is True
    assert sql["query"].startswith("SELECT ip.* FROM `RecipeDB_ingredient_phrase` AS ip")
    assert "`RecipeDB_micronutrients_v1`" not in sql["query"]
    assert sql["ir"]["root"] == "ingredients"


def test_non_recipe_root_fails_closed_for_unsupported_cross_root_filter():
    sql = build_sql_query(
        {
            "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
            "nutrition_filters": [{"nutrient": "calories", "operator": "<", "value": 300}],
            "dietary_styles": [],
            "sql_operators": [],
        },
        transcript="show ingredients under 300 calories",
    )

    assert sql["query"] is None
    assert sql["validation"]["ok"] is False
    assert any("cannot be rendered from root 'ingredients'" in error for error in sql["validation"]["errors"])


def test_sql_builder_uses_schema_linker_for_alias_inputs():
    tag_result = {
        "nutrition_filters": [{"nutrient": "kcal", "operator": "<", "value": 300}],
        "dietary_styles": ["vegetarian"],
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "sql_operators": [],
    }

    sql = build_sql_query(tag_result, transcript="vegetarian recipes under 300 kcal")

    assert sql["validation"]["ok"] is True
    assert "r.`calories` < ?" in sql["query"]
    assert "r.`ovo_lacto_vegetarian` IN" in sql["query"]
    assert sql["params"][0] == 300.0


def test_planner_uses_schema_linker_for_nutrition_alias_endpoint():
    config = load_recipe_db_config()
    tag_result = {
        "nutrition_filters": [{"nutrient": "kcal", "operator": "<", "value": 300}],
        "dietary_styles": [],
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "categories": [],
        "sql_operators": [],
    }

    plans = plan_queries(tag_result, transcript="recipes under 300 kcal")

    assert plans[0].path == endpoint_path("calories", config)


def test_sql_builder_uses_enriched_low_carb_filter():
    tag_result = {
        "ingredients": ["paneer"],
        "ingredient_logic": {"include": ["paneer"], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [],
        "dietary_styles": [],
        "sql_operators": [],
    }

    sql = build_sql_query(tag_result, transcript="low carb dinner with paneer")

    assert "`RecipeDB_ingredient_phrase`" in sql["query"]
    assert "r.`carbohydrate, by difference (g)` <= ?" in sql["query"]
    assert sql["params"] == ["paneer", "%paneer%", 50.0]
    assert sql["join_assumptions"][0]["confidence"] == "index_supported"


def test_planner_uses_enriched_low_carb_endpoint():
    config = load_recipe_db_config()
    tag_result = {
        "ingredients": ["paneer"],
        "ingredient_logic": {"include": ["paneer"], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [],
        "dietary_styles": [],
        "categories": [],
        "sql_operators": [],
    }

    plans = plan_queries(tag_result, transcript="low carb dinner with paneer")

    assert any(plan.path == endpoint_path("carbs", config) for plan in plans)


def test_sql_builder_uses_enriched_between_minutes_filter():
    tag_result = {
        "ingredients": ["garlic"],
        "ingredient_logic": {"include": ["garlic"], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [],
        "dietary_styles": [],
        "sql_operators": [],
    }

    sql = build_sql_query(
        tag_result,
        transcript="recipes between 20 and 40 minutes with garlic",
    )

    assert "r.`total_time` BETWEEN ? AND ?" in sql["query"]
    assert sql["params"] == ["garlic", "%garlic%", 20.0, 40.0]


def test_api_verifier_confirms_matching_recipe_against_ir():
    config = load_recipe_db_config()
    ingredient_list_field = next(field for field in config["search_text_fields"] if field == "ingredients")
    ingredient_value_field = config["nested_ingredient_fields"][0]
    dietary_value = config["diet_aliases"]["vegan"]
    recipe = {
        config["identity_fields"][0]: "1",
        config["dedupe_fallback_fields"]["title"][0]: "Spinach Garlic Soup",
        config["response_field_aliases"]["calories"][0]: "250",
        config["response_field_aliases"]["total_time"][0]: "20",
        dietary_value: "1.0",
        ingredient_list_field: [{ingredient_value_field: "spinach"}, {ingredient_value_field: "garlic"}],
    }
    query_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[
            Predicate("ingredients", "ingredient", "=", ("spinach",)),
            Predicate("nutrition", "calories", "<", (300,)),
            Predicate("base", "total_time", "<=", (30,)),
            Predicate("dietary", "dietary_tag", "=", (dietary_value,)),
        ],
    )

    verification = verify_recipes_against_ir([recipe], query_ir)

    assert verification.ok is True
    assert verification.verified is True
    assert verification.failed_count == 0
    assert verification.unknown_count == 0


def test_api_verifier_reports_response_that_contradicts_ir():
    config = load_recipe_db_config()
    ingredient_list_field = next(field for field in config["search_text_fields"] if field == "ingredients")
    ingredient_value_field = config["nested_ingredient_fields"][0]
    recipe = {
        config["identity_fields"][0]: "2",
        config["dedupe_fallback_fields"]["title"][0]: "Spinach Soup",
        config["response_field_aliases"]["calories"][0]: "450",
        ingredient_list_field: [{ingredient_value_field: "spinach"}],
    }
    query_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[Predicate("nutrition", "calories", "<", (300,))],
    )

    verification = verify_recipes_against_ir([recipe], query_ir)

    assert verification.ok is False
    assert verification.verified is False
    assert verification.failed_count == 1
    assert verification.failures[0].recipe_key.startswith(config["identity_fields"][0])


def test_top_limit_without_sort_cue_does_not_order_by_calories():
    tag_result = {
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [{"nutrient": "calories", "operator": "<", "value": 300}],
        "dietary_styles": [],
        "sql_operators": [{"canonical": "LIMIT", "text": "top 5"}],
    }

    sql = build_sql_query(tag_result, transcript="top 5 recipes under 300 calories")

    assert "ORDER BY" not in sql["query"]
    assert sql["query"].endswith("LIMIT ?")
    assert sql["params"] == [300.0, 5]


def test_prepared_time_from_most_to_least_orders_prep_time_desc():
    tag_result = {
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [],
        "dietary_styles": [],
        "sql_operators": [{"canonical": "ORDER BY", "text": "from most to least"}],
    }

    sql = build_sql_query(
        tag_result,
        transcript="give those recipes which can be have prepared time from most to least",
    )

    assert sql["validation"]["ok"] is True
    assert "ORDER BY r.`prep_time` DESC" in sql["query"]


def test_prepared_time_from_least_to_most_orders_prep_time_asc():
    tag_result = {
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [],
        "dietary_styles": [],
        "sql_operators": [{"canonical": "ORDER BY", "text": "from least to most"}],
    }

    sql = build_sql_query(
        tag_result,
        transcript="prepared time from least to most",
    )

    assert sql["validation"]["ok"] is True
    assert "ORDER BY r.`prep_time` ASC" in sql["query"]


def test_dietary_or_intent_uses_or_condition_not_and():
    tag_result = {
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [],
        "dietary_styles": ["vegan", "vegetarian"],
        "sql_operators": [],
    }

    sql = build_sql_query(tag_result, transcript="show vegan or vegetarian recipes")

    assert sql["validation"]["ok"] is True
    assert sql["ir"]["predicates"][0]["quantifier"] == "or_exists"
    assert "r.`vegan` IN" in sql["query"]
    assert " OR r.`ovo_lacto_vegetarian` IN" in sql["query"]


def test_ranker_uses_same_prepared_time_direction_rules():
    recipes = [
        {"Recipe_id": "1", "prep_time": "10"},
        {"Recipe_id": "2", "prep_time": "50"},
        {"Recipe_id": "3", "prep_time": "30"},
    ]

    ranked = rank_and_limit(
        recipes,
        {"sql_operators": [{"canonical": "ORDER BY", "text": "from most to least"}]},
        transcript="prepared time from most to least",
    )

    assert [recipe["Recipe_id"] for recipe in ranked] == ["2", "3", "1"]


def test_sql_validator_rejects_unknown_fields():
    query_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[
            Predicate(
                source="nutrition",
                field="mystery_nutrient",
                operator="<",
                values=(10,),
            )
        ],
    )

    result = validate_query_ir(query_ir)

    assert result.ok is False
    assert any("mystery_nutrient" in error for error in result.errors)


def test_sql_builder_accepts_pdf_proven_sugar_column():
    sql = build_sql_query(
        {
            "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
            "nutrition_filters": [{"nutrient": "sugar", "operator": "<", "value": 10}],
            "dietary_styles": [],
            "categories": [],
            "sql_operators": [],
        },
        transcript="low sugar recipes",
    )

    assert sql["validation"]["ok"] is True
    assert "r.`sugars, total (g)` < ?" in sql["query"]
    assert sql["params"] == [10.0]


def test_sql_validator_rejects_indirect_join_paths_until_supported():
    query_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[
            Predicate(
                source="ingredient_flavor",
                field="dietrx_category",
                operator="=",
                values=("spice",),
            )
        ],
    )

    result = validate_query_ir(query_ir)

    assert result.ok is False
    assert any("multi-hop join path" in error for error in result.errors)


def test_sql_validator_rejects_bad_operator_value_count():
    query_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[
            Predicate(
                source="nutrition",
                field="calories",
                operator="BETWEEN",
                values=(100,),
            )
        ],
    )

    result = validate_query_ir(query_ir)

    assert result.ok is False
    assert any("expects 2 value" in error for error in result.errors)


def test_sql_validator_rejects_numeric_values_outside_schema_range():
    query_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[
            Predicate(
                source="nutrition",
                field="calories",
                operator="<",
                values=(-10,),
            )
        ],
    )

    result = validate_query_ir(query_ir)

    assert result.ok is False
    assert any("below min" in error for error in result.errors)


def test_sql_builder_returns_validation_errors_without_compiling_bad_ir():
    import module3.sql_query_builder as builder

    bad_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[Predicate("nutrition", "not_real", "<", (1,))],
    )
    original = builder.build_query_ir
    try:
        builder.build_query_ir = lambda *args, **kwargs: bad_ir
        sql = builder.build_sql_query({}, transcript="bad query")
    finally:
        builder.build_query_ir = original

    assert sql["query"] is None
    assert sql["validation"]["ok"] is False


def test_sql_validator_can_require_virtual_evidence():
    query_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[
            Predicate(
                source="nutrition",
                field="fiber",
                operator=">",
                values=(5,),
            )
        ],
    )
    virtual_schema = {
        "virtual_tables": {
            "recipes": {"fields": {}},
            "recipe_nutrition": {"fields": {"calories": {}}},
            "recipe_dietary": {"fields": {}},
        }
    }

    result = validate_query_ir(
        query_ir,
        virtual_schema=virtual_schema,
        require_virtual_evidence=True,
    )

    assert result.ok is False
    assert any("no virtual-schema evidence" in error for error in result.errors)


def test_query_relaxer_uses_configured_progressive_policy():
    query_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[
            Predicate("ingredients", "ingredient", "=", ("paneer",)),
            Predicate("ingredients", "ingredient", "=", ("spinach",)),
            Predicate("ingredients", "ingredient", "IN", ("cream",), quantifier="not_exists"),
            Predicate("nutrition", "calories", "<", (300,)),
        ],
    )

    chain = relaxation_chain(query_ir)

    assert len(chain) == 1
    assert all(predicate.quantifier != "not_exists" for predicate in chain[0].predicates)

    full_chain = relaxation_chain(query_ir, max_steps=3)

    assert len(full_chain) == 3
    assert any(
        predicate.source == "ingredients"
        and predicate.operator == "IN"
        and predicate.quantifier == "exists"
        and predicate.values == ("paneer", "spinach")
        for predicate in full_chain[1].predicates
    )
    assert all(predicate.source != "nutrition" for predicate in full_chain[2].predicates)


def test_query_relaxer_can_drop_category_filters_from_schema_policy():
    query_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[
            Predicate("base", "region", "=", ("Indian Subcontinent",)),
            Predicate("nutrition", "calories", "<", (400,)),
        ],
    )

    relaxed = relax_query_ir(query_ir)

    assert relaxed is not None
    assert all(predicate.field != "region" for predicate in relaxed.predicates)
    assert any(predicate.source == "nutrition" for predicate in relaxed.predicates)


def test_query_relaxer_returns_none_when_no_policy_can_apply():
    query_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[Predicate("ingredients", "ingredient", "IN", ("paneer",))],
    )

    assert relax_query_ir(query_ir) is None


def test_dedupe_merges_empty_safe_values():
    recipes = [
        {"Recipe_id": "1", "Recipe_title": "A", "Protein (g)": ""},
        {"Recipe_id": "1", "Protein (g)": "12"},
    ]

    deduped = dedupe_recipes(recipes)

    assert len(deduped) == 1
    assert deduped[0]["Protein (g)"] == "12"


def test_schema_discovery_summarizes_response_fields():
    data = {
        "data": [
            {
                "Recipe_id": "1",
                "Recipe_title": "Spinach Salad",
                "Calories": "130.6",
                "vegan": "1.0",
            }
        ]
    }

    summary = _summarize_endpoint(data, data["data"])

    assert summary["recipe_count"] == 1
    assert summary["recipe_fields"]["Calories"]["numeric_like"] is True
    assert "string" in summary["recipe_fields"]["Recipe_title"]["types"]


def test_schema_discovery_builds_virtual_tables_from_aliases():
    config = load_recipe_db_config()
    fields = {
        "Recipe_id": {"types": ["string"], "nullable": False, "numeric_like": True, "examples": ["1"], "observations": 1},
        "Calories": {"types": ["string"], "nullable": False, "numeric_like": True, "examples": ["130.6"], "observations": 1},
        "vegan": {"types": ["string"], "nullable": False, "numeric_like": True, "examples": ["1.0"], "observations": 1},
    }

    tables = _build_virtual_tables(fields, config)

    assert tables["recipes"]["candidate_primary_keys"] == ["Recipe_id"]
    assert tables["recipe_nutrition"]["fields"]["calories"]["observed_as"] == "Calories"
    assert tables["recipe_dietary"]["fields"]["vegan"]["observed_as"] == "vegan"


def test_schema_discovery_does_not_overwrite_cache_on_token_exhaustion_without_evidence():
    result = {
        "stopped_early": True,
        "stop_reason": "Not enough tokens",
        "virtual_tables": {
            "recipes": {"fields": {}},
            "recipe_nutrition": {"fields": {}},
            "recipe_dietary": {"fields": {}},
        },
    }

    assert _should_write_cache(result) is False
