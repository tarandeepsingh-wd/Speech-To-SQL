import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
TEMP_DB_DIR = PROJECT_ROOT / "temp_recipe_db"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TEMP_DB_DIR))

from run_query import verify_generated_result
from speech_to_sql.m5_execution.query_planner import plan_queries
from speech_to_sql.m5_execution.api_verifier import verify_recipes_against_ir
from speech_to_sql.common.config import endpoint_path, load_recipe_db_config
from speech_to_sql.m3_query_ir.query_relaxer import relax_query_ir, relaxation_chain
from speech_to_sql.m5_execution.result_merger import dedupe_recipes, merge_api_results
from speech_to_sql.m5_execution.client import ApiResult
from speech_to_sql.m5_execution.query_planner import ApiQueryPlan
from speech_to_sql.m5_execution.post_filter import post_filter
from speech_to_sql.m5_execution.ranker import rank_and_limit
from speech_to_sql.m3_query_ir.intent_enricher import enrich_tag_result
from speech_to_sql.m3_query_ir.intent_grounder import ground_intent
from speech_to_sql.m3_query_ir.intent_preparer import prepare_intent
from speech_to_sql.m3_query_ir.query_ir_contract import (
    QueryIRContractError,
    query_ir_from_contract,
    query_ir_to_payload,
)
from speech_to_sql.m3_query_ir.schema_discovery import _build_virtual_tables, _should_write_cache, _summarize_endpoint
from speech_to_sql.m3_query_ir.schema_registry import load_schema_registry
from speech_to_sql.m4_sql.schema_sql import load_schema_sql
from speech_to_sql.m3_query_ir.schema_linker import link_tag_result
from speech_to_sql.m3_query_ir.sql_ir import Predicate, QueryIR, SelectExpr
from speech_to_sql.m4_sql.sql_plan_builder import build_sql_plan
from speech_to_sql.m4_sql.sql_query_builder import build_sql_query
from speech_to_sql.m4_sql.sql_validator import validate_query_ir
from speech_to_sql.m2_nlu.entity_tagger import tag_entities


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


def test_api_planner_sends_only_dietrx_categories_to_category_endpoint():
    query = "Find vegan recipes with spinach where spinach is categorized as a leafy green."
    prepared = prepare_intent(tag_entities(query), transcript=query)

    plans = plan_queries(prepared, transcript=query)
    ingredient_plan = next(plan for plan in plans if plan.label == "ingredient_category_title")

    assert ingredient_plan.params["includeIngredients"] == "spinach"
    assert "includeCategories" not in ingredient_plan.params


def test_api_planner_uses_dietrx_category_values_when_available():
    query = "Find recipes that use an ingredient from the seafood category."
    prepared = prepare_intent(tag_entities(query), transcript=query)

    plans = plan_queries(prepared, transcript=query)
    ingredient_plan = next(plan for plan in plans if plan.label == "ingredient_category_title")

    assert prepared["ingredient_logic"]["include"] == []
    assert ingredient_plan.params["includeCategories"] == "Seafood"
    assert "includeIngredients" not in ingredient_plan.params


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

    assert sql["query"].startswith("SELECT r.* FROM `recipes` AS r WHERE")
    assert "r.`vegan` IN" in sql["query"]
    assert "ORDER BY (SELECT n_order.`protein (g)`" in sql["query"]
    assert "FROM `micronutrients` AS n_order" in sql["query"]
    assert sql["query"].endswith("LIMIT ?")
    assert sql["params"][-1] == 5
    assert sql["ir"]["predicates"][0]["source"] == "dietary"
    assert sql["validation"]["ok"] is True


def test_plain_veg_excludes_egg_based_vegetarian_flags():
    tag_result = {
        "dietary_styles": ["vegetarian"],
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [],
        "sql_operators": [],
    }
    sql = build_sql_query(
        tag_result,
        transcript="show me veg food",
    )

    query = sql["query"]
    assert sql["validation"]["ok"] is True
    assert "r.`vegan` IN" in query
    assert "r.`lacto_vegetarian` IN" in query
    assert "r.`ovo_vegetarian` IN" not in query
    assert "r.`ovo_lacto_vegetarian` IN" not in query


def test_veg_with_egg_includes_ovo_vegetarian_flags():
    tag_result = {
        "ingredients": ["egg"],
        "ingredient_logic": {"include": ["egg"], "exclude": [], "include_operator": "AND"},
        "dietary_styles": ["vegetarian"],
        "nutrition_filters": [],
        "sql_operators": [],
    }
    sql = build_sql_query(
        tag_result,
        transcript="show vegetarian recipes with egg",
    )

    query = sql["query"]
    assert sql["validation"]["ok"] is True
    assert "r.`ovo_vegetarian` IN" in query
    assert "r.`ovo_lacto_vegetarian` IN" in query
    assert "r.`vegan` IN" not in query
    assert "r.`lacto_vegetarian` IN" not in query


def test_veg_with_dairy_prefers_lacto_vegetarian_flag():
    tag_result = {
        "ingredients": ["paneer"],
        "ingredient_logic": {"include": ["paneer"], "exclude": [], "include_operator": "AND"},
        "dietary_styles": ["vegetarian"],
        "nutrition_filters": [],
        "sql_operators": [],
    }
    sql = build_sql_query(
        tag_result,
        transcript="show vegetarian recipes with paneer",
    )

    query = sql["query"]
    assert sql["validation"]["ok"] is True
    assert "r.`lacto_vegetarian` IN" in query
    assert "r.`ovo_vegetarian` IN" not in query
    assert "r.`ovo_lacto_vegetarian` IN" not in query


def test_general_low_and_high_nutrition_phrases_handle_typos():
    base_tag = {
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [],
        "dietary_styles": [],
        "categories": [],
        "sql_operators": [],
    }
    cases = [
        ("food with low caorires", "calories", "<=", 500.0),
        ("food with high protien", "protein", ">=", 20.0),
        ("food with low cholestrol", "cholesterol", "<=", 20.0),
        ("food with high carbs", "carbohydrate_by_difference", ">=", 50.0),
        ("food with low fat", "total_lipid_fat", "<=", 3.0),
        ("food with high sodium", "sodium", ">=", 400.0),
        ("sugar free food", "sugar", "<=", 0.5),
    ]

    for transcript, field, operator, value in cases:
        sql = build_sql_query(dict(base_tag), transcript=transcript)
        predicates = sql["ir"]["predicates"]

        assert sql["validation"]["ok"] is True
        assert len(predicates) == 1
        assert predicates[0]["field"] == field
        assert predicates[0]["operator"] == operator
        assert predicates[0]["values"] == [value]


def test_relative_nutrition_phrases_sort_instead_of_thresholding():
    base_tag = {
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "nutrition_filters": [],
        "dietary_styles": [],
        "categories": [],
        "sql_operators": [],
    }
    cases = [
        ("food with least caorires", "calories", "ASC"),
        ("food with highest protien", "protein", "DESC"),
        ("food with lowest cholestrol", "cholesterol", "ASC"),
        ("food with min sugar", "sugar", "ASC"),
        ("food with max carbohydrates", "carbs", "DESC"),
    ]

    for transcript, field, direction in cases:
        sql = build_sql_query(dict(base_tag), transcript=transcript)

        assert sql["validation"]["ok"] is True
        assert sql["ir"]["predicates"] == []
        assert sql["ir"]["order_by"] == [{"field": field, "direction": direction}]


def test_numeric_min_max_nutrition_phrases_are_thresholds_not_sorts():
    cases = [
        ("minimum 40 grams protein", "protein", ">=", 40.0),
        ("maximum 60 grams carbs", "carbohydrate_by_difference", "<=", 60.0),
    ]

    for transcript, field, operator, value in cases:
        tag_result = tag_entities(transcript)
        sql = build_sql_query(tag_result, transcript=transcript)

        assert tag_result["ingredients"] == []
        assert tag_result["sql_operators"] == []
        assert sql["validation"]["ok"] is True
        assert sql["ir"]["order_by"] == []
        assert sql["ir"]["predicates"][0]["field"] == field
        assert sql["ir"]["predicates"][0]["operator"] == operator
        assert sql["ir"]["predicates"][0]["values"] == [value]


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


def test_nlu_recovers_diet_typo_and_resets_negation_scope():
    result = tag_entities(
        "recipies containing onion but no garlic contains milk and jackfruit "
        "but are vigan but has chiken"
    )

    assert result["dietary_styles"] == ["vegan"]
    assert result["ingredient_logic"]["include"] == [
        "onion",
        "milk",
        "jackfruit",
        "chicken",
    ]
    assert result["ingredient_logic"]["exclude"] == ["garlic"]

    sql = build_sql_query(result, transcript="vegan recipes with chicken and milk")
    warnings = " ".join(sql["validation"]["warnings"])
    assert "vegan" in warnings
    assert "chicken" in warnings
    assert "milk" in warnings


def test_negated_dietary_style_routes_to_excluded_diet_without_losing_utensil():
    query = (
        "recipes which contain onion but do not contain garlic but contain Milk "
        "but should not be vegan recipes but contain container"
    )

    result = tag_entities(query)
    sql = build_sql_query(result, transcript=query)

    assert result["dietary_styles"] == []
    assert result["excluded_dietary_styles"] == ["vegan"]
    assert result["ingredient_logic"]["include"] == ["onion", "milk"]
    assert result["ingredient_logic"]["exclude"] == ["garlic"]
    assert result["utensils"] == ["container"]
    assert sql["validation"]["ok"] is True
    assert "r.`vegan` NOT IN" in sql["query"]
    assert "CONCAT(?, r.`utensils`, ?) LIKE ?" in sql["query"]
    assert "%||container||%" in sql["params"]


def test_negated_dietary_style_after_do_not_contain_is_not_ingredient_or_positive_diet():
    query = (
        "recipes which contain onion but do not contain garlic but contain Milk "
        "but do not contain vegan recipes but contain container"
    )

    result = tag_entities(query)
    sql = build_sql_query(result, transcript=query)

    assert result["dietary_styles"] == []
    assert result["excluded_dietary_styles"] == ["vegan"]
    assert result["ingredient_logic"]["include"] == ["onion", "milk"]
    assert result["ingredient_logic"]["exclude"] == ["garlic"]
    assert "vegan" not in result["ingredients"]
    assert "container" not in result["ingredients"]
    assert result["utensils"] == ["container"]
    assert sql["validation"]["ok"] is True
    assert "r.`vegan` NOT IN" in sql["query"]
    assert "CONCAT(?, r.`utensils`, ?) LIKE ?" in sql["query"]


def test_query_preparation_preserves_empty_include_for_negated_ingredient():
    query = "give me those recipises that do not contain egg and are not vegan"

    result = tag_entities(query)
    prepared = prepare_intent(result, transcript=query)
    sql = build_sql_query(result, transcript=query)

    assert result["ingredient_logic"]["include"] == []
    assert result["ingredient_logic"]["exclude"] == ["egg"]
    assert prepared["ingredient_logic"]["include"] == []
    assert prepared["ingredient_logic"]["exclude"] == ["egg"]
    assert prepared["excluded_dietary_styles"] == ["vegan"]
    assert "EXISTS (SELECT 1 FROM `ingredient_phrase` AS ip1" not in sql["query"]
    assert "NOT EXISTS (SELECT 1 FROM `ingredient_phrase`" in sql["query"]
    assert "r.`vegan` NOT IN" in sql["query"]
    assert sql["params"] == ["egg", "%egg%", "1", "1.0", "true", "True", "yes", "Y"]


def test_sql_operator_words_are_not_reused_as_fuzzy_ingredients():
    query = "Show me the recipes which are Vegan but should not be vegan"

    result = tag_entities(query)
    sql = build_sql_query(result, transcript=query)

    assert result["ingredients"] == []
    assert result["ingredient_logic"]["include"] == []
    assert result["ingredient_logic"]["exclude"] == []
    assert result["raw_spans"] == []
    assert result["dietary_styles"] == ["vegan"]
    assert result["excluded_dietary_styles"] == ["vegan"]
    assert any(op["canonical"] == "NOT" for op in result["sql_operators"])
    assert "`ingredient_phrase`" not in sql["query"]
    assert "r.`vegan` IN" in sql["query"]
    assert "r.`vegan` NOT IN" in sql["query"]


def test_general_nutrition_label_phrases_apply_without_ingredient_collision():
    query = "show me sugar free food"

    result = tag_entities(query)
    prepared = prepare_intent(result, transcript=query)
    sql = build_sql_query(result, transcript=query)

    assert prepared["ingredient_logic"]["include"] == []
    assert prepared["nutrition_filters"] == [{
        "nutrient": "sugar",
        "operator": "<=",
        "value": 0.5,
        "source": "intent_enrichment",
        "table_source": "nutrition",
    }]
    assert "`sugars, total (g)` <= ?" in sql["query"]
    assert sql["params"] == [0.5]


def test_specific_fat_label_phrase_suppresses_only_generic_presence_filter():
    query = "show me low saturated fat recipes"

    prepared = prepare_intent(tag_entities(query), transcript=query)

    assert prepared["nutrition_filters"] == [{
        "nutrient": "fatty_acids_total_saturated",
        "operator": "<=",
        "value": 1,
        "source": "intent_enrichment",
        "table_source": "nutrition",
    }]


def test_relative_superlative_sorts_without_fake_nutrition_threshold_or_ingredient():
    query = "show me least sugar recipes"

    result = tag_entities(query)
    prepared = prepare_intent(result, transcript=query)
    sql = build_sql_query(result, transcript=query)

    assert prepared["ingredient_logic"]["include"] == []
    assert prepared["nutrition_filters"] == []
    assert "ORDER BY" in sql["query"]
    assert "`sugars, total (g)`" in sql["query"]
    assert sql["params"] == []


def test_general_time_effort_and_quickest_sort_are_distinct():
    quick = "show me quick food"
    quickest = "show me quickest recipes"
    prep = "show me food with less time to prepare"

    quick_prepared = prepare_intent(tag_entities(quick), transcript=quick)
    quickest_prepared = prepare_intent(tag_entities(quickest), transcript=quickest)
    prep_prepared = prepare_intent(tag_entities(prep), transcript=prep)
    quick_sql = build_sql_query(tag_entities(quick), transcript=quick)
    quickest_sql = build_sql_query(tag_entities(quickest), transcript=quickest)

    assert quick_prepared["time_filters"][0]["field"] == "total_time"
    assert quick_prepared["time_filters"][0]["max"] == 30
    assert "ORDER BY" not in quick_sql["query"]
    assert quickest_prepared["time_filters"] == []
    assert "ORDER BY" in quickest_sql["query"]
    assert "total_time" in quickest_sql["query"]
    assert prep_prepared["time_filters"][0]["field"] == "prep_time"
    assert prep_prepared["time_filters"][0]["max"] == 20


def test_non_veg_maps_to_not_vegan_and_not_vegetarian_flag_group():
    query = "show me non veg food"

    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(tag_entities(query), transcript=query)

    assert prepared["dietary_styles"] == []
    assert set(prepared["excluded_dietary_styles"]) == {
        "vegan",
        "ovo_vegetarian",
        "lacto_vegetarian",
        "ovo_lacto_vegetarian",
    }
    assert "r.`vegan` NOT IN" in sql["query"]
    assert "r.`ovo_vegetarian` NOT IN" in sql["query"]
    assert "r.`lacto_vegetarian` NOT IN" in sql["query"]
    assert "r.`ovo_lacto_vegetarian` NOT IN" in sql["query"]


def test_nlu_does_not_treat_at_least_one_ingredient_as_recipe_nutrition_continuation():
    result = tag_entities(
        "Return recipes with lentil where the full recipe has cholesterol below "
        "100 mg but at least one ingredient has thiamin above 0.2 mg."
    )

    assert result["nutrition_filters"] == [
        {"nutrient": "cholesterol", "operator": "<", "value": 100.0},
        {"nutrient": "thiamin", "operator": ">", "value": 0.2},
    ]

    sql = build_sql_query(result, transcript=result["_normalized_query"])

    assert "n2.`cholesterol (mg)` >= ?" not in sql["query"]
    assert sql["validation"]["ok"] is True


def test_dummy_db_verifier_treats_semantic_conflict_zero_rows_as_expected():
    compiled = {
        "validation": {
            "ok": True,
            "errors": [],
            "warnings": [
                "semantic conflict: dietary style 'vegan' usually excludes ingredient 'chicken'"
            ],
        }
    }

    result = verify_generated_result(TEMP_DB_DIR / "recipe_test.db", {}, compiled, [], 0)

    assert result["ok"] is True
    assert result["semantic_conflicts"] == compiled["validation"]["warnings"]
    result_count = next(check for check in result["checks"] if check["name"] == "result_count")
    assert result_count["ok"] is True


def test_dummy_db_verifier_keeps_normal_zero_rows_as_failure():
    compiled = {"validation": {"ok": True, "errors": [], "warnings": []}}

    result = verify_generated_result(TEMP_DB_DIR / "recipe_test.db", {}, compiled, [], 0)

    assert result["ok"] is False
    result_count = next(check for check in result["checks"] if check["name"] == "result_count")
    assert result_count["ok"] is False


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
    assert "CONCAT(?, r.`utensils`, ?) LIKE ?" in sql["query"]
    assert "CONCAT(?, r.`processes`, ?) LIKE ?" in sql["query"]
    assert sql["params"] == [
        "||",
        "||",
        "%||skillet||%",
        "||",
        "||",
        "%||fry||%",
    ]


def test_text_to_sql_uses_extracted_utensil_and_action_filters():
    tags = tag_entities("fried recipes made in skillet")
    sql = build_sql_query(tags, transcript="fried recipes made in skillet")

    assert any(
        action["action"] == "fry"
        and action["type"] == "state_change"
        and action["span"] == (0, 4)
        and action["confidence"] >= 0.5
        for action in tags["actions"]
    )
    assert tags["utensils"] == ["skillet"]
    assert "CONCAT(?, r.`utensils`, ?) LIKE ?" in sql["query"]
    assert "CONCAT(?, r.`processes`, ?) LIKE ?" in sql["query"]
    assert "%||skillet||%" in sql["params"]
    assert "%||fry||%" in sql["params"]


def test_utensil_sql_matches_pipe_delimited_database_tokens():
    sql = build_sql_query(
        {
            "ingredients": [],
            "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
            "utensils": ["pan"],
            "actions": [],
            "dietary_styles": [],
            "nutrition_filters": [],
            "sql_operators": [],
        },
        transcript="recipes using pan",
    )

    assert sql["validation"]["ok"] is True
    assert sql["query"] == (
        "SELECT r.* FROM `recipes` AS r WHERE "
        "(CONCAT(?, r.`utensils`, ?) LIKE ?)"
    )
    assert sql["params"] == ["||", "||", "%||pan||%"]


def test_utensil_word_is_not_normalized_into_numeric_comparator():
    query = (
        "show me the recipe that contain protein not above 40 "
        "and also required bowl to prepare"
    )

    tags = tag_entities(query)
    sql = build_sql_query(tags, transcript=query)

    assert "bowl" in tags["_normalized_query"]
    assert "below to prepare" not in tags["_normalized_query"]
    assert tags["utensils"] == ["bowl"]
    assert tags["nutrition_filters"] == [
        {"nutrient": "protein", "operator": "<=", "value": 40.0}
    ]
    assert "CONCAT(?, r.`utensils`, ?) LIKE ?" in sql["query"]
    assert "%||bowl||%" in sql["params"]


def test_multiple_required_utensils_are_conjoined_not_alternatives():
    query = (
        "show me the recipise that contain protein not above 40 "
        "and also required skillet and pan to prepare"
    )

    tags = tag_entities(query)
    sql = build_sql_query(tags, transcript=query)

    assert tags["utensils"] == ["skillet", "pan"]
    assert "%||skillet||%" in sql["params"]
    assert "%||pan||%" in sql["params"]
    assert "CONCAT(?, r.`utensils`, ?) LIKE ? AND CONCAT(?, r.`utensils`, ?) LIKE ?" in sql["query"]


def test_misspelled_utensil_uses_csv_vocab_without_ingredient_bleed():
    tags = tag_entities("recipes made in a skilit with garlic")

    assert tags["utensils"] == ["skillet"]
    assert tags["ingredients"] == ["garlic"]
    assert all(span["canonical"] != "skilit" for span in tags["raw_spans"])
    skillet_span = next(span for span in tags["raw_spans"] if span["canonical"] == "skillet")
    assert skillet_span["confidence"] >= 0.5


def test_short_utensil_typos_use_vocab_confidence_without_ingredient_bleed():
    oven_tags = tag_entities("recipes made in an ovne with potato")
    fork_tags = tag_entities("recipes made with a fokr and tomato")

    assert oven_tags["utensils"] == ["oven"]
    assert oven_tags["ingredients"] == ["potato"]
    assert fork_tags["utensils"] == ["fork"]
    assert fork_tags["ingredients"] == ["tomato"]


def test_noisy_utensil_csv_rows_return_stronger_canonical():
    tags = tag_entities("recipes that require a saucpan with onion")

    assert tags["utensils"] == ["saucepan"]
    assert tags["ingredients"] == ["onion"]
    assert all(span["canonical"] != "saucpan" for span in tags["raw_spans"])


def test_misspelled_instruction_action_uses_csv_vocab_without_ingredient_bleed():
    query = "Show recipes where instructions mention sautte with tomato."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert [item["action"] for item in prepared["actions"]] == ["saute"]
    assert prepared["ingredient_logic"]["include"] == ["tomato"]
    assert "`instructions`" in sql["query"]
    assert "%saute%" in sql["params"]


def test_instruction_action_context_wins_over_fuzzy_utensil_overlap():
    tags = tag_entities("recipes where instructions mention seve with tomato")

    assert tags["utensils"] == []
    assert [item["action"] for item in tags["actions"]] == ["serve"]
    assert tags["ingredients"] == ["tomato"]


def test_utensil_context_wins_over_overlapping_action_candidate():
    tags = tag_entities("recipes using a whisk with egg")

    assert tags["utensils"] == ["whisk"]
    assert tags["actions"] == []
    assert tags["ingredients"] == ["egg"]


def test_ingredient_typo_does_not_become_fuzzy_utensil():
    tags = tag_entities("recipes with paner")

    assert tags["utensils"] == []
    assert tags["ingredients"]


def test_exact_utensil_keeps_ingredient_terms_out_of_other_entity_lanes():
    tags = tag_entities("mixer cake with butter and sugar")

    assert tags["utensils"] == ["mixer"]
    assert {"cake", "butter", "sugar"}.issubset(tags["ingredients"])
    assert tags["actions"] == []
    mixer_span = next(span for span in tags["raw_spans"] if span["canonical"] == "mixer")
    assert mixer_span["confidence"] == 1.0


def test_utensil_and_normalizer_do_not_displace_exact_ingredients():
    tags = tag_entities("blender soup with tomato and cream")

    assert tags["utensils"] == ["blender"]
    assert {"soup", "tomato", "cream"}.issubset(tags["ingredients"])
    assert "korean" not in tags["ingredients"]


def test_wok_remains_csv_backed_utensil():
    tags = tag_entities("wok fried rice with egg")

    assert tags["utensils"] == ["wok"]
    assert {"rice", "egg"}.issubset(tags["ingredients"])


def test_text_to_sql_uses_pdf_enriched_nutrition_columns():
    tags = tag_entities("recipes with folate over 50")
    sql = build_sql_query(tags, transcript="recipes with folate over 50")

    assert tags["nutrition_filters"] == [
        {"nutrient": "folate_total", "operator": ">", "value": 50.0}
    ]
    assert "n1.`folate, total (g)` > ?" in sql["query"]
    assert sql["params"] == [50.0]
    assert sql["validation"]["ok"] is True


def test_nutrition_presence_does_not_mask_unresolved_numeric_comparator():
    tags = tag_entities("recipes have calories greater than banana")

    assert tags["nutrition_filters"] == []


def test_text_to_sql_uses_low_sugar_enrichment_without_category_conflict():
    tags = tag_entities("low sugar recipes")
    sql = build_sql_query(tags, transcript="low sugar recipes")

    assert tags["categories"] == []
    assert "n1.`sugars, total (g)` <= ?" in sql["query"]
    assert sql["params"] == [10]
    assert sql["validation"]["ok"] is True


def test_mixed_recipe_and_ingredient_nutrition_uses_scoped_tables():
    query = "Show recipes where the full recipe has low carbs but at least one ingredient has high protein."
    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    nutrition = {item["nutrient"]: item for item in prepared["nutrition_filters"]}

    assert tags["dietary_styles"] == []
    assert nutrition["carbs"]["table_source"] == "nutrition"
    assert nutrition["carbs"]["operator"] == "<="
    assert nutrition["protein"]["table_source"] == "ingredient_nutrition"
    assert nutrition["protein"]["operator"] == ">="
    assert sql["validation"]["ok"] is True
    assert "n1.`carbohydrate, by difference (g)` <= ?" in sql["query"]
    assert "`ingredient_nutrition`" in sql["query"]
    assert "`protein (g)` >= ?" in sql["query"]
    assert "ORDER BY" not in sql["query"]


def test_ingredient_level_fat_filter_is_not_recipe_level_or_category():
    query = "Find recipes with recipe-level cholesterol below 20 mg but an ingredient that contains fat."
    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    nutrition = {item["nutrient"]: item for item in prepared["nutrition_filters"]}

    assert tags["categories"] == []
    assert nutrition["cholesterol"]["table_source"] == "nutrition"
    assert nutrition["fat"]["table_source"] == "ingredient_nutrition"
    assert sql["validation"]["ok"] is True
    assert "n1.`cholesterol (mg)` < ?" in sql["query"]
    assert "`ingredient_nutrition`" in sql["query"]
    assert "`total lipid (fat) (g)` > ?" in sql["query"]


def test_high_fiber_recipe_with_ingredient_thiamin_scopes_each_nutrient():
    query = "Show high-fiber recipes where lentil contributes thiamin at the ingredient level."
    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    nutrition = {item["nutrient"]: item for item in prepared["nutrition_filters"]}

    assert nutrition["fiber"]["table_source"] == "nutrition"
    assert nutrition["fiber"]["operator"] == ">="
    assert nutrition["thiamin"]["table_source"] == "ingredient_nutrition"
    assert sql["validation"]["ok"] is True
    assert "n1.`fiber, total dietary (g)` >= ?" in sql["query"]
    assert "`ingredient_nutrition`" in sql["query"]
    assert "`thiamin (mg)` > ?" in sql["query"]


def test_seafood_category_and_ingredient_level_cholesterol_use_correct_sources():
    query = "Find recipes under 500 calories that use an ingredient from the seafood category and have ingredient-level cholesterol."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)
    nutrition = {item["nutrient"]: item for item in prepared["nutrition_filters"]}

    assert nutrition["calories"]["table_source"] == "nutrition"
    assert nutrition["cholesterol"]["table_source"] == "ingredient_nutrition"
    assert prepared["category_filters"] == [
        {
            "source": "ingredient_flavor",
            "field": "dietrx_category",
            "operator": "=",
            "value": "Seafood",
            "term": "seafood",
        }
    ]
    assert sql["validation"]["ok"] is True
    assert "n1.`calories` < ?" in sql["query"]
    assert "`ingredient_nutrition`" in sql["query"]
    assert "`ingredient_flavor`" in sql["query"]


def test_ingredient_level_numeric_nutrition_with_copula_uses_ingredient_table():
    query = "Show recipes with salmon where the ingredient-level protein is above 20g."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)
    nutrition = {item["nutrient"]: item for item in prepared["nutrition_filters"]}

    assert nutrition["protein"]["table_source"] == "ingredient_nutrition"
    assert nutrition["protein"]["operator"] == ">"
    assert nutrition["protein"]["value"] == 20.0
    assert sql["validation"]["ok"] is True
    assert "`ingredient_phrase`" in sql["query"]
    assert "`ingredient_nutrition`" in sql["query"]
    assert "`protein (g)` > ?" in sql["query"]


def test_greater_than_copula_numeric_nutrition_keeps_threshold():
    query = "Show recipes with lentil where ingredient-level thiamin is greater than 0.2 mg."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)
    nutrition = {item["nutrient"]: item for item in prepared["nutrition_filters"]}

    assert nutrition["thiamin"]["table_source"] == "ingredient_nutrition"
    assert nutrition["thiamin"]["operator"] == ">"
    assert nutrition["thiamin"]["value"] == 0.2
    assert sql["validation"]["ok"] is True
    assert "`ingredient_nutrition`" in sql["query"]
    assert "`thiamin (mg)` > ?" in sql["query"]


def test_total_recipe_vitamin_c_with_copula_stays_recipe_level():
    query = "Show recipes with tomato where the instructions mention saute and total recipe vitamin C is above 10 mg."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)
    nutrition = {item["nutrient"]: item for item in prepared["nutrition_filters"]}

    assert nutrition["vitamin_c"]["table_source"] == "nutrition"
    assert nutrition["vitamin_c"]["operator"] == ">"
    assert nutrition["vitamin_c"]["value"] == 10.0
    assert sql["validation"]["ok"] is True
    assert "n1.`vitamin c, total ascorbic acid (mg)` > ?" in sql["query"]
    assert "`instructions`" in sql["query"]
    assert "ins1.`steps` LIKE ?" in sql["query"]
    assert "r.`processes` LIKE ?" not in sql["query"]


def test_total_recipe_scope_does_not_become_sum_aggregate():
    query = "Show recipes with salmon where total recipe calories are under 500 and ingredient-level protein is above 20g."
    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)
    nutrition = {item["nutrient"]: item for item in prepared["nutrition_filters"]}

    assert all(item.get("canonical") != "SUM" for item in tags["sql_operators"])
    assert nutrition["calories"]["table_source"] == "nutrition"
    assert nutrition["protein"]["table_source"] == "ingredient_nutrition"
    assert sql["validation"]["ok"] is True
    assert "n1.`calories` < ?" in sql["query"]
    assert "`protein (g)` > ?" in sql["query"]


def test_same_nutrient_threshold_can_exist_at_recipe_and_ingredient_level():
    query = "Find recipes with lemon juice where total recipe vitamin C is above 10 mg and ingredient-level vitamin C is above 10 mg."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["lemon juice"]
    vitamin_c_filters = [
        item for item in prepared["nutrition_filters"]
        if item["nutrient"] == "vitamin_c"
    ]
    assert [item["table_source"] for item in vitamin_c_filters] == [
        "nutrition",
        "ingredient_nutrition",
    ]
    assert sql["validation"]["ok"] is True
    assert "n1.`vitamin c, total ascorbic acid (mg)` > ?" in sql["query"]
    assert "`vitamin c, total ascorbic acid (mg)` > ?" in sql["query"]


def test_leafy_green_category_does_not_create_grain_ingredient():
    query = "Find recipes with spinach where spinach is leafy green and steps mention chop."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["spinach"]
    assert prepared["category_filters"][0]["value"] == "leafy green"
    assert sql["validation"]["ok"] is True
    assert "%grain%" not in sql["params"]
    assert "`ingredient_flavor`" in sql["query"]
    assert "`instructions`" in sql["query"]


def test_repeated_ingredient_and_nutrient_term_prefers_scoped_nutrition_mention():
    query = "Show recipes with water where ingredient-level water is above 50g."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)
    nutrition = {item["nutrient"]: item for item in prepared["nutrition_filters"]}

    assert prepared["ingredient_logic"]["include"] == ["water"]
    assert nutrition["water"]["table_source"] == "ingredient_nutrition"
    assert sql["validation"]["ok"] is True
    assert "`ingredient_nutrition`" in sql["query"]
    assert "`water (g)` > ?" in sql["query"]


def test_ingredient_phrase_contains_nutrient_word_stays_ingredient_filter():
    query = "Show recipes where the instruction steps mention stir and the ingredient phrase contains sugar."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["sugar"]
    assert prepared["nutrition_filters"] == []
    assert sql["validation"]["ok"] is True
    assert "`ingredient_phrase`" in sql["query"]
    assert "`instructions`" in sql["query"]
    assert "`micronutrients`" not in sql["query"]


def test_plain_ingredient_pair_with_sugar_does_not_become_nutrition_filter():
    query = "Fetch recipes with lemon juice and sugar for review."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["lemon juice", "sugar"]
    assert prepared["nutrition_filters"] == []
    assert sql["validation"]["ok"] is True
    assert "`ingredient_phrase`" in sql["query"]
    assert "`micronutrients`" not in sql["query"]


def test_having_calorie_filter_does_not_create_fuzzy_asafoetida():
    query = "Search for recipes with tomato having no more than 600 calories for testing."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["tomato"]
    assert prepared["nutrition_filters"][0]["operator"] == "<="
    assert "asafoetida" not in sql["params"]


def test_project_suffix_words_do_not_become_instruction_actions():
    query = "List dishes where the steps mention saute for a pipeline check."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert [item["action"] for item in prepared["actions"]] == ["saute"]
    assert sql["params"] == ["%saute%"]


def test_project_suffix_words_do_not_become_ingredients():
    query = "Can you show meals whose instructions mention boil for a dummy data run."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == []
    assert sql["validation"]["ok"] is True
    assert "date" not in sql["params"]


def test_relative_clause_word_does_not_stick_to_ingredient():
    query = "Get me dishes with chicken that take no less than 5 minutes for validation."
    prepared = prepare_intent(tag_entities(query), transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["chicken"]
    assert prepared["time_filters"][0]["field"] == "total_time"


def test_time_instruction_and_water_phrase_keep_required_joins():
    query = "Show recipes under 60 minutes where the steps mention stir and the ingredient phrase contains water."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["water"]
    assert [action["action"] for action in prepared["actions"]] == ["stir"]
    assert sql["validation"]["ok"] is True
    assert "`recipes` AS r" in sql["query"]
    assert "`ingredient_phrase`" in sql["query"]
    assert "`instructions`" in sql["query"]
    assert "`micronutrients`" not in sql["query"]
    assert "%water%" in sql["params"]
    assert "%stir%" in sql["params"]


def test_long_project_context_does_not_drop_exact_ingredient_phrase_match():
    query = (
        "Bring back menu recipes under 60 minutes where where instructions contain stir "
        "and the ingredient phrase contains water from the local sample."
    )
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["water"]
    assert [action["action"] for action in prepared["actions"]] == ["stir"]
    assert sql["validation"]["ok"] is True
    assert "`ingredient_phrase`" in sql["query"]
    assert "`instructions`" in sql["query"]
    assert "%water%" in sql["params"]


def test_generated_flavor_aliases_use_ingredient_flavor_join():
    query = "Find vegan recipes with carrot where carrot is categorized as root vegetable."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["carrot"]
    assert prepared["category_filters"][0]["source"] == "ingredient_flavor"
    assert prepared["category_filters"][0]["field"] == "flavordb_category"
    assert prepared["category_filters"][0]["value"] == "root vegetable"
    assert sql["validation"]["ok"] is True
    assert "`ingredient_flavor`" in sql["query"]


def test_flavor_category_context_does_not_extract_flavor_as_ingredient():
    query = "Show recipes with tomato where tomato has flavor category nightshade for testing."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["tomato"]
    assert prepared["category_filters"][0]["value"] == "nightshade"
    assert "flavor" not in sql["params"]


def test_limit_cue_does_not_become_instruction_action():
    query = "Show top 4 recipes with tomato where instructions mention saute."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert [item["action"] for item in prepared["actions"]] == ["saute"]
    assert sql["validation"]["ok"] is True
    assert "%top%" not in sql["params"]
    assert "LIMIT ?" in sql["query"]
    assert sql["params"][-1] == 4


def test_flavor_category_and_instruction_mentions_use_joined_tables():
    query = "Show recipes that use garlic, where garlic is categorized as an allium, and the instructions mention saute."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["garlic"]
    assert prepared["category_filters"][0]["source"] == "ingredient_flavor"
    assert prepared["category_filters"][0]["field"] == "flavordb_category"
    assert sql["validation"]["ok"] is True
    assert "`ingredient_phrase`" in sql["query"]
    assert "`ingredient_flavor`" in sql["query"]
    assert "`instructions`" in sql["query"]
    assert "ins1.`steps` LIKE ?" in sql["query"]
    assert "r.`processes` LIKE ?" not in sql["query"]
    assert "%saute%" in sql["params"]


def test_instruction_scoped_utensil_does_not_duplicate_base_utensil_filter():
    query = "Find recipes where the ingredient phrase contains chicken and the instruction steps mention skillet."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["chicken"]
    assert prepared["utensils"] == ["skillet"]
    assert sql["validation"]["ok"] is True
    assert "`ingredient_phrase`" in sql["query"]
    assert "`instructions`" in sql["query"]
    assert "ins1.`steps` LIKE ?" in sql["query"]
    assert "r.`utensils` LIKE ?" not in sql["query"]
    assert "%skillet%" in sql["params"]


def test_later_instruction_clause_does_not_capture_base_utensil_or_time_word():
    query = (
        "Find Indian recipes with chicken, garlic, ginger, and tomato that use a saucepan, "
        "where the instructions mention simmer, and prep time is under 30 minutes."
    )
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["chicken", "garlic", "ginger", "tomato"]
    assert prepared["utensils"] == ["saucepan"]
    assert [item["action"] for item in prepared["actions"]] == ["simmer"]
    assert prepared["time_filters"] == [
        {
            "field": "prep_time",
            "min": 0,
            "max": 30.0,
            "source": "intent_enrichment",
        }
    ]
    assert "n1.`prep_time`" not in sql["query"]
    assert "CONCAT(?, r.`utensils`, ?) LIKE ?" in sql["query"]
    assert "r.`processes` LIKE ?" not in sql["query"]
    assert "ins1.`steps` LIKE ?" in sql["query"]
    assert "ins2.`steps` LIKE ?" not in sql["query"]
    assert "%tie%" not in sql["params"]
    assert sql["params"].count("%simmer%") == 1
    assert sql["params"].count("%||saucepan||%") == 1


def test_quantity_available_and_cooking_step_mentions_use_phrase_and_instruction_tables():
    query = "Show recipes with ingredient quantity available in the ingredient phrase and cooking steps that mention bake."
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == []
    assert sql["validation"]["ok"] is True
    assert "`ingredient_phrase`" in sql["query"]
    assert "ip1.`quantity` LIKE ?" in sql["query"]
    assert "`instructions`" in sql["query"]
    assert "ins1.`steps` LIKE ?" in sql["query"]
    assert "r.`processes` LIKE ?" not in sql["query"]
    assert "%bake%" in sql["params"]
    assert "%cook%" not in sql["params"]


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


def test_m2_nlu_tags_configured_cuisine_terms():
    result = tag_entities("Indian paneer recipes")

    assert any(
        item["term"] == "indian" and item["category"] == "cuisine"
        for item in result["categories"]
    )


def test_sql_intent_builder_does_not_import_planner_internals():
    source = Path("src/speech_to_sql/m3_query_ir/sql_intent_builder.py").read_text(encoding="utf-8")

    assert "query_planner" not in source


def test_schema_registry_uses_real_schema_recipe_join():
    registry = load_schema_registry()
    join = registry.relationship_for_relation("ingredients")

    assert registry.base_table["name"] == "recipes"
    assert join is not None
    assert join.from_table == "ingredient_phrase"
    assert join.from_column == "recipe_no"
    assert join.to_table == "recipes"
    assert join.to_column == "recipe_id"
    assert join.confidence == "schema"


def test_schema_sql_documents_real_tables_and_join_assumptions():
    schema_sql = load_schema_sql()

    assert "CREATE TABLE `micronutrients`" in schema_sql
    assert "CREATE TABLE `ingredient_phrase`" in schema_sql
    assert "CREATE INDEX" not in schema_sql
    assert "matching_values" not in schema_sql
    assert "ingredient_phrase.recipe_no -> recipes.recipe_id" in schema_sql


def test_architecture_doc_locks_model_boundary_to_query_ir():
    source = Path("docs/pipeline_architecture.md").read_text(encoding="utf-8")
    lowered = source.lower()

    assert "raw sql" in lowered
    assert "any other model" in lowered
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
    assert "n1.`calories` < ?" in sql["query"]
    assert "r.`vegan` IN" in sql["query"]
    assert "ORDER BY (SELECT n_order.`protein (g)`" in sql["query"]
    assert sql["params"][-1] == 3


def test_query_ir_contract_accepts_configured_aggregate_select():
    query_ir = query_ir_from_contract({"select": [{"expression": "COUNT(*)"}]})
    sql = build_sql_query({}, query_ir=query_ir)

    assert query_ir.select[0].alias == "count"
    assert sql["query"].startswith("SELECT COUNT(*) AS count FROM")


def test_query_ir_contract_rejects_raw_sql_payloads():
    payloads = [
        {"query": "SELECT * FROM micronutrients"},
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
    bad_select = {"select": [{"expression": "r.`not_real`"}]}
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
    source = Path("src/speech_to_sql/m4_sql/sql_compiler.py").read_text(encoding="utf-8")

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
    assert nutrition_plan.join_assumptions
    assert nutrition_plan.join_assumptions[0]["name"] == "micronutrients_to_recipes_by_recipe_id"


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
    assert " JOIN `ingredient_phrase` AS ip ON " in sql["query"]
    assert "ip.`ingredient` = ?" in sql["query"]
    assert "ip.`ingredient_phrase` LIKE ?" in sql["query"]
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
    assert "ip_u1.`ingredient` = ?" in sql["query"]
    assert "ip_u1.`ingredient_phrase` LIKE ?" in sql["query"]
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
    assert "ip_u1.`ingredient` = ?" in sql["query"]
    assert "ip_u1.`ingredient_phrase` LIKE ?" in sql["query"]
    assert "GROUP BY ip_union.`recipe_no`" in sql["query"]
    assert "HAVING COUNT(*) = ?" in sql["query"]
    assert "JOIN `ingredient_phrase` AS ip1" not in sql["query"]
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
    assert sql["query"].startswith("SELECT ip.* FROM `ingredient_phrase` AS ip")
    assert "`micronutrients`" not in sql["query"]
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


def test_food_item_query_with_scoped_ingredient_and_steps_stays_recipe_root():
    query = (
        "Show food items under 60 minutes where the steps mention stir "
        "and the ingredient phrase contains sugar."
    )
    prepared = prepare_intent(tag_entities(query), transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert sql["validation"]["ok"] is True
    assert sql["ir"]["root"] == "recipe"
    assert sql["query"].startswith("SELECT DISTINCT r.* FROM `recipes` AS r")
    assert "`ingredient_phrase`" in sql["query"]
    assert "`instructions`" in sql["query"]


def test_food_item_instruction_only_query_returns_recipes():
    query = "Show food items where the steps mention stir."
    sql = build_sql_query(tag_entities(query), transcript=query)

    assert sql["validation"]["ok"] is True
    assert sql["ir"]["root"] == "recipe"
    assert sql["query"].startswith("SELECT r.* FROM `recipes` AS r")


def test_sql_builder_uses_schema_linker_for_alias_inputs():
    tag_result = {
        "nutrition_filters": [{"nutrient": "kcal", "operator": "<", "value": 300}],
        "dietary_styles": ["vegetarian"],
        "ingredient_logic": {"include": [], "exclude": [], "include_operator": "AND"},
        "sql_operators": [],
    }

    sql = build_sql_query(tag_result, transcript="vegetarian recipes under 300 kcal")

    assert sql["validation"]["ok"] is True
    assert "n1.`calories` < ?" in sql["query"]
    assert sql["ir"]["predicates"][1]["quantifier"] == "or_exists"
    assert "r.`ovo_vegetarian` IN" in sql["query"]
    assert "r.`lacto_vegetarian` IN" in sql["query"]
    assert "r.`ovo_lacto_vegetarian` IN" in sql["query"]
    assert sql["params"][0] == 300.0


def test_veg_abbreviation_maps_to_vegetarian_without_breaking_non_veg():
    veg_query = "show me the top 100 veg recipies"
    veg_tags = tag_entities(veg_query)
    veg_sql = build_sql_query(veg_tags, transcript=veg_query)

    assert veg_tags["dietary_styles"] == ["vegetarian"]
    assert veg_tags["ingredients"] == []
    assert veg_sql["ir"]["predicates"][0]["values"] == [
        "ovo_vegetarian",
        "lacto_vegetarian",
        "ovo_lacto_vegetarian",
    ]
    assert "r.`ovo_vegetarian` IN" in veg_sql["query"]
    assert "r.`lacto_vegetarian` IN" in veg_sql["query"]
    assert "r.`ovo_lacto_vegetarian` IN" in veg_sql["query"]
    assert "LIMIT ?" in veg_sql["query"]
    assert veg_sql["params"][-1] == 100

    non_veg_tags = tag_entities("show me top 20 non veg recipes")
    assert non_veg_tags["dietary_styles"] == ["non-vegetarian"]
    assert "vegetarian" not in non_veg_tags["dietary_styles"]


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

    assert "`ingredient_phrase`" in sql["query"]
    assert "n1.`carbohydrate, by difference (g)` <= ?" in sql["query"]
    assert sql["params"] == ["paneer", "%paneer%", 50.0]
    assert sql["join_assumptions"][0]["confidence"] == "schema"


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

    assert "CAST(NULLIF(r.`total_time`, '') AS DOUBLE) BETWEEN ? AND ?" in sql["query"]
    assert sql["params"] == ["garlic", "%garlic%", 20.0, 40.0]


def test_sql_builder_handles_not_more_than_minutes_as_base_time_upper_bound():
    query = (
        "GIVE ME THE FOOD ITEMS THAT CAN BE PREPARED IN NOT MORE THAN "
        "30 MINS AND HAVE CALORIES LESS THAN 500"
    )

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["time_filters"] == [
        {
            "field": "total_time",
            "min": 0,
            "max": 30.0,
            "source": "intent_enrichment",
        }
    ]
    assert "n1.`calories` < ?" in sql["query"]
    assert "CAST(NULLIF(r.`total_time`, '') AS DOUBLE) <= ?" in sql["query"]
    assert "n2.`total_time`" not in sql["query"]
    assert sql["params"] == [500.0, 30.0]


def test_sql_builder_handles_not_more_than_calories_with_include_and_exclude():
    query = (
        "Give me the recipies which does not contain onion but should contain "
        "garlic having not more than 333 calories"
    )

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["garlic"]
    assert prepared["ingredient_logic"]["exclude"] == ["onion"]
    assert prepared["nutrition_filters"][0]["nutrient"] == "calories"
    assert prepared["nutrition_filters"][0]["operator"] == "<="
    assert prepared["nutrition_filters"][0]["value"] == 333.0
    assert "n1.`calories` <= ?" in sql["query"]
    assert "n1.`calories` > ?" not in sql["query"]
    assert sql["params"] == ["garlic", "%garlic%", "onion", "%onion%", 333.0]


def test_sql_builder_handles_not_less_than_calories_as_lower_bound():
    query = "Give me recipes with garlic not less than 333 calories"

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["garlic"]
    assert prepared["nutrition_filters"][0]["nutrient"] == "calories"
    assert prepared["nutrition_filters"][0]["operator"] == ">="
    assert prepared["nutrition_filters"][0]["value"] == 333.0
    assert "n1.`calories` >= ?" in sql["query"]
    assert "n1.`calories` < ?" not in sql["query"]
    assert sql["params"] == ["garlic", "%garlic%", 333.0]


def test_sql_builder_handles_negated_greater_than_calories_as_upper_bound():
    query = "Give me recipes with garlic not greater than 333 calories"

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["garlic"]
    assert prepared["nutrition_filters"][0]["nutrient"] == "calories"
    assert prepared["nutrition_filters"][0]["operator"] == "<="
    assert prepared["nutrition_filters"][0]["value"] == 333.0
    assert "n1.`calories` <= ?" in sql["query"]
    assert "n1.`calories` > ?" not in sql["query"]
    assert sql["params"] == ["garlic", "%garlic%", 333.0]


def test_sql_builder_handles_negated_below_calories_as_lower_bound():
    query = "Give me recipes with garlic not below 333 calories"

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["garlic"]
    assert prepared["nutrition_filters"][0]["nutrient"] == "calories"
    assert prepared["nutrition_filters"][0]["operator"] == ">="
    assert prepared["nutrition_filters"][0]["value"] == 333.0
    assert "n1.`calories` >= ?" in sql["query"]
    assert "n1.`calories` < ?" not in sql["query"]
    assert sql["params"] == ["garlic", "%garlic%", 333.0]


def test_nutrition_relation_negation_inverts_local_comparator_scope():
    cases = [
        (
            "give me those recipise that do not have calories more than 900",
            "calories",
            "<=",
            900.0,
            "n1.`calories` <= ?",
        ),
        (
            "recipes should not have protein less than 40",
            "protein",
            ">=",
            40.0,
            "n1.`protein (g)` >= ?",
        ),
        (
            "recipes do not contain fat above 20",
            "fat",
            "<=",
            20.0,
            "n1.`total lipid (fat) (g)` <= ?",
        ),
    ]

    for query, nutrient, operator, value, sql_fragment in cases:
        tags = tag_entities(query)
        prepared = prepare_intent(tags, transcript=query)
        sql = build_sql_query(prepared, transcript=query)

        assert [
            (item["nutrient"], item["operator"], item["value"])
            for item in prepared["nutrition_filters"]
        ] == [(nutrient, operator, value)]
        assert sql_fragment in sql["query"]
        assert sql["params"] == [value]


def test_nutrition_value_with_gm_of_nutrient_is_numeric_not_presence():
    query = (
        "recipise that have calories less than 900 and more than 500 "
        "and also have 20 gm of proteins"
    )

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert [
        (item["nutrient"], item["operator"], item["value"])
        for item in prepared["nutrition_filters"]
    ] == [
        ("calories", "<", 900.0),
        ("calories", ">", 500.0),
        ("protein", "=", 20.0),
    ]
    assert "n3.`protein (g)` = ?" in sql["query"]
    assert "n3.`protein (g)` > ?" not in sql["query"]
    assert sql["params"] == [900.0, 500.0, 20.0]


def test_nutrition_comparator_survives_punctuation_after_nutrient():
    queries = [
        "recipes have calories, more than 300 but less than 900",
        "recipes have calories: more than 300 but less than 900",
        "recipes have calories; more than 300 but less than 900",
        "recipes have calories - more than 300 but less than 900",
        "recipes have calories (more than 300) but less than 900",
    ]

    for query in queries:
        tags = tag_entities(query)
        prepared = prepare_intent(tags, transcript=query)
        sql = build_sql_query(prepared, transcript=query)

        assert [
            (item["nutrient"], item["operator"], item["value"])
            for item in prepared["nutrition_filters"]
        ] == [
            ("calories", ">", 300.0),
            ("calories", "<", 900.0),
        ]
        assert "n1.`calories` > ?" in sql["query"]
        assert "n2.`calories` < ?" in sql["query"]
        assert sql["params"] == [300.0, 900.0]


def test_complex_query_keeps_punctuated_calorie_range():
    query = (
        "Give me top 10 Indian recipes that do not contain onion, potato, garlic "
        "and must contain olives and jackfruit and lychee and paneer and has more "
        "than 30 grams of protein and have calories, more than 300 but less than "
        "900 and also contain vitamin D, vitamin B 12 and have prep time under "
        "50 minutes"
    )

    tags = tag_entities(query)
    transcript = tags["_normalized_query"]
    prepared = prepare_intent(tags, transcript=transcript)
    sql = build_sql_query(prepared, transcript=transcript)

    nutrition = [
        (item["nutrient"], item["operator"], item["value"])
        for item in prepared["nutrition_filters"]
    ]
    assert ("calories", ">", 300.0) in nutrition
    assert ("calories", "<", 900.0) in nutrition
    assert ("calories", ">", 0.0) not in nutrition
    assert "n2.`calories` > ?" in sql["query"]
    assert "n3.`calories` < ?" in sql["query"]
    assert 300.0 in sql["params"]
    assert 900.0 in sql["params"]


def test_sql_builder_handles_not_less_than_minutes_as_base_time_lower_bound():
    query = "recipes with garlic that take not less than 30 minutes"

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["time_filters"] == [
        {
            "field": "total_time",
            "min": 30.0,
            "max": None,
            "source": "intent_enrichment",
        }
    ]
    assert "CAST(NULLIF(r.`total_time`, '') AS DOUBLE) >= ?" in sql["query"]
    assert "r.`total_time` <= ?" not in sql["query"]
    assert sql["params"] == ["garlic", "%garlic%", 30.0]


def test_sql_builder_handles_negated_greater_than_minutes_as_base_time_upper_bound():
    query = "recipes with garlic that take not greater than 30 minutes"

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["time_filters"] == [
        {
            "field": "total_time",
            "min": 0,
            "max": 30.0,
            "source": "intent_enrichment",
        }
    ]
    assert "CAST(NULLIF(r.`total_time`, '') AS DOUBLE) <= ?" in sql["query"]
    assert "CAST(NULLIF(r.`total_time`, '') AS DOUBLE) >= ?" not in sql["query"]
    assert sql["params"] == ["garlic", "%garlic%", 30.0]


def test_sql_builder_handles_negated_below_minutes_as_base_time_lower_bound():
    query = "recipes with garlic that take not below 30 minutes"

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["time_filters"] == [
        {
            "field": "total_time",
            "min": 30.0,
            "max": None,
            "source": "intent_enrichment",
        }
    ]
    assert "CAST(NULLIF(r.`total_time`, '') AS DOUBLE) >= ?" in sql["query"]
    assert "CAST(NULLIF(r.`total_time`, '') AS DOUBLE) <= ?" not in sql["query"]
    assert sql["params"] == ["garlic", "%garlic%", 30.0]


def test_negated_ingredient_list_excludes_all_items():
    query = "Give me recipes containing chicken but not garlic, onion, mushrooms or peanuts."

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["chicken"]
    assert prepared["ingredient_logic"]["exclude"] == ["garlic", "onion", "mushroom", "peanut"]
    assert sql["validation"]["ok"] is True
    assert "NOT EXISTS" in sql["query"]
    assert sql["params"] == [
        "chicken", "%chicken%",
        "garlic", "%garlic%",
        "onion", "%onion%",
        "mushroom", "%mushroom%",
        "peanut", "%peanut%",
    ]


def test_mixed_ingredient_or_groups_render_as_grouped_exists():
    query = "Find recipes with rice or noodles and either chicken or shrimp."

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include_groups"] == [
        {"ingredients": ["rice", "pasta"], "operator": "OR"},
        {"ingredients": ["chicken", "shrimp"], "operator": "OR"},
    ]
    assert sql["validation"]["ok"] is True
    assert sql["query"].count("EXISTS (SELECT 1 FROM `ingredient_phrase`") == 2
    assert sql["params"] == [
        "rice", "%rice%",
        "pasta", "%pasta%",
        "chicken", "%chicken%",
        "shrimp", "%shrimp%",
    ]


def test_nutrition_continuation_does_not_cross_named_nutrients():
    query = (
        "Show recipes with more than 600 calories, at least 50 grams protein, "
        "less than 20 grams fat and under 40 grams carbohydrates."
    )

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    filters = [
        (item["nutrient"], item["operator"], item["value"])
        for item in prepared["nutrition_filters"]
    ]
    assert filters == [
        ("calories", ">", 600.0),
        ("protein", ">=", 50.0),
        ("fat", "<", 20.0),
        ("carbs", "<", 40.0),
    ]
    assert sql["validation"]["ok"] is True
    assert sql["params"] == [600.0, 50.0, 20.0, 40.0]


def test_multiple_time_constraints_use_local_time_fields():
    query = "Show recipes that can be prepared in less than 20 minutes and cooked within 15 minutes."

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["time_filters"] == [
        {"field": "prep_time", "min": 0, "max": 20.0, "source": "intent_enrichment"},
        {"field": "cook_time", "min": 0, "max": 15.0, "source": "intent_enrichment"},
    ]
    assert "r.`prep_time`" in sql["query"]
    assert "r.`cook_time`" in sql["query"]
    assert sql["params"] == [20.0, 15.0]


def test_ingredient_count_constraints_render_count_subquery():
    tags = tag_entities("Find recipes having fewer than eight ingredients.")
    transcript = tags["_normalized_query"]
    prepared = prepare_intent(tags, transcript=transcript)
    sql = build_sql_query(prepared, transcript=transcript)

    assert sql["validation"]["ok"] is True
    assert "COUNT(DISTINCT ip_count.`ingredient`)" in sql["query"]
    assert sql["params"] == [8.0]


def test_nutrition_between_generates_single_between_predicate():
    query = (
        "Give me recipes with protein above 40g, fat below 15g, "
        "carbohydrates below 30g, sodium below 400mg and calories between 300 and 600."
    )

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["nutrition_filters"][0] == {
        "nutrient": "calories",
        "operator": "BETWEEN",
        "values": (300.0, 600.0),
        "table_source": "nutrition",
    }
    assert "n1.`calories` BETWEEN ? AND ?" in sql["query"]
    assert sql["params"] == [300.0, 600.0, 40.0, 15.0, 30.0, 400.0]


def test_contradictory_include_and_exclude_mentions_are_preserved():
    query = "Show recipes containing garlic but not containing garlic."

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["ingredient_logic"]["include"] == ["garlic"]
    assert prepared["ingredient_logic"]["exclude"] == ["garlic"]
    assert "EXISTS" in sql["query"]
    assert "NOT EXISTS" in sql["query"]
    assert sql["params"] == ["garlic", "%garlic%", "garlic", "%garlic%"]


def test_typo_heavy_query_keeps_fuzzy_ingredients_and_sorts_by_protein():
    query = (
        "Can you please find me the top 5 recipies contaning chikcen, onnion, "
        "tomatos, spinach, cheeze and olive oil but excluding garlick, mashrooms, "
        "peenuts and butter, wich can be preparred within 30 minuts, having more "
        "then 45 grams protien, less then 550 calries, less then 20 grams fat, "
        "more then 10 grams fibre, and order them by highest protien first?"
    )

    tags = tag_entities(query)
    transcript = tags["_normalized_query"]
    prepared = prepare_intent(tags, transcript=transcript)
    sql = build_sql_query(prepared, transcript=transcript)

    assert "canning" not in prepared["ingredient_logic"]["include"]
    assert prepared["ingredient_logic"]["include"] == [
        "chicken", "onion", "tomato", "spinach", "cheese", "olive oil",
    ]
    assert prepared["ingredient_logic"]["exclude"] == ["garlic", "mushroom", "peanut", "butter"]
    assert prepared["time_filters"] == [
        {"field": "prep_time", "min": 0, "max": 30.0, "source": "intent_enrichment"}
    ]
    assert sql["validation"]["ok"] is True
    assert sql["ir"]["order_by"] == [{"field": "protein", "direction": "DESC"}]
    assert sql["ir"]["limit"] == 5
    assert "n4.`fiber, total dietary (g)` > ?" in sql["query"]


def test_quickest_recipes_sort_by_total_time():
    query = "Find the quickest beef recipes."

    tags = tag_entities(query)
    prepared = prepare_intent(tags, transcript=query)
    sql = build_sql_query(prepared, transcript=query)

    assert prepared["time_filters"] == []
    assert sql["validation"]["ok"] is True
    assert sql["ir"]["order_by"] == [{"field": "total_time", "direction": "ASC"}]
    assert "ORDER BY (SELECT n_order.`total_time`" in sql["query"]


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
        config["response_field_aliases"]["region"][0]: "Italian",
        config["response_field_aliases"]["utensils"][0]: "skillet,bowl",
        dietary_value: "1.0",
        ingredient_list_field: [{ingredient_value_field: "spinach"}, {ingredient_value_field: "garlic"}],
    }
    query_ir = QueryIR(
        select=[SelectExpr("r.*")],
        predicates=[
            Predicate("ingredients", "ingredient", "=", ("spinach",)),
            Predicate("nutrition", "calories", "<", (300,)),
            Predicate("base", "total_time", "<=", (30,)),
            Predicate("base", "region", "=", ("Italian",)),
            Predicate("base", "utensils", "LIKE", ("%skillet%",)),
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
    assert "ORDER BY (SELECT n_order.`prep_time`" in sql["query"]
    assert sql["query"].endswith("DESC")


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
    assert "ORDER BY (SELECT n_order.`prep_time`" in sql["query"]
    assert sql["query"].endswith("ASC")


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
    assert " OR r.`ovo_vegetarian` IN" in sql["query"]
    assert " OR r.`lacto_vegetarian` IN" in sql["query"]
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
    assert "n1.`sugars, total (g)` < ?" in sql["query"]
    assert sql["params"] == [10.0]


def test_sql_validator_accepts_supported_chained_ingredient_flavor_join():
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
    sql = build_sql_query({}, query_ir=query_ir)

    assert result.ok is True
    assert sql["validation"]["ok"] is True
    assert "ingredient_phrase" in sql["query"]
    assert "ingredient_flavor" in sql["query"]
    assert sql["join_assumptions"][0]["name"] == (
        "ingredient_phrase_to_recipes_by_recipe_no->ingredient_phrase_to_flavor"
    )


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
    import speech_to_sql.m4_sql.sql_query_builder as builder

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
