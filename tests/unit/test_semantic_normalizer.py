from speech_to_sql.m2_nlu.adaptive_lexicon import (
    TRANSCRIPT_FEEDBACK_FILE_ENV,
    normalize_noisy_text,
    reset_adaptive_cache,
)
from speech_to_sql.m2_nlu.entity_tagger import tag_entities


def test_schema_normalizer_corrects_operator_near_numeric_filter(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities("show recipes undr 100 calories")

    assert result["_normalized_query"] == "show recipes under 100 calories"
    assert result["nutrition_filters"] == [
        {"nutrient": "calories", "operator": "<", "value": 100.0}
    ]
    assert result["_asr_corrections"][0]["layer"] == "schema_lexicon"
    assert result["_asr_corrections"][0]["kind"] == "sql_operator"


def test_schema_normalizer_uses_context_for_weaker_operator_match(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    noisy = normalize_noisy_text("show recipes ander 100 calories")
    unrelated = normalize_noisy_text("show recipes with ander sauce")

    assert noisy["text"] == "show recipes under 100 calories"
    assert unrelated["text"] == "show recipes with ander sauce"


def test_schema_normalizer_corrects_cuisine_alias_from_config(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities("show me italin recipes")

    assert result["_normalized_query"] == "show me italian recipes"
    assert result["categories"] == [{"term": "italian", "category": "cuisine"}]
    assert result["_asr_corrections"][0]["evidence"] == "recipe_db.cuisine_aliases"


def test_schema_normalizer_recovers_noisy_operator_phrase_and_nutrient(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities("gib me fud les dam 564 kaloris")

    assert result["_normalized_query"] == "gib me fud less than 564 calories"
    assert result["nutrition_filters"] == [
        {"nutrient": "calories", "operator": "<", "value": 564.0}
    ]
    corrections = {(item["source"], item["target"], item["layer"]) for item in result["_asr_corrections"]}
    assert ("les dam", "less than", "schema_phrase_lexicon") in corrections
    assert ("kaloris", "calories", "schema_lexicon") in corrections


def test_schema_normalizer_protects_valid_operator_phrase(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities("show food kaloris less than 100")

    assert result["_normalized_query"] == "show food calories less than 100"
    assert result["nutrition_filters"] == [
        {"nutrient": "calories", "operator": "<", "value": 100.0}
    ]


def test_schema_normalizer_does_not_rewrite_operator_noise_without_numeric_context(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    assert normalize_noisy_text("les dam sauce")["text"] == "les dam sauce"
    assert normalize_noisy_text("show recipes with ander sauce")["text"] == "show recipes with ander sauce"


def test_schema_normalizer_converts_spoken_numbers_for_numeric_filters(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities("show recipes under eight hundred calories")

    assert result["_normalized_query"] == "show recipes under 800 calories"
    assert result["nutrition_filters"] == [
        {"nutrient": "calories", "operator": "<", "value": 800.0}
    ]
    assert result["_asr_corrections"][0]["layer"] == "spoken_number_grammar"


def test_schema_normalizer_converts_noisy_spoken_number_scales(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities("show recipes under eight hunderd calories")

    assert result["_normalized_query"] == "show recipes under 800 calories"
    assert result["nutrition_filters"] == [
        {"nutrient": "calories", "operator": "<", "value": 800.0}
    ]


def test_schema_normalizer_combines_noisy_number_words_with_noisy_operator_and_nutrient(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities("recipiz mor thn savan hundrad kaloris")

    assert result["_normalized_query"] == "recipiz more than 700 calories"
    assert result["nutrition_filters"] == [
        {"nutrient": "calories", "operator": ">", "value": 700.0}
    ]
    corrections = {(item["source"], item["target"], item["layer"]) for item in result["_asr_corrections"]}
    assert ("savan hundrad", "700", "spoken_number_grammar") in corrections
    assert ("mor thn", "more than", "schema_phrase_lexicon") in corrections
    assert ("kaloris", "calories", "schema_lexicon") in corrections


def test_schema_normalizer_recovers_short_noisy_number_words_in_numeric_phrase(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities("fuud which have kelories grater thn twu hundrad fafty")

    assert result["_normalized_query"] == "fuud which have calories greater than 250"
    assert result["nutrition_filters"] == [
        {"nutrient": "calories", "operator": ">", "value": 250.0}
    ]
    corrections = {(item["source"], item["target"], item["layer"]) for item in result["_asr_corrections"]}
    assert ("twu hundrad fafty", "250", "spoken_number_grammar") in corrections
    assert ("grater thn", "greater than", "schema_phrase_lexicon") in corrections
    assert ("kelories", "calories", "schema_lexicon") in corrections


def test_schema_normalizer_prioritizes_noisy_operator_before_noisy_number_phrase(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities("recipiz that habe chken leechee and have kelories lou thn nene hundrad")

    assert result["_normalized_query"] == (
        "recipiz that have chken leechee and have calories less than 900"
    )
    assert result["ingredients"] == ["chicken", "lychee"]
    assert result["nutrition_filters"] == [
        {"nutrient": "calories", "operator": "<", "value": 900.0}
    ]
    corrections = {(item["source"], item["target"], item["layer"]) for item in result["_asr_corrections"]}
    assert ("nene hundrad", "900", "spoken_number_grammar") in corrections
    assert ("lou thn", "less than", "schema_phrase_lexicon") in corrections
    assert ("kelories", "calories", "schema_lexicon") in corrections


def test_schema_normalizer_combines_spoken_numbers_with_noisy_sql_terms(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities("recipiz which have kelories mor thn eight hundred seventy two")

    assert result["_normalized_query"] == "recipiz which have calories more than 872"
    assert result["nutrition_filters"] == [
        {"nutrient": "calories", "operator": ">", "value": 872.0}
    ]
    corrections = {(item["target"], item["layer"]) for item in result["_asr_corrections"]}
    assert ("872", "spoken_number_grammar") in corrections
    assert ("more than", "schema_phrase_lexicon") in corrections
    assert ("calories", "schema_lexicon") in corrections


def test_schema_normalizer_does_not_convert_number_words_without_numeric_context(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = normalize_noisy_text("show thousand island dressing recipes")

    assert result["text"] == "show thousand island dressing recipes"
    assert result["corrections"] == []


def test_schema_normalizer_does_not_convert_food_words_that_sound_like_numbers(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = normalize_noisy_text("quick recipes with flour")

    assert result["text"] == "quick recipes with flour"
    assert result["corrections"] == []


def test_schema_normalizer_repairs_noisy_presence_cues_before_nutrients(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    cases = [
        ("show recipes that contane vitamin k", "show recipes that contain vitamin k", "contane", "contain"),
        ("show recipes habe vitamin k", "show recipes have vitamin k", "habe", "have"),
        ("show recipes wit vitamin k", "show recipes with vitamin k", "wit", "with"),
    ]

    for query, normalized, source, target in cases:
        result = tag_entities(query)

        assert result["_normalized_query"] == normalized
        assert result["nutrition_filters"] == [
            {"nutrient": "vitamin_k", "operator": ">", "value": 0.0}
        ]
        assert any(
            item["source"] == source
            and item["target"] == target
            and item["layer"] == "role_cue_lexicon"
            and item["kind"] == "nutrition_presence_cue"
            for item in result["_asr_corrections"]
        )


def test_schema_normalizer_keeps_noisy_presence_cues_without_nutrition_context(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = normalize_noisy_text("show recipes wit sauce")

    assert result["text"] == "show recipes wit sauce"
    assert result["corrections"] == []


def test_schema_normalizer_repairs_noisy_nutrition_negation_cue(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities("show recipes withot vitamin k")

    assert result["_normalized_query"] == "show recipes without vitamin k"
    assert result["nutrition_filters"] == []
    assert any(
        item["source"] == "withot"
        and item["target"] == "without"
        and item["layer"] == "role_cue_lexicon"
            and item["kind"] == "nutrition_negation_cue"
            for item in result["_asr_corrections"]
    )


def test_nlu_handles_continued_calorie_range_and_vitamin_presence_list(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    query = (
        "Give me all those Italian recipes that contain potato, garlic, jackfruit "
        "and have calories less than 600 but greater than 300 and contains "
        "vitamin K, vitamin C and also can be prepared under 47 minutes."
    )

    result = tag_entities(query)

    assert result["ingredients"] == ["potato", "garlic", "jackfruit"]
    assert result["categories"] == [{"term": "italian", "category": "cuisine"}]
    assert result["nutrition_filters"] == [
        {"nutrient": "calories", "operator": "<", "value": 600.0},
        {"nutrient": "calories", "operator": ">", "value": 300.0},
        {"nutrient": "vitamin_k", "operator": ">", "value": 0.0},
        {"nutrient": "vitamin_c", "operator": ">", "value": 0.0},
    ]


def test_nlu_stops_continued_nutrition_scope_at_time_phrase(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities("recipes with calories less than 600 but under 47 minutes")

    assert result["nutrition_filters"] == [
        {"nutrient": "calories", "operator": "<", "value": 600.0}
    ]


def test_nlu_stops_continued_nutrition_scope_at_next_nutrient(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities("recipes with calories less than 600 and protein greater than 20")

    assert result["nutrition_filters"] == [
        {"nutrient": "calories", "operator": "<", "value": 600.0},
        {"nutrient": "protein", "operator": ">", "value": 20.0},
    ]


def test_schema_normalizer_keeps_general_modal_word_near_numeric_filter(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities(
        "recipes with calories less than 600 but should be greater than 300"
    )

    assert result["_normalized_query"] == (
        "recipes with calories less than 600 but should be greater than 300"
    )
    assert result["nutrition_filters"] == [
        {"nutrient": "calories", "operator": "<", "value": 600.0},
        {"nutrient": "calories", "operator": ">", "value": 300.0},
    ]
    assert all(
        correction["source"] != "should"
        for correction in result["_asr_corrections"]
    )


def test_schema_normalizer_does_not_rewrite_instruction_action_as_operator(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = tag_entities(
        "Give me recipes under 60 minutes where the steps mention boil "
        "and the ingredient phrase contains tomato."
    )

    assert "steps mention boil" in result["_normalized_query"]
    assert "steps mention below" not in result["_normalized_query"]
    assert any(
        action["action"] == "boil" and action["type"] == "state_change"
        for action in result["actions"]
    )


def test_schema_normalizer_does_not_rewrite_ingredient_phrase_object_to_placeholder(monkeypatch, tmp_path):
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(tmp_path / "feedback.jsonl"))
    reset_adaptive_cache()

    result = normalize_noisy_text(
        "Show recipes under 60 minutes where the steps mention stir "
        "and the ingredient phrase contains water."
    )

    assert "ingredient phrase contains water" in result["text"]
    assert "contains pattern" not in result["text"]
    assert result["corrections"] == []
