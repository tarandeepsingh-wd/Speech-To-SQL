import json

from speech_to_sql.m2_nlu.adaptive_lexicon import (
    TRANSCRIPT_FEEDBACK_FILE_ENV,
    derive_transcript_corrections,
    normalize_noisy_text,
    reset_adaptive_cache,
)
from speech_to_sql.m2_nlu.entity_tagger import tag_entities


def test_derives_transcript_replacements_from_user_correction():
    corrections = derive_transcript_corrections(
        "show recipes with floor and pan ear",
        "show recipes with flour and paneer",
    )

    assert {"source": "floor", "target": "flour"} in corrections
    assert {"source": "pan ear", "target": "paneer"} in corrections


def test_applies_learned_transcript_feedback(monkeypatch, tmp_path):
    feedback_file = tmp_path / "transcript_feedback.jsonl"
    feedback_file.write_text(
        json.dumps({
            "original_query": "show recipes with floor",
            "corrected_query": "show recipes with flour",
            "corrections": [{"source": "floor", "target": "flour"}],
            "approved": True,
        }) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(feedback_file))
    reset_adaptive_cache()

    result = normalize_noisy_text("quick recipes with floor")

    assert result["text"] == "quick recipes with flour"
    assert result["corrections"] == [
        {"source": "floor", "target": "flour", "count": 1}
    ]


def test_tagger_uses_learned_transcript_feedback(monkeypatch, tmp_path):
    feedback_file = tmp_path / "transcript_feedback.jsonl"
    feedback_file.write_text(
        json.dumps({
            "original_query": "recipes with floor",
            "corrected_query": "recipes with flour",
            "corrections": [{"source": "floor", "target": "flour"}],
            "approved": True,
        }) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(TRANSCRIPT_FEEDBACK_FILE_ENV, str(feedback_file))
    reset_adaptive_cache()

    result = tag_entities("show recipes with floor")

    assert result["_normalized_query"] == "show recipes with flour"
    assert "flour" in result["ingredients"]
