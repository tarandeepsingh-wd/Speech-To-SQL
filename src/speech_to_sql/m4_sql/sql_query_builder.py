from __future__ import annotations

import re

from speech_to_sql.common.config import load_sql_schema_config
from speech_to_sql.m3_query_ir.intent_grounder import IntentGroundingError
from speech_to_sql.m4_sql.sql_compiler import compile_query_ir
from speech_to_sql.m3_query_ir.sql_intent_builder import build_query_ir
from speech_to_sql.m3_query_ir.sql_ir import QueryIR
from speech_to_sql.m4_sql.sql_validator import validate_query_ir


def build_sql_query(
    tag_result: dict,
    *,
    transcript: str | None = None,
    table_name: str | None = None,
    validate: bool = True,
    require_virtual_evidence: bool = False,
    query_ir: QueryIR | None = None,
) -> dict:
    """
    Build SQL by compiling a schema-aware QueryIR.

    `table_name` is kept for backward compatibility. The preferred way to
    change schema names is editing config/sql_schema.json.
    """
    try:
        query_ir = query_ir or build_query_ir(tag_result, transcript=transcript)
    except IntentGroundingError as error:
        return {
            "query": None,
            "params": [],
            "ir": None,
            "grounding": error.grounding,
            "validation": {
                "ok": False,
                "errors": error.errors,
                "warnings": [],
                "error_details": error.error_details,
            },
        }
    validation = validate_query_ir(
        query_ir,
        require_virtual_evidence=require_virtual_evidence,
    ) if validate else None
    if validation and not validation.ok:
        return {
            "query": None,
            "params": [],
            "ir": query_ir.as_dict(),
            "validation": validation.as_dict(),
        }

    compiled = compile_query_ir(query_ir)
    title_boost = _dish_title_boost(tag_result, transcript)
    if title_boost:
        compiled["dish_title_boost"] = title_boost
    if validation:
        compiled["validation"] = validation.as_dict()
    if table_name:
        schema = load_sql_schema_config()
        default_table = schema["base_table"]["name"]
        compiled["query"] = compiled["query"].replace(
            f" FROM {default_table} AS ",
            f" FROM {table_name} AS ",
            1,
        )
    return compiled


def _dish_title_boost(tag_result: dict, transcript: str | None) -> dict | None:
    """
    Infer a title-ranking hint from adjacent ingredient spans only.

    This deliberately does not classify the query as a dish/title query and it
    never adds a recipe_title filter. It only marks phrases like "butter
    chicken" when the matched ingredient spans are separated by whitespace.
    Punctuation and connector words such as "and", "or", "with", or
    "containing" leave a non-whitespace gap, so they do not trigger the boost.
    """
    text = transcript or tag_result.get("_normalized_query") or tag_result.get("transcript") or ""
    if not text:
        return None

    logic = tag_result.get("ingredient_logic") or {}
    include_operator = str(logic.get("include_operator") or "AND").upper()
    if include_operator != "AND":
        return None

    include = _unique_lower_strings(logic.get("include") or tag_result.get("ingredients") or [])
    if len(include) < 2:
        return None
    include_set = set(include)

    spans: list[tuple[int, int, str]] = []
    for item in tag_result.get("raw_spans") or []:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical") or "").strip().lower()
        if canonical not in include_set:
            continue
        span = item.get("span")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        try:
            start = int(span[0])
            end = int(span[1])
        except (TypeError, ValueError):
            continue
        if start < 0 or end < start or end >= len(text):
            continue
        spans.append((start, end, canonical))

    if len(spans) < 2:
        return None

    spans.sort(key=lambda item: (item[0], item[1]))
    best: list[tuple[int, int, str]] = []
    current: list[tuple[int, int, str]] = []
    current_terms: set[str] = set()
    for span in spans:
        if (
            current
            and span[2] not in current_terms
            and _whitespace_gap(text, current[-1][1], span[0])
        ):
            current.append(span)
            current_terms.add(span[2])
        else:
            if len(current) > len(best):
                best = current
            current = [span]
            current_terms = {span[2]}
    if len(current) > len(best):
        best = current

    if len(best) < 2:
        return None

    phrase = _normalize_title_phrase(" ".join(span[2] for span in best))
    tokens = _unique_lower_strings(phrase.split())
    if len(tokens) < 2:
        return None
    return {
        "phrase": phrase,
        "tokens": tokens,
        "source": "adjacent_ingredient_spans",
    }


def _whitespace_gap(text: str, previous_end: int, next_start: int) -> bool:
    if next_start <= previous_end:
        return False
    gap = text[previous_end + 1:next_start]
    return bool(gap) and gap.strip() == ""


def _normalize_title_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _unique_lower_strings(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
