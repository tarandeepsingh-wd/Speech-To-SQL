from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from module3.config import load_recipe_db_config, load_sql_schema_config
from module3.result_merger import recipe_key
from module3.sql_ir import Predicate, QueryIR


@dataclass(frozen=True)
class PredicateIssue:
    recipe_key: str
    predicate: dict[str, Any]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe_key": self.recipe_key,
            "predicate": self.predicate,
            "reason": self.reason,
        }


@dataclass
class ApiVerificationResult:
    ok: bool
    verified: bool
    recipe_count: int
    predicate_count: int
    predicate_checks: int
    failed_count: int = 0
    unknown_count: int = 0
    failures: list[PredicateIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "verified": self.verified,
            "recipe_count": self.recipe_count,
            "predicate_count": self.predicate_count,
            "predicate_checks": self.predicate_checks,
            "failed_count": self.failed_count,
            "unknown_count": self.unknown_count,
            "failures": [failure.as_dict() for failure in self.failures],
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class _Evaluation:
    passed: bool | None
    reason: str | None = None


def verify_recipes_against_ir(
    recipes: list[dict[str, Any]],
    query_ir: QueryIR,
    *,
    config: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
    max_issues: int = 20,
) -> ApiVerificationResult:
    config = config or load_recipe_db_config()
    schema = schema or load_sql_schema_config()
    predicate_count = len(query_ir.predicates)
    predicate_checks = len(recipes) * predicate_count
    failures: list[PredicateIssue] = []
    warnings: list[str] = []
    failed_count = 0
    unknown_count = 0

    if not recipes and predicate_count:
        warnings.append("No API recipes returned, so there is no response evidence to verify.")

    for index, recipe in enumerate(recipes):
        key = recipe_key(recipe) or f"result_index:{index}"
        for predicate in query_ir.predicates:
            evaluation = _evaluate_predicate(recipe, predicate, config, schema)
            if evaluation.passed is False:
                failed_count += 1
                if len(failures) < max_issues:
                    failures.append(PredicateIssue(
                        recipe_key=key,
                        predicate=predicate.as_dict(),
                        reason=evaluation.reason or "Predicate did not match the API recipe.",
                    ))
            elif evaluation.passed is None:
                unknown_count += 1
                if len(warnings) < max_issues:
                    warnings.append(
                        f"{key}: {evaluation.reason or 'Predicate could not be verified from API fields.'}"
                    )

    ok = failed_count == 0
    verified = ok and unknown_count == 0 and bool(recipes or predicate_count == 0)
    return ApiVerificationResult(
        ok=ok,
        verified=verified,
        recipe_count=len(recipes),
        predicate_count=predicate_count,
        predicate_checks=predicate_checks,
        failed_count=failed_count,
        unknown_count=unknown_count,
        failures=failures,
        warnings=warnings,
    )


def _evaluate_predicate(
    recipe: dict[str, Any],
    predicate: Predicate,
    config: dict[str, Any],
    schema: dict[str, Any],
) -> _Evaluation:
    positive = _evaluate_positive_predicate(recipe, predicate, config, schema)
    if positive.passed is None:
        return positive
    if predicate.quantifier == "not_exists":
        if positive.passed:
            return _Evaluation(False, f"Excluded {predicate.source}.{predicate.field} was present.")
        return _Evaluation(True)
    return positive


def _evaluate_positive_predicate(
    recipe: dict[str, Any],
    predicate: Predicate,
    config: dict[str, Any],
    schema: dict[str, Any],
) -> _Evaluation:
    if predicate.source == "ingredients":
        return _evaluate_text_predicate(_recipe_search_text(recipe, config), predicate)
    if predicate.source == "nutrition":
        return _evaluate_numeric_predicate(recipe, predicate, config)
    if predicate.source == "dietary":
        return _evaluate_dietary_predicate(recipe, predicate, config)
    if predicate.source == "base":
        return _evaluate_base_predicate(recipe, predicate, config, schema)
    return _Evaluation(None, f"Source {predicate.source!r} has no API verification rule.")


def _evaluate_text_predicate(text: str, predicate: Predicate) -> _Evaluation:
    if not text:
        return _Evaluation(None, "Configured API text fields were missing or empty.")
    values = [_normalize_text(str(value)) for value in predicate.values if str(value).strip()]
    if not values:
        return _Evaluation(None, f"{predicate.source}.{predicate.field} has no values to verify.")

    matches = [_contains_term(text, value) for value in values]
    if predicate.operator.upper() == "=":
        passed = all(matches)
    elif predicate.operator.upper() == "IN":
        passed = any(matches)
    else:
        return _Evaluation(None, f"Operator {predicate.operator!r} is not verifiable on text fields.")

    if not passed:
        return _Evaluation(False, f"Expected text value(s) {values!r} were not found.")
    return _Evaluation(True)


def _evaluate_numeric_predicate(
    recipe: dict[str, Any],
    predicate: Predicate,
    config: dict[str, Any],
) -> _Evaluation:
    value = _field_float(recipe, predicate.field, config)
    if value is None:
        return _Evaluation(None, f"Configured API field for {predicate.source}.{predicate.field} was missing or non-numeric.")
    return _compare_numeric_value(value, predicate)


def _evaluate_base_predicate(
    recipe: dict[str, Any],
    predicate: Predicate,
    config: dict[str, Any],
    schema: dict[str, Any],
) -> _Evaluation:
    field = schema.get("base_columns", {}).get(predicate.field, predicate.field)
    value = _field_float(recipe, field, config)
    if value is None:
        return _Evaluation(None, f"Configured API field for base.{predicate.field} was missing or non-numeric.")
    return _compare_numeric_value(value, predicate)


def _evaluate_dietary_predicate(
    recipe: dict[str, Any],
    predicate: Predicate,
    config: dict[str, Any],
) -> _Evaluation:
    values = [str(value).strip() for value in predicate.values if str(value).strip()]
    if not values:
        return _Evaluation(None, "Dietary predicate has no values to verify.")

    matched_flags: list[bool] = []
    saw_configured_field = False
    for value in values:
        for field in _dietary_response_fields(value, config):
            if field not in recipe:
                continue
            saw_configured_field = True
            matched_flags.append(_truthy_recipe_flag(recipe.get(field), config))

    if not saw_configured_field:
        return _Evaluation(None, "Configured dietary flag field was missing from the API recipe.")

    operator = predicate.operator.upper()
    if operator == "=":
        passed = all(matched_flags)
    elif operator == "IN":
        passed = any(matched_flags)
    else:
        return _Evaluation(None, f"Operator {predicate.operator!r} is not verifiable on dietary flags.")

    if not passed:
        return _Evaluation(False, f"Expected dietary value(s) {values!r} were not truthy.")
    return _Evaluation(True)


def _compare_numeric_value(value: float, predicate: Predicate) -> _Evaluation:
    operator = predicate.operator.upper()
    values = [_to_float(item) for item in predicate.values]
    if any(item is None for item in values):
        return _Evaluation(None, f"Predicate value(s) for {predicate.source}.{predicate.field} were non-numeric.")

    targets = [float(item) for item in values if item is not None]
    if operator == "BETWEEN" and len(targets) != 2:
        return _Evaluation(None, f"Operator {predicate.operator!r} needs two numeric values.")
    if operator != "IN" and operator != "BETWEEN" and len(targets) != 1:
        return _Evaluation(None, f"Operator {predicate.operator!r} needs one numeric value.")

    if operator == "<":
        passed = value < targets[0]
    elif operator == "<=":
        passed = value <= targets[0]
    elif operator == "=":
        passed = value == targets[0]
    elif operator == ">=":
        passed = value >= targets[0]
    elif operator == ">":
        passed = value > targets[0]
    elif operator == "BETWEEN":
        lower, upper = targets
        passed = lower <= value <= upper
    elif operator == "IN":
        passed = value in targets
    else:
        return _Evaluation(None, f"Operator {predicate.operator!r} is not verifiable on numeric fields.")

    if not passed:
        return _Evaluation(False, f"API value {value} did not satisfy {predicate.operator} {targets}.")
    return _Evaluation(True)


def _field_float(
    recipe: dict[str, Any],
    logical_field: str,
    config: dict[str, Any],
) -> float | None:
    for field in _response_field_candidates(logical_field, config):
        if field in recipe:
            return _to_float(recipe.get(field))
    return None


def _response_field_candidates(logical_field: str, config: dict[str, Any]) -> list[str]:
    candidates = list(config["response_field_aliases"].get(logical_field, []))
    if logical_field not in candidates:
        candidates.append(logical_field)
    return _unique_strings(candidates)


def _recipe_search_text(recipe: dict[str, Any], config: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in config["search_text_fields"]:
        value = recipe.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(_flatten_list_text(value, config))
    return _normalize_text(" ".join(parts))


def _flatten_list_text(items: list[Any], config: dict[str, Any]) -> list[str]:
    values = []
    for item in items:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, dict):
            for key in config["nested_ingredient_fields"]:
                value = item.get(key)
                if isinstance(value, str):
                    values.append(value)
    return values


def _dietary_response_fields(value: str, config: dict[str, Any]) -> list[str]:
    candidates = [value]
    for source, target in config["diet_aliases"].items():
        if value in {source, target}:
            candidates.append(target)
    return _unique_strings(candidates)


def _truthy_recipe_flag(value: Any, config: dict[str, Any]) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return str(value).strip().lower() in set(config["truthy_values"])


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())).strip()


def _contains_term(text: str, term: str) -> bool:
    if not term:
        return True
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def _unique_strings(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        result.append(text)
        seen.add(key)
    return result
