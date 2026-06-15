import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import ahocorasick
from rapidfuzz import fuzz, process

from speech_to_sql.m2_nlu.sql_ops.clause_classifier import predict_sql_clause_labels
from speech_to_sql.m2_nlu.sql_ops.ner import extract_sql_operator_ner_spans

WORD_BOUNDARIES = set(" ,.()\n\t!?;:'\"")

_LEXICON_PATH = Path(__file__).parents[1] / "data" / "sql_operator_lexicon.json"

_TOP_N_RE = re.compile(r"\b(top|first|limit)\s+\d+\b", re.IGNORECASE)
_TOTAL_SCOPE_PATTERN = re.compile(
    r"^\s+(?:"
    r"recipe(?:[-\s]+level)?|full\s+recipe|whole\s+recipe|per\s+recipe|"
    r"calor(?:y|ies)|kcal|energy|protein|carbs?|carbohydrates?|fat|"
    r"fiber|fibre|sodium|salt|sugar|cholesterol|magnesium|folate|"
    r"thiamin|vitamin(?:\s+[a-z0-9]+)?|time|total\s+time|cook\s+time|prep\s+time"
    r")\b",
    re.IGNORECASE,
)

def _load_lexicon() -> dict:
    with open(_LEXICON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_LEXICON = _load_lexicon()
SYMBOL_OPERATORS = {
    symbol: (payload["canonical"], payload["type"])
    for symbol, payload in _LEXICON["symbols"].items()
}
RULE_TERMS = {
    term: (payload["canonical"], payload["type"])
    for term, payload in _LEXICON["terms"].items()
}
FALLBACK_TERMS = dict(_LEXICON["fallback_terms"])

PRIORITY = {
    "rule_symbol": 0,
    "rule_keyword": 1,
    "ml_ner": 2,
    "ml_clause": 3,
    "fallback_synonym": 4,
}


@dataclass(frozen=True)
class SqlOperator:
    text: str
    canonical: str
    type: str
    span: tuple[int, int]
    layer: str
    confidence: float

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "canonical": self.canonical,
            "type": self.type,
            "span": [self.span[0], self.span[1]],
            "layer": self.layer,
            "confidence": self.confidence,
        }


def preprocess_sql_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _build_automaton() -> ahocorasick.Automaton:
    automaton = ahocorasick.Automaton()
    for term, payload in RULE_TERMS.items():
        automaton.add_word(term, (term, *payload))
    automaton.make_automaton()
    return automaton


SQL_AUTOMATON = _build_automaton()


def _has_boundaries(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end + 1] if end + 1 < len(text) else " "
    return before in WORD_BOUNDARIES and after in WORD_BOUNDARIES


def _extract_symbol_rules(text: str) -> list[SqlOperator]:
    results = []
    for symbol in sorted(SYMBOL_OPERATORS, key=len, reverse=True):
        canonical, op_type = SYMBOL_OPERATORS[symbol]
        for match in re.finditer(re.escape(symbol), text):
            results.append(SqlOperator(
                text=match.group(),
                canonical=canonical,
                type=op_type,
                span=(match.start(), match.end() - 1),
                layer="rule_symbol",
                confidence=1.0,
            ))
    return results


def _extract_keyword_rules(text: str) -> list[SqlOperator]:
    lowered = text.lower()
    results = []
    for end, (term, canonical, op_type) in SQL_AUTOMATON.iter(lowered):
        start = end - len(term) + 1
        if not _has_boundaries(lowered, start, end):
            continue
        results.append(SqlOperator(
            text=text[start:end + 1],
            canonical=canonical,
            type=op_type,
            span=(start, end),
            layer="rule_keyword",
            confidence=1.0,
        ))
    return results


def _extract_fallback_synonyms(text: str) -> list[SqlOperator]:
    lowered = text.lower()
    tokens = list(re.finditer(r"\b[a-z][a-z]*\b", lowered))
    results = []
    canonical_terms = list(RULE_TERMS.keys())

    for width in (4, 3, 2):
        for index in range(0, len(tokens) - width + 1):
            start = tokens[index].start()
            end = tokens[index + width - 1].end() - 1
            phrase = lowered[start:end + 1]
            if phrase in RULE_TERMS:
                continue
            direct = FALLBACK_TERMS.get(phrase)
            if direct:
                canonical, op_type = RULE_TERMS.get(direct, (direct, "COMPARE"))
                results.append(SqlOperator(
                    text=text[start:end + 1],
                    canonical=canonical,
                    type=op_type,
                    span=(start, end),
                    layer="fallback_synonym",
                    confidence=0.86,
                ))
                continue

            match = process.extractOne(
                phrase,
                canonical_terms,
                scorer=fuzz.WRatio,
                score_cutoff=91,
            )
            if match:
                canonical, op_type = RULE_TERMS[match[0]]
                results.append(SqlOperator(
                    text=text[start:end + 1],
                    canonical=canonical,
                    type=op_type,
                    span=(start, end),
                    layer="fallback_synonym",
                    confidence=round(match[1] / 100, 2),
                ))
    return results


def _normalize_ml_span(span_text: str, label_type: str) -> tuple[str, str] | None:
    lowered = span_text.lower().strip()
    if lowered in RULE_TERMS:
        return RULE_TERMS[lowered]

    candidates = [
        term for term, (_, op_type) in RULE_TERMS.items()
        if op_type == label_type
    ] or list(RULE_TERMS.keys())
    match = process.extractOne(
        lowered,
        candidates,
        scorer=fuzz.WRatio,
        score_cutoff=72,
    )
    if match:
        return RULE_TERMS[match[0]]
    return None


def _extract_ml_ner(text: str) -> list[SqlOperator]:
    results = []
    for span in extract_sql_operator_ner_spans(text):
        normalized = _normalize_ml_span(span["text"], span["type"])
        if not normalized:
            continue
        canonical, op_type = normalized
        results.append(SqlOperator(
            text=span["text"],
            canonical=canonical,
            type=op_type,
            span=tuple(span["span"]),
            layer="ml_ner",
            confidence=span["confidence"],
        ))
    return results


def _extract_clause_classifier(text: str) -> list[SqlOperator]:
    results = []
    for index, prediction in enumerate(predict_sql_clause_labels(text)):
        results.append(SqlOperator(
            text="",
            canonical=prediction["canonical"],
            type=prediction["type"],
            span=(-index - 1, -index - 1),
            layer="ml_clause",
            confidence=prediction["confidence"],
        ))
    return results


def resolve_sql_overlaps(operators: list[SqlOperator]) -> list[SqlOperator]:
    ranked = sorted(
        operators,
        key=lambda item: (
            PRIORITY.get(item.layer, 9),
            -(item.span[1] - item.span[0]),
            -item.confidence,
        ),
    )
    accepted = []
    used = set()
    for item in ranked:
        if item.span[0] < 0 or item.span[1] < 0:
            accepted.append(item)
            continue
        positions = set(range(item.span[0], item.span[1] + 1))
        if positions & used:
            continue
        accepted.append(item)
        used |= positions
    return sorted(accepted, key=lambda item: (item.span[0] < 0, item.span[0], item.canonical))


def _dedupe_implicit_clause_predictions(operators: list[SqlOperator]) -> list[SqlOperator]:
    # Collect all canonicals already found by rule/NER layers
    rule_canonicals = {
        op.canonical
        for op in operators
        if op.layer != "ml_clause"
    }

    seen_implicit = set()
    deduped = []
    for item in operators:
        if item.layer != "ml_clause":
            deduped.append(item)
            continue
        # Drop if rules already found this operator - no need to duplicate
        if item.canonical in rule_canonicals:
            continue
        if item.canonical in seen_implicit:
            continue
        deduped.append(item)
        seen_implicit.add(item.canonical)
    return deduped


def _context_supports(operator: SqlOperator, all_ops: list[SqlOperator], text: str) -> bool:
    if operator.layer == "ml_clause":
        return True
    if operator.confidence < 0.60:
        return False
    if operator.layer.startswith("rule_symbol"):
        return True
    if operator.canonical in {"WHERE", "GROUP BY", "ORDER BY", "JOIN", "BETWEEN", "IS NULL", "IS NOT NULL"}:
        return True
    if operator.type == "COMPARE":
        return True
    if operator.canonical in {"AND", "OR", "NOT"}:
        return operator.layer == "rule_keyword" or len(all_ops) > 1
    if operator.canonical == "IN":
        window = text[max(0, operator.span[0] - 16):operator.span[1] + 17].lower()
        return bool(re.search(r"\b(one of|not in|where|with|whose)\b", window))
    if operator.type == "AGG":
        if _is_total_scope_phrase(operator, text):
            return False
        if operator.span[0] == 0:
            return True
        window = text[max(0, operator.span[0] - 16):operator.span[1] + 17].lower()
        return bool(re.search(r"\b(of|by|where|select|show|get|find|what|list|give|tell|which|fetch)\b", window))
    return operator.confidence >= 0.80


def _is_total_scope_phrase(operator: SqlOperator, text: str) -> bool:
    if operator.canonical != "SUM":
        return False
    if operator.text.strip().lower() != "total":
        return False
    suffix = text[operator.span[1] + 1:operator.span[1] + 80]
    return bool(_TOTAL_SCOPE_PATTERN.match(suffix))


def extract_sql_operators(text: str) -> list[dict]:
    if not text or not text.strip():
        return []

    candidates = []
    candidates.extend(_extract_symbol_rules(text))
    candidates.extend(_extract_keyword_rules(text))
    for m in _TOP_N_RE.finditer(text):
        candidates.append(SqlOperator(
            text=m.group(0),
            canonical="LIMIT",
            type="CLAUSE",
            span=(m.start(), m.end() - 1),
            layer="rule_keyword",
            confidence=1.0,
        ))
    candidates.extend(_extract_fallback_synonyms(text))
    if os.getenv("SPEECH_TO_SQL_ENABLE_SQL_ML") == "1" and (
        not candidates or os.getenv("SPEECH_TO_SQL_FORCE_SQL_ML") == "1"
    ):
        candidates.extend(_extract_ml_ner(text))
    if os.getenv("SPEECH_TO_SQL_ENABLE_SQL_CLAUSE_ML") == "1":
        candidates.extend(_extract_clause_classifier(text))

    fused = _dedupe_implicit_clause_predictions(resolve_sql_overlaps(candidates))
    validated = [
        item for item in fused
        if _context_supports(item, fused, text)
    ]
    return [item.as_dict() for item in validated]
