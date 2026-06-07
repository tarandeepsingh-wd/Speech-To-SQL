from module3.query_planner import ApiQueryPlan, plan_queries
from module3.intent_enricher import enrich_tag_result
from module3.intent_grounder import IntentGroundingError, ground_intent
from module3.intent_preparer import prepare_intent
from module3.query_ir_contract import (
    QueryIRContractError,
    query_ir_from_contract,
    query_ir_from_payload,
    query_ir_to_payload,
)
from module3.schema_linker import link_tag_result
from module3.sql_query_builder import build_sql_query
from module3.api_verifier import verify_recipes_against_ir
from module3.query_relaxer import relax_query_ir, relaxation_chain

__all__ = [
    "ApiQueryPlan",
    "build_sql_query",
    "enrich_tag_result",
    "ground_intent",
    "IntentGroundingError",
    "link_tag_result",
    "prepare_intent",
    "plan_queries",
    "QueryIRContractError",
    "query_ir_from_contract",
    "query_ir_from_payload",
    "query_ir_to_payload",
    "relax_query_ir",
    "relaxation_chain",
    "search_recipes",
    "search_text",
    "verify_recipes_against_ir",
]


def __getattr__(name):
    if name in {"search_recipes", "search_text"}:
        from module3.search import search_recipes, search_text

        return {"search_recipes": search_recipes, "search_text": search_text}[name]
    raise AttributeError(f"module 'module3' has no attribute {name!r}")
