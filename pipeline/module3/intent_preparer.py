from __future__ import annotations

from typing import Any

from module3.config import load_recipe_db_config, load_sql_schema_config
from module3.intent_enricher import enrich_tag_result
from module3.intent_grounder import ground_intent
from module3.schema_linker import link_tag_result


def prepare_intent(
    tag_result: dict[str, Any],
    *,
    transcript: str | None = None,
    recipe_config: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if tag_result.get("_intent_prepared"):
        return tag_result

    recipe_config = recipe_config or load_recipe_db_config()
    schema = schema or load_sql_schema_config()
    prepared = enrich_tag_result(
        tag_result,
        transcript=transcript,
        recipe_config=recipe_config,
        schema=schema,
    )
    prepared = link_tag_result(
        prepared,
        transcript=transcript,
        recipe_config=recipe_config,
        schema=schema,
    )
    prepared = ground_intent(prepared, schema=schema)
    prepared["_intent_prepared"] = True
    return prepared
