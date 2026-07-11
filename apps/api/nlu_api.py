from pathlib import Path
import sys
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from speech_to_sql.m2_nlu.entity_tagger import tag_entities
from speech_to_sql.m2_nlu.adaptive_lexicon import (
    derive_transcript_corrections,
    feedback_summary,
    reset_adaptive_cache,
)
from speech_to_sql.m2_nlu.feedback_logger import log_ingredient_feedback, log_transcript_feedback
from speech_to_sql.m2_nlu.sql_ops import extract_sql_operators, predict_sql_clause_labels


app = FastAPI(
    title="Speech-to-SQL API",
    description="Entity extraction, intent parsing, ASR, SQL planning, and recipe search.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextQueryRequest(BaseModel):
    query: str


class SearchQueryRequest(BaseModel):
    query: str
    execute: bool = False
    max_api_calls: int = 3
    fetch_limit: int = 100
    final_limit: Optional[int] = None


class AthenaExecuteRequest(BaseModel):
    query: str
    max_rows: int = Field(default=20, ge=1, le=100)
    timeout_seconds: int = Field(default=90, ge=10, le=300)


class SqlOperatorRequest(BaseModel):
    query: str


class IngredientFeedbackRequest(BaseModel):
    query: str
    predicted_ingredients: list[str] = Field(default_factory=list)
    corrected_ingredients: list[str] = Field(default_factory=list)
    source: str = "api"


class SearchFeedbackRequest(BaseModel):
    original_query: str
    corrected_query: str = ""
    predicted_ingredients: list[str] = Field(default_factory=list)
    corrected_ingredients: list[str] = Field(default_factory=list)
    source: str = "api"


class TagResponse(BaseModel):
    ingredients: list
    utensils: list
    actions: list
    dietary_styles: list
    categories: list
    nutrition_filters: list
    sql_operators: list
    ingredient_logic: dict
    sql_filter_plan: dict
    unified_sql_plan: dict
    raw_spans: list


@app.post("/tag", response_model=TagResponse)
def tag_text(payload: TextQueryRequest):
    """Extract entities and SQL plans from a text query."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return tag_entities(payload.query)


@app.post("/query")
def query_text(payload: SearchQueryRequest) -> dict:
    """Run a text query through intent extraction, QueryIR, SQL preview, and optional API execution."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    from speech_to_sql.m5_execution.search import search_text

    return search_text(
        payload.query,
        execute=payload.execute,
        max_api_calls=payload.max_api_calls,
        fetch_limit=payload.fetch_limit,
        final_limit=payload.final_limit,
    )


@app.post("/athena/execute")
def execute_athena(payload: AthenaExecuteRequest) -> dict:
    """Execute paste-ready Athena SQL and return a small result preview."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Athena SQL cannot be empty.")

    from speech_to_sql.m5_execution.athena_client import (
        AthenaConfigError,
        AthenaQueryError,
        execute_athena_query,
    )

    try:
        return execute_athena_query(
            payload.query,
            max_rows=payload.max_rows,
            timeout_seconds=payload.timeout_seconds,
        )
    except AthenaConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AthenaQueryError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/transcribe-and-tag")
async def transcribe_and_tag(
    audio: UploadFile = File(...),
    language: Optional[str] = None,
    model_size: str = "small",
):
    """Run the full WAV audio to transcript to entity-tagging pipeline."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    from speech_to_sql.m1_speech.transcriber import transcribe_audio

    transcript = transcribe_audio(
        audio_bytes,
        language=language,
        model_size=model_size,
    )
    result = tag_entities(transcript)
    return {"transcript": transcript, **result}


@app.post("/extract_sql_operators")
def extract_sql_operator_endpoint(payload: SqlOperatorRequest) -> dict:
    """Legacy endpoint kept for backward compatibility."""
    return {"operators": extract_sql_operators(payload.query)}


@app.post("/extract_sql_clauses")
def extract_sql_clause_endpoint(payload: SqlOperatorRequest) -> dict:
    """Predict implicit SQL clauses/operators for a natural-language query."""
    return {"clauses": predict_sql_clause_labels(payload.query)}


@app.post("/feedback/ingredients")
def ingredient_feedback_endpoint(payload: IngredientFeedbackRequest) -> dict:
    """Save corrected ingredient labels for future reranker training."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    log_ingredient_feedback(
        query=payload.query,
        predicted=payload.predicted_ingredients,
        corrected_ingredients=payload.corrected_ingredients,
        source=payload.source,
    )
    return {"status": "saved"}


@app.post("/feedback/search")
def search_feedback_endpoint(payload: SearchFeedbackRequest) -> dict:
    """Save transcript and ingredient corrections from the UI."""
    original_query = payload.original_query.strip()
    corrected_query = payload.corrected_query.strip()
    if not original_query:
        raise HTTPException(status_code=400, detail="Original query cannot be empty.")
    if not corrected_query and not payload.corrected_ingredients:
        raise HTTPException(status_code=400, detail="No correction was provided.")

    corrected_query = corrected_query or original_query
    corrections = derive_transcript_corrections(original_query, corrected_query)
    log_transcript_feedback(
        original_query=original_query,
        corrected_query=corrected_query,
        predicted_ingredients=payload.predicted_ingredients,
        corrected_ingredients=payload.corrected_ingredients,
        corrections=corrections,
        source=payload.source,
    )
    if payload.corrected_ingredients:
        log_ingredient_feedback(
            query=corrected_query,
            predicted=payload.predicted_ingredients,
            corrected_ingredients=payload.corrected_ingredients,
            source=payload.source,
        )
    reset_adaptive_cache()
    return {
        "status": "saved",
        "corrections": corrections,
        "summary": feedback_summary(),
    }


@app.get("/feedback/summary")
def feedback_summary_endpoint() -> dict:
    return feedback_summary()


@app.get("/health")
def health():
    return {"status": "ok"}
