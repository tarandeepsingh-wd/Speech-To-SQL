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
from speech_to_sql.m2_nlu.feedback_logger import log_ingredient_feedback
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


class SqlOperatorRequest(BaseModel):
    query: str


class IngredientFeedbackRequest(BaseModel):
    query: str
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


@app.get("/health")
def health():
    return {"status": "ok"}
