from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from module1_asr.entity_tagger import tag_entities
from module1_asr.sql_ops import extract_sql_operators
from module1_asr.transcriber import transcribe_audio


app = FastAPI(
    title="TADKA Module 1 NLU Pipeline",
    description="Entity extraction, intent parsing, ASR, and SQL filter planning.",
    version="1.0.0",
)


class TextQueryRequest(BaseModel):
    query: str


class SqlOperatorRequest(BaseModel):
    query: str


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


@app.get("/health")
def health():
    return {"status": "ok"}
