from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from module1_asr.transcriber import transcribe_audio
from module1_asr.entity_tagger import tag_entities

app = FastAPI(title="TADKA Pipeline API 🌶️")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "TADKA Python pipeline running 🧠"}

@app.get("/health")
def health():
    return {
        "modules": {
            "module1_asr": "ready",
            "module2_kg": "pending DB",
            "module3_sql": "pending DB",
            "module4_eval": "pending DB"
        }
    }

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Receive audio file → transcribe → extract entities."""
    audio_bytes = await audio.read()
    transcript = transcribe_audio(audio_bytes)
    entities = tag_entities(transcript)
    return {
        "transcript": transcript,
        "entities": entities
    }