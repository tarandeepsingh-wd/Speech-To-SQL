import whisper
import tempfile
import os

# Load model once at startup (use "base" for speed, "small"/"medium" for accuracy)
model = whisper.load_model("base")

def transcribe_audio(audio_bytes: bytes) -> str:
    """Takes raw audio bytes, returns transcribed text."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path)
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)