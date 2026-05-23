# # import speech_recognition as sr
# # import tempfile
# # import os

# # recognizer = sr.Recognizer()

# # def transcribe_audio(audio_bytes: bytes) -> str:
# #     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
# #         tmp.write(audio_bytes)
# #         tmp_path = tmp.name
# #     try:
# #         with sr.AudioFile(tmp_path) as source:
# #             audio = recognizer.record(source)
# #         text = recognizer.recognize_google(audio)
# #         return text.strip()
# #     finally:
# #         os.unlink(tmp_path)
        
        
# import tempfile
# import os
# import whisper

# model = whisper.load_model("base")

# def transcribe_audio(audio_bytes: bytes) -> str:
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#         tmp.write(audio_bytes)
#         tmp_path = tmp.name
#     try:
#         result = model.transcribe(tmp_path)
        
#         return result["text"].strip()
#     finally:
#         os.unlink(tmp_path)
import tempfile
import os
import whisper

# ─────────────────────────────────────────────────────────
# CHANGE: Switched from SpeechRecognition + Google Web API
#         to OpenAI Whisper (local, offline model).
#
# WHY:
#   1. The proposal explicitly specifies "Whisper ASR".
#   2. Google SR sends audio to Google's servers — a privacy
#      concern for a research database context.
#   3. Google SR requires internet on every call. Whisper runs
#      fully offline after the one-time model download.
#   4. Whisper is significantly more accurate on domain-specific
#      food vocabulary and handles accents better.
#   5. Whisper returns word-level timestamps (useful for future
#      BIO-style span alignment if needed).
#
# MODEL CHOICE: "small" (244M params) is a good default —
# accurate enough for food queries, fast enough on CPU.
# Swap to "medium" or "large" if a GPU is available.
# ─────────────────────────────────────────────────────────

_model = None

def _get_model():
    """Lazy-load Whisper model on first use."""
    global _model
    if _model is None:
        _model = whisper.load_model("small")
    return _model


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe raw WAV audio bytes → cleaned text string.

    Args:
        audio_bytes: Raw WAV file content as bytes.

    Returns:
        Transcribed text, stripped of leading/trailing whitespace.
    """
    model = _get_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language="en")
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)
