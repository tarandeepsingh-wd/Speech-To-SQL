import speech_recognition as sr
import tempfile
import os

recognizer = sr.Recognizer()

def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text.strip()
    finally:
        os.unlink(tmp_path)
        
        
#         import tempfile
# import os
# import whisper

# model = whisper.load_model("small")

# def transcribe_audio(audio_bytes: bytes) -> str:
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#         tmp.write(audio_bytes)
#         tmp_path = tmp.name
#     try:
#         result = model.transcribe(tmp_path)
#         return result["text"].strip()
#     finally:
#         os.unlink(tmp_path)