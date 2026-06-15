import speech_recognition as sr
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speech_to_sql.m1_speech.transcriber import transcribe_audio
from speech_to_sql.m2_nlu.entity_tagger import tag_entities

def record_audio() -> bytes:
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 3

    with sr.Microphone() as source:
        print("\nAdjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Speak now...")
        audio = recognizer.listen(source)
        print("Recording complete.")

    return audio.get_wav_data()

def run():
    audio_bytes = record_audio()

    print("\nTranscribing...")
    transcript = transcribe_audio(audio_bytes)
    print("\nTRANSCRIPT:", transcript)

    print("\nExtracting entities...")
    entities = tag_entities(transcript)
    print("\nENTITIES:", entities)

if __name__ == "__main__":
    run()
