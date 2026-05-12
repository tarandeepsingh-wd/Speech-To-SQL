import os
import speech_recognition as sr

from transcriber import transcribe_audio
from entity_tagger import tag_entities

# RECORD AUDIO
def record_audio():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold=3
    with sr.Microphone() as source:
        print("\nSpeak now...")
        recognizer.adjust_for_ambient_noise(source)

        audio = recognizer.listen(source)

        print("Recording complete.")

    return audio.get_wav_data()


# GET AUDIO BYTES

audio_bytes = record_audio()


# TRANSCRIBE

print("\nTranscribing.")

transcript = transcribe_audio(audio_bytes)

print("\n TRANSCRIPT:")
print(transcript)


# ENTITY EXTRACTION

entities = tag_entities(transcript)

print("\n ENTITIES:")
print(entities)

# CLEANUP
os.remove(audio_path)
