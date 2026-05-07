import sounddevice as sd
from scipy.io.wavfile import write
import os

from transcriber import transcribe_audio
from entity_tagger import tag_entities


duration = 5
samplerate = 44100

print("\n🎤 Speak now...")

audio = sd.rec(
    int(duration * samplerate),
    samplerate=samplerate,
    channels=1,
    dtype="int16"
)

sd.wait()

print(" Recording complete.")


# SAVE AUDIO

audio_path = "temp_audio.wav"

write(audio_path, samplerate, audio)

# READ AUDIO BYTES

with open(audio_path, "rb") as f:
    audio_bytes = f.read()


# TRANSCRIBE

print("\n⏳ Transcribing...")

transcript = transcribe_audio(audio_bytes)

print("\n📝 TRANSCRIPT:")
print(transcript)


# ENTITY EXTRACTION

entities = tag_entities(transcript)

print("\n🔍 ENTITIES:")
print(entities)

# CLEANUP
os.remove(audio_path)
