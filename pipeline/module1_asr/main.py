import io
import wave
import numpy as np
import sounddevice as sd

from module1_asr.transcriber import transcribe_audio
from module1_asr.entity_tagger import tag_entities


SAMPLE_RATE = 16000
CHANNELS    = 1


def record_until_enter() -> np.ndarray:
    """
    Records audio from the default microphone until the user
    presses Enter. Returns a numpy array of float32 samples.
    """
    print("\n Recording... Press ENTER to stop.\n")

    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate = SAMPLE_RATE,
        channels   = CHANNELS,
        dtype      = "float32",
        callback   = callback,
    ):
        input()   # blocks until user presses Enter

    if not frames:
        return np.array([], dtype="float32")

    return np.concatenate(frames, axis=0)


def numpy_to_wav_bytes(audio: np.ndarray) -> bytes:
    """
    Converts a float32 numpy array → in-memory WAV bytes.
    Whisper's transcribe() needs a file path or file-like WAV.
    We write to a BytesIO buffer so no temp file is needed here
    (transcriber.py writes its own temp file from these bytes).
    """
    # Convert float32 [-1, 1] → int16 for standard WAV
    pcm = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)          # 2 bytes = int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    return buf.getvalue()


def process_audio(audio_bytes: bytes) -> dict:
    """Full pipeline: WAV bytes -> transcript -> structured entities."""
    text     = transcribe_audio(audio_bytes)
    entities = tag_entities(text)
    return {"transcript": text, **entities}


def process_text(text: str) -> dict:
    """Text-only path — skips ASR. Useful for quick testing."""
    entities = tag_entities(text)
    return {"transcript": text, **entities}


def print_result(result: dict):
    print(f"\n  Transcript       : {result['transcript']}")
    print(f"  Ingredients      : {result['ingredients']}")
    print(f"  Ingredient logic : {result.get('ingredient_logic', {})}")
    print(f"  Utensils         : {result['utensils']}")
    state_actions = [a['action'] for a in result['actions'] if a['type'] == 'state_change']
    print(f"  Cooking actions  : {state_actions}")
    print(f"  Dietary styles   : {result['dietary_styles']}")
    print(f"  Categories       : {[c['category'] for c in result['categories']]}")
    print(f"  Nutrition filters: {result['nutrition_filters']}")
    print(f"  SQL operators    : {result.get('sql_operators', [])}")


# ─────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  TADKA - Module 1: ASR + Entity Tagger")
    print("=" * 55)

    while True:
        print("\nOptions:")
        print("  [1] Speak a query (microphone)")
        print("  [2] Type a query  (text)")
        print("  [q] Quit")
        choice = input("\nChoice: ").strip().lower()

        if choice == "q":
            print("Bye!")
            break

        elif choice == "1":
            audio_np  = record_until_enter()
            if audio_np.size == 0:
                print("No audio captured. Try again.")
                continue
            wav_bytes = numpy_to_wav_bytes(audio_np)
            print("\nTranscribing with Whisper...")
            result    = process_audio(wav_bytes)
            print_result(result)

        elif choice == "2":
            text   = input("Type your query: ").strip()
            result = process_text(text)
            print_result(result)

        else:
            print("Invalid choice.")
