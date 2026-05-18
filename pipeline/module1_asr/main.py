from module1_asr.transcriber import transcribe_audio
from module1_asr.entity_tagger import tag_entities

def process_audio(audio_bytes: bytes) -> dict:
    text     = transcribe_audio(audio_bytes)
    entities = tag_entities(text)
    return {
        "transcript": text,
        "entities":   entities
    }

def process_text(text: str) -> dict:
    entities = tag_entities(text)
    return {
        "transcript": text,
        "entities":   entities
    }

if __name__ == "__main__":
    result = process_text("I want chicken curry with adrak and lal mirch, make it gluten-free")
    print(result)