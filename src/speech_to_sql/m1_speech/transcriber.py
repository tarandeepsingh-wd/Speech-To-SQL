import os
import tempfile

import whisper


_MODELS = {}

_RECIPE_PROMPT = (
    "Recipe search. Ingredients: cumin, coriander, turmeric, paneer, "
    "ghee, cardamom, ginger, garlic, onion, tomato, chicken, rice, lentil, "
    "spinach, yogurt, cream, saffron, masala, biryani, dal, sabji, adrak, "
    "haldi, jeera, lahsun, mirch."
)


def _get_model(size: str = "small"):
    """Lazy-load and cache Whisper models by size."""
    if size not in _MODELS:
        _MODELS[size] = whisper.load_model(size)
    return _MODELS[size]


def transcribe_audio(
    audio_bytes: bytes,
    language: str | None = None,
    model_size: str = "small",
) -> str:
    """
    Transcribe raw WAV audio bytes to a cleaned text string.

    Args:
        audio_bytes: Raw WAV file content as bytes.
        language: ISO 639-1 code such as "en" or "hi"; None enables
            Whisper auto-detection and is best for mixed Hindi-English.
        model_size: Whisper model size, defaulting to CPU-friendly "small".
    """
    model = _get_model(model_size)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(
            tmp_path,
            language=language,
            initial_prompt=_RECIPE_PROMPT,
            temperature=0.0,
            word_timestamps=True,
        )
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)
