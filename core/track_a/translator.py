"""
Translate text spans via Claude API with persistent JSONL cache.

Cache key: hash(source_text + target_lang)
"""

import hashlib
import json
from pathlib import Path

import anthropic

from core.config import get_api_key, get_model

_DEFAULT_CACHE_PATH = Path(__file__).parent.parent.parent / "cache" / "translations.jsonl"
_MODEL = get_model("claude-haiku-4-5-20251001")


def _cache_key(text: str, target_lang: str) -> str:
    return hashlib.sha256(f"{text}||{target_lang}".encode()).hexdigest()[:16]


def load_cache(cache_path: Path = _DEFAULT_CACHE_PATH) -> dict:
    cache: dict = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    cache[entry["key"]] = entry["translation"]
    return cache


def save_to_cache(key: str, translation: str, cache_path: Path = _DEFAULT_CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "translation": translation}, ensure_ascii=False) + "\n")


def translate_spans(
    spans: list[dict],
    target_lang: str = "Vietnamese",
    cache_path: Path = _DEFAULT_CACHE_PATH,
    batch_size: int = 50,
) -> list[dict]:
    """
    Translate a list of span dicts in-place (adds 'translation' field).
    Returns the same list with 'translation' populated.
    """
    client = anthropic.Anthropic(api_key=get_api_key())
    cache = load_cache(cache_path)

    # Split into batches to avoid hitting token limits
    for i in range(0, len(spans), batch_size):
        batch = spans[i : i + batch_size]
        to_translate = []
        for span in batch:
            key = _cache_key(span["text"], target_lang)
            if key in cache:
                span["translation"] = cache[key]
            else:
                to_translate.append((span, key))

        if not to_translate:
            continue

        # Build prompt: translate all texts in one API call
        texts = [s["text"] for s, _ in to_translate]
        prompt = (
            f"Translate each line below to {target_lang}. "
            "Return ONLY a JSON array of translated strings in the same order. "
            "Preserve formatting (whitespace, punctuation). Do not add explanations.\n\n"
            + "\n".join(f"{j+1}. {t}" for j, t in enumerate(texts))
        )

        message = client.messages.create(
            model=_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()

        # Parse JSON array response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        translations = json.loads(raw[start:end])

        for (span, key), translation in zip(to_translate, translations):
            span["translation"] = translation
            save_to_cache(key, translation, cache_path)

    return spans
