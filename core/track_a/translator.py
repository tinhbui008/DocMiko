"""
Translate text spans via the configured LLM provider with a persistent
JSONL cache. Provider (Anthropic / Ollama) is chosen in core.llm / core.config.

Cache key: hash(source_text + target_lang)
"""

import hashlib
import json
from pathlib import Path

from core.config import get_model
from core.llm import translate_batch

_DEFAULT_CACHE_PATH = Path(__file__).parent.parent.parent / "cache" / "translations.jsonl"


def _cache_key(text: str, target_lang: str) -> str:
    # Include the model so Claude and Ollama translations don't collide in the
    # shared cache (different providers -> different entries for the same text).
    return hashlib.sha256(f"{text}||{target_lang}||{get_model()}".encode()).hexdigest()[:16]


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
    chunk_size: int = 8,
) -> list[dict]:
    """
    Translate a list of span dicts in-place (adds 'translation' field).
    Returns the same list with 'translation' populated.

    `chunk_size` is forwarded to translate_batch: on dense pages a single large
    request makes local models drift the line numbering, dropping many lines
    back to the source language. Small chunks keep alignment.

    A translation equal to the source is NOT cached — it usually means a
    fall-back (the model failed to align that line), so it should be retried on
    the next run rather than frozen as "English" in the cache.
    """
    cache = load_cache(cache_path)

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

        texts = [s["text"] for s, _ in to_translate]
        translations = translate_batch(texts, target_lang, chunk_size=chunk_size)

        for (span, key), translation in zip(to_translate, translations):
            span["translation"] = translation
            if translation != span["text"]:  # don't cache fall-backs
                cache[key] = translation
                save_to_cache(key, translation, cache_path)

    return spans
