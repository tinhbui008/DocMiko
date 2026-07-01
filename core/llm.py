"""
Provider-agnostic text translation.

`translate_batch(texts, target_lang)` returns a translation for each input
string, in order. It dispatches on `config.get_provider()`:

  anthropic          -> Claude (messages API)
  ollama / openai    -> OpenAI-compatible chat API (Ollama at :11434/v1)

Both tracks now translate *text only* (Track B gets its text from RapidOCR),
so a local Ollama text model can stand in for Anthropic during testing — no
vision model required. Switch by setting in .env:

  LLM_PROVIDER=ollama
  LLM_BASE_URL=http://localhost:11434/v1
  LLM_MODEL=qwen2.5

Untranslatable placeholder text (e.g. "Lorem ipsum") should be returned
unchanged by the model; we also guard by falling back to the source string
whenever the response count doesn't line up.
"""

from __future__ import annotations

import re

from core.config import get_provider, get_model, make_client, make_openai_client

_PROMPT = (
    "You are a professional translator. Translate each numbered line into {lang}.\n"
    "Output rules:\n"
    "- Output exactly one line per input, in the SAME numbered format: `N. translation`.\n"
    "- Keep the same count and order. Do NOT merge, split, add, or drop lines.\n"
    "- Write natural {lang} with normal spaces between words.\n"
    "- Write ONLY in {lang}. Never output Chinese characters (unless {lang} is Chinese).\n"
    "- Keep placeholder/lorem-ipsum text unchanged. Never add notes or commentary.\n\n{body}"
)

_LINE_RE = re.compile(r"^\s*(\d+)\s*[\.\)\:\-]\s*(.*\S)\s*$")


def _build_prompt(texts: list[str], target_lang: str) -> str:
    body = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return _PROMPT.format(lang=target_lang, body=body)


def _parse_numbered(raw: str, texts: list[str]) -> list[str]:
    """
    Parse `N. translation` lines, matching by index. Robust to missing/extra
    lines and invalid JSON: any input line without a parsed match keeps its
    original text rather than corrupting the whole batch.
    """
    result = list(texts)
    if not raw:
        return result
    mp: dict[int, str] = {}
    for line in raw.splitlines():
        m = _LINE_RE.match(line)
        if m:
            val = m.group(2).strip().strip('"').strip()
            if val:
                mp[int(m.group(1))] = val
    for i in range(len(texts)):
        if (i + 1) in mp:
            result[i] = mp[i + 1]
    return result


def translate_batch(
    texts: list[str],
    target_lang: str = "Vietnamese",
    chunk_size: int | None = None,
) -> list[str]:
    """
    Translate a list of strings. Length-preserving; per-line source fallback.

    `chunk_size`: when set, translate at most this many lines per call. Smaller
    chunks stop weaker local models from drifting the line numbering (a line's
    translation landing on the wrong line). None = one call for all lines.
    """
    if not texts:
        return []

    if chunk_size and chunk_size > 0 and len(texts) > chunk_size:
        out: list[str] = []
        for i in range(0, len(texts), chunk_size):
            out.extend(_translate_once(texts[i : i + chunk_size], target_lang))
        return out
    return _translate_once(texts, target_lang)


def _translate_once(texts: list[str], target_lang: str) -> list[str]:
    prompt = _build_prompt(texts, target_lang)
    if get_provider() in ("ollama", "openai"):
        raw = _call_openai(prompt)
    else:
        raw = _call_anthropic(prompt)
    return _parse_numbered(raw, texts)


def _call_anthropic(prompt: str) -> str | None:
    try:
        client = make_client()
        msg = client.messages.create(
            model=get_model(),
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return None


def _call_openai(prompt: str) -> str | None:
    try:
        client = make_openai_client()
        resp = client.chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None
