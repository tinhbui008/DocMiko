"""
Vision-LLM OCR for image-based PDF pages.

Uses Claude vision to extract text blocks with position, style, and translation
in a single API call — no hardcoded deck-specific rules.

Output schema per block:
{
    "id": str,
    "bbox_norm": [x0, y0, x1, y1],  # normalized 0..1 relative to image size
    "text": str,
    "translation": str,
    "font_color_hex": str,           # e.g. "#FFFFFF"
    "bg_color_hex": str | null,
    "font_size_pt": float | null,
    "role": str,                     # "heading" | "body" | "caption" | "label" | ...
}
"""

import base64
import hashlib
import io
import json
from pathlib import Path

import fitz
from PIL import Image

from core.config import get_model, make_client

_MODEL = get_model("claude-sonnet-4-6")

# Persistent cache so a Vision-OCR call (which costs money) is never repeated
# for the same page image + target language. Keyed by hash(png bytes + lang).
_CACHE_PATH = Path(__file__).parent.parent.parent / "cache" / "ocr_blocks.jsonl"

_SYSTEM_TEMPLATE = """You are a PDF translation assistant.
Given a page image, identify every text block visible on the page.
For each block output a JSON object with these fields:
  id: unique string (e.g. "b01")
  bbox_norm: [x0, y0, x1, y1] normalized to image width/height (0.0-1.0)
  text: original text exactly as it appears
  translation: {target_lang} translation of the text
  font_color_hex: dominant text color as "#RRGGBB"
  bg_color_hex: background color directly behind the text as "#RRGGBB", or null if transparent/image
  font_size_pt: approximate font size in points, or null if unknown
  role: one of heading | subheading | body | caption | label | button | other

Return ONLY a valid JSON array of these objects - no markdown fences, no explanation."""


def _image_to_png_bytes(image: Image.Image) -> tuple[bytes, int, int]:
    buf = io.BytesIO()
    rgb = image.convert("RGB")
    rgb.save(buf, format="PNG")
    return buf.getvalue(), rgb.width, rgb.height


def _cache_key(png_bytes: bytes, target_lang: str) -> str:
    h = hashlib.sha256()
    h.update(png_bytes)
    h.update(target_lang.encode())
    return h.hexdigest()[:20]


def _cache_get(key: str) -> list[dict] | None:
    if not _CACHE_PATH.exists():
        return None
    with open(_CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if entry.get("key") == key:
                    return entry["blocks"]
    return None


def _cache_put(key: str, blocks: list[dict]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "blocks": blocks}, ensure_ascii=False) + "\n")


def ocr_image(
    image: Image.Image,
    target_lang: str = "Vietnamese",
    page_index: int | None = None,
) -> list[dict]:
    """
    Run Vision-LLM OCR on an already-rendered PIL image.
    Returns a list of text-block dicts (bbox_norm + translation + style).
    Results are cached by image+language so a paid call never repeats.
    """
    png_bytes, width, height = _image_to_png_bytes(image)
    key = _cache_key(png_bytes, target_lang)
    cached = _cache_get(key)
    if cached is not None:
        for block in cached:
            block["page_index"] = page_index
        return cached

    client = make_client()  # lazy — only builds the SDK client on a real call
    img_b64 = base64.standard_b64encode(png_bytes).decode()

    message = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=_SYSTEM_TEMPLATE.format(target_lang=target_lang),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract and translate all text blocks on this page.",
                    },
                ],
            }
        ],
    )

    raw = message.content[0].text.strip()
    start = raw.find("[")
    end = raw.rfind("]") + 1
    blocks = json.loads(raw[start:end])

    for block in blocks:
        block["image_width"] = width
        block["image_height"] = height
        block["page_index"] = page_index

    _cache_put(key, blocks)
    return blocks


def ocr_page(pdf_path: str, page_idx: int, dpi: int = 150, target_lang: str = "Vietnamese") -> list[dict]:
    """Standalone helper: rasterize a PDF page then OCR it."""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return ocr_image(image, target_lang=target_lang, page_index=page_idx)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m core.track_b.ocr <file.pdf> <page_index>")
        sys.exit(1)

    result = ocr_page(sys.argv[1], int(sys.argv[2]))
    print(json.dumps(result, indent=2, ensure_ascii=False))
