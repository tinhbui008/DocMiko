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
import json
import os
from pathlib import Path

import anthropic
import fitz


_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """You are a PDF translation assistant.
Given a page image, identify every text block visible on the page.
For each block output a JSON object with these fields:
  id: unique string (e.g. "b01")
  bbox_norm: [x0, y0, x1, y1] normalized to image width/height (0.0–1.0)
  text: original text exactly as it appears
  translation: Vietnamese translation of the text
  font_color_hex: dominant text color as "#RRGGBB"
  bg_color_hex: background color directly behind the text as "#RRGGBB", or null if transparent/image
  font_size_pt: approximate font size in points, or null if unknown
  role: one of heading | subheading | body | caption | label | button | other

Return ONLY a valid JSON array of these objects — no markdown fences, no explanation."""


def page_to_image_b64(pdf_path: str, page_idx: int, dpi: int = 150) -> tuple[bytes, int, int]:
    """Render a PDF page to a PNG image, return (b64_bytes, width_px, height_px)."""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes, pix.width, pix.height


def ocr_page(pdf_path: str, page_idx: int, dpi: int = 150) -> list[dict]:
    """
    Run Vision-LLM OCR on a single PDF page.
    Returns list of text block dicts with translations.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    img_bytes, width, height = page_to_image_b64(pdf_path, page_idx, dpi)
    img_b64 = base64.standard_b64encode(img_bytes).decode()

    message = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
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

    # Attach image dimensions for downstream use
    for block in blocks:
        block["image_width"] = width
        block["image_height"] = height
        block["page_index"] = page_idx

    return blocks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m core.track_b.ocr <file.pdf> <page_index>")
        sys.exit(1)

    result = ocr_page(sys.argv[1], int(sys.argv[2]))
    print(json.dumps(result, indent=2, ensure_ascii=False))
