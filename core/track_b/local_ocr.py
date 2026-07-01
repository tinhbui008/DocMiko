"""
Vision-free Track B OCR+translate.

Instead of a Vision-LLM, this builds the same block schema from local pieces:
  * RapidOCR  -> precise per-line boxes + recognized source text (local, free)
  * translate_batch -> translation via the configured provider (Ollama/Anthropic)
  * raster sampling -> actual glyph colour (captures gradient/pattern fills)

Because the boxes already come from the detector, no separate detect pass is
needed (use it as the pipeline `ocr_fn` with `detect_fn=None`). This lets a
local Ollama *text* model drive Track B end-to-end — no vision model required.

Block schema matches core.track_b.ocr.ocr_image so the rest of the pipeline
(inpaint / render / recompose) is unchanged.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from core.llm import translate_batch
from core.track_b.detector import detect_text_boxes, merge_lines_to_blocks


def _sample_glyph_rgb(arr: np.ndarray, box: list) -> tuple[int, int, int]:
    """Median colour of the glyph pixels in a box as an (r, g, b) tuple."""
    x0, y0, x1, y1 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(arr.shape[1], x1), min(arr.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0)
    region = arr[y0:y1, x0:x1].reshape(-1, 3).astype(np.int32)
    bg = np.median(region, axis=0)
    dist = np.sqrt(((region - bg) ** 2).sum(axis=1))
    glyph = region[dist > 45.0]
    if glyph.shape[0] < max(8, int(0.02 * region.shape[0])):
        glyph = region
    med = np.median(glyph, axis=0).astype(int)
    return (int(med[0]), int(med[1]), int(med[2]))


def _sample_glyph_color(arr: np.ndarray, box: list) -> str:
    """Median glyph colour -> '#RRGGBB'."""
    return "#{:02x}{:02x}{:02x}".format(*_sample_glyph_rgb(arr, box))


def _desegment(text: str) -> str:
    """
    RapidOCR loses spaces on tightly-tracked text ("acrossthecountrytoprovide").
    Re-insert word boundaries with wordninja, but only on long single tokens
    (>=12 letters) so short acronyms/brands (BOFA) are left intact. wordninja
    preserves original character casing.
    """
    try:
        import wordninja
    except Exception:
        return text
    out = []
    for tok in text.split():
        core = tok.strip(".,:;()[]'\"")
        if len(core) >= 12 and core.replace("'", "").isalpha():
            parts = wordninja.split(tok)
            if len(parts) > 1:
                out.append(" ".join(parts))
                continue
        out.append(tok)
    return " ".join(out)


def _is_brandish(text: str) -> bool:
    """
    Heuristic: short ALL-CAPS text (with or without spaces) is almost always a
    logo / acronym / brand (BFFB, BOFA, "LONGMAN GROUP") — keep it untranslated
    so it is neither erased nor redrawn. The length cap avoids skipping long
    all-caps headings that genuinely need translating.
    """
    t = text.strip()
    return bool(t) and t.isupper() and any(c.isalpha() for c in t) and len(t) <= 15


def _role_for(px_height: float) -> str:
    if px_height >= 44:
        return "heading"
    if px_height >= 30:
        return "subheading"
    return "body"


def ocr_translate_local(
    image: Image.Image,
    target_lang: str = "Vietnamese",
    page_index: int | None = None,
    min_conf: float = 0.5,
) -> list[dict]:
    """Detect + translate every text line locally, returning OCR block dicts."""
    arr = np.array(image.convert("RGB"))
    W, H = image.width, image.height

    det = [d for d in detect_text_boxes(image) if d["conf"] >= min_conf]
    if not det:
        return []

    # Repair space-less OCR runs, then merge line-boxes of the same paragraph so
    # a split sentence is translated as one coherent unit. Attach each line's
    # glyph colour so headings (distinct colour) don't merge into body text.
    for d in det:
        d["text"] = _desegment(d["text"])
        d["_rgb"] = _sample_glyph_rgb(arr, d["bbox"])
    merged = merge_lines_to_blocks(det)

    # Only send real prose to the translator; keep brand/acronym tokens as-is.
    to_translate = [m for m in merged if not _is_brandish(m["text"])]
    tr_map = {}
    if to_translate:
        # chunk_size=1 -> one call per paragraph: no cross-line numbering drift.
        results = translate_batch(
            [m["text"] for m in to_translate], target_lang, chunk_size=1
        )
        tr_map = {id(m): t for m, t in zip(to_translate, results)}

    blocks: list[dict] = []
    for i, m in enumerate(merged):
        x0, y0, x1, y1 = m["bbox"]
        line_h = m.get("line_h", y1 - y0)
        tr = tr_map.get(id(m), m["text"])
        blocks.append(
            {
                "id": f"b{i:02d}",
                "bbox_norm": [x0 / W, y0 / H, x1 / W, y1 / H],
                "text": m["text"],
                "translation": tr,
                "font_color_hex": _sample_glyph_color(arr, m["bbox"]),
                "bg_color_hex": None,
                # Renderer uses this as a pixel size; base it on a single line.
                "font_size_pt": max(8, int(line_h * 0.8)),
                "role": _role_for(line_h),
                "image_width": W,
                "image_height": H,
                "page_index": page_index,
                "_matched": True,  # boxes are already precise
            }
        )
    return blocks
