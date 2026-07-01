"""
Write translated text back into PDF text-layer (in-place replacement).

Strategy: redact the original span bbox, then insert translated text with
the same font/size/color at the same position.

Two entry points:
  write_page(page, spans)           — edit one page of an already-open document
  write_translations(path, spans..) — open/save wrapper (kept for standalone use)
"""

import io

import fitz
import numpy as np
from PIL import Image

# DPI used to rasterize the page for background sampling + inpainting.
_SAMPLE_DPI = 150


def write_page(page: "fitz.Page", spans: list[dict], strip_source: bool = False) -> None:
    """
    Apply translated spans to a single (already-open) page.

    Hiding strategy — *inpaint-and-redraw* rather than redaction:
    design-tool exports (Canva/Figma/InDesign) usually draw text inside Form
    XObjects, and PyMuPDF's ``apply_redactions`` does not descend into XObjects,
    so the original text survives. Instead we rasterize the page, **telea-inpaint
    the original glyph pixels** to reconstruct the background (gradients/photos
    included), patch that clean background over each original span as a small
    image, then draw the translation as native (vector) text on top. This is
    robust regardless of nesting and — unlike a flat sampled-colour rectangle —
    leaves no visible box over gradient/photo backgrounds. (Same background
    handling as Track B.)

    Long translations are re-flowed onto multiple lines when they would
    overflow the original box width (Vietnamese runs wider than English).

    `strip_source`: also run a text-only redaction pass to delete the underlying
    original text from the content stream (so it isn't selectable/searchable).
    This works for page-level text (Word/InDesign); it is a harmless no-op for
    text locked inside XObjects, where the cover rectangle does the hiding.

    `spans` must all belong to this page and carry a 'translation' field.
    """
    changed = [s for s in spans if s.get("translation", s["text"]) != s["text"]]
    if not changed:
        return

    if strip_source:
        for span in changed:
            page.add_redact_annot(fitz.Rect(span["bbox"]))
        # Remove text only; keep images and vector art untouched.
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            text=fitz.PDF_REDACT_TEXT_REMOVE,
        )

    # Rasterize the *original* page once, before any drawing, to sample from.
    scale = _SAMPLE_DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    covers = []  # (span, cover_pdf_rect, cover_px, bg)
    boxes_px = []
    for span in changed:
        bbox = fitz.Rect(span["bbox"])
        bg = _sample_bg(img, scale, bbox, pix.width, pix.height)
        # Actual glyph color comes from the pixels, not span["color"]: gradient/
        # pattern-filled text reports as black (0) in the text layer even when it
        # renders white/coloured. Trust the raster; fall back to the layer color.
        fg = _sample_fg(img, scale, bbox, bg)
        span["_draw_color"] = fg if fg is not None else _unpack_color(span["color"])
        # Pad slightly so ascenders/descenders of the original are fully hidden.
        pad = bbox.height * 0.12
        cover = fitz.Rect(bbox.x0 - pad, bbox.y0 - pad, bbox.x1 + pad, bbox.y1 + pad)
        px = _px_box(cover, scale, pix.width, pix.height)
        covers.append((span, cover, px, bg))
        if px is not None:
            boxes_px.append(px)

    # Reconstruct the background behind the original text (erase glyphs) once.
    cleaned = _inpaint_cover(img, boxes_px)

    for span, cover, px, bg in covers:
        if cleaned is not None and px is not None:
            x0, y0, x1, y1 = px
            crop = np.ascontiguousarray(cleaned[y0:y1, x0:x1])
            page.insert_image(cover, stream=_png_bytes(crop), keep_proportion=False)
        else:
            # Fallback (no cv2): flat sampled-colour rectangle.
            page.draw_rect(cover, color=bg, fill=bg, width=0)

    for span in changed:
        _draw_translation(page, span, span["_draw_color"])


def _px_box(rect: "fitz.Rect", scale: float, w: int, h: int):
    x0 = max(0, int(rect.x0 * scale))
    y0 = max(0, int(rect.y0 * scale))
    x1 = min(w, int(rect.x1 * scale))
    y1 = min(h, int(rect.y1 * scale))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _png_bytes(arr: "np.ndarray") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _inpaint_cover(img: "np.ndarray", boxes_px: list) -> "np.ndarray | None":
    """
    Telea-inpaint the glyph pixels inside each box so the background (incl.
    gradients/photos) is reconstructed. Returns a cleaned copy, or None if
    OpenCV is unavailable (caller falls back to flat fill).
    """
    if not boxes_px:
        return img
    try:
        import cv2
    except Exception:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for x0, y0, x1, y1 in boxes_px:
        region = gray[y0:y1, x0:x1].astype(np.int32)
        bg = np.median(region)
        diff = np.abs(region - bg)
        thr = max(28.0, float(diff.std()))
        mask[y0:y1, x0:x1] = np.maximum(
            mask[y0:y1, x0:x1], (diff > thr).astype(np.uint8) * 255
        )
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    if mask.sum() == 0:
        return img
    return cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def _draw_translation(page: "fitz.Page", span: dict, color: tuple) -> None:
    """
    Draw one translated span. Priority order:
      1. Fits at the original size  -> draw at the original baseline.
      2. Slightly too wide          -> shrink on ONE line (down to 60%),
                                        keeping the baseline. Headings/labels
                                        stay single-line as designed.
      3. Too long even at 60%       -> wrap into a modest box (≤ ~2.2× the
                                        original height) with a fitted size.
    """
    text = span["translation"]
    fontfile = _font_for(span)
    size = float(span["size"])
    bbox = fitz.Rect(span["bbox"])
    origin = fitz.Point(span["origin"])

    def _line(sz: float) -> None:
        page.insert_text(origin, text, fontsize=sz, color=color,
                         fontname="notos", fontfile=fontfile)

    try:
        font = fitz.Font(fontfile=fontfile)
    except Exception:
        _line(size)
        return

    if bbox.width <= 0:
        _line(size)
        return

    full = font.text_length(text, fontsize=size)
    if full <= bbox.width:
        _line(size)
        return

    shrink = size * (bbox.width / full)
    if shrink >= size * 0.6:
        _line(shrink)
        return

    # Genuinely long: wrap into a modest downward box, fitting the font to it.
    max_height = bbox.height * 2.2
    box = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y0 + max_height)
    fitted = _fit_wrapped_size(font, text, bbox.width, max_height, size)
    rc = page.insert_textbox(box, text, fontsize=fitted, color=color,
                             fontname="notos", fontfile=fontfile, align=0)
    if rc < 0:
        _line(max(shrink, size * 0.45))


def _sample_fg(
    img: "np.ndarray", scale: float, bbox: "fitz.Rect", bg_rgb: tuple
) -> tuple[float, float, float] | None:
    """
    Estimate the actual glyph color as the median color of the pixels inside the
    bbox that stand out strongly from the sampled background. Returns None when
    too few glyph pixels are found (thin/small text) so the caller can fall back
    to the text-layer color.
    """
    x0, y0 = max(0, int(bbox.x0 * scale)), max(0, int(bbox.y0 * scale))
    x1, y1 = int(bbox.x1 * scale), int(bbox.y1 * scale)
    x1 = min(img.shape[1], x1)
    y1 = min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.int32)
    bg = np.array([c * 255.0 for c in bg_rgb])
    dist = np.sqrt(((crop - bg) ** 2).sum(axis=1))
    glyph = crop[dist > 60.0]  # pixels clearly different from background
    if glyph.shape[0] < max(10, int(0.015 * crop.shape[0])):
        return None
    med = np.median(glyph, axis=0) / 255.0
    return (float(med[0]), float(med[1]), float(med[2]))


def _wrap_lines(font: "fitz.Font", text: str, size: float, width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if font.text_length(test, fontsize=size) <= width or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _fit_wrapped_size(
    font: "fitz.Font", text: str, width: float, max_height: float, start: float
) -> float:
    """Largest size (down to 50% of start) whose wrapped text fits max_height."""
    size = start
    floor = start * 0.5
    while size >= floor:
        lines = _wrap_lines(font, text, size, width)
        if len(lines) * (size * 1.15) <= max_height:
            return size
        size -= 0.5
    return floor


def _sample_bg(
    img: "np.ndarray", scale: float, bbox: "fitz.Rect", pix_w: int, pix_h: int
) -> tuple[float, float, float]:
    """
    Estimate the background color behind a text span as the median color of a
    thin ring just outside the span's bbox (avoids the glyph pixels themselves).
    Falls back to white if the ring is empty.
    """
    x0, y0 = int(bbox.x0 * scale), int(bbox.y0 * scale)
    x1, y1 = int(bbox.x1 * scale), int(bbox.y1 * scale)
    m = max(3, int((y1 - y0) * 0.15))
    ys0, ys1 = max(0, y0 - m), min(pix_h, y1 + m)
    xs0, xs1 = max(0, x0 - m), min(pix_w, x1 + m)
    if x1 <= x0 or y1 <= y0:
        return (1.0, 1.0, 1.0)

    parts = [
        img[ys0:y0, xs0:xs1].reshape(-1, 3),  # top band
        img[y1:ys1, xs0:xs1].reshape(-1, 3),  # bottom band
        img[y0:y1, xs0:x0].reshape(-1, 3),    # left band
        img[y0:y1, x1:xs1].reshape(-1, 3),    # right band
    ]
    ring = np.concatenate([p for p in parts if p.size], axis=0) if any(
        p.size for p in parts
    ) else np.empty((0, 3))
    if ring.size == 0:
        return (1.0, 1.0, 1.0)
    med = np.median(ring, axis=0) / 255.0
    return (float(med[0]), float(med[1]), float(med[2]))


# --- font selection -------------------------------------------------------

from pathlib import Path

_FONTS_DIR = Path(__file__).parent.parent.parent / "assets" / "fonts"


def _font_for(span: dict) -> str:
    """Pick a NotoSans weight roughly matching the original font name."""
    name = (span.get("font") or "").lower()
    bold = "bold" in name or "black" in name or "heavy" in name
    italic = "italic" in name or "oblique" in name
    if bold and italic:
        f = "NotoSans-BoldItalic.ttf"
    elif bold:
        f = "NotoSans-Bold.ttf"
    elif italic:
        f = "NotoSans-Italic.ttf"
    else:
        f = "NotoSans-Regular.ttf"
    path = _FONTS_DIR / f
    if not path.exists():
        path = _FONTS_DIR / "NotoSans-Regular.ttf"
    return str(path)


def write_translations(
    pdf_path: str,
    spans: list[dict],
    output_path: str,
    fonts_dir: str | None = None,
) -> None:
    """Open a PDF, apply translated spans page by page, save to output_path."""
    doc = fitz.open(pdf_path)

    by_page: dict[int, list[dict]] = {}
    for span in spans:
        by_page.setdefault(span["page"], []).append(span)

    for page_idx, page_spans in by_page.items():
        write_page(doc[page_idx], page_spans)

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()


def _unpack_color(packed: int) -> tuple[float, float, float]:
    r = ((packed >> 16) & 0xFF) / 255.0
    g = ((packed >> 8) & 0xFF) / 255.0
    b = (packed & 0xFF) / 255.0
    return (r, g, b)
