#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
babeldoc_like_pdf_translator.py

A practical BabelDOC-like PDF translation pipeline:

- Parse PDF with PyMuPDF.
- Build an intermediate representation (IR): page -> blocks -> lines -> spans.
- Reconstruct paragraphs using reading-order/layout heuristics.
- Translate paragraphs in batches through a pluggable Translator interface.
- Preserve the original PDF as the visual base.
- Cover original text regions.
- Overlay translated text with adaptive font fitting.
- Save a translated PDF.

This file is intentionally provider-agnostic. Replace DummyTranslator with your own
LLM / Google / DeepL / internal translation service.

Install:
    pip install pymupdf

Usage:
    python babeldoc_like_pdf_translator.py input.pdf output.pdf --target vi

Optional:
    python babeldoc_like_pdf_translator.py input.pdf output.pdf --font fonts/NotoSans-Regular.ttf --font-bold fonts/NotoSans-Bold.ttf
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import fitz  # PyMuPDF


# ============================================================
# Types
# ============================================================

BBox = Tuple[float, float, float, float]  # x0, y0, x1, y1 in PyMuPDF page coordinates, origin top-left
RGB = Tuple[float, float, float]          # 0..1


@dataclass
class TextSpan:
    text: str
    bbox: BBox
    font: str
    size: float
    color: int
    flags: int = 0
    is_bold: bool = False
    is_italic: bool = False


@dataclass
class TextLine:
    bbox: BBox
    spans: List[TextSpan] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans).strip()

    @property
    def font_size(self) -> float:
        sizes = [s.size for s in self.spans if s.size > 0]
        return median(sizes) if sizes else 10.0

    @property
    def color(self) -> int:
        colors = [s.color for s in self.spans]
        return colors[0] if colors else 0

    @property
    def main_font(self) -> str:
        fonts = [s.font for s in self.spans if s.font]
        return most_common(fonts) or "helv"

    @property
    def is_bold(self) -> bool:
        return any(s.is_bold for s in self.spans)


@dataclass
class TextBlock:
    id: str
    page_index: int
    bbox: BBox
    lines: List[TextLine] = field(default_factory=list)
    role: str = "body"       # body/title/caption/header/footer/page_number/table/formula/unknown
    order: int = 0
    original_text: str = ""
    translated_text: str = ""
    align: str = "left"      # left/center/right

    @property
    def font_size(self) -> float:
        sizes = [line.font_size for line in self.lines if line.font_size > 0]
        return median(sizes) if sizes else 10.0

    @property
    def color(self) -> int:
        colors = [line.color for line in self.lines]
        return most_common(colors) if colors else 0

    @property
    def is_bold(self) -> bool:
        return any(line.is_bold for line in self.lines)

    @property
    def main_font(self) -> str:
        fonts = [line.main_font for line in self.lines if line.main_font]
        return most_common(fonts) or "helv"


@dataclass
class PageIR:
    page_index: int
    width: float
    height: float
    rotation: int
    blocks: List[TextBlock] = field(default_factory=list)


@dataclass
class DocumentIR:
    source_pdf: str
    pages: List[PageIR] = field(default_factory=list)


# ============================================================
# Translation interface
# ============================================================

class Translator(Protocol):
    def translate_batch(
        self,
        items: List[Dict[str, str]],
        source_lang: str,
        target_lang: str,
        glossary: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Input:
            [{"id": "p0_b1", "text": "Hello world"}]

        Output:
            [{"id": "p0_b1", "translated": "Xin chào thế giới"}]

        Implement this with your real translation provider.
        """
        ...


class DummyTranslator:
    """
    Safe placeholder translator.
    It DOES NOT translate. It only marks the text so you can test layout/rendering.

    Replace this with a real translator.
    """
    def translate_batch(
        self,
        items: List[Dict[str, str]],
        source_lang: str,
        target_lang: str,
        glossary: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        return [
            {
                "id": item["id"],
                "translated": item["text"],
            }
            for item in items
        ]


class JsonMapTranslator:
    """
    Translator for testing from a prepared JSON file:
        {
          "p0_b1": "Bản dịch...",
          "p0_b2": "..."
        }

    Usage:
        python babeldoc_like_pdf_translator.py in.pdf out.pdf --translation-map translated.json
    """
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.translation_map = json.load(f)

    def translate_batch(
        self,
        items: List[Dict[str, str]],
        source_lang: str,
        target_lang: str,
        glossary: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        output = []
        for item in items:
            output.append({
                "id": item["id"],
                "translated": self.translation_map.get(item["id"], item["text"]),
            })
        return output


# ============================================================
# Utility
# ============================================================

def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def most_common(values: Sequence):
    if not values:
        return None
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def rect_union(rects: Iterable[BBox]) -> BBox:
    rects = list(rects)
    if not rects:
        return (0, 0, 0, 0)
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def bbox_width(b: BBox) -> float:
    return max(0.0, b[2] - b[0])


def bbox_height(b: BBox) -> float:
    return max(0.0, b[3] - b[1])


def expand_bbox(b: BBox, padding: float, page_w: float, page_h: float) -> BBox:
    return (
        max(0.0, b[0] - padding),
        max(0.0, b[1] - padding),
        min(page_w, b[2] + padding),
        min(page_h, b[3] + padding),
    )


def int_color_to_rgb(color: int) -> RGB:
    """
    PyMuPDF rawdict color is usually int: 0xRRGGBB.
    """
    r = ((color >> 16) & 255) / 255
    g = ((color >> 8) & 255) / 255
    b = (color & 255) / 255
    return (r, g, b)


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def detect_span_style(span: dict) -> Tuple[bool, bool]:
    font = str(span.get("font", "")).lower()
    flags = int(span.get("flags", 0) or 0)

    is_bold = any(k in font for k in ["bold", "black", "semibold", "demibold", "heavy"])
    is_italic = any(k in font for k in ["italic", "oblique"])

    # PyMuPDF flags vary by font. Keep font-name heuristic as main signal.
    if flags & 2:
        is_italic = True

    return is_bold, is_italic


def is_probably_page_number(text: str) -> bool:
    t = text.strip()
    return bool(re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", t))


def estimate_alignment(block_bbox: BBox, line_bboxes: List[BBox], page_w: float) -> str:
    if not line_bboxes:
        return "left"

    lefts = [b[0] for b in line_bboxes]
    rights = [b[2] for b in line_bboxes]

    left_var = max(lefts) - min(lefts)
    right_var = max(rights) - min(rights)

    # Very rough heuristic.
    if right_var < 4 and left_var > 12:
        return "right"

    block_center = (block_bbox[0] + block_bbox[2]) / 2
    page_center = page_w / 2
    widths = [bbox_width(b) for b in line_bboxes]
    avg_w = sum(widths) / max(1, len(widths))

    if abs(block_center - page_center) < page_w * 0.08 and avg_w < page_w * 0.65:
        return "center"

    return "left"


# ============================================================
# PDF parsing -> IR
# ============================================================

def parse_pdf_to_ir(source_pdf: str) -> DocumentIR:
    doc = fitz.open(source_pdf)
    ir = DocumentIR(source_pdf=source_pdf)

    for page_index, page in enumerate(doc):
        page_ir = PageIR(
            page_index=page_index,
            width=float(page.rect.width),
            height=float(page.rect.height),
            rotation=int(page.rotation or 0),
        )

        raw = page.get_text("rawdict")
        block_counter = 0

        for raw_block in raw.get("blocks", []):
            if raw_block.get("type") != 0:
                continue

            lines: List[TextLine] = []

            for raw_line in raw_block.get("lines", []):
                spans: List[TextSpan] = []

                for raw_span in raw_line.get("spans", []):
                    chars = raw_span.get("chars", [])
                    if chars:
                        text = "".join(ch.get("c", "") for ch in chars)
                    else:
                        text = raw_span.get("text", "")

                    text = text.replace("\x00", "")
                    if not text:
                        continue

                    bbox = tuple(float(v) for v in raw_span.get("bbox", (0, 0, 0, 0)))
                    is_bold, is_italic = detect_span_style(raw_span)

                    spans.append(TextSpan(
                        text=text,
                        bbox=bbox,  # type: ignore
                        font=str(raw_span.get("font", "")),
                        size=float(raw_span.get("size", 10.0)),
                        color=int(raw_span.get("color", 0) or 0),
                        flags=int(raw_span.get("flags", 0) or 0),
                        is_bold=is_bold,
                        is_italic=is_italic,
                    ))

                line_text = "".join(s.text for s in spans).strip()
                if not line_text:
                    continue

                line_bbox = tuple(float(v) for v in raw_line.get("bbox", rect_union(s.bbox for s in spans)))
                lines.append(TextLine(bbox=line_bbox, spans=spans))  # type: ignore

            if not lines:
                continue

            block_bbox = tuple(float(v) for v in raw_block.get("bbox", rect_union(l.bbox for l in lines)))
            original_text = normalize_text("\n".join(line.text for line in lines))
            if not original_text:
                continue

            block = TextBlock(
                id=f"p{page_index}_b{block_counter}",
                page_index=page_index,
                bbox=block_bbox,  # type: ignore
                lines=lines,
                order=block_counter,
                original_text=original_text,
            )
            block.align = estimate_alignment(block.bbox, [l.bbox for l in lines], page_ir.width)
            block.role = classify_block_role(block, page_ir)
            page_ir.blocks.append(block)
            block_counter += 1

        page_ir.blocks = sort_blocks_reading_order(page_ir.blocks)
        for i, b in enumerate(page_ir.blocks):
            b.order = i

        ir.pages.append(page_ir)

    doc.close()
    return ir


def classify_block_role(block: TextBlock, page: PageIR) -> str:
    text = block.original_text.strip()
    x0, y0, x1, y1 = block.bbox

    if is_probably_page_number(text):
        if y0 < page.height * 0.12 or y1 > page.height * 0.88:
            return "page_number"

    if y1 < page.height * 0.08:
        return "header"

    if y0 > page.height * 0.92:
        return "footer"

    if block.font_size >= 16 or (block.is_bold and block.font_size >= 13):
        return "title"

    # Very rough caption detection.
    if re.match(r"^\s*(fig\.|figure|table|chart|ảnh|hình|bảng)\s+\d+", text, flags=re.I):
        return "caption"

    return "body"


def sort_blocks_reading_order(blocks: List[TextBlock]) -> List[TextBlock]:
    """
    Basic reading order. Good enough for simple one-column/two-column PDFs.
    For complex docs, replace this with a real reading-order model.
    """
    if not blocks:
        return []

    # Try column grouping by x center.
    centers = [((b.bbox[0] + b.bbox[2]) / 2) for b in blocks]
    page_mid = median(centers)

    def col_id(b: TextBlock):
        cx = (b.bbox[0] + b.bbox[2]) / 2
        return 0 if cx < page_mid else 1

    # If almost everything spans full width, this still sorts top-to-bottom.
    return sorted(blocks, key=lambda b: (col_id(b), b.bbox[1], b.bbox[0]))


# ============================================================
# Paragraph reconstruction
# ============================================================

def should_merge_blocks(prev: TextBlock, curr: TextBlock, page: PageIR) -> bool:
    """
    Merge consecutive text blocks into a paragraph if they look like the same paragraph.
    This is conservative. Aggressive merging can hurt rendering.
    """
    if prev.role != "body" or curr.role != "body":
        return False

    prev_text = prev.original_text.strip()
    curr_text = curr.original_text.strip()

    if not prev_text or not curr_text:
        return False

    # Do not merge bullets too blindly.
    if re.match(r"^(\d+\.|[•\-*])\s+", curr_text):
        return False

    px0, py0, px1, py1 = prev.bbox
    cx0, cy0, cx1, cy1 = curr.bbox

    same_column = abs(px0 - cx0) < 12 and abs(px1 - cx1) < page.width * 0.18
    vertical_gap = cy0 - py1
    close_gap = 0 <= vertical_gap <= max(prev.font_size, curr.font_size) * 1.6
    similar_size = abs(prev.font_size - curr.font_size) <= 1.2

    # If prev ends with hard sentence punctuation, less likely same paragraph.
    prev_ends_sentence = bool(re.search(r"[.!?。！？:]$", prev_text))

    return same_column and close_gap and similar_size and not prev_ends_sentence


def build_paragraph_blocks(page: PageIR) -> List[TextBlock]:
    """
    Combine adjacent body blocks into paragraph blocks.
    Non-body blocks are passed through.
    """
    result: List[TextBlock] = []
    current: Optional[TextBlock] = None

    for block in page.blocks:
        if block.role in {"header", "footer", "page_number"}:
            # You may skip these from translation by not adding them.
            result.append(block)
            current = None
            continue

        if current is None:
            current = copy.deepcopy(block)
            result.append(current)
            continue

        if should_merge_blocks(current, block, page):
            current.lines.extend(copy.deepcopy(block.lines))
            current.bbox = rect_union([current.bbox, block.bbox])
            current.original_text = normalize_text(current.original_text + " " + block.original_text)
            current.id = current.id + "__" + block.id
        else:
            current = copy.deepcopy(block)
            result.append(current)

    # Rename IDs after merge for safety.
    for i, block in enumerate(result):
        block.id = f"p{page.page_index}_para{i}"

    return result


def rebuild_document_paragraphs(ir: DocumentIR) -> DocumentIR:
    new_ir = copy.deepcopy(ir)
    for page in new_ir.pages:
        page.blocks = build_paragraph_blocks(page)
        for i, block in enumerate(page.blocks):
            block.order = i
    return new_ir


# ============================================================
# Placeholder protection
# ============================================================

PLACEHOLDER_PATTERNS = [
    r"https?://[^\s)]+",
    r"[\w.\-]+@[\w.\-]+\.\w+",
    r"\b\d+(?:[.,]\d+)?%?\b",
    r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b",
    r"\([A-Za-z0-9,\s.;:-]+\)",  # simple citation-ish parentheses
]


def protect_placeholders(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace fragile tokens with placeholders before translation.
    Restore after translation.
    """
    mapping: Dict[str, str] = {}
    protected = text

    counter = 0

    for pattern in PLACEHOLDER_PATTERNS:
        regex = re.compile(pattern)

        def repl(match):
            nonlocal counter
            value = match.group(0)
            key = f"<PH_{counter}>"
            mapping[key] = value
            counter += 1
            return key

        protected = regex.sub(repl, protected)

    return protected, mapping


def restore_placeholders(text: str, mapping: Dict[str, str]) -> str:
    for key, value in mapping.items():
        text = text.replace(key, value)
    return text


# ============================================================
# Translation pipeline
# ============================================================

def iter_translatable_blocks(ir: DocumentIR, translate_headers_footers: bool = False) -> Iterable[TextBlock]:
    skip_roles = {"page_number"}
    if not translate_headers_footers:
        skip_roles.update({"header", "footer"})

    for page in ir.pages:
        for block in page.blocks:
            if block.role in skip_roles:
                continue
            if not block.original_text.strip():
                continue
            yield block


def translate_ir(
    ir: DocumentIR,
    translator: Translator,
    source_lang: str = "auto",
    target_lang: str = "vi",
    glossary: Optional[Dict[str, str]] = None,
    batch_size: int = 20,
    translate_headers_footers: bool = False,
) -> DocumentIR:
    new_ir = copy.deepcopy(ir)
    blocks = list(iter_translatable_blocks(new_ir, translate_headers_footers=translate_headers_footers))

    for start in range(0, len(blocks), batch_size):
        batch = blocks[start:start + batch_size]

        protected_payload = []
        placeholder_maps: Dict[str, Dict[str, str]] = {}

        for block in batch:
            protected_text, mapping = protect_placeholders(block.original_text)
            placeholder_maps[block.id] = mapping
            protected_payload.append({
                "id": block.id,
                "role": block.role,
                "text": protected_text,
            })

        translated = translator.translate_batch(
            items=protected_payload,
            source_lang=source_lang,
            target_lang=target_lang,
            glossary=glossary,
        )

        translated_by_id = {item["id"]: item.get("translated", "") for item in translated}

        for block in batch:
            raw_translation = translated_by_id.get(block.id, block.original_text)
            block.translated_text = restore_placeholders(raw_translation, placeholder_maps.get(block.id, {})).strip()

    return new_ir


# ============================================================
# Background sampling / cover
# ============================================================

def sample_background_color(page: fitz.Page, rect: fitz.Rect, zoom: float = 1.5) -> RGB:
    """
    Simple background sampler around a rect. This works best on flat backgrounds.
    For complex image backgrounds, you may need inpainting or blurred local patches.
    """
    page_rect = page.rect
    pad = 2.0
    sample_rect = fitz.Rect(
        max(page_rect.x0, rect.x0 - pad),
        max(page_rect.y0, rect.y0 - pad),
        min(page_rect.x1, rect.x1 + pad),
        min(page_rect.y1, rect.y1 + pad),
    )

    if sample_rect.is_empty or sample_rect.width <= 0 or sample_rect.height <= 0:
        return (1, 1, 1)

    matrix = fitz.Matrix(zoom, zoom)
    try:
        pix = page.get_pixmap(matrix=matrix, clip=sample_rect, alpha=False)
    except Exception:
        return (1, 1, 1)

    if pix.width <= 0 or pix.height <= 0:
        return (1, 1, 1)

    data = pix.samples
    n = pix.n

    # Sample border pixels to avoid sampling the old text in the center.
    pixels = []

    def add_pixel(x: int, y: int):
        idx = (y * pix.width + x) * n
        if idx + 2 < len(data):
            pixels.append((data[idx], data[idx + 1], data[idx + 2]))

    for x in range(pix.width):
        add_pixel(x, 0)
        add_pixel(x, pix.height - 1)

    for y in range(pix.height):
        add_pixel(0, y)
        add_pixel(pix.width - 1, y)

    if not pixels:
        return (1, 1, 1)

    r = int(median([p[0] for p in pixels])) / 255
    g = int(median([p[1] for p in pixels])) / 255
    b = int(median([p[2] for p in pixels])) / 255
    return (r, g, b)


def cover_original_text(
    page: fitz.Page,
    block: TextBlock,
    page_ir: PageIR,
    padding: float = 1.2,
    use_sampled_bg: bool = True,
    fallback_color: RGB = (1, 1, 1),
):
    rect = fitz.Rect(*expand_bbox(block.bbox, padding, page_ir.width, page_ir.height))
    fill = sample_background_color(page, rect) if use_sampled_bg else fallback_color
    page.draw_rect(rect, color=None, fill=fill, overlay=True)


# ============================================================
# Text measuring / wrapping / fitting
# ============================================================

class FontResolver:
    """
    Resolve PDF font style to actual Vietnamese-capable fonts.
    """
    def __init__(
        self,
        regular_font: Optional[str] = None,
        bold_font: Optional[str] = None,
        fallback_fontname: str = "helv",
    ):
        self.regular_font = regular_font
        self.bold_font = bold_font or regular_font
        self.fallback_fontname = fallback_fontname

        self._regular_fit_font = None
        self._bold_fit_font = None

        if self.regular_font and Path(self.regular_font).exists():
            self._regular_fit_font = fitz.Font(fontfile=self.regular_font)

        if self.bold_font and Path(self.bold_font).exists():
            self._bold_fit_font = fitz.Font(fontfile=self.bold_font)

    def fontfile_for(self, block: TextBlock) -> Optional[str]:
        if block.is_bold and self.bold_font and Path(self.bold_font).exists():
            return self.bold_font
        if self.regular_font and Path(self.regular_font).exists():
            return self.regular_font
        return None

    def fontname_for(self, block: TextBlock) -> str:
        # Built-in Helvetica fallback has limited Vietnamese support.
        # Use fontfile whenever possible.
        return self.fallback_fontname

    def fitz_font_for(self, block: TextBlock) -> fitz.Font:
        if block.is_bold and self._bold_fit_font:
            return self._bold_fit_font
        if self._regular_fit_font:
            return self._regular_fit_font
        return fitz.Font(self.fallback_fontname)


def split_long_word(word: str, font: fitz.Font, fontsize: float, max_width: float) -> List[str]:
    if font.text_length(word, fontsize=fontsize) <= max_width:
        return [word]

    parts = []
    current = ""

    for ch in word:
        candidate = current + ch
        if current and font.text_length(candidate, fontsize=fontsize) > max_width:
            parts.append(current)
            current = ch
        else:
            current = candidate

    if current:
        parts.append(current)

    return parts


def wrap_text_to_width(text: str, font: fitz.Font, fontsize: float, max_width: float) -> List[str]:
    paragraphs = text.splitlines() or [text]
    lines: List[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            lines.append("")
            continue

        words = para.split()
        current = ""

        for word in words:
            word_parts = split_long_word(word, font, fontsize, max_width)

            for part in word_parts:
                candidate = part if not current else current + " " + part
                if font.text_length(candidate, fontsize=fontsize) <= max_width:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = part

        if current:
            lines.append(current)

    return lines


@dataclass
class TextLayout:
    lines: List[str]
    fontsize: float
    line_height: float
    rect: fitz.Rect
    align: str
    fits: bool


def compute_text_layout(
    text: str,
    rect: fitz.Rect,
    font: fitz.Font,
    original_fontsize: float,
    align: str = "left",
    min_font_scale: float = 0.62,
) -> TextLayout:
    """
    Adaptive fitting:
    - Try near-original font size first.
    - Reduce line-height slightly before shrinking too much.
    - Shrink font by 0.25pt steps.
    """
    if rect.width <= 1 or rect.height <= 1:
        return TextLayout([], original_fontsize, 1.1, rect, align, False)

    max_fontsize = max(4.0, original_fontsize)
    min_fontsize = max(5.0, original_fontsize * min_font_scale)

    # Prefer small changes in line-height before significant font shrink.
    line_height_candidates = [1.15, 1.10, 1.05, 1.0, 0.96]

    size = max_fontsize
    while size >= min_fontsize:
        for lh in line_height_candidates:
            lines = wrap_text_to_width(text, font, size, rect.width)
            needed_h = len(lines) * size * lh

            if needed_h <= rect.height + 0.5:
                return TextLayout(lines, size, lh, rect, align, True)

        size -= 0.25

    # Fallback: smallest size, may overflow.
    lines = wrap_text_to_width(text, font, min_fontsize, rect.width)
    return TextLayout(lines, min_fontsize, 0.96, rect, align, False)


def draw_layout(
    page: fitz.Page,
    layout: TextLayout,
    font_resolver: FontResolver,
    block: TextBlock,
    color: RGB,
):
    if not layout.lines:
        return

    fontfile = font_resolver.fontfile_for(block)
    fontname = font_resolver.fontname_for(block)

    x0, y0, x1, y1 = layout.rect
    y = y0 + layout.fontsize

    measure_font = font_resolver.fitz_font_for(block)

    for line in layout.lines:
        if y > y1 + layout.fontsize:
            break

        if layout.align == "center":
            line_w = measure_font.text_length(line, fontsize=layout.fontsize)
            x = x0 + max(0, (layout.rect.width - line_w) / 2)
        elif layout.align == "right":
            line_w = measure_font.text_length(line, fontsize=layout.fontsize)
            x = x1 - line_w
        else:
            x = x0

        page.insert_text(
            point=fitz.Point(x, y),
            text=line,
            fontsize=layout.fontsize,
            fontname=fontname,
            fontfile=fontfile,
            color=color,
            overlay=True,
        )

        y += layout.fontsize * layout.line_height


# ============================================================
# Rendering
# ============================================================

def render_translated_pdf(
    source_pdf: str,
    translated_ir: DocumentIR,
    output_pdf: str,
    regular_font: Optional[str] = None,
    bold_font: Optional[str] = None,
    cover_text: bool = True,
    sampled_background: bool = True,
    translate_headers_footers: bool = False,
):
    pdf = fitz.open(source_pdf)
    font_resolver = FontResolver(regular_font=regular_font, bold_font=bold_font)

    for page_ir in translated_ir.pages:
        page = pdf[page_ir.page_index]

        for block in page_ir.blocks:
            if block.role == "page_number":
                continue

            if not translate_headers_footers and block.role in {"header", "footer"}:
                continue

            text = block.translated_text.strip() or block.original_text.strip()
            if not text:
                continue

            rect = fitz.Rect(*block.bbox)

            # Expand rect a little to reduce clipping.
            rect = fitz.Rect(*expand_bbox(tuple(rect), 0.5, page_ir.width, page_ir.height))

            if cover_text:
                cover_original_text(
                    page=page,
                    block=block,
                    page_ir=page_ir,
                    padding=1.2,
                    use_sampled_bg=sampled_background,
                )

            measure_font = font_resolver.fitz_font_for(block)
            layout = compute_text_layout(
                text=text,
                rect=rect,
                font=measure_font,
                original_fontsize=block.font_size,
                align=block.align,
            )

            color = int_color_to_rgb(block.color)

            draw_layout(
                page=page,
                layout=layout,
                font_resolver=font_resolver,
                block=block,
                color=color,
            )

    pdf.save(output_pdf, garbage=4, deflate=True)
    pdf.close()


# ============================================================
# QA helpers
# ============================================================

def render_pdf_pages(pdf_path: str, out_dir: str, dpi: int = 160, max_pages: Optional[int] = None):
    """
    Render pages to PNG for manual QA.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72

    for i, page in enumerate(doc):
        if max_pages is not None and i >= max_pages:
            break
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(str(out / f"page_{i+1:04d}.png"))

    doc.close()


def export_ir_json(ir: DocumentIR, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(ir), f, ensure_ascii=False, indent=2)


# ============================================================
# Main pipeline
# ============================================================

def translate_pdf(
    input_pdf: str,
    output_pdf: str,
    translator: Translator,
    source_lang: str = "auto",
    target_lang: str = "vi",
    regular_font: Optional[str] = None,
    bold_font: Optional[str] = None,
    glossary: Optional[Dict[str, str]] = None,
    export_ir_path: Optional[str] = None,
    render_preview_dir: Optional[str] = None,
    batch_size: int = 20,
    cover_text: bool = True,
    sampled_background: bool = True,
    translate_headers_footers: bool = False,
):
    print(f"[1/5] Parse PDF -> IR: {input_pdf}")
    ir = parse_pdf_to_ir(input_pdf)

    print("[2/5] Rebuild paragraph blocks")
    ir = rebuild_document_paragraphs(ir)

    if export_ir_path:
        print(f"      Export IR before translation: {export_ir_path}")
        export_ir_json(ir, export_ir_path)

    print("[3/5] Translate IR")
    translated_ir = translate_ir(
        ir=ir,
        translator=translator,
        source_lang=source_lang,
        target_lang=target_lang,
        glossary=glossary,
        batch_size=batch_size,
        translate_headers_footers=translate_headers_footers,
    )

    if export_ir_path:
        translated_path = str(Path(export_ir_path).with_name(Path(export_ir_path).stem + "_translated.json"))
        print(f"      Export translated IR: {translated_path}")
        export_ir_json(translated_ir, translated_path)

    print("[4/5] Render translated PDF")
    render_translated_pdf(
        source_pdf=input_pdf,
        translated_ir=translated_ir,
        output_pdf=output_pdf,
        regular_font=regular_font,
        bold_font=bold_font,
        cover_text=cover_text,
        sampled_background=sampled_background,
        translate_headers_footers=translate_headers_footers,
    )

    if render_preview_dir:
        print(f"[5/5] Render preview PNGs: {render_preview_dir}")
        render_pdf_pages(output_pdf, render_preview_dir)

    print(f"Done: {output_pdf}")


def load_glossary(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BabelDOC-like PDF translator base file using PyMuPDF overlay rendering."
    )

    parser.add_argument("input_pdf", help="Source PDF path")
    parser.add_argument("output_pdf", help="Output translated PDF path")

    parser.add_argument("--source", default="auto", help="Source language, default: auto")
    parser.add_argument("--target", default="vi", help="Target language, default: vi")

    parser.add_argument("--font", default=None, help="Vietnamese-capable regular TTF/OTF font path")
    parser.add_argument("--font-bold", default=None, help="Vietnamese-capable bold TTF/OTF font path")

    parser.add_argument("--translation-map", default=None, help="JSON map {block_id: translated_text} for testing")
    parser.add_argument("--glossary", default=None, help="JSON glossary map {source_term: target_term}")

    parser.add_argument("--export-ir", default=None, help="Export IR JSON path")
    parser.add_argument("--preview-dir", default=None, help="Render output PDF pages to PNG for QA")

    parser.add_argument("--batch-size", type=int, default=20)

    parser.add_argument("--no-cover", action="store_true", help="Do not cover original text")
    parser.add_argument("--no-sampled-bg", action="store_true", help="Use white cover instead of sampled background")
    parser.add_argument("--translate-headers-footers", action="store_true")

    return parser


def main(argv: Optional[List[str]] = None):
    args = build_arg_parser().parse_args(argv)

    if not Path(args.input_pdf).exists():
        raise FileNotFoundError(args.input_pdf)

    if args.translation_map:
        translator: Translator = JsonMapTranslator(args.translation_map)
    else:
        translator = DummyTranslator()
        print(
            "WARNING: Using DummyTranslator. The PDF layout will be processed, "
            "but text will not actually be translated.\n"
            "Pass --translation-map or implement a real Translator class.",
            file=sys.stderr,
        )

    glossary = load_glossary(args.glossary)

    translate_pdf(
        input_pdf=args.input_pdf,
        output_pdf=args.output_pdf,
        translator=translator,
        source_lang=args.source,
        target_lang=args.target,
        regular_font=args.font,
        bold_font=args.font_bold,
        glossary=glossary,
        export_ir_path=args.export_ir,
        render_preview_dir=args.preview_dir,
        batch_size=args.batch_size,
        cover_text=not args.no_cover,
        sampled_background=not args.no_sampled_bg,
        translate_headers_footers=args.translate_headers_footers,
    )


if __name__ == "__main__":
    main()
