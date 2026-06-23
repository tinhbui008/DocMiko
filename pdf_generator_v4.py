#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_generator_v4_babeldoc_like.py

Improved BabelDOC-like PDF translation base.

Main fixes compared with v1:
1. Do NOT rewrite text when the translation is unchanged.
   This keeps the original PDF pixel-close when using DummyTranslator.
2. Skip logo / decorative / hidden text blocks such as "POWERED BY".
3. Preserve the original PDF as the base; only changed translatable blocks are covered and redrawn.
4. Better alignment detection: left-aligned headings are no longer forced to center.
5. Better font fallback on Windows: Arial / Arial Bold / Arial Narrow Bold.
6. Optional translation-map for testing without an API.

Install:
    pip install pymupdf

Usage:
    python pdf_generator_v4_babeldoc_like.py test_vietnamese.pdf output_v2.pdf

With prepared translation map:
    python pdf_generator_v4_babeldoc_like.py test_vietnamese.pdf output_v2.pdf --translation-map translated.json

With custom Vietnamese fonts:
    python pdf_generator_v4_babeldoc_like.py test_vietnamese.pdf output_v2.pdf --font fonts/NotoSans-Regular.ttf --font-bold fonts/NotoSans-Bold.ttf --font-title fonts/NotoSansCondensed-Bold.ttf
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import fitz  # PyMuPDF

BBox = Tuple[float, float, float, float]
RGB = Tuple[float, float, float]


# ============================================================
# Data schema / IR
# ============================================================

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
        return "".join(s.text for s in self.spans).strip()

    @property
    def font_size(self) -> float:
        return median([s.size for s in self.spans if s.size > 0]) or 10.0

    @property
    def color(self) -> int:
        values = [s.color for s in self.spans]
        return most_common(values) if values else 0

    @property
    def is_bold(self) -> bool:
        return any(s.is_bold for s in self.spans)

    @property
    def main_font(self) -> str:
        values = [s.font for s in self.spans if s.font]
        return most_common(values) or "helv"


@dataclass
class TextBlock:
    id: str
    page_index: int
    bbox: BBox
    lines: List[TextLine] = field(default_factory=list)
    role: str = "body"  # title/body/caption/cta/logo/header/footer/page_number/hidden
    order: int = 0
    align: str = "left"
    original_text: str = ""
    translated_text: str = ""

    @property
    def font_size(self) -> float:
        return median([ln.font_size for ln in self.lines]) or 10.0

    @property
    def color(self) -> int:
        values = [ln.color for ln in self.lines]
        return most_common(values) if values else 0

    @property
    def is_bold(self) -> bool:
        return any(ln.is_bold for ln in self.lines)

    @property
    def main_font(self) -> str:
        values = [ln.main_font for ln in self.lines if ln.main_font]
        return most_common(values) or "helv"


@dataclass
class PageIR:
    page_index: int
    width: float
    height: float
    rotation: int
    image_rects: List[BBox] = field(default_factory=list)
    blocks: List[TextBlock] = field(default_factory=list)


@dataclass
class DocumentIR:
    source_pdf: str
    pages: List[PageIR] = field(default_factory=list)


# ============================================================
# Translator interface
# ============================================================

class Translator(Protocol):
    def translate_batch(
        self,
        items: List[Dict[str, str]],
        source_lang: str,
        target_lang: str,
        glossary: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        ...


class DummyTranslator:
    """For layout testing only. It returns the original text unchanged."""

    def translate_batch(
        self,
        items: List[Dict[str, str]],
        source_lang: str,
        target_lang: str,
        glossary: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        return [{"id": item["id"], "translated": item["text"]} for item in items]


class JsonMapTranslator:
    """
    Use a JSON file for testing real changed text:
        {
          "p0_para0": "Vietnamese translation...",
          "p0_para1": "..."
        }
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
        result = []
        for item in items:
            result.append({
                "id": item["id"],
                "translated": self.translation_map.get(item["id"], item["text"]),
            })
        return result


# ============================================================
# General utilities
# ============================================================

def median(values: Sequence[float]) -> float:
    values = sorted([v for v in values if v is not None])
    if not values:
        return 0.0
    n = len(values)
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2


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
        return (0.0, 0.0, 0.0, 0.0)
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


def bbox_center(b: BBox) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def bbox_contains(outer: BBox, inner: BBox, tolerance: float = 0.0) -> bool:
    return (
        inner[0] >= outer[0] - tolerance and
        inner[1] >= outer[1] - tolerance and
        inner[2] <= outer[2] + tolerance and
        inner[3] <= outer[3] + tolerance
    )


def expand_bbox(b: BBox, padding: float, page_w: float, page_h: float) -> BBox:
    return (
        max(0.0, b[0] - padding),
        max(0.0, b[1] - padding),
        min(page_w, b[2] + padding),
        min(page_h, b[3] + padding),
    )


def int_color_to_rgb(color: int) -> RGB:
    r = ((color >> 16) & 255) / 255
    g = ((color >> 8) & 255) / 255
    b = (color & 255) / 255
    return (r, g, b)


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def comparable_text(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_is_same(a: str, b: str) -> bool:
    return comparable_text(a) == comparable_text(b)


def detect_span_style(span: dict) -> Tuple[bool, bool]:
    font = str(span.get("font", "")).lower()
    flags = int(span.get("flags", 0) or 0)
    is_bold = any(k in font for k in ["bold", "black", "semibold", "demibold", "heavy"])
    is_italic = any(k in font for k in ["italic", "oblique"])
    if flags & 2:
        is_italic = True
    return is_bold, is_italic


def compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def is_probably_page_number(text: str) -> bool:
    return bool(re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", text.strip()))


def is_logo_or_decorative_text(text: str) -> bool:
    c = compact_text(text)
    if c in {"poweredby", "poweredbyfloridacommerce", "floridacommerce"}:
        return True
    if "poweredby" in c:
        return True
    # Business logos often extract as short all-caps fragments. Keep conservative.
    if len(c) <= 4 and text.strip().isupper():
        return True
    return False


def rect_inside_any_large_image(block_bbox: BBox, image_rects: List[BBox], page_w: float, page_h: float) -> bool:
    page_area = page_w * page_h
    for img in image_rects:
        if bbox_width(img) * bbox_height(img) < page_area * 0.08:
            continue
        if bbox_contains(img, block_bbox, tolerance=2.0):
            return True
    return False


# ============================================================
# PDF parsing
# ============================================================

def extract_image_rects(page: fitz.Page) -> List[BBox]:
    rects: List[BBox] = []
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        try:
            for r in page.get_image_rects(xref):
                rects.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
        except Exception:
            continue
    return rects


def parse_pdf_to_ir(source_pdf: str) -> DocumentIR:
    pdf = fitz.open(source_pdf)
    ir = DocumentIR(source_pdf=source_pdf)

    for page_index, page in enumerate(pdf):
        page_ir = PageIR(
            page_index=page_index,
            width=float(page.rect.width),
            height=float(page.rect.height),
            rotation=int(page.rotation or 0),
            image_rects=extract_image_rects(page),
        )

        raw = page.get_text("rawdict")
        block_idx = 0

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

                    if not text or not text.strip():
                        continue

                    is_bold, is_italic = detect_span_style(raw_span)
                    bbox = tuple(float(v) for v in raw_span.get("bbox", (0, 0, 0, 0)))

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

                if not spans:
                    continue

                line_bbox = tuple(float(v) for v in raw_line.get("bbox", rect_union(s.bbox for s in spans)))
                line = TextLine(bbox=line_bbox, spans=spans)  # type: ignore
                if line.text:
                    lines.append(line)

            if not lines:
                continue

            block_bbox = tuple(float(v) for v in raw_block.get("bbox", rect_union(l.bbox for l in lines)))
            original_text = normalize_text("\n".join(line.text for line in lines))
            if not original_text:
                continue

            block = TextBlock(
                id=f"p{page_index}_b{block_idx}",
                page_index=page_index,
                bbox=block_bbox,  # type: ignore
                lines=lines,
                order=block_idx,
                original_text=original_text,
            )
            block.align = estimate_alignment(block, page_ir)
            block.role = classify_block_role(block, page_ir)
            page_ir.blocks.append(block)
            block_idx += 1

        page_ir.blocks = sort_blocks_reading_order(page_ir.blocks)
        for i, block in enumerate(page_ir.blocks):
            block.order = i
        ir.pages.append(page_ir)

    pdf.close()
    return ir


def estimate_alignment(block: TextBlock, page: PageIR) -> str:
    line_boxes = [ln.bbox for ln in block.lines]
    if not line_boxes:
        return "left"

    lefts = [b[0] for b in line_boxes]
    rights = [b[2] for b in line_boxes]
    centers = [(b[0] + b[2]) / 2 for b in line_boxes]

    left_var = max(lefts) - min(lefts)
    right_var = max(rights) - min(rights)
    center_var = max(centers) - min(centers)

    # Strong left-alignment signal: every line starts at nearly same x.
    # This fixes left headings that v1 accidentally centered.
    if left_var <= 6:
        return "left"

    # Right-alignment signal.
    if right_var <= 6 and left_var > 10:
        return "right"

    page_center = page.width / 2
    avg_center = sum(centers) / len(centers)

    if abs(avg_center - page_center) < page.width * 0.06 and center_var <= 10:
        return "center"

    return "left"


def classify_block_role(block: TextBlock, page: PageIR) -> str:
    text = block.original_text.strip()
    c = compact_text(text)
    x0, y0, x1, y1 = block.bbox

    if is_logo_or_decorative_text(text):
        return "logo"

    if is_probably_page_number(text):
        if y0 < page.height * 0.12 or y1 > page.height * 0.88:
            return "page_number"

    # Hidden text under a large photo/image is common in PDFs with duplicate objects.
    # Do not touch it unless it is obviously normal text.
    if rect_inside_any_large_image(block.bbox, page.image_rects, page.width, page.height):
        if block.font_size <= 9 or is_logo_or_decorative_text(text):
            return "hidden"

    if y1 < page.height * 0.055:
        return "header"
    if y0 > page.height * 0.94:
        return "footer"

    if block.font_size >= 18 or (block.is_bold and block.font_size >= 15):
        return "title"

    if re.match(r"^\s*(fig\.|figure|table|chart|ảnh|hình|bảng)\s+\d+", text, flags=re.I):
        return "caption"

    # CTA / QR side text, still translatable.
    if re.search(r"learn more|apply|email|contact", text, flags=re.I):
        return "cta"

    return "body"


def sort_blocks_reading_order(blocks: List[TextBlock]) -> List[TextBlock]:
    # Simple and stable: top-to-bottom, left-to-right.
    # For complex two-column docs, replace with DocLayout-YOLO reading order.
    return sorted(blocks, key=lambda b: (round(b.bbox[1] / 8), b.bbox[0]))


# ============================================================
# Paragraph reconstruction
# ============================================================

def should_merge_blocks(prev: TextBlock, curr: TextBlock, page: PageIR) -> bool:
    if prev.role != "body" or curr.role != "body":
        return False

    prev_text = prev.original_text.strip()
    curr_text = curr.original_text.strip()
    if not prev_text or not curr_text:
        return False

    if re.match(r"^(\d+\.|[•\-*])\s+", curr_text):
        return False

    px0, py0, px1, py1 = prev.bbox
    cx0, cy0, cx1, cy1 = curr.bbox

    same_left = abs(px0 - cx0) < 10
    similar_width = abs(bbox_width(prev.bbox) - bbox_width(curr.bbox)) < page.width * 0.18
    vertical_gap = cy0 - py1
    close_gap = 0 <= vertical_gap <= max(prev.font_size, curr.font_size) * 1.35
    similar_size = abs(prev.font_size - curr.font_size) <= 1.0
    prev_ends_sentence = bool(re.search(r"[.!?。！？:]$", prev_text))

    return same_left and similar_width and close_gap and similar_size and not prev_ends_sentence


def build_paragraph_blocks(page: PageIR) -> List[TextBlock]:
    result: List[TextBlock] = []
    current: Optional[TextBlock] = None

    for block in page.blocks:
        if block.role != "body":
            result.append(copy.deepcopy(block))
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
        else:
            current = copy.deepcopy(block)
            result.append(current)

    for i, block in enumerate(result):
        block.id = f"p{page.page_index}_para{i}"
        block.order = i

    return result


def rebuild_document_paragraphs(ir: DocumentIR) -> DocumentIR:
    new_ir = copy.deepcopy(ir)
    for page in new_ir.pages:
        page.blocks = build_paragraph_blocks(page)
    return new_ir


# ============================================================
# Placeholder protection
# ============================================================

PLACEHOLDER_PATTERNS = [
    r"https?://[^\s)]+",
    r"[\w.\-]+@[\w.\-]+\.\w+",
    r"\b\d+(?:[.,]\d+)?%?\b",
    r"\$\s?\d+(?:[.,]\d+)?",
    r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b",
]


def protect_placeholders(text: str) -> Tuple[str, Dict[str, str]]:
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
# Translation
# ============================================================

def is_translatable_block(block: TextBlock, translate_headers_footers: bool = False) -> bool:
    if block.role in {"logo", "hidden", "page_number"}:
        return False
    if block.role in {"header", "footer"} and not translate_headers_footers:
        return False
    if not block.original_text.strip():
        return False
    return True


def iter_translatable_blocks(ir: DocumentIR, translate_headers_footers: bool = False) -> Iterable[TextBlock]:
    for page in ir.pages:
        for block in page.blocks:
            if is_translatable_block(block, translate_headers_footers):
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
    blocks = list(iter_translatable_blocks(new_ir, translate_headers_footers))

    for start in range(0, len(blocks), batch_size):
        batch = blocks[start:start + batch_size]
        payload = []
        placeholder_maps: Dict[str, Dict[str, str]] = {}

        for block in batch:
            protected, mapping = protect_placeholders(block.original_text)
            placeholder_maps[block.id] = mapping
            payload.append({"id": block.id, "role": block.role, "text": protected})

        translated = translator.translate_batch(payload, source_lang, target_lang, glossary)
        translated_by_id = {item["id"]: item.get("translated", "") for item in translated}

        for block in batch:
            raw_translation = translated_by_id.get(block.id, block.original_text)
            block.translated_text = restore_placeholders(raw_translation, placeholder_maps.get(block.id, {})).strip()

    return new_ir


# ============================================================
# Font resolving
# ============================================================

class FontResolver:
    def __init__(
        self,
        regular_font: Optional[str] = None,
        bold_font: Optional[str] = None,
        title_font: Optional[str] = None,
        fallback_fontname: str = "helv",
    ):
        # IMPORTANT v3 fix:
        # PyMuPDF ignores `fontfile=` if `fontname` is a built-in name like "helv".
        # That was why Vietnamese glyphs became "Ch··ng", "Kho·n", etc.
        # We now use custom font names whenever a real font file is available.
        self.regular_font = self._first_existing([regular_font, *self._system_font_candidates("regular")])
        self.bold_font = self._first_existing([bold_font, *self._system_font_candidates("bold"), self.regular_font])
        self.title_font = self._first_existing([title_font, *self._system_font_candidates("title"), self.bold_font, self.regular_font])
        self.fallback_fontname = fallback_fontname
        self._font_cache: Dict[str, fitz.Font] = {}

    @staticmethod
    def _font_supports_vietnamese(path: str) -> bool:
        # Some fonts exist but do not contain Vietnamese glyphs.
        # Example: LiberationSansNarrow-Bold renders "Chương" as "Ch□□ng".
        # Avoid selecting those fonts for translated Vietnamese text.
        probe = "ăâêôơưđĂÂÊÔƠƯĐáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        try:
            font = fitz.Font(fontfile=path)
            return all(font.has_glyph(ord(ch)) for ch in probe)
        except Exception:
            return False

    @classmethod
    def _first_existing(cls, paths: List[Optional[str]]) -> Optional[str]:
        fallback_existing = None
        for p in paths:
            if not p:
                continue
            path = Path(p)
            if not path.exists():
                continue
            if fallback_existing is None:
                fallback_existing = str(path)
            if cls._font_supports_vietnamese(str(path)):
                return str(path)
        # If no Vietnamese-capable font is found, return a real font anyway;
        # caller will still work, but glyphs may be broken.
        return fallback_existing

    @staticmethod
    def _system_font_candidates(kind: str) -> List[str]:
        """
        Cross-platform Vietnamese-capable font candidates.
        Windows users usually hit Arial / Arial Narrow here.
        Linux/macOS candidates make the script safer in server/docker environments.
        """
        win = Path("C:/Windows/Fonts")

        linux = Path("/usr/share/fonts")
        candidates: List[str] = []

        if kind == "title":
            candidates += [
                str(win / "arialnb.ttf"),
                str(win / "ARIALNB.TTF"),
                str(win / "arialbd.ttf"),
                str(win / "ARIALBD.TTF"),
                str(linux / "truetype/dejavu/DejaVuSansCondensed-Bold.ttf"),
                str(linux / "truetype/noto/NotoSans-CondensedBold.ttf"),
                str(linux / "truetype/noto/NotoSans-CondensedSemiBold.ttf"),
                str(linux / "truetype/liberation/LiberationSansNarrow-Bold.ttf"),
            ]
        elif kind == "bold":
            candidates += [
                str(win / "arialbd.ttf"),
                str(win / "ARIALBD.TTF"),
                str(win / "calibrib.ttf"),
                str(win / "CALIBRIB.TTF"),
                str(linux / "truetype/dejavu/DejaVuSans-Bold.ttf"),
                str(linux / "truetype/liberation/LiberationSans-Bold.ttf"),
                str(linux / "truetype/noto/NotoSans-Bold.ttf"),
            ]
        else:
            candidates += [
                str(win / "arial.ttf"),
                str(win / "ARIAL.TTF"),
                str(win / "calibri.ttf"),
                str(win / "CALIBRI.TTF"),
                str(linux / "truetype/dejavu/DejaVuSans.ttf"),
                str(linux / "truetype/liberation/LiberationSans-Regular.ttf"),
                str(linux / "truetype/noto/NotoSans-Regular.ttf"),
            ]

        # macOS common paths
        candidates += [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ]
        return candidates

    def fontfile_for(self, block: TextBlock) -> Optional[str]:
        if block.role == "title" and self.title_font:
            return self.title_font
        if block.is_bold and self.bold_font:
            return self.bold_font
        if self.regular_font:
            return self.regular_font
        return None

    def fontname_for(self, block: TextBlock) -> str:
        # DO NOT return "helv" when using fontfile. Built-in names ignore fontfile
        # and break Vietnamese diacritics. Use stable custom resource names.
        if block.role == "title" and self.title_font:
            return "FTitleVN"
        if block.is_bold and self.bold_font:
            return "FBoldVN"
        if self.regular_font:
            return "FRegularVN"
        return self.fallback_fontname

    def fitz_font_for(self, block: TextBlock) -> fitz.Font:
        fontfile = self.fontfile_for(block)
        key = self.fontname_for(block) + "|" + (fontfile or self.fallback_fontname)
        if key in self._font_cache:
            return self._font_cache[key]
        if fontfile:
            font = fitz.Font(fontfile=fontfile)
        else:
            font = fitz.Font(self.fallback_fontname)
        self._font_cache[key] = font
        return font


# ============================================================
# Background cover
# ============================================================

def sample_background_color(page: fitz.Page, rect: fitz.Rect, zoom: float = 1.5) -> RGB:
    page_rect = page.rect
    sample_rect = fitz.Rect(
        max(page_rect.x0, rect.x0 - 2),
        max(page_rect.y0, rect.y0 - 2),
        min(page_rect.x1, rect.x1 + 2),
        min(page_rect.y1, rect.y1 + 2),
    )

    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=sample_rect, alpha=False)
    except Exception:
        return (1, 1, 1)

    if pix.width <= 0 or pix.height <= 0:
        return (1, 1, 1)

    data = pix.samples
    n = pix.n
    pixels = []

    def add_pixel(x: int, y: int):
        idx = (y * pix.width + x) * n
        if idx + 2 < len(data):
            pixels.append((data[idx], data[idx + 1], data[idx + 2]))

    # Border pixels: avoid old text in the center of the block.
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


def cover_text_block(page: fitz.Page, block: TextBlock, page_ir: PageIR, sampled_background: bool = True):
    # Cover per block, not full page. It preserves images/vectors.
    pad = 1.0 if block.role != "title" else 1.5
    rect = fitz.Rect(*expand_bbox(block.bbox, pad, page_ir.width, page_ir.height))
    fill = sample_background_color(page, rect) if sampled_background else (1, 1, 1)
    page.draw_rect(rect, color=None, fill=fill, overlay=True)


# ============================================================
# Text layout / drawing
# ============================================================

@dataclass
class LayoutResult:
    lines: List[str]
    fontsize: float
    line_height: float
    rect: fitz.Rect
    align: str
    fits: bool


def split_long_word(word: str, font: fitz.Font, fontsize: float, max_width: float) -> List[str]:
    if font.text_length(word, fontsize=fontsize) <= max_width:
        return [word]

    parts = []
    cur = ""
    for ch in word:
        test = cur + ch
        if cur and font.text_length(test, fontsize=fontsize) > max_width:
            parts.append(cur)
            cur = ch
        else:
            cur = test
    if cur:
        parts.append(cur)
    return parts


def wrap_text_to_width(text: str, font: fitz.Font, fontsize: float, max_width: float) -> List[str]:
    output: List[str] = []
    raw_lines = text.splitlines() or [text]

    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            output.append("")
            continue

        current = ""
        for word in raw.split():
            for part in split_long_word(word, font, fontsize, max_width):
                test = part if not current else current + " " + part
                if font.text_length(test, fontsize=fontsize) <= max_width:
                    current = test
                else:
                    if current:
                        output.append(current)
                    current = part
        if current:
            output.append(current)

    return output


def compute_layout(
    text: str,
    block: TextBlock,
    rect: fitz.Rect,
    font: fitz.Font,
) -> LayoutResult:
    if rect.width <= 1 or rect.height <= 1:
        return LayoutResult([], block.font_size, 1.1, rect, block.align, False)

    original_size = max(5.0, block.font_size)

    if block.role == "title":
        min_scale = 0.55
        line_heights = [1.02, 0.98, 0.94]
    else:
        min_scale = 0.62
        line_heights = [1.15, 1.10, 1.05, 1.0, 0.96]

    size = original_size
    min_size = max(5.0, original_size * min_scale)

    while size >= min_size:
        for lh in line_heights:
            lines = wrap_text_to_width(text, font, size, rect.width)
            needed_h = len(lines) * size * lh
            if needed_h <= rect.height + 0.8:
                return LayoutResult(lines, size, lh, rect, block.align, True)
        size -= 0.25

    size = min_size
    lines = wrap_text_to_width(text, font, size, rect.width)
    return LayoutResult(lines, size, line_heights[-1], rect, block.align, False)


def draw_layout(page: fitz.Page, layout: LayoutResult, block: TextBlock, resolver: FontResolver, color: RGB):
    if not layout.lines:
        return

    fontfile = resolver.fontfile_for(block)
    fontname = resolver.fontname_for(block)
    measure_font = resolver.fitz_font_for(block)

    x0, y0, x1, y1 = layout.rect
    y = y0 + layout.fontsize

    for line in layout.lines:
        if y > y1 + layout.fontsize:
            break

        if layout.align == "center":
            w = measure_font.text_length(line, fontsize=layout.fontsize)
            x = x0 + max(0, (layout.rect.width - w) / 2)
        elif layout.align == "right":
            w = measure_font.text_length(line, fontsize=layout.fontsize)
            x = x1 - w
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
# Render PDF
# ============================================================

def should_render_changed_translation(block: TextBlock, force_render: bool = False) -> bool:
    if not block.translated_text.strip():
        return False
    if force_render:
        return True
    # Most important v2 fix: do not rewrite identical text.
    # Otherwise the PDF will look different even when no translation has happened.
    return not text_is_same(block.original_text, block.translated_text)


def render_translated_pdf(
    input_pdf: str,
    translated_ir: DocumentIR,
    output_pdf: str,
    regular_font: Optional[str] = None,
    bold_font: Optional[str] = None,
    title_font: Optional[str] = None,
    cover_text: bool = True,
    sampled_background: bool = True,
    translate_headers_footers: bool = False,
    force_render: bool = False,
):
    pdf = fitz.open(input_pdf)
    resolver = FontResolver(regular_font=regular_font, bold_font=bold_font, title_font=title_font)

    changed_count = 0
    skipped_same = 0
    skipped_role = 0

    for page_ir in translated_ir.pages:
        page = pdf[page_ir.page_index]

        for block in page_ir.blocks:
            if not is_translatable_block(block, translate_headers_footers):
                skipped_role += 1
                continue

            if not should_render_changed_translation(block, force_render=force_render):
                skipped_same += 1
                continue

            text = block.translated_text.strip()
            rect = fitz.Rect(*expand_bbox(block.bbox, 0.5, page_ir.width, page_ir.height))

            if cover_text:
                cover_text_block(page, block, page_ir, sampled_background=sampled_background)

            font = resolver.fitz_font_for(block)
            layout = compute_layout(text, block, rect, font)
            color = int_color_to_rgb(block.color)
            draw_layout(page, layout, block, resolver, color)
            changed_count += 1

    pdf.save(output_pdf, garbage=4, deflate=True)
    pdf.close()

    print(f"Render summary: changed={changed_count}, skipped_same={skipped_same}, skipped_role={skipped_role}")


# ============================================================
# QA / export
# ============================================================

def export_ir_json(ir: DocumentIR, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(ir), f, ensure_ascii=False, indent=2)


def render_pdf_pages(pdf_path: str, out_dir: str, dpi: int = 160, max_pages: Optional[int] = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open(pdf_path)
    zoom = dpi / 72
    for i, page in enumerate(pdf):
        if max_pages is not None and i >= max_pages:
            break
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(str(out / f"page_{i + 1:04d}.png"))
    pdf.close()


def load_json_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    title_font: Optional[str] = None,
    glossary: Optional[Dict[str, str]] = None,
    batch_size: int = 20,
    export_ir_path: Optional[str] = None,
    preview_dir: Optional[str] = None,
    cover_text: bool = True,
    sampled_background: bool = True,
    translate_headers_footers: bool = False,
    force_render: bool = False,
):
    print(f"[1/5] Parse PDF -> IR: {input_pdf}")
    ir = parse_pdf_to_ir(input_pdf)

    print("[2/5] Rebuild paragraph blocks")
    ir = rebuild_document_paragraphs(ir)

    if export_ir_path:
        print(f"      Export IR: {export_ir_path}")
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
        input_pdf=input_pdf,
        translated_ir=translated_ir,
        output_pdf=output_pdf,
        regular_font=regular_font,
        bold_font=bold_font,
        title_font=title_font,
        cover_text=cover_text,
        sampled_background=sampled_background,
        translate_headers_footers=translate_headers_footers,
        force_render=force_render,
    )

    if preview_dir:
        print(f"[5/5] Render preview PNGs: {preview_dir}")
        render_pdf_pages(output_pdf, preview_dir)

    print(f"Done: {output_pdf}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Improved BabelDOC-like PDF translation base using PyMuPDF overlay.")
    p.add_argument("input_pdf", help="Source PDF path")
    p.add_argument("output_pdf", help="Output PDF path")
    p.add_argument("--source", default="auto")
    p.add_argument("--target", default="vi")
    p.add_argument("--font", default=None, help="Regular TTF/OTF font path")
    p.add_argument("--font-bold", default=None, help="Bold TTF/OTF font path")
    p.add_argument("--font-title", default=None, help="Condensed/bold title font path")
    p.add_argument("--translation-map", default=None, help="JSON map: {block_id: translated_text}")
    p.add_argument("--glossary", default=None, help="JSON glossary: {source_term: target_term}")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--export-ir", default=None)
    p.add_argument("--preview-dir", default=None)
    p.add_argument("--no-cover", action="store_true")
    p.add_argument("--no-sampled-bg", action="store_true")
    p.add_argument("--translate-headers-footers", action="store_true")
    p.add_argument("--force-render", action="store_true", help="Rewrite even unchanged text. Usually only for debugging.")
    return p


def main(argv: Optional[List[str]] = None):
    args = build_arg_parser().parse_args(argv)

    if not Path(args.input_pdf).exists():
        raise FileNotFoundError(args.input_pdf)

    if args.translation_map:
        translator: Translator = JsonMapTranslator(args.translation_map)
    else:
        translator = DummyTranslator()
        print(
            "WARNING: Using DummyTranslator. No real translation will happen.\n"
            "In v2, unchanged text will NOT be rewritten, so output should stay visually close to source.\n"
            "Use --translation-map or implement a real Translator class.",
            file=sys.stderr,
        )

    glossary = load_json_map(args.glossary)

    translate_pdf(
        input_pdf=args.input_pdf,
        output_pdf=args.output_pdf,
        translator=translator,
        source_lang=args.source,
        target_lang=args.target,
        regular_font=args.font,
        bold_font=args.font_bold,
        title_font=args.font_title,
        glossary=glossary,
        batch_size=args.batch_size,
        export_ir_path=args.export_ir,
        preview_dir=args.preview_dir,
        cover_text=not args.no_cover,
        sampled_background=not args.no_sampled_bg,
        translate_headers_footers=args.translate_headers_footers,
        force_render=args.force_render,
    )


if __name__ == "__main__":
    main()
