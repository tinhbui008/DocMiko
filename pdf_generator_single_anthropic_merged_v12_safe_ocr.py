#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_generator_single_anthropic_merged_v12_safe_ocr.py

Single-file BabelDOC-like PDF translator.

Includes:
- V8 PDF parser / renderer core
- V9 fit-aware rendering + compact translation memory
- Anthropic Claude native Messages API support
- OpenAI-compatible fallback support

.env for Anthropic:
    LLM_PROVIDER=anthropic
    LLM_API_KEY=sk-ant-...
    LLM_BASE_URL=https://api.anthropic.com
    LLM_MODEL=<your Claude model id>
    LLM_MAX_TOKENS=4096
    LLM_TEMPERATURE=0

Run:
    python pdf_generator_single_anthropic_merged_v12_safe_ocr.py input.pdf output.pdf `
      --font "fonts/NotoSans-Regular.ttf" `
      --font-bold "fonts/NotoSans-Bold.ttf" `
      --font-title "fonts/NotoSansCondensed-Bold.ttf" `
      --export-ir "ir_single.json"
"""

# VERSION_MARKER = 'merged_v12_safe_ocr_filtering'
# ==================== V8 CORE ====================
import argparse
import copy
import json
import os
import tempfile
import re
import sys
import time
import urllib.error
import urllib.request
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
# .env + OpenAI-compatible LLM translator
# ============================================================

def load_dotenv_file(path: str = ".env", override: bool = False) -> None:
    """Minimal .env loader to avoid adding python-dotenv."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value



def protection_instruction_v7() -> str:
    """Prompt snippet based on PDF_TRANSLATOR_PROTECT_TERMS."""
    protect_terms = os.getenv("PDF_TRANSLATOR_PROTECT_TERMS", "1").strip().lower() not in {"0", "false", "no", "n"}
    if protect_terms:
        return (
            "- Preserve product/technical names when they are placeholders or common product terms: "
            "CXVIEW, CXVIEW GPT Box, GPT Box, AI Video Analytics, CCTV, POS, API, RTSP, ONVIF, VMS, ROI, POC, PPE, USD.\\n"
            "- Use the glossary exactly. For this domain, translate camera as 'camera' not 'máy ảnh'; site as 'địa điểm' not 'trang web'; billing as 'thanh toán' not 'hóa đơn'; Edge Intelligence as 'trí tuệ biên'."
        )
    return (
        "- Translate all normal human-readable text into the target language, including technical/business phrases.\\n"
        "- Preserve only URLs, emails, numbers, currencies, and explicit placeholders like <PH_0>.\\n"
        "- Do not leave English words unless they are brand names, model names, URLs, emails, numbers, or glossary-protected terms."
    )


class OpenAICompatibleTranslator:
    """
    OpenAI-compatible Chat Completions translator.

    Reads:
        LLM_API_KEY
        LLM_BASE_URL, e.g. https://api.groq.com/openai/v1
        LLM_MODEL, e.g. llama-3.3-70b-versatile
    """

    def __init__(self, temperature: float = 0.1, timeout: int = 120, max_retries: int = 3):
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
        self.model = os.getenv("LLM_MODEL", "").strip()
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        missing = [k for k, v in {
            "LLM_API_KEY": self.api_key,
            "LLM_BASE_URL": self.base_url,
            "LLM_MODEL": self.model,
        }.items() if not v]
        if missing:
            raise ValueError(f"Missing LLM env vars: {', '.join(missing)}")

    def translate_batch(
        self,
        items: List[Dict[str, str]],
        source_lang: str,
        target_lang: str,
        glossary: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        payload_items = []
        for item in items:
            payload_items.append({
                "id": item["id"],
                "role": item.get("role", "body"),
                "max_chars": item.get("max_chars"),
                "max_lines": item.get("max_lines"),
                "box_width_pt": item.get("box_width_pt"),
                "box_height_pt": item.get("box_height_pt"),
                "font_size_pt": item.get("font_size_pt"),
                "text": item["text"],
            })

        glossary_text = ""
        if glossary:
            glossary_text = "\nGlossary, must follow exactly:\n" + json.dumps(glossary, ensure_ascii=False, indent=2)

        system_prompt = (
            "You are a professional PDF document translator. "
            "Translate text blocks while preserving meaning, formatting intent, and IDs. "
            "Return valid JSON only."
        )
        user_prompt = f"""
Translate the following PDF text blocks.

Source language: {source_lang}
Target language: {target_lang}

Rules:
- Return a JSON array only. Do not wrap it in markdown.
- Each item must be: {{"id": "...", "translated": "..."}}
- Keep exactly the same IDs. Do not add, remove, merge, split, or reorder IDs.
- Preserve placeholders exactly, e.g. <PH_0>, <PH_1>.
- Preserve URLs, emails, numbers, currencies, formulas, and organization/product names unless the glossary says otherwise.
- This is a layout-preserving PDF translation. Every item includes max_chars and max_lines. Keep the translation within those constraints whenever possible.
- Preserve placeholders exactly, including non-breaking placeholders like <PH_0>.
{protection_instruction_v7()}
- Avoid word-by-word literal Vietnamese. Prefer natural B2B/technical Vietnamese suitable for a sales deck.
- For Detection labels, use compact patterns like 'Phát hiện ...', 'Nhận dạng ...', 'Giám sát ...', 'Đếm ...'.
- Never output awkward mixed phrases like 'leo climbing', 'Đấu tranh' for fighting, 'vật thể không có người', 'Linhh hoạt hóa đơn', or 'CXVIEW GPT Hộp'.
- For role=title: concise marketing headline, no explanatory wording.
- For role=cta: concise call-to-action.
- For role=label: very short Vietnamese label, preferably 1-4 words, no full sentence.
- For small boxes where max_chars <= 30: use the shortest natural equivalent; keep English term if Vietnamese would be too long.
- For role=body: use natural Vietnamese suitable for a formal flyer/document, but avoid unnecessary length.
- Do not translate hidden/logo/decorative text; those should not appear here.
- Avoid awkward literal translations such as translating product word "Box" into "Hộp" when it is part of the product name.
{glossary_text}

Input JSON:
{json.dumps(payload_items, ensure_ascii=False, indent=2)}
""".strip()

        raw = self._chat_completion(system_prompt, user_prompt)
        parsed = self._parse_json_array(raw)
        by_id = {str(obj.get("id")): str(obj.get("translated", "")) for obj in parsed if isinstance(obj, dict)}

        result = []
        for item in items:
            translated = by_id.get(item["id"], item["text"])
            translated = postprocess_translation(
                translated,
                original_text=item.get("text", ""),
                role=item.get("role", "body"),
                target_lang=target_lang,
            )
            result.append({"id": item["id"], "translated": translated})

        if os.getenv("PDF_TRANSLATOR_QUALITY_PASS", "1") != "0":
            result = self._quality_review_batch(payload_items, result, target_lang, glossary)

        return result

    def _quality_review_batch(
        self,
        payload_items: List[Dict[str, object]],
        drafts: List[Dict[str, str]],
        target_lang: str,
        glossary: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        # Only review problematic items to save tokens and cost.
        draft_by_id = {d["id"]: d.get("translated", "") for d in drafts}
        problematic = []
        for item in payload_items:
            text = str(item.get("text", ""))
            draft = draft_by_id.get(str(item.get("id")), "")
            role = str(item.get("role", "body"))
            if translation_has_quality_issue(text, draft, role, target_lang):
                problematic.append({**item, "draft": draft})

        if not problematic:
            return drafts

        glossary_text = ""
        if glossary:
            glossary_text = "\nGlossary, must follow exactly:\n" + json.dumps(glossary, ensure_ascii=False, indent=2)

        system_prompt = (
            "You are a senior Vietnamese localization reviewer for B2B AI video analytics sales decks. "
            "Fix inaccurate, awkward, or literal translations. Return valid JSON only."
        )
        user_prompt = f"""
Review and fix only the Vietnamese translations below.

Rules:
- Return a JSON array only. Each item: {{"id":"...", "translated":"..."}}
- Preserve the same IDs.
- Preserve placeholders exactly.
{protection_instruction_v7()}
- Fix literal/wrong phrases: 'leo climbing', 'Đấu tranh', 'vật thể không có người', 'thao túng camera', 'Linhh hoạt hóa đơn', 'GPT Hộp'.
- Keep label translations compact to fit the given max_chars/max_lines.
{glossary_text}

Input JSON:
{json.dumps(problematic, ensure_ascii=False, indent=2)}
""".strip()
        try:
            raw = self._chat_completion(system_prompt, user_prompt)
            parsed = self._parse_json_array(raw)
            reviewed = {str(obj.get("id")): str(obj.get("translated", "")) for obj in parsed if isinstance(obj, dict)}
        except Exception:
            reviewed = {}

        output = []
        for d in drafts:
            fixed = reviewed.get(d["id"], d.get("translated", ""))
            # Apply deterministic cleanup again after review.
            role = "body"
            original = ""
            for item in payload_items:
                if str(item.get("id")) == d["id"]:
                    role = str(item.get("role", "body"))
                    original = str(item.get("text", ""))
                    break
            fixed = postprocess_translation(fixed, original, role, target_lang)
            output.append({"id": d["id"], "translated": fixed})
        return output

    def _chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        url = self.base_url + "/chat/completions"
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = json.dumps(body).encode("utf-8")
        # Groq / Cloudflare can reject bare urllib clients with HTTP 403 / error code 1010
        # if the request has no browser-like User-Agent. Keep this explicit.
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": os.getenv(
                "LLM_USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36 pdf-translator-mikotech/0.6",
            ),
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_text = response.read().decode("utf-8")
                parsed = json.loads(response_text)
                return parsed["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                body_text = e.read().decode("utf-8", errors="replace")
                hint = ""
                if e.code == 403 and "1010" in body_text:
                    hint = (
                        " | Hint: Groq/Cloudflare blocked this HTTP client signature. "
                        "v6 adds User-Agent headers; if it still happens, test from another network/VPN off, "
                        "or contact Groq support with the CF-RAY / error details."
                    )
                last_error = RuntimeError(f"HTTP {e.code}: {body_text[:800]}{hint}")
            except Exception as e:
                last_error = e
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"LLM translation failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _parse_json_array(text: str) -> List[Dict[str, str]]:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and isinstance(obj.get("items"), list):
                return obj["items"]
        except Exception:
            pass
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise ValueError("LLM did not return a valid JSON array")


# ============================================================
# Embedded font extraction utilities
# ============================================================

def clean_font_name(name: str) -> str:
    return re.sub(r"^[A-Z]{6}\+", "", str(name or "").strip())


def safe_font_resource_name(name: str, prefix: str = "FOrig") -> str:
    clean = clean_font_name(name)
    clean = re.sub(r"[^A-Za-z0-9_]", "_", clean)
    return prefix + "_" + (clean[:42] or "Font")


def extract_embedded_fonts(pdf_path: str, out_dir: str = "extracted_fonts") -> Dict[str, str]:
    """
    Extract embedded fonts from source PDF for local reuse.
    FontResolver will only reuse extracted fonts that actually support Vietnamese.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open(pdf_path)
    font_map: Dict[str, str] = {}

    for page in pdf:
        for info in page.get_fonts(full=True):
            xref = int(info[0])
            basefont = clean_font_name(info[3] if len(info) > 3 else f"font_{xref}")
            if not basefont or basefont in font_map:
                continue
            try:
                extracted = pdf.extract_font(xref)
            except Exception:
                continue
            if isinstance(extracted, tuple):
                name, ext, font_type, content = extracted
            else:
                name = extracted.get("name", basefont)
                ext = extracted.get("ext", "ttf")
                content = extracted.get("content", b"")
            if not content:
                continue
            clean = clean_font_name(name or basefont) or basefont
            ext = str(ext or "ttf").lower().lstrip(".")
            file_name = re.sub(r"[^A-Za-z0-9_.-]", "_", clean) + "." + ext
            path = out / file_name
            try:
                path.write_bytes(content)
                font_map[basefont] = str(path)
                font_map[clean] = str(path)
            except Exception:
                continue

    pdf.close()
    return font_map


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


def line_bboxes_look_like_one_paragraph(lines: List[TextLine]) -> bool:
    """
    True when a raw PDF text block behaves like a normal paragraph/title:
    multiple lines stacked vertically with roughly the same left edge.

    False for table/header/grid blocks where PyMuPDF grouped several distant
    text islands into a single block. Those must be split before translation,
    otherwise all translated labels get rendered into one small rectangle.
    """
    if len(lines) <= 1:
        return True

    lefts = [ln.bbox[0] for ln in lines]
    rights = [ln.bbox[2] for ln in lines]
    tops = [ln.bbox[1] for ln in lines]
    bottoms = [ln.bbox[3] for ln in lines]
    sizes = [ln.font_size for ln in lines]
    avg_size = max(1.0, sum(sizes) / len(sizes))

    left_var = max(lefts) - min(lefts)
    right_var = max(rights) - min(rights)

    # Classic paragraph: each line starts nearly at the same x.
    if left_var <= max(8.0, avg_size * 0.9):
        return True

    # Right aligned multi-line paragraph.
    if right_var <= max(8.0, avg_size * 0.9):
        return True

    # Detect overlapping/same-row lines spread across columns, e.g. table headers.
    overlaps = 0
    sorted_lines = sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    for prev, curr in zip(sorted_lines, sorted_lines[1:]):
        if curr.bbox[1] < prev.bbox[3] - 1.0:
            overlaps += 1

    vertical_span = max(bottoms) - min(tops)
    horizontal_span = max(rights) - min(lefts)

    if overlaps > 0 and horizontal_span > avg_size * 12:
        return False

    # If all lines live in a shallow band but x positions jump a lot, this is a grid.
    if vertical_span <= avg_size * 3.5 and left_var > avg_size * 5:
        return False

    # Conservative fallback: if x jumps a lot, split.
    if left_var > avg_size * 10 and len(lines) >= 3:
        return False

    return True


def should_split_raw_text_block(lines: List[TextLine]) -> bool:
    return len(lines) > 1 and not line_bboxes_look_like_one_paragraph(lines)


def make_block_from_lines(page_index: int, block_id: str, order: int, lines: List[TextLine]) -> TextBlock:
    bbox = rect_union(ln.bbox for ln in lines)
    return TextBlock(
        id=block_id,
        page_index=page_index,
        bbox=bbox,  # type: ignore
        lines=copy.deepcopy(lines),
        order=order,
        original_text=normalize_text("\n".join(ln.text for ln in lines)),
    )


def layout_hints_for_block(block: TextBlock) -> Dict[str, object]:
    """Approximate constraints passed to the LLM so it translates for the box."""
    w = bbox_width(block.bbox)
    h = bbox_height(block.bbox)
    fs = max(4.0, block.font_size)
    max_lines = max(1, int(h / max(1.0, fs * 1.05)))

    # Rough chars-per-line estimate for Vietnamese-capable sans fonts.
    chars_per_line = max(4, int(w / max(1.0, fs * 0.48)))
    max_chars = max(6, chars_per_line * max_lines)

    # Small labels should be extra compact.
    if fs <= 8.0 or max_chars <= 30:
        max_chars = max(6, int(max_chars * 0.85))

    return {
        "max_chars": int(max_chars),
        "max_lines": int(max_lines),
        "box_width_pt": round(w, 1),
        "box_height_pt": round(h, 1),
        "font_size_pt": round(fs, 1),
    }


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

            # PyMuPDF sometimes groups visually unrelated text islands into one raw block
            # (very common in tables / slide grids). Split those before translation.
            if should_split_raw_text_block(lines):
                for one_line in lines:
                    split_block = make_block_from_lines(
                        page_index=page_index,
                        block_id=f"p{page_index}_b{block_idx}",
                        order=block_idx,
                        lines=[one_line],
                    )
                    split_block.align = estimate_alignment(split_block, page_ir)
                    split_block.role = classify_block_role(split_block, page_ir)
                    page_ir.blocks.append(split_block)
                    block_idx += 1
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

    # Some PDFs contain duplicate invisible text under images, but slide decks often place
    # real visible text on top of a full-page image background. Therefore image-overlap
    # hidden detection is OFF by default. Enable only when you know the PDF has duplicate
    # hidden text: set PDF_TRANSLATOR_DETECT_HIDDEN_BY_IMAGE=1.
    if os.getenv("PDF_TRANSLATOR_DETECT_HIDDEN_BY_IMAGE", "0") == "1":
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
    if re.search(r"learn more|apply|email|contact|book|demo", text, flags=re.I):
        return "cta"

    # Small grid / icon labels should be translated very compactly.
    if block.font_size <= 8.2 and bbox_width(block.bbox) <= 140 and bbox_height(block.bbox) <= 45:
        return "label"

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
    # Preserve product / technical terms that should not be literally translated.
    r"\bCXVIEW GPT Box\b",
    r"\bGPT Box\b",
    r"\bAI Video Analytics\b",
    r"\bCXVIEW\b",
    r"\bCCTV\b",
    r"\bPOS\b",
    r"\bAPI\b",
    r"\bRTSP\b",
    r"\bONVIF\b",
    r"\bVMS\b",
    r"\bROI\b",
    r"\bPOC\b",
    r"\bPPE\b",
    r"https?://[^\s)]+",
    r"[\w.\-]+@[\w.\-]+\.\w+",
    r"\b\d+(?:[.,]\d+)?%?\b",
    r"\$\s?\d+(?:[.,]\d+)?",
    r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b",
]



def _active_placeholder_patterns_v7() -> List[str]:
    """Return placeholder patterns according to protection mode.

    PDF_TRANSLATOR_PROTECT_TERMS=1:
        Preserve product/technical terms as previous versions did.

    PDF_TRANSLATOR_PROTECT_TERMS=0:
        Preserve only structural tokens that should never be altered:
        URLs, emails, numbers, currency. This lets the LLM translate all
        normal text; glossary can be used later to keep/standardize terms.
    """
    protect_terms = os.getenv("PDF_TRANSLATOR_PROTECT_TERMS", "1").strip().lower() not in {"0", "false", "no", "n"}

    structural_patterns = [
        r"https?://[^\s)]+",
        r"[\w.\-]+@[\w.\-]+\.\w+",
        r"\$\s?\d+(?:[.,]\d+)?",
        r"\b\d+(?:[.,]\d+)?%?\b",
    ]

    if not protect_terms:
        return structural_patterns

    return PLACEHOLDER_PATTERNS


def protect_placeholders(text: str) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    protected = text
    counter = 0

    for pattern in _active_placeholder_patterns_v7():
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

# Built-in terminology and translation memory
# These are not meant to replace the LLM. They are deterministic guardrails
# for short labels, table headers, and domain terms where LLMs often become literal.

BUILTIN_GLOSSARY_VI: Dict[str, str] = {
    "camera": "camera",
    "cameras": "camera",
    "site": "địa điểm",
    "sites": "địa điểm",
    "edge intelligence": "trí tuệ biên",
    "edge device": "thiết bị biên",
    "on premise": "tại chỗ",
    "on-premise": "tại chỗ",
    "AI video analytics": "AI Video Analytics",
    "CXVIEW GPT Box": "CXVIEW GPT Box",
    "CCTV": "CCTV",
    "POS": "POS",
    "API": "API",
    "RTSP": "RTSP",
    "ONVIF": "ONVIF",
    "VMS": "VMS",
    "ROI": "ROI",
    "POC": "POC",
    "PPE": "PPE",
    "USD": "USD",
    "billing flexibility": "Linh hoạt thanh toán",
    "concurrent AI stream ratio": "Tỷ lệ luồng AI đồng thời",
}

# Normalized exact phrase memory for small labels and stable UI/table text.
# This dramatically improves slide/flyer PDFs where each label has a tiny bbox.
EXACT_TRANSLATION_MEMORY_VI: Dict[str, str] = {
    # Page 1 / overview
    "client challenges. cxview solutions. business impacts.": "Thách thức khách hàng. Giải pháp CXVIEW. Tác động kinh doanh.",
    "from conservative cctv infrastructure to real-time intelligence, operational automation and business-performance reporting.": "Từ hạ tầng CCTV truyền thống đến trí tuệ thời gian thực, tự động hóa vận hành và báo cáo hiệu quả kinh doanh.",
    "about cxview": "GIỚI THIỆU CXVIEW",
    "cxview gpt box & ai video analytics": "CXVIEW GPT Box & AI Video Analytics",
    "cxview gpt box": "CXVIEW GPT Box",
    "& ai video analytics": "& AI Video Analytics",

    # Page 2 / pricing table
    "package": "Gói",
    "tier": "Cấp",
    "cameras": "Camera",
    "concurrent ai streams active on ai models existed": "Luồng AI đồng thời\ntrên mô hình AI hiện có",
    "concurrent ai streams": "Luồng AI đồng thời",
    "active on ai models existed": "trên mô hình AI hiện có",
    "monthly (usd)": "Hàng tháng (USD)",
    "24-month (usd)": "24 tháng (USD)",
    "36-month (usd)": "36 tháng (USD)",
    "concurrent ai stream ratio": "Tỷ lệ luồng AI đồng thời",
    "billing flexibility": "Linh hoạt thanh toán",
    "cxview smart ai video analytics solution": "CXVIEW SMART AI VIDEO ANALYTICS SOLUTION",

    # Page 3 / solution labels
    "our solutions": "GIẢI PHÁP CỦA CHÚNG TÔI",
    "vehicle plate recognition": "Nhận dạng biển số xe",
    "fence and wall climbing detection": "Phát hiện leo rào/tường",
    "intrusion detection": "Phát hiện xâm nhập",
    "camera tampering detection": "Phát hiện can thiệp camera",
    "ppe and uniform detection": "Phát hiện PPE & đồng phục",
    "forklift and vehicle safety detection": "An toàn xe nâng & phương tiện",
    "falls and slips detection": "Phát hiện té ngã/trượt",
    "fighting detection": "Phát hiện đánh nhau",
    "unusual crowd detection": "Phát hiện đám đông bất thường",
    "smoke and fire detection": "Phát hiện khói/lửa",
    "ai camera in security": "AI CAMERA\nAN NINH",
    "ai camera in operation": "AI CAMERA\nVẬN HÀNH",
    "ai camera in safety": "AI CAMERA\nAN TOÀN",
    "smart workstation monitoring": "Giám sát trạm làm việc",
    "automated product counting": "Đếm sản phẩm tự động",
    "product quality inspection": "Kiểm tra chất lượng sản phẩm",
    "heat maps & route maps analysis": "Phân tích bản đồ nhiệt/lộ trình",
    "heat maps and route maps analysis": "Phân tích bản đồ nhiệt/lộ trình",
    "dwell time report": "Báo cáo thời gian lưu lại",
    "traffic counting": "Đếm lưu lượng",
    "customer demographic analysis": "Phân tích nhân khẩu học",
    "customer engagement detection": "Phát hiện tương tác khách hàng",
    "table cleaning detection": "Phát hiện dọn bàn",
    "patrol automatic report": "Báo cáo tuần tra tự động",
    "unattended object detection": "Phát hiện vật thể bỏ quên",

    # Page 4 / process
    "our process": "QUY TRÌNH",
    "strategic understanding": "Thấu hiểu chiến lược",
    "tailored solution architecture": "Kiến trúc giải pháp tùy chỉnh",
    "pilot (poc) & validation": "Pilot (POC) & xác thực",
    "scale & sustain": "Mở rộng & duy trì",
    "book your live demo": "Đặt lịch demo trực tiếp",
    "near-zero latency": "Độ trễ gần bằng 0",
    "data sovereignty": "Chủ quyền dữ liệu",
    "cost efficiency": "Tối ưu chi phí",
    "bandwidth savings": "Tiết kiệm băng thông",

    # Page 5 / platform
    "platform & technology": "NỀN TẢNG & CÔNG NGHỆ",
    "how cxview delivers the transformation": "CXVIEW triển khai chuyển đổi như thế nào",
    "why a transformation partner is essential": "Vì sao cần một đối tác chuyển đổi",
    "core deployment benefits": "Lợi ích triển khai cốt lõi",
    "customer on - premise ai vision infrastructure": "Hạ tầng AI Vision tại chỗ của khách hàng",
    "customer on-premise ai vision infrastructure": "Hạ tầng AI Vision tại chỗ của khách hàng",
    "no camera replacement required when streams are available": "Không cần thay camera khi có sẵn luồng video",
    "on-premise processing supports latency, bandwidth control and data sovereignty": "Xử lý tại chỗ hỗ trợ độ trễ thấp, kiểm soát băng thông và chủ quyền dữ liệu",
    "automated reports reduce manual debate with employees, vendors and site teams": "Báo cáo tự động giảm tranh luận thủ công với nhân viên, nhà cung cấp và đội ngũ tại địa điểm",
    "management receives objective before/after performance data for every site": "Ban quản lý nhận dữ liệu hiệu suất trước/sau khách quan cho từng địa điểm",
}

BAD_PHRASE_REPLACEMENTS_VI: List[Tuple[str, str]] = [
    ("Linhh hoạt hóa đơn", "Linh hoạt thanh toán"),
    ("Linhh hoạt Thanh toán", "Linh hoạt thanh toán"),
    ("Linh hoạt hóa đơn", "Linh hoạt thanh toán"),
    ("hóa đơn hàng tháng", "thanh toán hàng tháng"),
    ("chu kỳ hóa đơn", "chu kỳ thanh toán"),
    ("mẫu AI", "mô hình AI"),
    ("các mẫu AI", "các mô hình AI"),
    ("máy ảnh", "camera"),
    ("Máy ảnh", "Camera"),
    ("thao túng Camera", "can thiệp camera"),
    ("thao túng camera", "can thiệp camera"),
    ("Nhận dạng Biển xe", "Nhận dạng biển số xe"),
    ("Biển xe", "biển số xe"),
    ("vật thể không có người", "vật thể bỏ quên"),
    ("Đối tượng Bỏ quên", "vật thể bỏ quên"),
    ("leo climbing", "leo"),
    ("Phát hiện Đấu tranh", "Phát hiện đánh nhau"),
    ("Phát hiện đấu tranh", "Phát hiện đánh nhau"),
    ("đánh tranh", "đánh nhau"),
    ("Làm sạch bàn", "dọn bàn"),
    ("Tầm nhìn Trên chỗ", "AI Vision tại chỗ"),
    ("Trên chỗ", "tại chỗ"),
    ("trên chỗ", "tại chỗ"),
    ("thiết bị cạnh", "thiết bị biên"),
    ("trí tuệ Edge", "trí tuệ biên"),
    ("động cơ quyết định", "hệ thống ra quyết định"),
    ("cơ sở hạ tầng bảo thủ CCTV", "hạ tầng CCTV truyền thống"),
    ("bảo thủ CCTV", "CCTV truyền thống"),
    ("nhóm trang web", "đội ngũ tại địa điểm"),
    ("mọi trang web", "mọi địa điểm"),
    ("trang web", "địa điểm"),
    ("trang web.", "địa điểm."),
    ("site teams", "đội ngũ tại địa điểm"),
    ("thúc bách", "chặt chẽ"),
    ("cấp bách hoạt động", "tính cấp thiết trong vận hành"),
]

BANNED_TRANSLATION_FRAGMENTS_VI = [
    "Linhh", "leo climbing", "Đấu tranh", "đấu tranh", "thao túng Camera", "thao túng camera",
    "vật thể không có người", "Biển xe", "máy ảnh", "Máy ảnh", "trang web", "Trên chỗ", "trên chỗ",
]


def normalize_translation_key(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("\n", " ")
    text = text.replace("&", " & ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def exact_translation_override(text: str, target_lang: str = "vi") -> Optional[str]:
    if not target_lang.lower().startswith("vi"):
        return None
    key = normalize_translation_key(text)
    return EXACT_TRANSLATION_MEMORY_VI.get(key)


def merged_glossary_for_target(user_glossary: Optional[Dict[str, str]], target_lang: str = "vi") -> Optional[Dict[str, str]]:
    if not target_lang.lower().startswith("vi"):
        return user_glossary
    merged = dict(BUILTIN_GLOSSARY_VI)
    if user_glossary:
        merged.update(user_glossary)
    return merged


def postprocess_translation(text: str, original_text: str, role: str = "body", target_lang: str = "vi") -> str:
    if not text:
        return text
    if not target_lang.lower().startswith("vi"):
        return text

    t = text.strip()
    for bad, good in BAD_PHRASE_REPLACEMENTS_VI:
        t = t.replace(bad, good)

    # Spacing around bullets: keep compact for flyer labels.
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([•✓ü])\s*", r"\1 ", t)
    t = re.sub(r"\s+", " ", t) if "\n" not in t else re.sub(r"[ \t]+", " ", t)

    # Keep protected product names clean if the model inserted odd spacing/casing.
    t = re.sub(r"CXVIEW\s+GPT\s+Box", "CXVIEW GPT Box", t, flags=re.I)
    t = re.sub(r"AI\s+Video\s+Analytics", "AI Video Analytics", t, flags=re.I)
    t = re.sub(r"\bGPT\s+Hộp\b", "GPT Box", t, flags=re.I)
    t = re.sub(r"CXVIEW\s+GPT\s+Hộp", "CXVIEW GPT Box", t, flags=re.I)

    # Tiny labels should not end with periods.
    if role in {"label", "title", "cta"}:
        t = t.strip().rstrip(".")

    return t.strip()


def translation_has_quality_issue(original_text: str, translated_text: str, role: str = "body", target_lang: str = "vi") -> bool:
    if not target_lang.lower().startswith("vi"):
        return False
    if not translated_text.strip():
        return False
    # Placeholders must survive.
    orig_ph = set(re.findall(r"<PH_\d+>", original_text))
    trans_ph = set(re.findall(r"<PH_\d+>", translated_text))
    if not orig_ph.issubset(trans_ph):
        return True
    for frag in BANNED_TRANSLATION_FRAGMENTS_VI:
        if frag in translated_text:
            return True
    # For short labels, avoid unnecessary English leftovers, except protected acronyms/products.
    if role == "label":
        leftover = re.findall(r"\b[A-Za-z]{4,}\b", translated_text)
        allowed = {"CXVIEW", "CCTV", "RTSP", "ONVIF", "VMS", "PPE", "POC", "ROI", "USD", "Video", "Analytics", "GPT", "Box"}
        if any(w not in allowed for w in leftover):
            return True
    return False


def is_translatable_block(block: TextBlock, translate_headers_footers: bool = False) -> bool:
    if not block.original_text.strip():
        return False

    translate_all = os.getenv("PDF_TRANSLATOR_TRANSLATE_ALL_TEXT", "1").strip().lower() not in {"0", "false", "no", "n"}

    # Default v7 behavior: translate every extractable text object.
    # Keep only hidden/page numbers skipped. Logos are also translated unless
    # PDF_TRANSLATOR_SKIP_LOGO_TEXT=1, because the user's goal is full-text translation.
    if translate_all:
        if block.role in {"hidden", "page_number"}:
            return False
        if block.role == "logo" and os.getenv("PDF_TRANSLATOR_SKIP_LOGO_TEXT", "0") == "1":
            return False
        return True

    # Legacy behavior.
    if block.role in {"logo", "hidden", "page_number"}:
        return False
    if block.role in {"header", "footer"} and not translate_headers_footers:
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
    effective_glossary = merged_glossary_for_target(glossary, target_lang)

    # Step 1: exact translation memory for stable labels/table headers.
    pending: List[TextBlock] = []
    exact_count = 0
    for block in blocks:
        exact = exact_translation_override(block.original_text, target_lang)
        if exact is not None:
            block.translated_text = exact
            exact_count += 1
        else:
            pending.append(block)

    if exact_count:
        print(f"      Translation memory exact hits: {exact_count}")

    # Step 2: LLM translation for remaining text.
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        payload = []
        placeholder_maps: Dict[str, Dict[str, str]] = {}

        for block in batch:
            protected, mapping = protect_placeholders(block.original_text)
            placeholder_maps[block.id] = mapping
            hints = layout_hints_for_block(block)
            payload.append({
                "id": block.id,
                "role": block.role,
                "text": protected,
                "source_text": block.original_text,
                **hints,
            })

        translated = translator.translate_batch(payload, source_lang, target_lang, effective_glossary)
        translated_by_id = {item["id"]: item.get("translated", "") for item in translated}

        for block in batch:
            raw_translation = translated_by_id.get(block.id, block.original_text)
            restored = restore_placeholders(raw_translation, placeholder_maps.get(block.id, {})).strip()
            block.translated_text = postprocess_translation(restored, block.original_text, block.role, target_lang)

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
        embedded_fonts: Optional[Dict[str, str]] = None,
        prefer_original_fonts: bool = True,
        fallback_fontname: str = "helv",
    ):
        self.embedded_fonts = {clean_font_name(k): v for k, v in (embedded_fonts or {}).items() if v}
        self.prefer_original_fonts = prefer_original_fonts
        self.regular_font = self._first_existing([regular_font, *self._system_font_candidates("regular")])
        self.bold_font = self._first_existing([bold_font, *self._system_font_candidates("bold"), self.regular_font])
        self.title_font = self._first_existing([title_font, *self._system_font_candidates("title"), self.bold_font, self.regular_font])
        self.fallback_fontname = fallback_fontname
        self._font_cache: Dict[str, fitz.Font] = {}
        self._supports_cache: Dict[str, bool] = {}

    @staticmethod
    def _font_supports_vietnamese(path: str) -> bool:
        probe = "ăâêôơưđĂÂÊÔƠƯĐáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        try:
            font = fitz.Font(fontfile=path)
            return all(font.has_glyph(ord(ch)) for ch in probe)
        except Exception:
            return False

    def _supports_vietnamese_cached(self, path: Optional[str]) -> bool:
        if not path:
            return False
        if path not in self._supports_cache:
            self._supports_cache[path] = self._font_supports_vietnamese(path)
        return self._supports_cache[path]

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
        return fallback_existing

    @staticmethod
    def _system_font_candidates(kind: str) -> List[str]:
        win = Path("C:/Windows/Fonts")
        linux = Path("/usr/share/fonts")
        candidates: List[str] = []
        if kind == "title":
            candidates += [
                str(win / "arialnb.ttf"), str(win / "ARIALNB.TTF"),
                str(win / "arialbd.ttf"), str(win / "ARIALBD.TTF"),
                str(linux / "truetype/dejavu/DejaVuSansCondensed-Bold.ttf"),
                str(linux / "truetype/noto/NotoSans-CondensedBold.ttf"),
                str(linux / "truetype/noto/NotoSansCondensed-Bold.ttf"),
                str(linux / "truetype/liberation/LiberationSansNarrow-Bold.ttf"),
            ]
        elif kind == "bold":
            candidates += [
                str(win / "arialbd.ttf"), str(win / "ARIALBD.TTF"),
                str(win / "calibrib.ttf"), str(win / "CALIBRIB.TTF"),
                str(linux / "truetype/dejavu/DejaVuSans-Bold.ttf"),
                str(linux / "truetype/liberation/LiberationSans-Bold.ttf"),
                str(linux / "truetype/noto/NotoSans-Bold.ttf"),
            ]
        else:
            candidates += [
                str(win / "arial.ttf"), str(win / "ARIAL.TTF"),
                str(win / "calibri.ttf"), str(win / "CALIBRI.TTF"),
                str(linux / "truetype/dejavu/DejaVuSans.ttf"),
                str(linux / "truetype/liberation/LiberationSans-Regular.ttf"),
                str(linux / "truetype/noto/NotoSans-Regular.ttf"),
            ]
        candidates += [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ]
        return candidates

    def _original_font_path_for(self, block: TextBlock) -> Optional[str]:
        if not self.prefer_original_fonts:
            return None
        clean = clean_font_name(block.main_font)
        path = self.embedded_fonts.get(clean)
        if path:
            return path
        compact = re.sub(r"[^a-z0-9]", "", clean.lower())
        for name, p in self.embedded_fonts.items():
            if re.sub(r"[^a-z0-9]", "", name.lower()) == compact:
                return p
        return None

    def fontfile_for(self, block: TextBlock) -> Optional[str]:
        original = self._original_font_path_for(block)
        if original and self._supports_vietnamese_cached(original):
            return original
        if block.role == "title" and self.title_font:
            return self.title_font
        if block.is_bold and self.bold_font:
            return self.bold_font
        if self.regular_font:
            return self.regular_font
        return None

    def fontname_for(self, block: TextBlock) -> str:
        original = self._original_font_path_for(block)
        if original and self._supports_vietnamese_cached(original):
            return safe_font_resource_name(block.main_font, prefix="FOrig")
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
        font = fitz.Font(fontfile=fontfile) if fontfile else fitz.Font(self.fallback_fontname)
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
    elif block.role == "label":
        min_scale = 0.50
        line_heights = [1.05, 1.0, 0.96, 0.92]
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



def find_symbol_font_file() -> Optional[str]:
    candidates = [
        os.getenv("PDF_TRANSLATOR_SYMBOL_FONT", ""),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
        "C:/Windows/Fonts/seguisym.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for c in candidates:
        if c and Path(c).exists():
            try:
                f = fitz.Font(fontfile=c)
                if f.has_glyph(ord("✓")):
                    return c
            except Exception:
                continue
    return None

_SYMBOL_FONT_FILE_CACHE: Optional[str] = None
_SYMBOL_FITZ_FONT_CACHE: Optional[fitz.Font] = None


def get_symbol_font() -> Tuple[Optional[str], Optional[fitz.Font]]:
    global _SYMBOL_FONT_FILE_CACHE, _SYMBOL_FITZ_FONT_CACHE
    if _SYMBOL_FONT_FILE_CACHE is None:
        _SYMBOL_FONT_FILE_CACHE = find_symbol_font_file() or ""
    if _SYMBOL_FONT_FILE_CACHE and _SYMBOL_FITZ_FONT_CACHE is None:
        try:
            _SYMBOL_FITZ_FONT_CACHE = fitz.Font(fontfile=_SYMBOL_FONT_FILE_CACHE)
        except Exception:
            _SYMBOL_FITZ_FONT_CACHE = None
    return (_SYMBOL_FONT_FILE_CACHE or None), _SYMBOL_FITZ_FONT_CACHE


def draw_layout(page: fitz.Page, layout: LayoutResult, block: TextBlock, resolver: FontResolver, color: RGB):
    if not layout.lines:
        return

    fontfile = resolver.fontfile_for(block)
    fontname = resolver.fontname_for(block)
    measure_font = resolver.fitz_font_for(block)
    symbol_file, symbol_font = get_symbol_font()

    x0, y0, x1, y1 = layout.rect
    y = y0 + layout.fontsize

    for line in layout.lines:
        if y > y1 + layout.fontsize:
            break

        draw_line = line
        leading_check = False
        if draw_line.startswith("✓ "):
            leading_check = True
            draw_line = draw_line[2:].lstrip()
        elif draw_line == "✓":
            leading_check = True
            draw_line = ""

        full_line_for_measure = ("✓ " + draw_line) if leading_check else draw_line
        if layout.align == "center":
            w = measure_font.text_length(full_line_for_measure.replace("✓", "•"), fontsize=layout.fontsize)
            x = x0 + max(0, (layout.rect.width - w) / 2)
        elif layout.align == "right":
            w = measure_font.text_length(full_line_for_measure.replace("✓", "•"), fontsize=layout.fontsize)
            x = x1 - w
        else:
            x = x0

        if leading_check:
            if symbol_file and symbol_font:
                page.insert_text(
                    point=fitz.Point(x, y),
                    text="✓",
                    fontsize=layout.fontsize,
                    fontname="FSymbolVN",
                    fontfile=symbol_file,
                    color=color,
                    overlay=True,
                )
                x += symbol_font.text_length("✓ ", fontsize=layout.fontsize) + 1.0
            else:
                # Last-resort symbol that most sans fonts support.
                page.insert_text(
                    point=fitz.Point(x, y),
                    text="•",
                    fontsize=layout.fontsize,
                    fontname=fontname,
                    fontfile=fontfile,
                    color=color,
                    overlay=True,
                )
                x += measure_font.text_length("• ", fontsize=layout.fontsize) + 1.0

        if draw_line:
            page.insert_text(
                point=fitz.Point(x, y),
                text=draw_line,
                fontsize=layout.fontsize,
                fontname=fontname,
                fontfile=fontfile,
                color=color,
                overlay=True,
            )
        y += layout.fontsize * layout.line_height


def render_translated_pdf(
    input_pdf: str,
    translated_ir: DocumentIR,
    output_pdf: str,
    regular_font: Optional[str] = None,
    bold_font: Optional[str] = None,
    title_font: Optional[str] = None,
    embedded_fonts: Optional[Dict[str, str]] = None,
    prefer_original_fonts: bool = True,
    cover_text: bool = True,
    sampled_background: bool = True,
    translate_headers_footers: bool = False,
    force_render: bool = False,
):
    pdf = fitz.open(input_pdf)
    resolver = FontResolver(
        regular_font=regular_font,
        bold_font=bold_font,
        title_font=title_font,
        embedded_fonts=embedded_fonts,
        prefer_original_fonts=prefer_original_fonts,
    )

    changed_count = 0
    skipped_same = 0
    skipped_role = 0
    skipped_unfit = 0

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

            font = resolver.fitz_font_for(block)
            layout = compute_layout(text, block, rect, font)

            # Safety guard: if a translation cannot fit without becoming unreadable,
            # keep the original text instead of destroying the layout.
            min_safe_scale = 0.45 if block.role == "label" else 0.55
            if (not layout.fits) and not force_render and (layout.fontsize / max(1.0, block.font_size) < min_safe_scale):
                skipped_unfit += 1
                continue

            if cover_text:
                cover_text_block(page, block, page_ir, sampled_background=sampled_background)

            color = int_color_to_rgb(block.color)
            draw_layout(page, layout, block, resolver, color)
            changed_count += 1

    pdf.save(output_pdf, garbage=4, deflate=True)
    pdf.close()

    print(f"Render summary: changed={changed_count}, skipped_same={skipped_same}, skipped_role={skipped_role}, skipped_unfit={skipped_unfit}")


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
    extract_fonts: bool = True,
    embedded_font_dir: str = "extracted_fonts",
    prefer_original_fonts: bool = True,
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

    embedded_fonts: Dict[str, str] = {}
    if extract_fonts:
        print(f"      Extract embedded fonts -> {embedded_font_dir}")
        embedded_fonts = extract_embedded_fonts(input_pdf, embedded_font_dir)
        print(f"      Extracted font records: {len(embedded_fonts)}")

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
        embedded_fonts=embedded_fonts,
        prefer_original_fonts=prefer_original_fonts,
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
    p.add_argument("--env-file", default=".env", help="Path to .env containing LLM_API_KEY / LLM_BASE_URL / LLM_MODEL")
    p.add_argument("--no-llm", action="store_true", help="Disable env-based LLM translation and use DummyTranslator unless --translation-map is provided")
    p.add_argument("--llm-temperature", type=float, default=0.1)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-retries", type=int, default=3)
    p.add_argument("--embedded-font-dir", default="extracted_fonts")
    p.add_argument("--no-extract-fonts", action="store_true")
    p.add_argument("--no-prefer-original-fonts", action="store_true", help="Do not try to reuse extracted original fonts")
    return p


def main(argv: Optional[List[str]] = None):
    args = build_arg_parser().parse_args(argv)

    if not Path(args.input_pdf).exists():
        raise FileNotFoundError(args.input_pdf)

    # Warn early for optional font args that do not exist. The renderer has fallbacks,
    # but missing font files explain why typography may not match the source.
    for font_arg_name in [
        "font", "font_bold", "font_title", "font_condensed", "font_condensed_bold",
        "font_condensed_semibold", "font_medium", "font_semibold", "font_black", "font_symbol",
    ]:
        fp = getattr(args, font_arg_name, None)
        if fp and not Path(fp).exists():
            print(f"WARNING: font file not found for --{font_arg_name.replace('_', '-')}: {fp}", file=sys.stderr)

    load_dotenv_file(args.env_file)

    if args.translation_map:
        translator: Translator = JsonMapTranslator(args.translation_map)
        print("Translator: JSON translation map")
    elif not args.no_llm and os.getenv("LLM_API_KEY") and os.getenv("LLM_BASE_URL") and os.getenv("LLM_MODEL"):
        translator = OpenAICompatibleTranslator(
            temperature=args.llm_temperature,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
        )
        print(f"Translator: LLM model={os.getenv('LLM_MODEL')} base_url={os.getenv('LLM_BASE_URL')}")
    else:
        translator = DummyTranslator()
        print(
            "WARNING: Using DummyTranslator. No real translation will happen.\n"
            "Add LLM_API_KEY, LLM_BASE_URL, LLM_MODEL to .env, or pass --translation-map.\n"
            "Use --no-llm intentionally only when testing layout.",
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
        extract_fonts=not args.no_extract_fonts,
        embedded_font_dir=args.embedded_font_dir,
        prefer_original_fonts=not args.no_prefer_original_fonts,
        export_ir_path=args.export_ir,
        preview_dir=args.preview_dir,
        cover_text=not args.no_cover,
        sampled_background=not args.no_sampled_bg,
        translate_headers_footers=args.translate_headers_footers,
        force_render=args.force_render,
    )


# ============================================================
# V8 OVERRIDES - style fidelity, segmentation, special chars
# ============================================================

# Keep references to v7 functions for fallback where useful.
_v7_postprocess_translation = postprocess_translation
_v7_translation_has_quality_issue = translation_has_quality_issue


def detect_span_style(span: dict) -> Tuple[bool, bool]:
    """More faithful weight detection.

    PyMuPDF flags often mark bold as bit 16. Font names such as Roboto-Medium
    and Roboto-Black are visually weighted but v7 treated Medium/Black poorly
    in some cases. For PDF translation, it is better to map Medium/Black to a
    Vietnamese-capable medium/bold fallback than to regular.
    """
    font = str(span.get("font", "")).lower()
    flags = int(span.get("flags", 0) or 0)
    is_bold = bool(flags & 16) or any(
        k in font for k in ["bold", "black", "semibold", "demibold", "heavy", "medium"]
    )
    is_italic = bool(flags & 2) or any(k in font for k in ["italic", "oblique"])
    return is_bold, is_italic


def normalize_special_chars(text: str) -> str:
    """Normalize PDF-extracted special glyphs before translation/rendering."""
    if not text:
        return text
    t = text.replace("\u00a0", " ")
    t = t.replace("\uf0fc", "✓")
    # Many PDFs using Wingdings expose checkmarks as ü. Treat it as a checkmark
    # only at bullet/check positions so normal words are not affected.
    t = re.sub(r"(?m)^\s*ü\s*", "✓ ", t)
    t = re.sub(r"(?m)([\n\r])\s*ü\s*", r"\1✓ ", t)
    # Normalize common bullet lookalikes.
    t = t.replace("·", "•")
    return t


def normalize_text(text: str) -> str:
    text = normalize_special_chars(text)
    text = text.replace("\u00ad", "")
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def split_line_into_text_islands(line: TextLine) -> List[TextLine]:
    """Split one PDF line into visual text islands when large horizontal gaps exist.

    This fixes tables / multi-column slides where PyMuPDF groups several cells
    or columns into one text line. We keep bullet glyph + following text together.
    """
    spans = [s for s in sorted(line.spans, key=lambda sp: sp.bbox[0]) if s.text.strip()]
    if len(spans) <= 1:
        return [line]

    groups: List[List[TextSpan]] = []
    cur: List[TextSpan] = [spans[0]]
    fs = max(4.0, line.font_size)

    for sp in spans[1:]:
        prev = cur[-1]
        gap = sp.bbox[0] - prev.bbox[2]
        prev_text = prev.text.strip()
        next_text = sp.text.strip()
        # Keep bullet/checkmark and its text together.
        keep_with_prev = prev_text in {"•", "✓", "ü", "-", "–"}
        # Large gaps are usually table cells or separate slide columns.
        split_gap = max(18.0, fs * 3.2)
        if gap > split_gap and not keep_with_prev and next_text not in {",", ".", ":", ";"}:
            groups.append(cur)
            cur = [sp]
        else:
            cur.append(sp)
    groups.append(cur)

    if len(groups) == 1:
        return [line]

    out: List[TextLine] = []
    for group in groups:
        out.append(TextLine(bbox=rect_union(s.bbox for s in group), spans=copy.deepcopy(group)))  # type: ignore
    return out


def line_bboxes_look_like_one_paragraph(lines: List[TextLine]) -> bool:
    if len(lines) <= 1:
        return True

    # If two or more lines are on the same horizontal band but far apart, this is
    # a table/grid row, not a paragraph.
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a, b = lines[i].bbox, lines[j].bbox
            vertical_overlap = min(a[3], b[3]) - max(a[1], b[1])
            if vertical_overlap > min(bbox_height(a), bbox_height(b)) * 0.45:
                gap = max(a[0], b[0]) - min(a[2], b[2])
                if gap > max(18.0, lines[i].font_size * 3.0):
                    return False

    lefts = [b.bbox[0] for b in lines]
    widths = [bbox_width(b.bbox) for b in lines]
    tops = [b.bbox[1] for b in lines]
    left_var = max(lefts) - min(lefts)
    avg_width = sum(widths) / max(1, len(widths))

    # Paragraphs have mostly increasing y positions.
    ordered_y = all(tops[i] <= tops[i + 1] + 1.5 for i in range(len(tops) - 1))
    if not ordered_y:
        return False

    if left_var > max(12.0, avg_width * 0.18):
        return False
    return True


def should_split_raw_text_block(lines: List[TextLine]) -> bool:
    if len(lines) <= 1:
        return False
    return not line_bboxes_look_like_one_paragraph(lines)


def parse_pdf_to_ir(source_pdf: str) -> DocumentIR:
    """V8 parser: split text into visual islands before classification."""
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

            visual_lines: List[TextLine] = []
            for raw_line in raw_block.get("lines", []):
                spans: List[TextSpan] = []
                for raw_span in raw_line.get("spans", []):
                    chars = raw_span.get("chars", [])
                    if chars:
                        text = "".join(ch.get("c", "") for ch in chars)
                    else:
                        text = raw_span.get("text", "")
                    text = normalize_special_chars(text)
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
                for island in split_line_into_text_islands(line):
                    if island.text:
                        visual_lines.append(island)

            if not visual_lines:
                continue

            # Split non-paragraph raw blocks into one visual island per block.
            if should_split_raw_text_block(visual_lines):
                for one_line in visual_lines:
                    split_block = make_block_from_lines(
                        page_index=page_index,
                        block_id=f"p{page_index}_b{block_idx}",
                        order=block_idx,
                        lines=[one_line],
                    )
                    split_block.align = estimate_alignment(split_block, page_ir)
                    split_block.role = classify_block_role(split_block, page_ir)
                    page_ir.blocks.append(split_block)
                    block_idx += 1
                continue

            block_bbox = rect_union(l.bbox for l in visual_lines)
            original_text = normalize_text("\n".join(line.text for line in visual_lines))
            if not original_text:
                continue
            block = TextBlock(
                id=f"p{page_index}_b{block_idx}",
                page_index=page_index,
                bbox=block_bbox,  # type: ignore
                lines=visual_lines,
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


# Stronger, more compact terminology memory for slide labels and tiny boxes.
EXACT_TRANSLATION_MEMORY_VI.update({
    "our solutions": "GIẢI PHÁP",
    "vehicle plate recognition": "Nhận dạng biển số",
    "fence and wall climbing detection": "Leo rào/tường",
    "intrusion detection": "Xâm nhập",
    "camera tampering detection": "Can thiệp camera",
    "ppe and uniform detection": "PPE & đồng phục",
    "forklift and vehicle safety detection": "An toàn xe nâng",
    "falls and slips detection": "Té ngã/trượt",
    "fighting detection": "Đánh nhau",
    "unusual crowd detection": "Đám đông bất thường",
    "smoke and fire detection": "Khói/lửa",
    "smart workstation monitoring": "Giám sát trạm",
    "automated product counting": "Đếm sản phẩm",
    "product quality inspection": "Kiểm tra chất lượng",
    "heat maps & route maps analysis": "Bản đồ nhiệt/lộ trình",
    "heat maps and route maps analysis": "Bản đồ nhiệt/lộ trình",
    "dwell time report": "Thời gian lưu lại",
    "traffic counting": "Đếm lưu lượng",
    "customer demographic analysis": "Nhân khẩu học",
    "customer engagement detection": "Tương tác khách hàng",
    "table cleaning detection": "Dọn bàn",
    "patrol automatic report": "Báo cáo tuần tra",
    "unattended object detection": "Vật thể bỏ quên",
    "package": "Gói",
    "tier": "Cấp",
    "cameras": "Camera",
    "concurrent ai streams active on ai models existed": "Luồng AI đồng thời\ntrên mô hình hiện có",
    "concurrent ai streams": "Luồng AI đồng thời",
    "active on ai models existed": "trên mô hình hiện có",
    "monthly (usd)": "Hàng tháng\n(USD)",
    "24-month (usd)": "24 tháng\n(USD)",
    "36-month (usd)": "36 tháng\n(USD)",
    "billing flexibility": "Thanh toán linh hoạt",
    "concurrent ai stream ratio": "Tỷ lệ luồng AI đồng thời",
    "book your live demo": "Đặt lịch demo",
    "customer on - premise ai vision infrastructure": "Hạ tầng AI Vision tại chỗ",
    "customer on-premise ai vision infrastructure": "Hạ tầng AI Vision tại chỗ",
    "no camera replacement required when streams are available": "Không cần thay camera khi có luồng video",
    "on-premise processing supports latency, bandwidth control and data sovereignty": "Xử lý tại chỗ: độ trễ thấp, tiết kiệm băng thông, chủ quyền dữ liệu",
    "automated reports reduce manual debate with employees, vendors and site teams": "Báo cáo tự động giảm tranh luận với nhân viên, nhà cung cấp và đội ngũ tại địa điểm",
    "management receives objective before/after performance data for every site": "Ban quản lý có dữ liệu trước/sau khách quan cho từng địa điểm",
    "üno camera replacement required when streams are available": "✓ Không cần thay camera khi có luồng video",
    "üon-premise processing supports latency, bandwidth control and data sovereignty": "✓ Xử lý tại chỗ: độ trễ thấp, tiết kiệm băng thông, chủ quyền dữ liệu",
    "üautomated reports reduce manual debate with employees, vendors and site teams": "✓ Báo cáo tự động giảm tranh luận với nhân viên, nhà cung cấp và đội ngũ tại địa điểm",
    "ümanagement receives objective before/after performance data for every site": "✓ Ban quản lý có dữ liệu trước/sau khách quan cho từng địa điểm",
})

BAD_PHRASE_REPLACEMENTS_VI.extend([
    ("Tác động kinh doanh", "Tác động kinh doanh"),
    ("thời gian biểu tham vọng", "lộ trình triển khai nhanh"),
    ("sự tính cấp thiết", "tính cấp thiết"),
    ("thông tin hành động", "thông tin có thể hành động"),
    ("theo dõi bảng điều khiển", "dashboard"),
    ("AI mô hình", "mô hình AI"),
    ("CXVIEW AI mô hình", "mô hình CXVIEW AI"),
])
BANNED_TRANSLATION_FRAGMENTS_VI.extend(["thời gian biểu tham vọng", "sự tính cấp thiết", "AI mô hình"])


def normalize_translation_key(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("\n", " ")
    text = text.replace("&", " & ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    # Match Wingdings checkmark extraction exposed as ü.
    text = re.sub(r"^✓\s*", "ü", text)
    return text


def postprocess_translation(text: str, original_text: str, role: str = "body", target_lang: str = "vi") -> str:
    if not text or not target_lang.lower().startswith("vi"):
        return text
    t = normalize_special_chars(text.strip())

    # Preserve leading bullets/checkmarks from the source when the model drops them.
    original_clean = normalize_special_chars(original_text.strip())
    if original_clean.startswith("•") and not t.startswith("•"):
        t = "• " + t.lstrip("• ").strip()
    if original_clean.startswith("✓") and not t.startswith("✓"):
        t = "✓ " + t.lstrip("✓ ").strip()

    for bad, good in BAD_PHRASE_REPLACEMENTS_VI:
        t = t.replace(bad, good)

    # Product and acronym cleanup.
    t = re.sub(r"CXVIEW\s+GPT\s+Box", "CXVIEW GPT Box", t, flags=re.I)
    t = re.sub(r"CXVIEW\s+GPT\s+Hộp", "CXVIEW GPT Box", t, flags=re.I)
    t = re.sub(r"\bGPT\s+Hộp\b", "GPT Box", t, flags=re.I)
    t = re.sub(r"AI\s+Video\s+Analytics", "AI Video Analytics", t, flags=re.I)
    t = re.sub(r"\b24\s*-\s*tháng\b", "24 tháng", t, flags=re.I)
    t = re.sub(r"\b36\s*-\s*tháng\b", "36 tháng", t, flags=re.I)

    # Normalize spacing without destroying deliberate line breaks.
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([•✓])\s*", r"\1 ", t)
    if "\n" in t:
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r" *\n *", "\n", t)
    else:
        t = re.sub(r"\s+", " ", t)

    # Tiny labels: no trailing punctuation and avoid verbose prefix where memory failed.
    if role in {"label", "title", "cta"}:
        t = t.strip().rstrip(".")
        t = re.sub(r"^Phát hiện\s+(can thiệp camera|xâm nhập|khói/lửa|đánh nhau|vật thể bỏ quên)$", r"\1", t, flags=re.I)

    return t.strip()


def translation_has_quality_issue(original_text: str, translated_text: str, role: str = "body", target_lang: str = "vi") -> bool:
    if _v7_translation_has_quality_issue(original_text, translated_text, role, target_lang):
        return True
    if not target_lang.lower().startswith("vi"):
        return False
    t = normalize_special_chars(translated_text)
    for frag in BANNED_TRANSLATION_FRAGMENTS_VI:
        if frag and frag in t:
            return True
    # Detect accidental leftover of the whole source in translated labels/headers.
    if role in {"label", "title", "cta"}:
        src_words = set(re.findall(r"[A-Za-z]{5,}", original_text))
        trg_words = set(re.findall(r"[A-Za-z]{5,}", t))
        allowed = {"CXVIEW", "CCTV", "RTSP", "ONVIF", "VMS", "Video", "Analytics"}
        if any(w in trg_words and w not in allowed for w in src_words):
            return True
    return False


def layout_hints_for_block(block: TextBlock) -> Dict[str, object]:
    w = bbox_width(block.bbox)
    h = bbox_height(block.bbox)
    fs = max(4.0, block.font_size)
    max_lines = max(1, int(h / max(1.0, fs * 1.02)))
    chars_per_line = max(3, int(w / max(1.0, fs * 0.55)))
    max_chars = max(4, chars_per_line * max_lines)
    if block.role == "label" or fs <= 8.2:
        max_chars = max(4, int(max_chars * 0.62))
    elif block.role in {"title", "cta"}:
        max_chars = max(8, int(max_chars * 0.82))
    else:
        max_chars = max(10, int(max_chars * 0.92))
    return {
        "max_chars": int(max_chars),
        "max_lines": int(max_lines),
        "box_width_pt": round(w, 1),
        "box_height_pt": round(h, 1),
        "font_size_pt": round(fs, 1),
    }




def font_supports_vietnamese(path: str) -> bool:
    if not path or not Path(path).exists():
        return False
    try:
        font = fitz.Font(fontfile=path)
        sample = "ăâđêôơưÁÀẢÃẠếệộợửữĐ✓•"
        return all(font.has_glyph(ord(ch)) for ch in sample if not ch.isspace())
    except Exception:
        return False

class FontResolver:
    """V8 font resolver: original if usable, otherwise visually similar VN fallback.

    The CXVIEW source PDF embeds subset Roboto fonts that do not contain
    Vietnamese glyphs, so v8 maps RobotoCondensed/Roboto-Black to condensed or
    bold Vietnamese fallback fonts while preserving original size/weight intent.
    """
    def __init__(
        self,
        regular_font: Optional[str] = None,
        bold_font: Optional[str] = None,
        title_font: Optional[str] = None,
        embedded_fonts: Optional[Dict[str, str]] = None,
        prefer_original_fonts: bool = True,
    ):
        self.regular_font = self._resolve_existing_font(regular_font) or self._windows_font("arial.ttf")
        self.bold_font = self._resolve_existing_font(bold_font) or self._windows_font("arialbd.ttf") or self.regular_font
        self.title_font = self._resolve_existing_font(title_font) or self._windows_font("arialbd.ttf") or self.bold_font
        self.embedded_fonts = embedded_fonts or {}
        self.prefer_original_fonts = prefer_original_fonts
        self._support_cache: Dict[str, bool] = {}
        self._font_cache: Dict[str, fitz.Font] = {}

        self.condensed_regular_font = self._discover_condensed_regular() or self.regular_font
        self.condensed_bold_font = self.title_font or self.bold_font
        self.medium_font = self._discover_variant(["Medium", "SemiBold", "Semibold"]) or self.bold_font
        self.black_font = self._discover_variant(["Black", "ExtraBold", "Bold"]) or self.condensed_bold_font

    @staticmethod
    def _resolve_existing_font(path: Optional[str]) -> Optional[str]:
        if path and Path(path).exists():
            return path
        return None

    @staticmethod
    def _windows_font(name: str) -> Optional[str]:
        p = Path("C:/Windows/Fonts") / name
        return str(p) if p.exists() else None

    def _discover_variant(self, style_names: List[str]) -> Optional[str]:
        candidates: List[Path] = []
        for base in [self.regular_font, self.bold_font, self.title_font]:
            if not base:
                continue
            p = Path(base)
            parent = p.parent
            stem = p.stem
            # Generate likely NotoSans / NotoSansCondensed variants.
            roots = [
                re.sub(r"(Regular|Bold|Medium|SemiBold|Semibold|Black)$", "", stem),
                "NotoSans-",
                "NotoSansCondensed-",
                "NotoSansDisplay-Condensed",
            ]
            for root in roots:
                for style in style_names:
                    candidates.append(parent / f"{root}{style}{p.suffix}")
                    candidates.append(parent / f"{root}-{style}{p.suffix}")
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def _discover_condensed_regular(self) -> Optional[str]:
        for base in [self.title_font, self.regular_font]:
            if not base:
                continue
            p = Path(base)
            candidates = [
                p.with_name(re.sub(r"Bold|SemiBold|Semibold|Black|Medium", "Regular", p.name)),
                p.with_name("NotoSansCondensed-Regular" + p.suffix),
                p.with_name("NotoSansDisplay-CondensedRegular" + p.suffix),
                p.with_name("NotoSansDisplay-Condensed-Regular" + p.suffix),
            ]
            for c in candidates:
                if c.exists():
                    return str(c)
        return None

    def _original_font_path_for(self, block: TextBlock) -> Optional[str]:
        clean = clean_font_name(block.main_font)
        return self.embedded_fonts.get(clean)

    def _supports_vietnamese_cached(self, path: str) -> bool:
        if path not in self._support_cache:
            self._support_cache[path] = font_supports_vietnamese(path)
        return self._support_cache[path]

    def _fallback_font_for_block(self, block: TextBlock) -> Optional[str]:
        name = clean_font_name(block.main_font).lower()
        if block.role == "title":
            return self.title_font or self.condensed_bold_font or self.bold_font
        if "black" in name or "heavy" in name:
            return self.black_font or self.condensed_bold_font or self.bold_font
        if "condensed" in name:
            if block.is_bold or "bold" in name or "medium" in name:
                return self.condensed_bold_font or self.bold_font
            return self.condensed_regular_font or self.regular_font
        if block.role == "label":
            return self.condensed_bold_font if block.is_bold else (self.condensed_regular_font or self.regular_font)
        if "medium" in name or block.is_bold:
            return self.medium_font or self.bold_font
        return self.regular_font

    def fontfile_for(self, block: TextBlock) -> Optional[str]:
        original = self._original_font_path_for(block)
        if self.prefer_original_fonts and original and self._supports_vietnamese_cached(original):
            return original
        return self._fallback_font_for_block(block)

    def fontname_for(self, block: TextBlock) -> str:
        original = self._original_font_path_for(block)
        if self.prefer_original_fonts and original and self._supports_vietnamese_cached(original):
            return safe_font_resource_name(block.main_font, prefix="FOrig")
        name = clean_font_name(block.main_font).lower()
        if block.role == "title":
            return "FTitleVN"
        if "condensed" in name and not block.is_bold:
            return "FCondensedRegularVN"
        if "condensed" in name or block.role == "label":
            return "FCondensedBoldVN" if block.is_bold else "FCondensedRegularVN"
        if block.is_bold or "medium" in name or "black" in name:
            return "FBoldVN"
        return "FRegularVN"

    def fitz_font_for(self, block: TextBlock) -> fitz.Font:
        fontfile = self.fontfile_for(block)
        key = self.fontname_for(block) + "|" + (fontfile or self.fallback_fontname if hasattr(self, "fallback_fontname") else "helv")
        if key in self._font_cache:
            return self._font_cache[key]
        font = fitz.Font(fontfile=fontfile) if fontfile else fitz.Font("helv")
        self._font_cache[key] = font
        return font


def compute_layout(text: str, block: TextBlock, rect: fitz.Rect, font: fitz.Font) -> LayoutResult:
    if rect.width <= 1 or rect.height <= 1:
        return LayoutResult([], block.font_size, 1.05, rect, block.align, False)

    original_size = max(5.0, block.font_size)
    if block.role == "title":
        min_scale = 0.78
        line_heights = [1.00, 0.96, 0.92]
    elif block.role == "label":
        min_scale = 0.72
        line_heights = [1.00, 0.96, 0.92, 0.88]
    elif block.role == "cta":
        min_scale = 0.72
        line_heights = [1.05, 1.0, 0.96]
    else:
        min_scale = 0.74
        line_heights = [1.12, 1.06, 1.0, 0.96]

    min_size = max(5.2, original_size * min_scale)
    # Prefer preserving original font size. Only shrink after all line-height
    # attempts fail. This keeps font-size/weight closer to source.
    size = original_size
    while size >= min_size:
        for lh in line_heights:
            lines = wrap_text_to_width(text, font, size, rect.width)
            needed_h = len(lines) * size * lh
            if needed_h <= rect.height + 0.5:
                return LayoutResult(lines, size, lh, rect, block.align, True)
        size -= 0.25

    lines = wrap_text_to_width(text, font, min_size, rect.width)
    return LayoutResult(lines, min_size, line_heights[-1], rect, block.align, False)



def find_symbol_font_file() -> Optional[str]:
    candidates = [
        os.getenv("PDF_TRANSLATOR_SYMBOL_FONT", ""),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
        "C:/Windows/Fonts/seguisym.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for c in candidates:
        if c and Path(c).exists():
            try:
                f = fitz.Font(fontfile=c)
                if f.has_glyph(ord("✓")):
                    return c
            except Exception:
                continue
    return None

_SYMBOL_FONT_FILE_CACHE: Optional[str] = None
_SYMBOL_FITZ_FONT_CACHE: Optional[fitz.Font] = None


def get_symbol_font() -> Tuple[Optional[str], Optional[fitz.Font]]:
    global _SYMBOL_FONT_FILE_CACHE, _SYMBOL_FITZ_FONT_CACHE
    if _SYMBOL_FONT_FILE_CACHE is None:
        _SYMBOL_FONT_FILE_CACHE = find_symbol_font_file() or ""
    if _SYMBOL_FONT_FILE_CACHE and _SYMBOL_FITZ_FONT_CACHE is None:
        try:
            _SYMBOL_FITZ_FONT_CACHE = fitz.Font(fontfile=_SYMBOL_FONT_FILE_CACHE)
        except Exception:
            _SYMBOL_FITZ_FONT_CACHE = None
    return (_SYMBOL_FONT_FILE_CACHE or None), _SYMBOL_FITZ_FONT_CACHE


def draw_layout(page: fitz.Page, layout: LayoutResult, block: TextBlock, resolver: FontResolver, color: RGB):
    if not layout.lines:
        return

    fontfile = resolver.fontfile_for(block)
    fontname = resolver.fontname_for(block)
    measure_font = resolver.fitz_font_for(block)
    symbol_file, symbol_font = get_symbol_font()

    x0, y0, x1, y1 = layout.rect
    y = y0 + layout.fontsize

    for line in layout.lines:
        if y > y1 + layout.fontsize:
            break

        draw_line = line
        leading_check = False
        if draw_line.startswith("✓ "):
            leading_check = True
            draw_line = draw_line[2:].lstrip()
        elif draw_line == "✓":
            leading_check = True
            draw_line = ""

        full_line_for_measure = ("✓ " + draw_line) if leading_check else draw_line
        if layout.align == "center":
            w = measure_font.text_length(full_line_for_measure.replace("✓", "•"), fontsize=layout.fontsize)
            x = x0 + max(0, (layout.rect.width - w) / 2)
        elif layout.align == "right":
            w = measure_font.text_length(full_line_for_measure.replace("✓", "•"), fontsize=layout.fontsize)
            x = x1 - w
        else:
            x = x0

        if leading_check:
            if symbol_file and symbol_font:
                page.insert_text(
                    point=fitz.Point(x, y),
                    text="✓",
                    fontsize=layout.fontsize,
                    fontname="FSymbolVN",
                    fontfile=symbol_file,
                    color=color,
                    overlay=True,
                )
                x += symbol_font.text_length("✓ ", fontsize=layout.fontsize) + 1.0
            else:
                # Last-resort symbol that most sans fonts support.
                page.insert_text(
                    point=fitz.Point(x, y),
                    text="•",
                    fontsize=layout.fontsize,
                    fontname=fontname,
                    fontfile=fontfile,
                    color=color,
                    overlay=True,
                )
                x += measure_font.text_length("• ", fontsize=layout.fontsize) + 1.0

        if draw_line:
            page.insert_text(
                point=fitz.Point(x, y),
                text=draw_line,
                fontsize=layout.fontsize,
                fontname=fontname,
                fontfile=fontfile,
                color=color,
                overlay=True,
            )
        y += layout.fontsize * layout.line_height


def render_translated_pdf(
    input_pdf: str,
    translated_ir: DocumentIR,
    output_pdf: str,
    regular_font: Optional[str] = None,
    bold_font: Optional[str] = None,
    title_font: Optional[str] = None,
    embedded_fonts: Optional[Dict[str, str]] = None,
    prefer_original_fonts: bool = True,
    cover_text: bool = True,
    sampled_background: bool = True,
    translate_headers_footers: bool = False,
    force_render: bool = False,
):
    pdf = fitz.open(input_pdf)
    resolver = FontResolver(
        regular_font=regular_font,
        bold_font=bold_font,
        title_font=title_font,
        embedded_fonts=embedded_fonts,
        prefer_original_fonts=prefer_original_fonts,
    )

    changed_count = skipped_same = skipped_role = skipped_unfit = 0

    for page_ir in translated_ir.pages:
        page = pdf[page_ir.page_index]
        for block in page_ir.blocks:
            if not is_translatable_block(block, translate_headers_footers):
                skipped_role += 1
                continue
            if not should_render_changed_translation(block, force_render=force_render):
                skipped_same += 1
                continue

            text = postprocess_translation(block.translated_text.strip(), block.original_text, block.role, "vi")
            rect = fitz.Rect(*expand_bbox(block.bbox, 0.35, page_ir.width, page_ir.height))
            font = resolver.fitz_font_for(block)
            layout = compute_layout(text, block, rect, font)

            # If not fit at a readable size, keep original instead of destroying
            # typography. Translation memory / glossary should make labels fit.
            min_safe_scale = 0.70 if block.role in {"label", "cta"} else 0.73
            if (not layout.fits or layout.fontsize / max(1.0, block.font_size) < min_safe_scale) and not force_render:
                skipped_unfit += 1
                continue

            if cover_text:
                cover_text_block(page, block, page_ir, sampled_background=sampled_background)
            color = int_color_to_rgb(block.color)
            draw_layout(page, layout, block, resolver, color)
            changed_count += 1

    pdf.save(output_pdf, garbage=4, deflate=True)
    pdf.close()
    print(f"Render summary: changed={changed_count}, skipped_same={skipped_same}, skipped_role={skipped_role}, skipped_unfit={skipped_unfit}")


# ==================== V9 PATCH ====================
ORIGINAL_POSTPROCESS_TRANSLATION = postprocess_translation

# ============================================================
# V9: stable terms and compact translation memory
# ============================================================

# These overrides intentionally prioritize layout-fit for slides.
# They are shorter than the full faithful translation when the bbox is small.
COMPACT_TRANSLATION_MEMORY_VI: Dict[str, str] = {
    # global headers
    "our solutions": "GIẢI PHÁP",
    "our process": "QUY TRÌNH",
    "platform & technology": "NỀN TẢNG & CÔNG NGHỆ",
    "about cxview": "GIỚI THIỆU CXVIEW",

    # pricing table
    "package": "Gói",
    "tier": "Cấp",
    "cameras": "Camera",
    "concurrent ai streams": "Luồng AI đồng thời",
    "active on ai models existed": "trên mô hình AI hiện có",
    "concurrent ai streams active on ai models existed": "Luồng AI đồng thời\ntrên mô hình AI hiện có",
    "monthly (usd)": "Hàng tháng\n(USD)",
    "24-month (usd)": "24 tháng\n(USD)",
    "36-month (usd)": "36 tháng\n(USD)",
    "concurrent ai stream ratio": "Tỷ lệ luồng AI đồng thời",
    "billing flexibility": "Linh hoạt thanh toán",

    # page 3 categories
    "ai camera in security": "AI CAMERA\nAN NINH",
    "ai camera in operation": "AI CAMERA\nVẬN HÀNH",
    "ai camera in safety": "AI CAMERA\nAN TOÀN",

    # page 3 solution labels - ultra short versions
    "camera tampering detection": "Can thiệp camera",
    "vehicle plate recognition": "Nhận dạng biển số",
    "patrol automatic report": "Báo cáo tuần tra",
    "unattended object detection": "Vật thể bỏ quên",
    "fence and wall climbing detection": "Leo rào/tường",
    "intrusion detection": "Xâm nhập",
    "product quality inspection": "Kiểm tra chất lượng",
    "customer demographic analysis": "Nhân khẩu học",
    "customer engagement detection": "Tương tác khách hàng",
    "smart workstation monitoring": "Giám sát trạm",
    "heat maps & route maps analysis": "Bản đồ nhiệt/lộ trình",
    "heat maps and route maps analysis": "Bản đồ nhiệt/lộ trình",
    "traffic counting": "Đếm lưu lượng",
    "automated product counting": "Đếm sản phẩm",
    "dwell time report": "Thời gian lưu lại",
    "table cleaning detection": "Dọn bàn",
    "falls and slips detection": "Té ngã/trượt",
    "unusual crowd detection": "Đám đông bất thường",
    "ppe and uniform detection": "PPE & đồng phục",
    "forklift and vehicle safety detection": "An toàn xe nâng",
    "smoke and fire detection": "Khói/lửa",
    "fighting detection": "Đánh nhau",

    # page 4 process
    "strategic understanding": "Thấu hiểu chiến lược",
    "tailored solution architecture": "Kiến trúc giải pháp tùy chỉnh",
    "pilot (poc) & validation": "Pilot (POC) & xác thực",
    "scale & sustain": "Mở rộng & duy trì",
    "book your live demo": "Đặt lịch\ndemo",
    "near-zero latency": "Độ trễ gần 0",
    "data sovereignty": "Chủ quyền dữ liệu",
    "cost efficiency": "Tối ưu chi phí",
    "bandwidth savings": "Tiết kiệm băng thông",

    # page 5
    "how cxview delivers the transformation": "CXVIEW triển khai chuyển đổi như thế nào",
    "why a transformation partner is essential": "Vì sao cần đối tác chuyển đổi",
    "core deployment benefits": "Lợi ích triển khai cốt lõi",
    "customer on - premise ai vision infrastructure": "Hạ tầng AI Vision tại chỗ",
    "customer on-premise ai vision infrastructure": "Hạ tầng AI Vision tại chỗ",
    "no camera replacement required when streams are available": "Không cần thay camera khi có luồng video",
    "on-premise processing supports latency, bandwidth control and data sovereignty": "Xử lý tại chỗ: độ trễ thấp, tiết kiệm băng thông, chủ quyền dữ liệu",
    "automated reports reduce manual debate with employees, vendors and site teams": "Báo cáo tự động giảm tranh luận với nhân viên, nhà cung cấp và đội ngũ tại địa điểm",
    "management receives objective before/after performance data for every site": "Ban quản lý có dữ liệu trước/sau khách quan cho từng địa điểm",
}


# V_USERFIX_HEADER_MEMORY
EXACT_TRANSLATION_MEMORY_VI.update({
    "how cxview delivers the transformation": "CXVIEW triển khai chuyển đổi như thế nào",
    "understand your goals, challenges and objectives.": "Hiểu mục tiêu, thách thức và mục tiêu của bạn.",
    "assess current systems & environment.": "Đánh giá hệ thống & môi trường hiện tại.",
})
COMPACT_TRANSLATION_MEMORY_VI.update({
    "how cxview delivers the transformation": "CXVIEW triển khai chuyển đổi như thế nào",
    "understand your goals, challenges and objectives.": "Hiểu mục tiêu & thách thức của bạn.",
    "assess current systems & environment.": "Đánh giá hệ thống & môi trường hiện tại.",
})



# V7 all-text translation memory for extractable headers / marketing copy.
EXACT_TRANSLATION_MEMORY_VI.update({
    "client challenges. cxview solutions. business impacts.": "Thách thức khách hàng. Giải pháp CXVIEW. Tác động kinh doanh.",
    "how cxview delivers the transformation": "CXVIEW triển khai chuyển đổi như thế nào",
    "cxview smart ai video analytics solution": "GIẢI PHÁP AI VIDEO ANALYTICS THÔNG MINH CXVIEW",
    "cxview gpt box & ai video analytics": "CXVIEW GPT Box & AI Video Analytics",
    "& ai video analytics": "& AI Video Analytics",
    "the new era of physical edge ai": "Kỷ nguyên mới của AI biên vật lý",
    "near-zero latency": "Độ trễ gần bằng 0",
    "data sovereignty": "Chủ quyền dữ liệu",
    "cost efficiency": "Tối ưu chi phí",
    "bandwidth savings": "Tiết kiệm băng thông",
})

COMPACT_TRANSLATION_MEMORY_VI.update({
    "client challenges. cxview solutions. business impacts.": "Thách thức khách hàng. Giải pháp CXVIEW. Tác động kinh doanh.",
    "how cxview delivers the transformation": "CXVIEW triển khai chuyển đổi như thế nào",
    "cxview smart ai video analytics solution": "GIẢI PHÁP AI VIDEO ANALYTICS THÔNG MINH CXVIEW",
    "the new era of physical edge ai": "Kỷ nguyên mới của AI biên vật lý",
    "near-zero latency": "Độ trễ gần 0",
    "data sovereignty": "Chủ quyền dữ liệu",
    "cost efficiency": "Tối ưu chi phí",
    "bandwidth savings": "Tiết kiệm băng thông",
})


# Some terms should remain in English because they are product/technical terms.
PROTECTED_TERMS = {
    "CXVIEW", "CXVIEW GPT Box", "GPT Box", "AI Video Analytics", "CCTV", "POS", "API",
    "RTSP", "ONVIF", "VMS", "ROI", "POC", "PPE", "USD", "AI Vision",
}

BAD_TO_GOOD = [
    ("\u00a0", " "),
    ("\u202f", " "),
    ("\u2007", " "),
    ("\u2009", " "),
    ("\u200b", ""),
    ("\ufeff", ""),
    ("\u00ad", ""),
    ("ﬁ", "fi"),
    ("ﬂ", "fl"),
    ("–", "-"),
    ("—", "-"),
    ("“", '"'),
    ("”", '"'),
    ("‘", "'"),
    ("’", "'"),
    ("minimum 24 tháng", "tối thiểu 24 tháng"),
    ("minimum 24-tháng", "tối thiểu 24 tháng"),
    ("CXVIEW mô hình AI", "mô hình AI của CXVIEW"),
    ("CXVIEW AI mô hình", "mô hình AI của CXVIEW"),
    ("máy ảnh", "camera"),
    ("Máy ảnh", "Camera"),
    ("hóa đơn", "thanh toán"),
    ("trang web", "địa điểm"),
    ("nhóm site", "đội ngũ tại địa điểm"),
    ("site teams", "đội ngũ tại địa điểm"),
    ("thiết bị cạnh", "thiết bị biên"),
    ("trí tuệ Edge", "trí tuệ biên"),
    ("Đấu tranh", "đánh nhau"),
    ("đấu tranh", "đánh nhau"),
    ("leo climbing", "leo"),
    ("vật thể không có người", "vật thể bỏ quên"),
    ("thao túng camera", "can thiệp camera"),
]


# ============================================================
# V9: text sanitation / quality utilities
# ============================================================

def sanitize_text_v9(text: str, original_text: str = "", role: str = "body") -> str:
    if not text:
        return ""
    t = str(text)
    for bad, good in BAD_TO_GOOD:
        t = t.replace(bad, good)

    # Convert checkmark glyphs extracted as ü only when they behave like list markers.
    t = re.sub(r"(^|\n|\s)[ü✓]\s*", lambda m: m.group(1) + "✓ ", t)

    # Remove duplicated source-English suffixes accidentally attached by the model or merged blocks.
    src = re.sub(r"\s+", " ", original_text or "").strip()
    if src and len(src) <= 120:
        t = re.sub(rf"\s*{re.escape(src)}\s*$", "", t, flags=re.I).strip()

    tails = [
        "OUR SOLUTIONS", "OUR PROCESS", "PLATFORM & TECHNOLOGY", "ABOUT CXVIEW",
        "CLIENT CHALLENGES", "CXVIEW SOLUTIONS", "BUSINESS IMPACTS",
    ]
    for tail in tails:
        t = re.sub(rf"\s+{re.escape(tail)}\s*$", "", t, flags=re.I).strip()

    # Remove pure English label leftovers if a Vietnamese phrase is already present.
    t = re.sub(r"GIẢI PHÁP\s+OUR\s+SOLUTIONS", "GIẢI PHÁP", t, flags=re.I)
    t = re.sub(r"QUY TRÌNH\s+OUR\s+PROCESS", "QUY TRÌNH", t, flags=re.I)
    t = re.sub(r"NỀN TẢNG\s*&\s*CÔNG NGHỆ\s+PLATFORM\s*&\s*TECHNOLOGY", "NỀN TẢNG & CÔNG NGHỆ", t, flags=re.I)

    # Stable spacing, preserving explicit newlines.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in t.splitlines()]
    t = "\n".join(line for line in lines if line != "")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    # Tiny labels should not end with sentence punctuation.
    if role in {"label", "title", "cta", "table", "table_cell", "header"}:
        t = t.rstrip(" .。")
    return t


def is_small_box(block: TextBlock) -> bool:
    return (
        getattr(block, "role", "body") in {"label", "table", "table_cell"}
        or block.font_size <= 8.5
        or (bbox_width(block.bbox) <= 150 and bbox_height(block.bbox) <= 45)
    )


def compact_override_for_block(block: TextBlock, target_lang: str = "vi") -> Optional[str]:
    if not target_lang.lower().startswith("vi"):
        return None

    key = normalize_translation_key(block.original_text)

    # V8: explicit memory always wins. This guarantees known visible English
    # strings are translated even if they are classified as body/logo/header.
    if key in COMPACT_TRANSLATION_MEMORY_VI:
        return COMPACT_TRANSLATION_MEMORY_VI[key]
    if key in EXACT_TRANSLATION_MEMORY_VI:
        return EXACT_TRANSLATION_MEMORY_VI[key]

    # Prefer compact memory for small boxes and slide/table labels.
    if is_small_box(block) and key in COMPACT_TRANSLATION_MEMORY_VI:
        return COMPACT_TRANSLATION_MEMORY_VI[key]

    # For headers/titles, use compact if available to avoid source-title tails.
    if block.role in {"title", "header", "label", "cta"} and key in COMPACT_TRANSLATION_MEMORY_VI:
        return COMPACT_TRANSLATION_MEMORY_VI[key]

    exact = exact_translation_override(block.original_text, target_lang)
    if exact:
        return exact
    return None


def has_english_leak(text: str, original_text: str = "", role: str = "body") -> bool:
    protect_terms = os.getenv("PDF_TRANSLATOR_PROTECT_TERMS", "1").strip().lower() not in {"0", "false", "no", "n"}

    if protect_terms:
        allowed = {"CXVIEW", "GPT", "Box", "AI", "Video", "Analytics", "CCTV", "POS", "API", "RTSP", "ONVIF", "VMS", "ROI", "POC", "PPE", "USD", "Basic", "Silver", "Gold"}
    else:
        allowed = {"CXVIEW", "GPT", "AI", "USD", "Basic", "Silver", "Gold"}

    words = re.findall(r"\b[A-Za-z][A-Za-z\-]{2,}\b", text)
    suspicious = [w for w in words if w not in allowed and w.upper() not in allowed]

    if role in {"label", "table", "table_cell", "title", "cta", "header", "footer"}:
        return bool(suspicious)

    if not protect_terms:
        return bool(suspicious)

    bad_words = {"minimum", "managed", "channel", "customer", "stream", "site", "deployment"}
    return any(w.lower() in bad_words for w in suspicious)


def postprocess_translation_v9(text: str, original_text: str, role: str = "body", target_lang: str = "vi") -> str:
    t = ORIGINAL_POSTPROCESS_TRANSLATION(text, original_text, role, target_lang)
    t = sanitize_text_v9(t, original_text, role)
    return t


# ============================================================
# V9: translate IR with compact memory and stricter LLM payload
# ============================================================

def translate_ir_v9(
    ir: DocumentIR,
    translator,
    source_lang: str = "auto",
    target_lang: str = "vi",
    glossary: Optional[Dict[str, str]] = None,
    batch_size: int = 20,
    translate_headers_footers: bool = False,
) -> DocumentIR:
    new_ir = copy.deepcopy(ir)
    # blocks = list(base.iter_translatable_blocks(new_ir, translate_headers_footers))
    blocks = list(iter_translatable_blocks(new_ir, translate_headers_footers))
    effective_glossary = merged_glossary_for_target(glossary, target_lang)

    pending: List[TextBlock] = []
    exact_count = 0
    for block in blocks:
        override = compact_override_for_block(block, target_lang)
        if override is not None:
            block.translated_text = postprocess_translation_v9(override, block.original_text, block.role, target_lang)
            exact_count += 1
        else:
            pending.append(block)

    if exact_count:
        print(f"      V9 compact/exact translation-memory hits: {exact_count}")

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        payload = []
        placeholder_maps: Dict[str, Dict[str, str]] = {}

        for block in batch:
            protected, mapping = protect_placeholders(sanitize_text_v9(block.original_text))
            placeholder_maps[block.id] = mapping
            hints = layout_hints_for_block(block)
            # Make constraints stricter for small boxes so the LLM does not over-translate.
            if is_small_box(block):
                hints["max_chars"] = max(4, min(int(hints.get("max_chars", 20)), int(bbox_width(block.bbox) / max(1, block.font_size * 0.72))))
                hints["max_lines"] = min(int(hints.get("max_lines", 2)), max(1, len(block.lines)))
            payload.append({
                "id": block.id,
                "role": "label" if is_small_box(block) else block.role,
                "text": protected,
                "source_text": block.original_text,
                "max_chars": hints.get("max_chars"),
                "max_lines": hints.get("max_lines"),
                "box_width_pt": hints.get("box_width_pt"),
                "box_height_pt": hints.get("box_height_pt"),
                "font_size_pt": hints.get("font_size_pt"),
                "instruction": (
                    "Translate this entire visible text into compact Vietnamese. "
                    "Return only the translated string for this block. No NBSP. "
                    "Do not leave English unless it is a brand/model name, number, currency, URL, email, or explicit placeholder."
                ),
            })

        translated = translator.translate_batch(payload, source_lang, target_lang, effective_glossary)
        translated_by_id = {item["id"]: item.get("translated", "") for item in translated}

        for block in batch:
            raw_translation = translated_by_id.get(block.id, block.original_text)
            restored = restore_placeholders(raw_translation, placeholder_maps.get(block.id, {})).strip()
            fixed = postprocess_translation_v9(restored, block.original_text, block.role, target_lang)

            # If the LLM output for a tiny label leaks English or is too long, prefer deterministic compact memory if available.
            override = compact_override_for_block(block, target_lang)
            if override and (is_small_box(block) or has_english_leak(fixed, block.original_text, block.role)):
                fixed = override
            block.translated_text = postprocess_translation_v9(fixed, block.original_text, block.role, target_lang)

    return new_ir


# ============================================================
# V9: no-fill redaction helpers
# ============================================================

def _rect_from_bbox(bbox) -> fitz.Rect:
    return fitz.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _expanded_rect(rect: fitz.Rect, pad: float, page_ir: PageIR) -> fitz.Rect:
    return fitz.Rect(
        max(0, rect.x0 - pad),
        max(0, rect.y0 - pad),
        min(page_ir.width, rect.x1 + pad),
        min(page_ir.height, rect.y1 + pad),
    )


def redaction_rects_for_block_v9(block: TextBlock, page_ir: PageIR) -> List[fitz.Rect]:
    by_span = os.getenv("PDF_TRANSLATOR_REDACT_BY_SPAN", "0") == "1"

    # Small padding is enough because redaction removes text objects, not background.
    role = getattr(block, "role", "body")
    if role == "title":
        pad = max(0.35, block.font_size * 0.025)
    elif is_small_box(block) or role in {"cta", "table", "table_cell"}:
        pad = max(0.25, block.font_size * 0.025)
    else:
        pad = max(0.35, block.font_size * 0.03)

    rects: List[fitz.Rect] = []
    for line in getattr(block, "lines", []) or []:
        if by_span and getattr(line, "spans", None):
            for span in line.spans:
                if not (getattr(span, "text", "") or "").strip():
                    continue
                r = _rect_from_bbox(span.bbox)
                if r.width > 0 and r.height > 0:
                    rects.append(_expanded_rect(r, pad, page_ir))
        else:
            if not (getattr(line, "text", "") or "").strip():
                continue
            r = _rect_from_bbox(line.bbox)
            if r.width > 0 and r.height > 0:
                rects.append(_expanded_rect(r, pad, page_ir))

    if not rects:
        rects.append(_expanded_rect(_rect_from_bbox(block.bbox), pad, page_ir))
    return rects


def add_no_fill_redactions(page: fitz.Page, rects: List[fitz.Rect]):
    for rect in rects:
        if rect.is_empty or rect.width <= 0 or rect.height <= 0:
            continue
        try:
            page.add_redact_annot(rect, fill=None, cross_out=False)
        except TypeError:
            page.add_redact_annot(rect, fill=None)


def apply_no_fill_redactions(page: fitz.Page):
    images_none = getattr(fitz, "PDF_REDACT_IMAGE_NONE", 0)
    graphics_none = getattr(fitz, "PDF_REDACT_LINE_ART_NONE", 0)
    text_remove = getattr(fitz, "PDF_REDACT_TEXT_REMOVE", 0)
    page.apply_redactions(images=images_none, graphics=graphics_none, text=text_remove)


# ============================================================
# V9: layout and candidate selection
# ============================================================

def original_line_count(block: TextBlock) -> int:
    return max(1, len([l for l in getattr(block, "lines", []) if (getattr(l, "text", "") or "").strip()]))


def role_policy_v9(block: TextBlock) -> Tuple[float, List[float], int]:
    role = "label" if is_small_box(block) else getattr(block, "role", "body")
    text_key = normalize_translation_key(getattr(block, "original_text", ""))

    # Marketing callouts often need to preserve a 2-line emphasis structure.
    if "we promise results" in text_key or "within 30 days" in text_key:
        return 0.80, [0.98, 0.94, 0.90], 1

    if role == "title":
        return 0.84, [1.00, 0.96, 0.92], 1
    if role in {"label", "table", "table_cell"}:
        return 0.74, [0.98, 0.94, 0.90, 0.86], 1
    if role == "cta":
        return 0.80, [1.02, 0.98, 0.94], 1
    return 0.78, [1.08, 1.02, 0.98, 0.94], 2


def compute_layout_v9(text: str, block: TextBlock, rect: fitz.Rect, font: fitz.Font) -> LayoutResult:
    if rect.width <= 1 or rect.height <= 1:
        return LayoutResult([], block.font_size, 1.0, rect, block.align, False)

    original_size = max(4.8, float(block.font_size or 8.0))
    min_scale, line_heights, extra_lines = role_policy_v9(block)
    min_size = max(4.6, original_size * min_scale)
    max_lines_allowed = original_line_count(block) + extra_lines
    role = "label" if is_small_box(block) else getattr(block, "role", "body")

    size = original_size
    while size >= min_size:
        for lh in line_heights:
            lines = wrap_text_to_width(text, font, size, rect.width)
            needed_h = len(lines) * size * lh
            # Body text may need more lines in Vietnamese; labels must stay compact.
            if role == "body":
                line_count_ok = len(lines) <= max(max_lines_allowed, int(rect.height / max(1, size * lh)))
            else:
                line_count_ok = len(lines) <= max_lines_allowed
            if line_count_ok and needed_h <= rect.height + 0.35:
                return LayoutResult(lines, size, lh, rect, block.align, True)
        size -= 0.20

    lines = wrap_text_to_width(text, font, min_size, rect.width)
    return LayoutResult(lines, min_size, line_heights[-1], rect, block.align, False)


def ultra_compact_label(text: str) -> str:
    t = sanitize_text_v9(text)
    replacements = [
        ("Phát hiện ", ""),
        ("Báo cáo ", ""),
        ("Giám sát ", ""),
        ("Phân tích ", ""),
        ("Kiểm tra ", ""),
        ("Nhận dạng ", ""),
        ("tự động", ""),
        ("khách hàng", ""),
        ("thời gian ", ""),
        ("sản phẩm", "SP"),
        ("camera", "camera"),
    ]
    for a, b in replacements:
        t = t.replace(a, b)
    t = re.sub(r"\s+", " ", t).strip(" -")
    return t


def translation_candidates_for_block(block: TextBlock, translated: str) -> List[str]:
    candidates: List[str] = []

    def add(x: Optional[str]):
        if not x:
            return
        x = sanitize_text_v9(x, block.original_text, block.role)
        if x and x not in candidates:
            candidates.append(x)

    # Small boxes should try compact TM first.
    add(compact_override_for_block(block, "vi"))
    add(translated)

    # If the base exact translation is longer than the compact override, keep it as second fallback.
    exact = exact_translation_override(block.original_text, "vi")
    add(exact)

    if is_small_box(block) or block.role in {"label", "cta", "title"}:
        for c in list(candidates):
            add(ultra_compact_label(c))

    # Final deterministic fixes for common headers.
    key = normalize_translation_key(block.original_text)
    if key in COMPACT_TRANSLATION_MEMORY_VI:
        add(COMPACT_TRANSLATION_MEMORY_VI[key])

    return candidates


def choose_layout_for_block(block: TextBlock, text: str, rect: fitz.Rect, font: fitz.Font) -> Tuple[Optional[LayoutResult], str, bool]:
    """Return layout, chosen_text, used_fallback_unfit."""
    candidates = translation_candidates_for_block(block, text)
    best_unfit: Optional[Tuple[LayoutResult, str]] = None

    for cand in candidates:
        layout = compute_layout_v9(cand, block, rect, font)
        if layout.fits:
            return layout, cand, False
        if best_unfit is None:
            best_unfit = (layout, cand)

    # For tiny labels/table cells, draw the most compact candidate even if not perfect; better than leaving English.
    if best_unfit and (is_small_box(block) or block.role in {"label", "table", "table_cell", "cta"}):
        return best_unfit[0], best_unfit[1], True

    return (best_unfit[0], best_unfit[1], True) if best_unfit else (None, "", True)


# ============================================================
# V9 render
# ============================================================

def render_translated_pdf_v9(
    input_pdf: str,
    translated_ir: DocumentIR,
    output_pdf: str,
    regular_font: Optional[str] = None,
    bold_font: Optional[str] = None,
    title_font: Optional[str] = None,
    embedded_fonts: Optional[Dict[str, str]] = None,
    prefer_original_fonts: bool = True,
    cover_text: bool = True,
    sampled_background: bool = True,  # compatibility; no-fill redaction does not use sampled fill
    translate_headers_footers: bool = False,
    force_render: bool = False,
):
    pdf = fitz.open(input_pdf)
    resolver = FontResolver(
        regular_font=regular_font,
        bold_font=bold_font,
        title_font=title_font,
        embedded_fonts=embedded_fonts,
        prefer_original_fonts=prefer_original_fonts,
    )

    force_unfit = os.getenv("PDF_TRANSLATOR_FORCE_UNFIT", "0") == "1"
    render_small_unfit = os.getenv("PDF_TRANSLATOR_RENDER_SMALL_UNFIT", "1") != "0"

    rendered_count = 0
    redacted_count = 0
    skipped_role = 0
    skipped_no_translation = 0
    skipped_same = 0
    skipped_unfit = 0
    drawn_unfit = 0
    compact_used = 0

    for page_ir in translated_ir.pages:
        page = pdf[page_ir.page_index]
        draw_jobs = []
        redaction_rects: List[fitz.Rect] = []

        for block in page_ir.blocks:
            if not is_translatable_block(block, translate_headers_footers):
                skipped_role += 1
                continue

            if not (block.translated_text and block.translated_text.strip()):
                skipped_no_translation += 1
                continue

            if text_is_same(block.original_text, block.translated_text) and not force_render:
                key_same = normalize_translation_key(block.original_text)
                forced_override = compact_override_for_block(block, "vi")
                translate_all_mode = os.getenv("PDF_TRANSLATOR_TRANSLATE_ALL_TEXT", "1").strip().lower() not in {"0", "false", "no", "n"}

                if translate_all_mode and forced_override and not text_is_same(block.original_text, forced_override):
                    block.translated_text = forced_override
                else:
                    skipped_same += 1
                    continue

            raw_text = postprocess_translation_v9(block.translated_text.strip(), block.original_text, block.role, "vi")
            if not raw_text:
                skipped_no_translation += 1
                continue

            rect = fitz.Rect(*expand_bbox(block.bbox, 0.18, page_ir.width, page_ir.height))
            font = resolver.fitz_font_for(block)
            layout, chosen_text, unfit = choose_layout_for_block(block, raw_text, rect, font)

            if layout is None or not chosen_text:
                skipped_unfit += 1
                continue

            # Skip unsafe body blocks rather than destroying the slide. Small labels get compact fallback.
            translate_all_mode = os.getenv("PDF_TRANSLATOR_TRANSLATE_ALL_TEXT", "1").strip().lower() not in {"0", "false", "no", "n"}
            render_unfit_alltext = os.getenv("PDF_TRANSLATOR_RENDER_UNFIT_ALLTEXT", "1").strip().lower() not in {"0", "false", "no", "n"}

            if unfit and not (
                force_unfit
                or force_render
                or (render_small_unfit and is_small_box(block))
                or (translate_all_mode and render_unfit_alltext)
            ):
                skipped_unfit += 1
                continue

            if chosen_text != raw_text:
                compact_used += 1

            # Mutate the layout lines to match the selected text if candidate changed.
            if chosen_text != raw_text:
                layout = compute_layout_v9(chosen_text, block, rect, font)

            if cover_text:
                redaction_rects.extend(redaction_rects_for_block_v9(block, page_ir))

            color = int_color_to_rgb(block.color)
            draw_jobs.append((layout, block, color))
            if unfit:
                drawn_unfit += 1

        if cover_text and redaction_rects:
            add_no_fill_redactions(page, redaction_rects)
            apply_no_fill_redactions(page)
            redacted_count += len(redaction_rects)

        for layout, block, color in draw_jobs:
            draw_layout(page, layout, block, resolver, color)
            rendered_count += 1

    pdf.save(output_pdf, garbage=4, deflate=True)
    pdf.close()

    print("Render summary v9:")
    print(f"  rendered={rendered_count}")
    print(f"  redaction_rects={redacted_count}")
    print(f"  compact_candidates_used={compact_used}")
    print(f"  drawn_unfit_small={drawn_unfit}")
    print(f"  skipped_role={skipped_role}")
    print(f"  skipped_same={skipped_same}")
    print(f"  skipped_no_translation={skipped_no_translation}")
    print(f"  skipped_unfit={skipped_unfit}")





# ============================================================
# Anthropic / OpenAI-compatible provider-aware translator
# ============================================================

ANTHROPIC_PROVIDER_NAMES = {"anthropic", "claude", "claude_native"}
_OriginalOpenAICompatibleTranslator = OpenAICompatibleTranslator


def _normalize_provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "openai_compatible").strip().lower()


def _is_anthropic_provider() -> bool:
    return _normalize_provider_name() in ANTHROPIC_PROVIDER_NAMES


def _prepare_anthropic_env() -> None:
    """
    Let users use either LLM_API_KEY or ANTHROPIC_API_KEY.
    """
    if not _is_anthropic_provider():
        return

    if not os.getenv("LLM_API_KEY") and os.getenv("ANTHROPIC_API_KEY"):
        os.environ["LLM_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")

    if not os.getenv("LLM_BASE_URL"):
        os.environ["LLM_BASE_URL"] = "https://api.anthropic.com"

    os.environ["LLM_BASE_URL"] = os.getenv("LLM_BASE_URL", "https://api.anthropic.com").strip().rstrip("/")


class ProviderAwareTranslator(_OriginalOpenAICompatibleTranslator):
    """
    Provider-aware translator.

    If LLM_PROVIDER=anthropic:
        Uses Anthropic native Messages API:
        POST {LLM_BASE_URL}/v1/messages

    Otherwise:
        Falls back to the original OpenAI-compatible /chat/completions flow.
    """

    def __init__(self, temperature: float = 0.1, timeout: int = 120, max_retries: int = 3):
        _prepare_anthropic_env()
        self.provider = _normalize_provider_name()

        if self.provider not in ANTHROPIC_PROVIDER_NAMES:
            super().__init__(temperature=temperature, timeout=timeout, max_retries=max_retries)
            return

        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.anthropic.com").strip().rstrip("/")
        self.model = os.getenv("LLM_MODEL", "").strip()
        self.temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))
        self.timeout = int(os.getenv("LLM_TIMEOUT", str(timeout)))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", str(max_retries)))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        self.anthropic_version = os.getenv("ANTHROPIC_VERSION", "2023-06-01").strip()

        missing = [k for k, v in {
            "LLM_API_KEY": self.api_key,
            "LLM_BASE_URL": self.base_url,
            "LLM_MODEL": self.model,
        }.items() if not v]
        if missing:
            raise ValueError(f"Missing Anthropic LLM env vars: {', '.join(missing)}")

    def _chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        if getattr(self, "provider", "") not in ANTHROPIC_PROVIDER_NAMES:
            return super()._chat_completion(system_prompt, user_prompt)

        return self._anthropic_messages_completion(system_prompt, user_prompt)

    def _anthropic_messages_completion(self, system_prompt: str, user_prompt: str) -> str:
        url = self.base_url.rstrip("/") + "/v1/messages"

        # Important:
        # Newer Anthropic models such as claude-opus-4-8 can reject `temperature`.
        # Therefore the field is not sent by default. Enable only when needed:
        #   LLM_SEND_TEMPERATURE=1
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        }

        if os.getenv("LLM_SEND_TEMPERATURE", "0").strip().lower() in {"1", "true", "yes", "y"}:
            body["temperature"] = self.temperature

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
            "accept": "application/json",
            "user-agent": os.getenv("LLM_USER_AGENT", "pdf-translator-mikotech/1.1 anthropic-native"),
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_text = response.read().decode("utf-8", errors="replace")

                parsed = json.loads(response_text)
                return self._extract_anthropic_text(parsed)

            except urllib.error.HTTPError as e:
                body_text = e.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"Anthropic HTTP {e.code}: {body_text[:1200]}")

                if e.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break

                retry_after = e.headers.get("retry-after")
                if retry_after:
                    try:
                        sleep_for = float(retry_after)
                    except ValueError:
                        sleep_for = min(2 ** attempt, 20)
                else:
                    sleep_for = min(2 ** attempt, 20)

                time.sleep(sleep_for)

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 20))

        raise RuntimeError(f"Anthropic translation failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _extract_anthropic_text(parsed: dict) -> str:
        content = parsed.get("content", [])
        parts = []

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))

        text = "\n".join(p for p in parts if p).strip()
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```\s*$", "", text).strip()

        if not text:
            raise ValueError(f"Anthropic response did not contain text content: {parsed}")

        return text


# Replace original translator with provider-aware translator.
OpenAICompatibleTranslator = ProviderAwareTranslator



# ============================================================
# Apply V9 patch in this single-file build
# ============================================================

postprocess_translation = postprocess_translation_v9
translate_ir = translate_ir_v9
compute_layout = compute_layout_v9
render_translated_pdf = render_translated_pdf_v9


def _print_provider_info_once():
    provider = _normalize_provider_name()
    if provider in ANTHROPIC_PROVIDER_NAMES:
        print(
            "Translator provider: Anthropic native Messages API | "
            f"model={os.getenv('LLM_MODEL')} | base_url={os.getenv('LLM_BASE_URL', 'https://api.anthropic.com')}"
        )
    elif os.getenv("LLM_API_KEY") and os.getenv("LLM_BASE_URL") and os.getenv("LLM_MODEL"):
        print(
            "Translator provider: OpenAI-compatible | "
            f"model={os.getenv('LLM_MODEL')} | base_url={os.getenv('LLM_BASE_URL')}"
        )



# ============================================================
# V6 MERGED - span-aware weight detection + one-file CLI
# ============================================================

WEIGHT_THIN = 100
WEIGHT_LIGHT = 300
WEIGHT_REGULAR = 400
WEIGHT_MEDIUM = 500
WEIGHT_SEMIBOLD = 600
WEIGHT_BOLD = 700
WEIGHT_BLACK = 900


def detect_font_weight(font_name: str, flags: int) -> int:
    """Return CSS-like weight (100-900) instead of only bool bold."""
    name = str(font_name or "").lower()

    # Order matters: most specific first.
    if any(k in name for k in ["black", "heavy", "extrabold", "extra-bold"]):
        return WEIGHT_BLACK
    if any(k in name for k in ["semibold", "semi-bold", "demibold", "demi-bold"]):
        return WEIGHT_SEMIBOLD
    if "medium" in name:
        return WEIGHT_MEDIUM
    if "bold" in name:
        return WEIGHT_BOLD
    if any(k in name for k in ["light", "thin"]):
        return WEIGHT_LIGHT

    # PyMuPDF flag bit 16 usually means bold.
    if int(flags or 0) & 16:
        return WEIGHT_BOLD

    return WEIGHT_REGULAR


def detect_span_style_v6(span: dict) -> Tuple[bool, bool, int]:
    """Return (is_bold, is_italic, weight)."""
    font = str(span.get("font", ""))
    flags = int(span.get("flags", 0) or 0)
    weight = detect_font_weight(font, flags)
    is_bold = weight >= WEIGHT_SEMIBOLD
    is_italic = bool(flags & 2) or any(k in font.lower() for k in ["italic", "oblique"])
    return is_bold, is_italic, weight


def _dominant_weight_of_block(block: TextBlock) -> int:
    """Most visually influential font weight across spans, weighted by char count."""
    weights = []
    for line in block.lines:
        for span in line.spans:
            w = int(getattr(span, "weight", WEIGHT_REGULAR) or WEIGHT_REGULAR)
            char_count = len((span.text or "").strip())
            weights.extend([w] * max(1, char_count))
    if not weights:
        return WEIGHT_REGULAR
    return most_common(weights)


def parse_pdf_to_ir_v6(source_pdf: str) -> DocumentIR:
    """Parser with per-span weight capture.

    Note: weights are attached as dynamic attributes on TextSpan/TextBlock during
    the live run. They are not serialized by dataclasses.asdict unless you later
    add weight fields to the dataclasses.
    """
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

            visual_lines: List[TextLine] = []
            for raw_line in raw_block.get("lines", []):
                spans: List[TextSpan] = []

                for raw_span in raw_line.get("spans", []):
                    chars = raw_span.get("chars", [])
                    if chars:
                        txt = "".join(ch.get("c", "") for ch in chars)
                    else:
                        txt = raw_span.get("text", "")

                    txt = normalize_special_chars(txt)
                    if not txt or not txt.strip():
                        continue

                    is_bold, is_italic, weight = detect_span_style_v6(raw_span)
                    bbox = tuple(float(v) for v in raw_span.get("bbox", (0, 0, 0, 0)))

                    span = TextSpan(
                        text=txt,
                        bbox=bbox,  # type: ignore
                        font=str(raw_span.get("font", "")),
                        size=float(raw_span.get("size", 10.0)),
                        color=int(raw_span.get("color", 0) or 0),
                        flags=int(raw_span.get("flags", 0) or 0),
                        is_bold=is_bold,
                        is_italic=is_italic,
                    )
                    setattr(span, "weight", weight)
                    spans.append(span)

                if not spans:
                    continue

                line_bbox = tuple(float(v) for v in raw_line.get("bbox", rect_union(s.bbox for s in spans)))
                line = TextLine(bbox=line_bbox, spans=spans)  # type: ignore

                for island in split_line_into_text_islands(line):
                    if island.text:
                        visual_lines.append(island)

            if not visual_lines:
                continue

            if should_split_raw_text_block(visual_lines):
                for one_line in visual_lines:
                    split_block = make_block_from_lines(
                        page_index=page_index,
                        block_id=f"p{page_index}_b{block_idx}",
                        order=block_idx,
                        lines=[one_line],
                    )
                    split_block.align = estimate_alignment(split_block, page_ir)
                    split_block.role = classify_block_role(split_block, page_ir)
                    setattr(split_block, "dominant_weight", _dominant_weight_of_block(split_block))
                    page_ir.blocks.append(split_block)
                    block_idx += 1
                continue

            block_bbox = rect_union(l.bbox for l in visual_lines)
            original_text = normalize_text("\n".join(line.text for line in visual_lines))
            if not original_text:
                continue

            block = TextBlock(
                id=f"p{page_index}_b{block_idx}",
                page_index=page_index,
                bbox=block_bbox,  # type: ignore
                lines=visual_lines,
                order=block_idx,
                original_text=original_text,
            )
            block.align = estimate_alignment(block, page_ir)
            block.role = classify_block_role(block, page_ir)
            setattr(block, "dominant_weight", _dominant_weight_of_block(block))
            page_ir.blocks.append(block)
            block_idx += 1

        page_ir.blocks = sort_blocks_reading_order(page_ir.blocks)
        for i, block in enumerate(page_ir.blocks):
            block.order = i
        ir.pages.append(page_ir)

    pdf.close()
    return ir


def font_supports_text_v6(path: Optional[str], text: str) -> bool:
    """Check the exact text, not only a generic Vietnamese probe."""
    if not path or not Path(path).exists():
        return False
    try:
        font = fitz.Font(fontfile=path)
        for ch in text or "":
            if ch.isspace():
                continue
            if ord(ch) in {0x00AD, 0x200B, 0x200C, 0x200D, 0xFEFF}:
                continue
            if not font.has_glyph(ord(ch)):
                return False
        return True
    except Exception:
        return False


class FontResolverV6:
    """Weight-aware font resolver for the merged single-file build.

    This is still block-level rendering after translation. It uses span-level
    weights to choose the dominant block font. True per-span translated rendering
    would require segment-preserving translation, which is a different pipeline.
    """

    def __init__(
        self,
        regular_font: Optional[str] = None,
        bold_font: Optional[str] = None,
        title_font: Optional[str] = None,
        condensed_font: Optional[str] = None,
        condensed_bold_font: Optional[str] = None,
        condensed_semibold_font: Optional[str] = None,
        medium_font: Optional[str] = None,
        semibold_font: Optional[str] = None,
        black_font: Optional[str] = None,
        symbol_font: Optional[str] = None,
        embedded_fonts: Optional[Dict[str, str]] = None,
        prefer_original_fonts: bool = True,
    ):
        def resolve(p: Optional[str]) -> Optional[str]:
            if p and Path(p).exists():
                return str(Path(p))
            return None

        self.regular_font = resolve(regular_font) or self._windows("arial.ttf")
        self.bold_font = resolve(bold_font) or self._windows("arialbd.ttf") or self.regular_font
        self.title_font = resolve(title_font) or self.bold_font
        self.condensed_regular_font = resolve(condensed_font) or self._discover_condensed_regular() or self.regular_font
        self.condensed_bold_font = resolve(condensed_bold_font) or self.title_font or self.bold_font
        self.condensed_semibold_font = resolve(condensed_semibold_font) or self._discover_condensed_semibold() or self.condensed_bold_font
        self.medium_font = resolve(medium_font) or self._discover_variant(["Medium"]) or self.regular_font
        self.semibold_font = resolve(semibold_font) or self._discover_variant(["SemiBold", "Semibold", "DemiBold"]) or self.bold_font
        self.black_font = resolve(black_font) or self._discover_variant(["Black", "ExtraBold", "Heavy"]) or self.bold_font
        self.symbol_font = resolve(symbol_font) or self._windows("seguisym.ttf") or self.regular_font

        self.embedded_fonts = embedded_fonts or {}
        self.prefer_original_fonts = prefer_original_fonts
        self._support_cache: Dict[Tuple[str, str], bool] = {}
        self._font_cache: Dict[str, fitz.Font] = {}
        self._scale_cache: Dict[Tuple[str, str, str], float] = {}

        if self.symbol_font:
            os.environ["PDF_TRANSLATOR_SYMBOL_FONT"] = self.symbol_font

        print("FontResolverV6 merged:")
        print(f"  regular={self.regular_font}")
        print(f"  medium={self.medium_font}")
        print(f"  semibold={self.semibold_font}")
        print(f"  bold={self.bold_font}")
        print(f"  black={self.black_font}")
        print(f"  title={self.title_font}")
        print(f"  condensed={self.condensed_regular_font}")
        print(f"  condensed_semibold={self.condensed_semibold_font}")
        print(f"  condensed_bold={self.condensed_bold_font}")
        print(f"  symbol={self.symbol_font}")

    @staticmethod
    def _windows(name: str) -> Optional[str]:
        p = Path("C:/Windows/Fonts") / name
        return str(p) if p.exists() else None

    def _discover_variant(self, style_names: List[str]) -> Optional[str]:
        candidates: List[Path] = []
        for base_font in [self.regular_font, self.bold_font, self.title_font]:
            if not base_font:
                continue
            p = Path(base_font)
            parent, suffix, stem = p.parent, p.suffix, p.stem
            roots = [
                re.sub(r"(Regular|Bold|Medium|SemiBold|Semibold|Black|ExtraBold|Heavy)$", "", stem),
                re.sub(r"[-_ ]?(Regular|Bold|Medium|SemiBold|Semibold|Black|ExtraBold|Heavy)$", "", stem),
                "NotoSans-",
                "NotoSansCondensed-",
                "Roboto-",
                "RobotoCondensed-",
            ]
            for root in roots:
                for style in style_names:
                    candidates.append(parent / f"{root}{style}{suffix}")
                    candidates.append(parent / f"{root}-{style}{suffix}")
                    candidates.append(parent / f"{root}_{style}{suffix}")
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def _discover_condensed_regular(self) -> Optional[str]:
        for base_font in [self.title_font, self.regular_font, self.bold_font]:
            if not base_font:
                continue
            p = Path(base_font)
            for name in ["NotoSansCondensed-Regular", "NotoSansCondensed", "RobotoCondensed-Regular", "RobotoCondensed"]:
                c = p.parent / f"{name}{p.suffix}"
                if c.exists():
                    return str(c)
            c = p.with_name(re.sub(r"Bold|SemiBold|Semibold|Black|Medium|ExtraBold|Heavy", "Regular", p.name))
            if c.exists():
                return str(c)
        return None

    def _discover_condensed_semibold(self) -> Optional[str]:
        for base_font in [self.condensed_bold_font, self.condensed_regular_font, self.title_font]:
            if not base_font:
                continue
            p = Path(base_font)
            for name in ["NotoSansCondensed-SemiBold", "NotoSansCondensed-Semibold", "NotoSansCondensed-Medium", "RobotoCondensed-SemiBold", "RobotoCondensed-Medium"]:
                c = p.parent / f"{name}{p.suffix}"
                if c.exists():
                    return str(c)
        return None

    def _supports_text_cached(self, path: Optional[str], text: str) -> bool:
        if not path:
            return False
        key = (path, text or "")
        if key not in self._support_cache:
            self._support_cache[key] = font_supports_text_v6(path, text or "")
        return self._support_cache[key]

    def _original_font_path_for(self, block: TextBlock) -> Optional[str]:
        clean = clean_font_name(block.main_font)
        return self.embedded_fonts.get(clean)

    def _is_condensed_context(self, block: TextBlock) -> bool:
        name = clean_font_name(block.main_font).lower()
        return (
            "condensed" in name
            or "narrow" in name
            or block.role in {"label", "table", "table_cell"}
            or is_small_box(block)
        )

    def _weight_to_font(self, weight: int, condensed: bool = False) -> Optional[str]:
        if condensed:
            if weight >= WEIGHT_BOLD:
                return self.condensed_bold_font or self.bold_font
            if weight >= WEIGHT_SEMIBOLD:
                return self.condensed_semibold_font or self.condensed_bold_font or self.semibold_font
            if weight >= WEIGHT_MEDIUM:
                return self.condensed_semibold_font or self.condensed_regular_font or self.medium_font
            return self.condensed_regular_font or self.regular_font

        if weight >= WEIGHT_BLACK:
            return self.black_font or self.bold_font
        if weight >= WEIGHT_BOLD:
            return self.bold_font
        if weight >= WEIGHT_SEMIBOLD:
            return self.semibold_font or self.bold_font
        if weight >= WEIGHT_MEDIUM:
            return self.medium_font or self.regular_font
        return self.regular_font

    def fontfile_for(self, block: TextBlock, text: Optional[str] = None) -> Optional[str]:
        render_text = text or getattr(block, "_selected_text_for_font", None) or block.translated_text or block.original_text or ""
        original = self._original_font_path_for(block)
        if self.prefer_original_fonts and original and self._supports_text_cached(original, render_text):
            return original

        weight = int(getattr(block, "dominant_weight", WEIGHT_REGULAR) or WEIGHT_REGULAR)
        condensed = self._is_condensed_context(block)
        candidate_order = [
            self._weight_to_font(weight, condensed=condensed),
            self._weight_to_font(weight, condensed=False),
            self.regular_font,
            self.bold_font,
        ]
        for candidate in candidate_order:
            if candidate and self._supports_text_cached(candidate, render_text):
                return candidate
        return self.regular_font or self.bold_font

    def fontname_for(self, block: TextBlock, text: Optional[str] = None) -> str:
        render_text = text or getattr(block, "_selected_text_for_font", None) or block.translated_text or block.original_text or ""
        original = self._original_font_path_for(block)
        if self.prefer_original_fonts and original and self._supports_text_cached(original, render_text):
            return safe_font_resource_name(block.main_font, prefix="FOrig")

        chosen = self.fontfile_for(block, render_text)
        weight = int(getattr(block, "dominant_weight", WEIGHT_REGULAR) or WEIGHT_REGULAR)
        condensed = self._is_condensed_context(block)

        if chosen == self.symbol_font:
            return "FSymbolVN"
        if condensed:
            if chosen == self.condensed_bold_font or weight >= WEIGHT_BOLD:
                return "FCondensedBoldVN"
            if chosen == self.condensed_semibold_font or weight >= WEIGHT_MEDIUM:
                return "FCondensedSemiBoldVN"
            return "FCondensedRegularVN"

        if chosen == self.black_font or weight >= WEIGHT_BLACK:
            return "FBlackVN"
        if chosen == self.bold_font or weight >= WEIGHT_BOLD:
            return "FBoldVN"
        if chosen == self.semibold_font or weight >= WEIGHT_SEMIBOLD:
            return "FSemiBoldVN"
        if chosen == self.medium_font or weight >= WEIGHT_MEDIUM:
            return "FMediumVN"
        return "FRegularVN"

    def fitz_font_for(self, block: TextBlock, text: Optional[str] = None) -> fitz.Font:
        fontfile = self.fontfile_for(block, text)
        key = self.fontname_for(block, text) + "|" + (fontfile or "helv")
        if key in self._font_cache:
            return self._font_cache[key]
        font = fitz.Font(fontfile=fontfile) if fontfile else fitz.Font("helv")
        self._font_cache[key] = font
        return font

    def font_size_scale_for(self, block: TextBlock, text: Optional[str] = None) -> float:
        render_text = sanitize_text_v9(text or getattr(block, "_selected_text_for_font", None) or block.translated_text or block.original_text or "")
        fallback_path = self.fontfile_for(block, render_text)
        original_path = self._original_font_path_for(block)

        if not fallback_path or not original_path or fallback_path == original_path:
            base_scale = 1.0
        else:
            key = (clean_font_name(block.main_font), fallback_path, sanitize_text_v9(block.original_text or "")[:80])
            if key in self._scale_cache:
                base_scale = self._scale_cache[key]
            else:
                try:
                    source_font = fitz.Font(fontfile=original_path)
                    fallback_font = self.fitz_font_for(block, render_text)
                    sample = sanitize_text_v9(block.original_text or "")
                    sample = re.sub(r"\s+", " ", sample).strip()
                    if len(sample) < 4:
                        sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
                    source_w = max(0.01, source_font.text_length(sample, fontsize=1.0))
                    fallback_w = max(0.01, fallback_font.text_length(sample, fontsize=1.0))
                    raw_scale = source_w / fallback_w

                    if self._is_condensed_context(block):
                        lo, hi = 0.86, 1.10
                    elif getattr(block, "role", "body") == "title":
                        lo, hi = 0.88, 1.06
                    else:
                        lo, hi = 0.88, 1.04
                    base_scale = max(lo, min(hi, raw_scale))
                except Exception:
                    base_scale = 1.0
                self._scale_cache[key] = base_scale

        # Role-based tuning knobs.
        role = "label" if is_small_box(block) else getattr(block, "role", "body")
        global_scale = float(os.getenv("PDF_TRANSLATOR_SIZE_SCALE_GLOBAL", "1.0"))
        if role in {"label", "table", "table_cell"}:
            role_scale = float(os.getenv("PDF_TRANSLATOR_SIZE_SCALE_LABEL", "1.00"))
        elif role == "title":
            role_scale = float(os.getenv("PDF_TRANSLATOR_SIZE_SCALE_TITLE", "1.00"))
        elif role == "cta":
            role_scale = float(os.getenv("PDF_TRANSLATOR_SIZE_SCALE_CTA", "1.00"))
        else:
            role_scale = float(os.getenv("PDF_TRANSLATOR_SIZE_SCALE_BODY", "0.96"))

        scale = base_scale * role_scale * global_scale
        if role in {"label", "table", "table_cell"}:
            return max(0.84, min(1.12, scale))
        if role == "body":
            return max(0.84, min(1.04, scale))
        return max(0.84, min(1.10, scale))



def line_weight_v9(line: TextLine) -> int:
    weights: List[int] = []
    for span in getattr(line, "spans", []) or []:
        w = int(getattr(span, "weight", WEIGHT_REGULAR) or WEIGHT_REGULAR)
        count = max(1, len((getattr(span, "text", "") or "").strip()))
        weights.extend([w] * count)
    if weights:
        return most_common(weights)
    if getattr(line, "is_bold", False):
        return WEIGHT_BOLD
    return WEIGHT_REGULAR


def block_has_line_style_variation_v9(block: TextBlock) -> bool:
    lines = [ln for ln in getattr(block, "lines", []) if (getattr(ln, "text", "") or "").strip()]
    if len(lines) < 2:
        return False

    colors = {getattr(ln, "color", 0) for ln in lines}
    sizes = [float(getattr(ln, "font_size", 0) or 0) for ln in lines]
    weights = {line_weight_v9(ln) for ln in lines}
    fonts = {clean_font_name(getattr(ln, "main_font", "") or "") for ln in lines}

    if len(colors) > 1:
        return True
    if len(weights) > 1:
        return True
    if sizes and (max(sizes) - min(sizes) >= 0.45):
        return True
    if len(fonts) > 1 and len(lines) <= 3:
        return True

    key = normalize_translation_key(getattr(block, "original_text", ""))
    if "we promise results" in key or "within 30 days" in key:
        return True
    return False


def should_use_line_aware_render_v9(block: TextBlock, layout: LayoutResult) -> bool:
    if not layout.lines or len(layout.lines) < 2:
        return False
    if getattr(block, "role", "body") in {"label", "table", "table_cell"}:
        return False
    if len(getattr(block, "lines", []) or []) > 5:
        return False
    return block_has_line_style_variation_v9(block)


def source_line_for_rendered_line_v9(block: TextBlock, render_index: int, total_render_lines: int) -> Optional[TextLine]:
    src_lines = [ln for ln in getattr(block, "lines", []) if (getattr(ln, "text", "") or "").strip()]
    if not src_lines:
        return None

    key = normalize_translation_key(getattr(block, "original_text", ""))

    # Special case: preserve callout hierarchy.
    # Source:
    #   We promise results       -> black/smaller
    #   within 30 days...        -> purple/bolder
    if ("we promise results" in key or "within 30 days" in key) and len(src_lines) >= 2:
        return src_lines[0] if render_index == 0 else src_lines[1]

    if len(src_lines) == total_render_lines and render_index < len(src_lines):
        return src_lines[render_index]

    # If translation wrapped into more lines than source, keep first line style
    # for the first rendered line and the final/emphasis source style for the rest.
    if len(src_lines) >= 2 and total_render_lines >= 2:
        if render_index == 0:
            return src_lines[0]
        if render_index >= total_render_lines - 1:
            return src_lines[-1]
        # Middle lines use the closest proportional source style.
        mapped = round(render_index * (len(src_lines) - 1) / max(1, total_render_lines - 1))
        return src_lines[min(len(src_lines) - 1, max(0, mapped))]

    return src_lines[min(render_index, len(src_lines) - 1)]


def style_block_from_line_v9(block: TextBlock, line: Optional[TextLine], text_for_font: str = "") -> TextBlock:
    if line is None:
        return block
    style_block = copy.copy(block)
    style_block.lines = [line]
    # Preserve role/align/bbox/original/translated but make style reflect this source line.
    setattr(style_block, "dominant_weight", line_weight_v9(line))
    if text_for_font:
        setattr(style_block, "_selected_text_for_font", text_for_font)
    setattr(style_block, "_font_size_scale", getattr(block, "_font_size_scale", 1.0))
    return style_block


def line_font_size_v9(block: TextBlock, source_line: Optional[TextLine], layout_fontsize: float) -> float:
    if source_line is None:
        return layout_fontsize
    block_size = max(0.01, float(getattr(block, "font_size", layout_fontsize) or layout_fontsize))
    src_size = max(0.01, float(getattr(source_line, "font_size", block_size) or block_size))
    ratio = src_size / block_size

    # Conservative clamp so line-aware styling does not destroy fit.
    key = normalize_translation_key(getattr(block, "original_text", ""))
    if "we promise results" in key or "within 30 days" in key:
        ratio = max(0.82, min(1.20, ratio))
    else:
        ratio = max(0.86, min(1.14, ratio))
    return max(4.2, layout_fontsize * ratio)


def draw_layout_v6(page: fitz.Page, layout: LayoutResult, block: TextBlock, resolver: FontResolverV6, color: RGB):
    """Draw translated layout using V9 line-aware typography.

    For normal blocks, this behaves like the previous V6 renderer.
    For short multi-line blocks with source style variation, it maps rendered
    lines back to source-line style: font weight, font size, and color.
    This fixes callouts like:
        We promise results
        within 30 days and ROI in 60 days.
    """
    if not layout.lines:
        return

    use_line_aware = should_use_line_aware_render_v9(block, layout)

    base_fontfile = resolver.fontfile_for(block)
    base_fontname = resolver.fontname_for(block)
    base_measure_font = resolver.fitz_font_for(block)
    symbol_file, symbol_font = get_symbol_font()

    x0, y0, x1, y1 = layout.rect
    y = y0

    total_lines = len(layout.lines)

    for i, line in enumerate(layout.lines):
        if not line:
            y += layout.fontsize * layout.line_height
            continue

        source_line = source_line_for_rendered_line_v9(block, i, total_lines) if use_line_aware else None
        style_block = style_block_from_line_v9(block, source_line, line) if source_line is not None else block

        if source_line is not None:
            fontfile = resolver.fontfile_for(style_block, line)
            fontname = resolver.fontname_for(style_block, line)
            measure_font = resolver.fitz_font_for(style_block, line)
            line_color = int_color_to_rgb(source_line.color)
            fontsize = line_font_size_v9(block, source_line, layout.fontsize)
        else:
            fontfile = base_fontfile
            fontname = base_fontname
            measure_font = base_measure_font
            line_color = color
            fontsize = layout.fontsize

        baseline = y + fontsize
        if baseline > y1 + fontsize:
            break

        draw_line = line
        leading_check = False
        if draw_line.startswith("✓ "):
            leading_check = True
            draw_line = draw_line[2:].lstrip()
        elif draw_line == "✓":
            leading_check = True
            draw_line = ""

        full_for_measure = ("✓ " + draw_line) if leading_check else draw_line
        measure_line = full_for_measure.replace("✓", "•")

        if layout.align == "center":
            w = measure_font.text_length(measure_line, fontsize=fontsize)
            x = x0 + max(0, (layout.rect.width - w) / 2)
        elif layout.align == "right":
            w = measure_font.text_length(measure_line, fontsize=fontsize)
            x = x1 - w
        else:
            x = x0

        if leading_check:
            if symbol_file and symbol_font:
                page.insert_text(
                    point=fitz.Point(x, baseline),
                    text="✓",
                    fontsize=fontsize,
                    fontname="FSymbolVN",
                    fontfile=symbol_file,
                    color=line_color,
                    overlay=True,
                )
                x += symbol_font.text_length("✓ ", fontsize=fontsize) + 1.0
            else:
                page.insert_text(
                    point=fitz.Point(x, baseline),
                    text="•",
                    fontsize=fontsize,
                    fontname=fontname,
                    fontfile=fontfile,
                    color=line_color,
                    overlay=True,
                )
                x += measure_font.text_length("• ", fontsize=fontsize) + 1.0

        if draw_line:
            page.insert_text(
                point=fitz.Point(x, baseline),
                text=draw_line,
                fontsize=fontsize,
                fontname=fontname,
                fontfile=fontfile,
                color=line_color,
                overlay=True,
            )

        # Use the larger of base and source-line size to keep rhythm and avoid overlaps.
        y += max(layout.fontsize, fontsize) * layout.line_height

def render_translated_pdf_v6(
    input_pdf: str,
    translated_ir: DocumentIR,
    output_pdf: str,
    regular_font: Optional[str] = None,
    bold_font: Optional[str] = None,
    title_font: Optional[str] = None,
    condensed_font: Optional[str] = None,
    condensed_bold_font: Optional[str] = None,
    condensed_semibold_font: Optional[str] = None,
    medium_font: Optional[str] = None,
    semibold_font: Optional[str] = None,
    black_font: Optional[str] = None,
    symbol_font: Optional[str] = None,
    embedded_fonts: Optional[Dict[str, str]] = None,
    prefer_original_fonts: bool = True,
    cover_text: bool = True,
    sampled_background: bool = True,
    translate_headers_footers: bool = False,
    force_render: bool = False,
):
    """Render with V9 redaction + V6 weight-aware font selection."""
    pdf = fitz.open(input_pdf)
    resolver = FontResolverV6(
        regular_font=regular_font,
        bold_font=bold_font,
        title_font=title_font,
        condensed_font=condensed_font,
        condensed_bold_font=condensed_bold_font,
        condensed_semibold_font=condensed_semibold_font,
        medium_font=medium_font,
        semibold_font=semibold_font,
        black_font=black_font,
        symbol_font=symbol_font,
        embedded_fonts=embedded_fonts,
        prefer_original_fonts=prefer_original_fonts,
    )

    force_unfit = os.getenv("PDF_TRANSLATOR_FORCE_UNFIT", "0") == "1"
    render_small_unfit = os.getenv("PDF_TRANSLATOR_RENDER_SMALL_UNFIT", "1") != "0"

    rendered_count = 0
    redacted_count = 0
    skipped_role = 0
    skipped_no_translation = 0
    skipped_same = 0
    skipped_unfit = 0
    drawn_unfit = 0
    compact_used = 0
    weight_distribution: Dict[int, int] = {}

    for page_ir in translated_ir.pages:
        page = pdf[page_ir.page_index]
        draw_jobs = []
        redaction_rects: List[fitz.Rect] = []

        for block in page_ir.blocks:
            if not is_translatable_block(block, translate_headers_footers):
                skipped_role += 1
                continue

            if not (block.translated_text and block.translated_text.strip()):
                skipped_no_translation += 1
                continue

            if text_is_same(block.original_text, block.translated_text) and not force_render:
                skipped_same += 1
                continue

            raw_text = postprocess_translation_v9(block.translated_text.strip(), block.original_text, block.role, "vi")
            if not raw_text:
                skipped_no_translation += 1
                continue

            rect = fitz.Rect(*expand_bbox(block.bbox, 0.18, page_ir.width, page_ir.height))

            setattr(block, "_selected_text_for_font", raw_text)
            font = resolver.fitz_font_for(block, raw_text)
            setattr(block, "_font_size_scale", resolver.font_size_scale_for(block, raw_text))
            layout, chosen_text, unfit = choose_layout_for_block(block, raw_text, rect, font)

            if layout is None or not chosen_text:
                skipped_unfit += 1
                continue

            if unfit and not (force_unfit or force_render or (render_small_unfit and is_small_box(block))):
                skipped_unfit += 1
                continue

            if chosen_text != raw_text:
                compact_used += 1
                setattr(block, "_selected_text_for_font", chosen_text)
                font = resolver.fitz_font_for(block, chosen_text)
                setattr(block, "_font_size_scale", resolver.font_size_scale_for(block, chosen_text))
                layout = compute_layout_v9(chosen_text, block, rect, font)

            weight = int(getattr(block, "dominant_weight", WEIGHT_REGULAR) or WEIGHT_REGULAR)
            weight_distribution[weight] = weight_distribution.get(weight, 0) + 1

            if cover_text:
                redaction_rects.extend(redaction_rects_for_block_v9(block, page_ir))

            color = int_color_to_rgb(block.color)
            draw_jobs.append((layout, block, color))
            if unfit:
                drawn_unfit += 1

        if cover_text and redaction_rects:
            add_no_fill_redactions(page, redaction_rects)
            apply_no_fill_redactions(page)
            redacted_count += len(redaction_rects)

        for layout, block, color in draw_jobs:
            draw_layout_v6(page, layout, block, resolver, color)
            rendered_count += 1

    pdf.save(output_pdf, garbage=4, deflate=True)
    pdf.close()

    print("Render summary v6 merged (weight-aware):")
    print(f"  rendered={rendered_count}")
    print(f"  redaction_rects={redacted_count}")
    print(f"  compact_candidates_used={compact_used}")
    print(f"  weight_distribution={weight_distribution}")
    print(f"  drawn_unfit={drawn_unfit}")
    print(f"  skipped_role={skipped_role}")
    print(f"  skipped_same={skipped_same}")
    print(f"  skipped_no_translation={skipped_no_translation}")
    print(f"  skipped_unfit={skipped_unfit}")



# ============================================================
# V11.3 CACHE / REUSE MODE
# ============================================================

def dict_to_text_span_v113(d: Dict) -> TextSpan:
    span = TextSpan(
        text=str(d.get("text", "")),
        bbox=tuple(float(x) for x in d.get("bbox", (0, 0, 0, 0))),  # type: ignore
        font=str(d.get("font", "")),
        size=float(d.get("size", 10.0) or 10.0),
        color=int(d.get("color", 0) or 0),
        flags=int(d.get("flags", 0) or 0),
        is_bold=bool(d.get("is_bold", False)),
        is_italic=bool(d.get("is_italic", False)),
    )
    # Rebuild dynamic V6 weight even though dataclass export did not save it.
    try:
        setattr(span, "weight", detect_font_weight(span.font, span.flags))
    except Exception:
        setattr(span, "weight", WEIGHT_BOLD if span.is_bold else WEIGHT_REGULAR)
    return span


def dict_to_text_line_v113(d: Dict) -> TextLine:
    return TextLine(
        bbox=tuple(float(x) for x in d.get("bbox", (0, 0, 0, 0))),  # type: ignore
        spans=[dict_to_text_span_v113(s) for s in d.get("spans", [])],
    )


def dict_to_text_block_v113(d: Dict) -> TextBlock:
    block = TextBlock(
        id=str(d.get("id", "")),
        page_index=int(d.get("page_index", 0) or 0),
        bbox=tuple(float(x) for x in d.get("bbox", (0, 0, 0, 0))),  # type: ignore
        lines=[dict_to_text_line_v113(ln) for ln in d.get("lines", [])],
        role=str(d.get("role", "body")),
        order=int(d.get("order", 0) or 0),
        align=str(d.get("align", "left")),
        original_text=str(d.get("original_text", "")),
        translated_text=str(d.get("translated_text", "")),
    )
    try:
        setattr(block, "dominant_weight", _dominant_weight_of_block(block))
    except Exception:
        setattr(block, "dominant_weight", WEIGHT_BOLD if block.is_bold else WEIGHT_REGULAR)
    return block


def load_ir_json_v113(path: str) -> DocumentIR:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ir = DocumentIR(source_pdf=str(data.get("source_pdf", "")))
    for p in data.get("pages", []):
        page = PageIR(
            page_index=int(p.get("page_index", 0) or 0),
            width=float(p.get("width", 0) or 0),
            height=float(p.get("height", 0) or 0),
            rotation=int(p.get("rotation", 0) or 0),
            image_rects=[
                tuple(float(x) for x in box)  # type: ignore
                for box in p.get("image_rects", [])
            ],
            blocks=[dict_to_text_block_v113(b) for b in p.get("blocks", [])],
        )
        ir.pages.append(page)
    return ir


class DiskCacheTranslator:
    """JSONL disk cache wrapper for LLM translation batches.

    This prevents re-spending API credits when the same text/instruction/model
    is translated again after a later render/OCR failure.
    """

    def __init__(self, wrapped: Translator, cache_path: str, source_lang: str, target_lang: str):
        self.wrapped = wrapped
        self.cache_path = Path(cache_path)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model = os.getenv("LLM_MODEL", "")
        self.provider = os.getenv("LLM_PROVIDER", os.getenv("PDF_TRANSLATOR_PROVIDER", ""))
        self.cache: Dict[str, str] = {}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    k = str(obj.get("key", ""))
                    v = str(obj.get("translated", ""))
                    if k:
                        self.cache[k] = v
        except Exception as e:
            print(f"WARNING: could not load translation cache {self.cache_path}: {e}", file=sys.stderr)

    def _make_key(self, item: Dict, source_lang: str, target_lang: str, glossary: Optional[Dict[str, str]]) -> str:
        import hashlib
        payload = {
            "provider": self.provider,
            "model": self.model,
            "source": source_lang,
            "target": target_lang,
            "text": str(item.get("text", "")),
            "role": str(item.get("role", "")),
            "instruction": str(item.get("instruction", "")),
            "max_chars": item.get("max_chars", None),
            "max_lines": item.get("max_lines", None),
            "glossary": glossary or {},
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _append(self, key: str, item: Dict, translated: str, source_lang: str, target_lang: str):
        obj = {
            "key": key,
            "provider": self.provider,
            "model": self.model,
            "source": source_lang,
            "target": target_lang,
            "id": item.get("id"),
            "text": item.get("text"),
            "translated": translated,
            "ts": time.time(),
        }
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def translate_batch(
        self,
        items: List[Dict[str, str]],
        source_lang: str,
        target_lang: str,
        glossary: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        if not items:
            return []

        output_by_id: Dict[str, str] = {}
        missing: List[Dict[str, str]] = []
        missing_keys: Dict[str, str] = {}

        for item in items:
            item_id = str(item.get("id", ""))
            k = self._make_key(item, source_lang, target_lang, glossary)
            if k in self.cache:
                output_by_id[item_id] = self.cache[k]
            else:
                missing.append(item)
                missing_keys[item_id] = k

        if missing:
            print(f"      Translation cache: hit={len(items)-len(missing)} miss={len(missing)} path={self.cache_path}")
            translated = self.wrapped.translate_batch(missing, source_lang, target_lang, glossary)
            item_by_id = {str(item.get("id", "")): item for item in missing}
            for obj in translated:
                item_id = str(obj.get("id", ""))
                val = str(obj.get("translated", ""))
                output_by_id[item_id] = val
                key = missing_keys.get(item_id)
                if key:
                    self.cache[key] = val
                    self._append(key, item_by_id.get(item_id, {"id": item_id}), val, source_lang, target_lang)
        else:
            print(f"      Translation cache: hit={len(items)} miss=0 path={self.cache_path}")

        return [{"id": str(item.get("id", "")), "translated": output_by_id.get(str(item.get("id", "")), str(item.get("text", "")))} for item in items]



# ============================================================
# V12 SAFE OCR FILTERING
# ============================================================

def rect_area_v12(rect: Tuple[float, float, float, float]) -> float:
    return max(0.0, float(rect[2]) - float(rect[0])) * max(0.0, float(rect[3]) - float(rect[1]))


def rect_intersection_area_v12(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def expand_rect_tuple_v12(
    rect: Tuple[float, float, float, float],
    pad: float,
    page_w: Optional[float] = None,
    page_h: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(v) for v in rect]
    x0 -= pad
    y0 -= pad
    x1 += pad
    y1 += pad
    if page_w is not None:
        x0 = max(0.0, min(float(page_w), x0))
        x1 = max(0.0, min(float(page_w), x1))
    if page_h is not None:
        y0 = max(0.0, min(float(page_h), y0))
        y1 = max(0.0, min(float(page_h), y1))
    return (x0, y0, x1, y1)


def collect_text_layer_exclusion_rects_v12(ir: DocumentIR, pad: float = 1.4) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """Collect text-layer bboxes so OCR does not duplicate already translated text.

    This is the key quality fix: OCR should target image/raster text only,
    not text-layer content that was already translated/rendered by the main pipeline.
    """
    out: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for page in ir.pages:
        rects: List[Tuple[float, float, float, float]] = []
        for block in page.blocks:
            if not (getattr(block, "original_text", "") or "").strip():
                continue
            if getattr(block, "role", "") in {"hidden", "page_number"}:
                continue

            # Use line rects instead of full block rect where possible. This avoids
            # excluding nearby image-only labels just because a paragraph bbox is large.
            line_rects = []
            for ln in getattr(block, "lines", []) or []:
                if (getattr(ln, "text", "") or "").strip():
                    line_rects.append(tuple(float(x) for x in ln.bbox))
            if line_rects:
                for r in line_rects:
                    rects.append(expand_rect_tuple_v12(r, pad, page.width, page.height))
            else:
                rects.append(expand_rect_tuple_v12(tuple(float(x) for x in block.bbox), pad, page.width, page.height))

        out[page.page_index] = rects
    return out


def build_ocr_exclusion_rects_v12(
    source_pdf: str,
    exclude_ir_path: Optional[str] = None,
    enabled: bool = True,
) -> Dict[int, List[Tuple[float, float, float, float]]]:
    if not enabled:
        return {}
    try:
        if exclude_ir_path:
            print(f"      OCR exclusion: load text-layer rects from IR: {exclude_ir_path}")
            ir = load_ir_json_v113(exclude_ir_path)
        else:
            print("      OCR exclusion: parse source PDF text-layer rects")
            ir = parse_pdf_to_ir_v6(source_pdf)
        rects = collect_text_layer_exclusion_rects_v12(ir)
        print(f"      OCR exclusion rects: {sum(len(v) for v in rects.values())}")
        return rects
    except Exception as e:
        print(f"WARNING: could not build OCR exclusion rects: {e}", file=sys.stderr)
        return {}


def has_vietnamese_marks_v12(s: str) -> bool:
    marks = set("ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    return any(ch in marks for ch in s or "")


def english_word_count_v12(s: str) -> int:
    words = re.findall(r"\b[A-Za-z][A-Za-z-]{2,}\b", s or "")
    allowed = {"CXVIEW", "GPT", "AI", "ROI", "POC", "CCTV", "VMS", "RTSP", "ONVIF", "POS", "API", "USD"}
    return sum(1 for w in words if w.upper() not in allowed)


def ocr_block_is_safe_candidate_v12(
    block: Dict,
    strict_english: bool = True,
    max_chars: int = 70,
    max_coverage: float = 0.20,
    exclude_rects: Optional[List[Tuple[float, float, float, float]]] = None,
) -> Tuple[bool, str]:
    """Return (keep, reason)."""
    text = re.sub(r"\s+", " ", str(block.get("text", "") or "")).strip()
    if not text:
        return False, "empty"

    if len(text) > max_chars:
        return False, "too_long"

    # After text-layer translation, OCR fallback should only handle English leftovers.
    # If it already has Vietnamese marks, it is almost certainly text-layer output OCR read again.
    if strict_english and has_vietnamese_marks_v12(text):
        return False, "has_vietnamese"

    # Avoid OCR-ing paragraphs/body. OCR fallback should mostly target image labels/buttons.
    role = str(block.get("role", "label"))
    if role == "body" and strict_english:
        return False, "body_block"

    # Require at least one non-protected English word in strict mode.
    if strict_english and english_word_count_v12(text) < 1:
        return False, "no_english_words"

    bbox = tuple(float(x) for x in block.get("bbox", (0, 0, 0, 0)))
    area = rect_area_v12(bbox)
    if area <= 0:
        return False, "bad_bbox"

    # Main quality gate: skip OCR bboxes that overlap existing text-layer bboxes.
    for r in exclude_rects or []:
        inter = rect_intersection_area_v12(bbox, r)
        if inter <= 0:
            continue
        coverage = inter / max(1.0, area)
        if coverage >= max_coverage:
            return False, f"overlap_text_layer_{coverage:.2f}"

    return True, "keep"


def filter_ocr_blocks_v12(
    blocks: List[Dict],
    page_index: int,
    exclude_rects_by_page: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None,
    strict_english: bool = True,
    max_chars: int = 70,
    max_coverage: float = 0.20,
) -> List[Dict]:
    exclude_rects = (exclude_rects_by_page or {}).get(page_index, [])
    kept: List[Dict] = []
    stats: Dict[str, int] = {}

    for b in blocks:
        keep, reason = ocr_block_is_safe_candidate_v12(
            b,
            strict_english=strict_english,
            max_chars=max_chars,
            max_coverage=max_coverage,
            exclude_rects=exclude_rects,
        )
        stats[reason] = stats.get(reason, 0) + 1
        if keep:
            kept.append(b)

    print(f"      OCR filter page {page_index + 1}: in={len(blocks)} kept={len(kept)} skipped={len(blocks)-len(kept)} reasons={stats}")
    return kept


def build_arg_parser_v6() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-file V6 weight-aware PDF translator")
    p.add_argument("input_pdf")
    p.add_argument("output_pdf")
    p.add_argument("--source", default="auto")
    p.add_argument("--target", default="vi")
    p.add_argument("--font", default=None)
    p.add_argument("--font-bold", default=None)
    p.add_argument("--font-title", default=None)
    p.add_argument("--font-condensed", default=None)
    p.add_argument("--font-condensed-bold", default=None)
    p.add_argument("--font-condensed-semibold", default=None)
    p.add_argument("--font-medium", default=None)
    p.add_argument("--font-semibold", default=None)
    p.add_argument("--font-black", default=None)
    p.add_argument("--font-symbol", default=None)
    p.add_argument("--translation-map", default=None)
    p.add_argument("--glossary", default=None)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--export-ir", default=None)
    p.add_argument("--preview-dir", default=None)
    p.add_argument("--no-cover", action="store_true")
    p.add_argument("--no-sampled-bg", action="store_true")
    p.add_argument("--translate-headers-footers", action="store_true")
    p.add_argument("--force-render", action="store_true")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--llm-temperature", type=float, default=0.1)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-retries", type=int, default=3)
    p.add_argument("--embedded-font-dir", default="extracted_fonts")
    p.add_argument("--no-extract-fonts", action="store_true")
    p.add_argument("--no-prefer-original-fonts", action="store_true")
    p.add_argument("--no-translate-all-text", action="store_true", help="Legacy mode: skip headers/footers/logos unless explicitly enabled")
    p.add_argument("--no-protect-terms", action="store_true", help="Do not hard-preserve built-in technical terms; use glossary later instead")
    p.add_argument("--skip-logo-text", action="store_true", help="Skip logo text even in translate-all mode")

    # OCR fallback mode: translates English text still visible in raster/images/diagrams
    # after the normal text-layer pass.
    p.add_argument("--ocr", action="store_true", help="Enable OCR fallback pass after text-layer translation")
    p.add_argument("--ocr-engine", default="paddle", choices=["paddle", "tesseract"], help="OCR engine. Default: paddle")
    p.add_argument("--ocr-lang", default="en", help="OCR language. Paddle default: en; Tesseract default equivalent: eng")
    p.add_argument("--ocr-dpi", type=int, default=220, help="Render DPI for OCR, default: 220")
    p.add_argument("--ocr-min-conf", type=float, default=45.0, help="Minimum OCR word confidence")
    p.add_argument("--ocr-min-chars", type=int, default=3, help="Minimum OCR line characters")
    p.add_argument("--ocr-output", default=None, help="Optional final PDF path after OCR. Default: overwrite output_pdf via temp file")
    p.add_argument("--paddle-textline-orientation", action="store_true", help="Enable PaddleOCR textline orientation classifier")
    p.add_argument("--paddle-cpu-threads", type=int, default=4, help="CPU threads for PaddleOCR; default 4")
    p.add_argument("--tesseract-cmd", default=None, help="Path to tesseract.exe on Windows; only used when --ocr-engine tesseract")

    # Cache / reuse modes to avoid re-spending LLM credits after OCR/render failures.
    p.add_argument("--translation-cache", default=".translation_cache.jsonl", help="JSONL disk cache for LLM translations")
    p.add_argument("--no-translation-cache", action="store_true", help="Disable JSONL translation cache")
    p.add_argument("--reuse-translated-ir", default=None, help="Load *_translated.json and skip parse/rebuild/translate")
    p.add_argument("--ocr-only-from", default=None, help="Skip text-layer translation/render; run OCR only on an existing rendered PDF")

    # Safe OCR filtering. Enabled by default because raw OCR over the rendered
    # PDF duplicates/overwrites already translated text.
    p.add_argument("--ocr-exclude-ir", default=None, help="Optional IR JSON used to exclude text-layer bboxes from OCR")
    p.add_argument("--no-ocr-exclude-text-layer", action="store_true", help="Do not exclude source text-layer bboxes from OCR")
    p.add_argument("--no-ocr-strict-english", action="store_true", help="Allow OCR candidates that contain Vietnamese/body text")
    p.add_argument("--ocr-max-chars", type=int, default=70, help="Max OCR candidate chars in strict mode")
    p.add_argument("--ocr-max-overlap", type=float, default=0.20, help="Skip OCR block if this much overlaps text-layer bbox")
    return p


def main_v6(argv: Optional[List[str]] = None):
    args = build_arg_parser_v6().parse_args(argv)

    if not Path(args.input_pdf).exists():
        raise FileNotFoundError(args.input_pdf)

    load_dotenv_file(args.env_file)

    # V7 defaults: translate all extractable text.
    os.environ["PDF_TRANSLATOR_TRANSLATE_ALL_TEXT"] = "0" if args.no_translate_all_text else "1"
    if args.no_protect_terms:
        os.environ["PDF_TRANSLATOR_PROTECT_TERMS"] = "0"
    else:
        os.environ.setdefault("PDF_TRANSLATOR_PROTECT_TERMS", "1")
    os.environ["PDF_TRANSLATOR_SKIP_LOGO_TEXT"] = "1" if args.skip_logo_text else os.getenv("PDF_TRANSLATOR_SKIP_LOGO_TEXT", "0")

    _prepare_anthropic_env()
    _print_provider_info_once()
    os.environ.setdefault("PDF_TRANSLATOR_RENDER_UNFIT_ALLTEXT", "1")

    print(
        "Text mode: translate_all_text="
        f"{os.getenv('PDF_TRANSLATOR_TRANSLATE_ALL_TEXT')} "
        f"protect_terms={os.getenv('PDF_TRANSLATOR_PROTECT_TERMS')} "
        f"skip_logo_text={os.getenv('PDF_TRANSLATOR_SKIP_LOGO_TEXT')} "
        f"render_unfit_alltext={os.getenv('PDF_TRANSLATOR_RENDER_UNFIT_ALLTEXT')}"
    )

    if args.translation_map:
        translator: Translator = JsonMapTranslator(args.translation_map)
        print("Translator: JSON translation map")
    elif not args.no_llm and os.getenv("LLM_API_KEY") and os.getenv("LLM_BASE_URL") and os.getenv("LLM_MODEL"):
        translator = OpenAICompatibleTranslator(
            temperature=args.llm_temperature,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
        )
        print(f"Translator: LLM model={os.getenv('LLM_MODEL')} base_url={os.getenv('LLM_BASE_URL')}")
    else:
        translator = DummyTranslator()
        print("WARNING: DummyTranslator. No real translation will happen.", file=sys.stderr)

    glossary = load_json_map(args.glossary)

    if not args.no_translation_cache:
        translator = DiskCacheTranslator(
            wrapped=translator,
            cache_path=args.translation_cache,
            source_lang=args.source,
            target_lang=args.target,
        )
        print(f"Translation cache enabled: {args.translation_cache}")

    ocr_exclude_rects_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
    if args.ocr or args.ocr_only_from:
        ocr_exclude_rects_by_page = build_ocr_exclusion_rects_v12(
            source_pdf=args.input_pdf,
            exclude_ir_path=args.ocr_exclude_ir,
            enabled=not args.no_ocr_exclude_text_layer,
        )

    if args.ocr_only_from:
        print(f"[OCR-only] Skip text-layer parse/translate/render. Input rendered PDF: {args.ocr_only_from}")

        if not Path(args.ocr_only_from).exists():
            raise FileNotFoundError(args.ocr_only_from)

        final_ocr_output = args.ocr_output or args.output_pdf
        ocr_output_pdf = final_ocr_output

        if args.ocr_engine == "paddle":
            os.environ["PDF_TRANSLATOR_PADDLE_CPU_THREADS"] = str(args.paddle_cpu_threads)
            os.environ.setdefault("FLAGS_use_mkldnn", "False")
            os.environ.setdefault("FLAGS_use_onednn", "False")
            apply_ocr_fallback_paddle_v11(
                pdf_path=args.ocr_only_from,
                output_pdf=ocr_output_pdf,
                translator=translator,
                source_lang=args.source,
                target_lang=args.target,
                glossary=glossary,
                dpi=args.ocr_dpi,
                lang=args.ocr_lang,
                min_conf=args.ocr_min_conf,
                min_chars=args.ocr_min_chars,
                batch_size=args.batch_size,
                regular_font=args.font,
                bold_font=args.font_bold,
                title_font=args.font_title,
                condensed_font=args.font_condensed,
                condensed_bold_font=args.font_condensed_bold,
                exclude_rects_by_page=ocr_exclude_rects_by_page,
                strict_english=not args.no_ocr_strict_english,
                max_chars=args.ocr_max_chars,
                max_overlap=args.ocr_max_overlap,
                use_textline_orientation=args.paddle_textline_orientation,
            )
        else:
            _configure_tesseract_cmd(args.tesseract_cmd)
            tess_lang = "eng" if args.ocr_lang == "en" else args.ocr_lang
            apply_ocr_fallback_v10(
                pdf_path=args.ocr_only_from,
                output_pdf=ocr_output_pdf,
                translator=translator,
                source_lang=args.source,
                target_lang=args.target,
                glossary=glossary,
                dpi=args.ocr_dpi,
                lang=tess_lang,
                min_conf=args.ocr_min_conf,
                min_chars=args.ocr_min_chars,
                batch_size=args.batch_size,
                regular_font=args.font,
                bold_font=args.font_bold,
                title_font=args.font_title,
                condensed_font=args.font_condensed,
                condensed_bold_font=args.font_condensed_bold,
                exclude_rects_by_page=ocr_exclude_rects_by_page,
                strict_english=not args.no_ocr_strict_english,
                max_chars=args.ocr_max_chars,
                max_overlap=args.ocr_max_overlap,
            )

        if args.preview_dir:
            print(f"[OCR-only] Render preview PNGs: {args.preview_dir}")
            render_pdf_pages(final_ocr_output, args.preview_dir)

        print(f"Done: {final_ocr_output}")
        return

    if args.reuse_translated_ir:
        print(f"[1/4] Load translated IR cache: {args.reuse_translated_ir}")
        translated_ir = load_ir_json_v113(args.reuse_translated_ir)

        embedded_fonts: Dict[str, str] = {}
        if not args.no_extract_fonts:
            print(f"      Extract embedded fonts -> {args.embedded_font_dir}")
            embedded_fonts = extract_embedded_fonts(args.input_pdf, args.embedded_font_dir)
            print(f"      Extracted font records: {len(embedded_fonts)}")

        print("[2/4] Render translated PDF from cached IR")
        render_translated_pdf_v6(
            input_pdf=args.input_pdf,
            translated_ir=translated_ir,
            output_pdf=args.output_pdf,
            regular_font=args.font,
            bold_font=args.font_bold,
            title_font=args.font_title,
            condensed_font=args.font_condensed,
            condensed_bold_font=args.font_condensed_bold,
            condensed_semibold_font=args.font_condensed_semibold,
            medium_font=args.font_medium,
            semibold_font=args.font_semibold,
            black_font=args.font_black,
            symbol_font=args.font_symbol,
            embedded_fonts=embedded_fonts,
            prefer_original_fonts=not args.no_prefer_original_fonts,
            cover_text=not args.no_cover,
            sampled_background=not args.no_sampled_bg,
            translate_headers_footers=args.translate_headers_footers,
            force_render=args.force_render,
        )

        if args.ocr:
            print(f"[3/4] OCR fallback pass ({args.ocr_engine})")
            final_ocr_output = args.ocr_output or args.output_pdf
            if args.ocr_output:
                ocr_input_pdf = args.output_pdf
                ocr_output_pdf = args.ocr_output
            else:
                ocr_input_pdf = args.output_pdf
                ocr_output_pdf = str(Path(args.output_pdf).with_suffix(".ocr_tmp.pdf"))

            if args.ocr_engine == "paddle":
                os.environ["PDF_TRANSLATOR_PADDLE_CPU_THREADS"] = str(args.paddle_cpu_threads)
                os.environ.setdefault("FLAGS_use_mkldnn", "False")
                os.environ.setdefault("FLAGS_use_onednn", "False")
                apply_ocr_fallback_paddle_v11(
                    pdf_path=ocr_input_pdf,
                    output_pdf=ocr_output_pdf,
                    translator=translator,
                    source_lang=args.source,
                    target_lang=args.target,
                    glossary=glossary,
                    dpi=args.ocr_dpi,
                    lang=args.ocr_lang,
                    min_conf=args.ocr_min_conf,
                    min_chars=args.ocr_min_chars,
                    batch_size=args.batch_size,
                    regular_font=args.font,
                    bold_font=args.font_bold,
                    title_font=args.font_title,
                    condensed_font=args.font_condensed,
                    condensed_bold_font=args.font_condensed_bold,
                exclude_rects_by_page=ocr_exclude_rects_by_page,
                strict_english=not args.no_ocr_strict_english,
                max_chars=args.ocr_max_chars,
                max_overlap=args.ocr_max_overlap,
                    use_textline_orientation=args.paddle_textline_orientation,
                )
            else:
                _configure_tesseract_cmd(args.tesseract_cmd)
                tess_lang = "eng" if args.ocr_lang == "en" else args.ocr_lang
                apply_ocr_fallback_v10(
                    pdf_path=ocr_input_pdf,
                    output_pdf=ocr_output_pdf,
                    translator=translator,
                    source_lang=args.source,
                    target_lang=args.target,
                    glossary=glossary,
                    dpi=args.ocr_dpi,
                    lang=tess_lang,
                    min_conf=args.ocr_min_conf,
                    min_chars=args.ocr_min_chars,
                    batch_size=args.batch_size,
                    regular_font=args.font,
                    bold_font=args.font_bold,
                    title_font=args.font_title,
                    condensed_font=args.font_condensed,
                    condensed_bold_font=args.font_condensed_bold,
                exclude_rects_by_page=ocr_exclude_rects_by_page,
                strict_english=not args.no_ocr_strict_english,
                max_chars=args.ocr_max_chars,
                max_overlap=args.ocr_max_overlap,
                )

            if not args.ocr_output:
                Path(ocr_output_pdf).replace(args.output_pdf)
            print(f"      OCR final output: {final_ocr_output}")

        if args.preview_dir:
            print(f"[4/4] Render preview PNGs: {args.preview_dir}")
            render_pdf_pages(args.output_pdf if not args.ocr_output else args.ocr_output, args.preview_dir)

        print(f"Done: {args.output_pdf if not args.ocr_output else args.ocr_output}")
        return

    print(f"[1/5] Parse PDF -> IR: {args.input_pdf}")
    ir = parse_pdf_to_ir_v6(args.input_pdf)

    print("[2/5] Rebuild paragraph blocks")
    ir = rebuild_document_paragraphs(ir)

    embedded_fonts: Dict[str, str] = {}
    if not args.no_extract_fonts:
        print(f"      Extract embedded fonts -> {args.embedded_font_dir}")
        embedded_fonts = extract_embedded_fonts(args.input_pdf, args.embedded_font_dir)
        print(f"      Extracted font records: {len(embedded_fonts)}")

    if args.export_ir:
        print(f"      Export IR: {args.export_ir}")
        export_ir_json(ir, args.export_ir)

    print("[3/5] Translate IR")
    translated_ir = translate_ir_v9(
        ir=ir,
        translator=translator,
        source_lang=args.source,
        target_lang=args.target,
        glossary=glossary,
        batch_size=args.batch_size,
        translate_headers_footers=args.translate_headers_footers,
    )

    if args.export_ir:
        tpath = str(Path(args.export_ir).with_name(Path(args.export_ir).stem + "_translated.json"))
        print(f"      Export translated IR: {tpath}")
        export_ir_json(translated_ir, tpath)

    print("[4/5] Render translated PDF")
    render_translated_pdf_v6(
        input_pdf=args.input_pdf,
        translated_ir=translated_ir,
        output_pdf=args.output_pdf,
        regular_font=args.font,
        bold_font=args.font_bold,
        title_font=args.font_title,
        condensed_font=args.font_condensed,
        condensed_bold_font=args.font_condensed_bold,
        condensed_semibold_font=args.font_condensed_semibold,
        medium_font=args.font_medium,
        semibold_font=args.font_semibold,
        black_font=args.font_black,
        symbol_font=args.font_symbol,
        embedded_fonts=embedded_fonts,
        prefer_original_fonts=not args.no_prefer_original_fonts,
        cover_text=not args.no_cover,
        sampled_background=not args.no_sampled_bg,
        translate_headers_footers=args.translate_headers_footers,
        force_render=args.force_render,
    )

    if args.ocr:
        print(f"[5/6] OCR fallback pass ({args.ocr_engine})")

        final_ocr_output = args.ocr_output or args.output_pdf
        if args.ocr_output:
            ocr_input_pdf = args.output_pdf
            ocr_output_pdf = args.ocr_output
        else:
            ocr_input_pdf = args.output_pdf
            tmp_ocr = str(Path(args.output_pdf).with_suffix(".ocr_tmp.pdf"))
            ocr_output_pdf = tmp_ocr

        if args.ocr_engine == "paddle":
            os.environ["PDF_TRANSLATOR_PADDLE_CPU_THREADS"] = str(args.paddle_cpu_threads)
            os.environ.setdefault("FLAGS_use_mkldnn", "False")
            os.environ.setdefault("FLAGS_use_onednn", "False")
            apply_ocr_fallback_paddle_v11(
                pdf_path=ocr_input_pdf,
                output_pdf=ocr_output_pdf,
                translator=translator,
                source_lang=args.source,
                target_lang=args.target,
                glossary=glossary,
                dpi=args.ocr_dpi,
                lang=args.ocr_lang,
                min_conf=args.ocr_min_conf,
                min_chars=args.ocr_min_chars,
                batch_size=args.batch_size,
                regular_font=args.font,
                bold_font=args.font_bold,
                title_font=args.font_title,
                condensed_font=args.font_condensed,
                condensed_bold_font=args.font_condensed_bold,
                exclude_rects_by_page=ocr_exclude_rects_by_page,
                strict_english=not args.no_ocr_strict_english,
                max_chars=args.ocr_max_chars,
                max_overlap=args.ocr_max_overlap,
                use_textline_orientation=args.paddle_textline_orientation,
            )
        else:
            _configure_tesseract_cmd(args.tesseract_cmd)
            # Tesseract language uses "eng" while Paddle uses "en".
            tess_lang = "eng" if args.ocr_lang == "en" else args.ocr_lang
            apply_ocr_fallback_v10(
                pdf_path=ocr_input_pdf,
                output_pdf=ocr_output_pdf,
                translator=translator,
                source_lang=args.source,
                target_lang=args.target,
                glossary=glossary,
                dpi=args.ocr_dpi,
                lang=tess_lang,
                min_conf=args.ocr_min_conf,
                min_chars=args.ocr_min_chars,
                batch_size=args.batch_size,
                regular_font=args.font,
                bold_font=args.font_bold,
                title_font=args.font_title,
                condensed_font=args.font_condensed,
                condensed_bold_font=args.font_condensed_bold,
                exclude_rects_by_page=ocr_exclude_rects_by_page,
                strict_english=not args.no_ocr_strict_english,
                max_chars=args.ocr_max_chars,
                max_overlap=args.ocr_max_overlap,
            )

        if not args.ocr_output:
            Path(ocr_output_pdf).replace(args.output_pdf)
        print(f"      OCR final output: {final_ocr_output}")

    if args.preview_dir:
        step = "[6/6]" if args.ocr else "[5/5]"
        print(f"{step} Render preview PNGs: {args.preview_dir}")
        render_pdf_pages(args.output_pdf if not args.ocr_output else args.ocr_output, args.preview_dir)

    print(f"Done: {args.output_pdf if not args.ocr_output else args.ocr_output}")



# ============================================================
# V8 forced all-text memory overrides
# ============================================================
# These are intentionally added late so they override earlier entries that kept
# product phrases in English. Glossary can be added later for preferred terms.
EXACT_TRANSLATION_MEMORY_VI.update({
    "cxview gpt box": "Hộp CXVIEW GPT",
    "& ai video analytics": "& Phân tích Video AI",
    "cxview gpt box & ai video analytics": "Hộp CXVIEW GPT & Phân tích Video AI",
    "client challenges. cxview solutions. business impacts.": "Thách thức KH. Giải pháp CXVIEW. Tác động KD.",
    "how cxview delivers the transformation": "CXVIEW triển khai chuyển đổi",
})

COMPACT_TRANSLATION_MEMORY_VI.update({
    "cxview gpt box": "Hộp CXVIEW GPT",
    "& ai video analytics": "& Phân tích Video AI",
    "cxview gpt box & ai video analytics": "Hộp CXVIEW GPT & Phân tích Video AI",
    "client challenges. cxview solutions. business impacts.": "Thách thức KH. Giải pháp CXVIEW. Tác động KD.",
    "how cxview delivers the transformation": "CXVIEW triển khai chuyển đổi",
})



# ============================================================
# V9 line-aware marketing/callout translation overrides
# ============================================================
# These are late overrides so they win over previous memories.
EXACT_TRANSLATION_MEMORY_VI.update({
    "we promise results within 30 days and roi in 60 days.": "Chúng tôi cam kết mang lại kết quả\ntrong 30 ngày và ROI trong 60 ngày.",
    "cxview delivers measurable business impact on an aggressive timeline that respects your operational urgency, going beyond mere software deployment.": "CXVIEW mang lại tác động kinh doanh có thể đo lường\nvới tiến độ triển khai nhanh, phù hợp nhu cầu vận hành\ncấp thiết của bạn, vượt xa việc chỉ triển khai phần mềm.",
})

COMPACT_TRANSLATION_MEMORY_VI.update({
    "we promise results within 30 days and roi in 60 days.": "Chúng tôi cam kết mang lại kết quả\ntrong 30 ngày và ROI trong 60 ngày.",
    "cxview delivers measurable business impact on an aggressive timeline that respects your operational urgency, going beyond mere software deployment.": "CXVIEW mang lại tác động kinh doanh có thể đo lường\nvới tiến độ triển khai nhanh, phù hợp nhu cầu vận hành\ncấp thiết của bạn, vượt xa việc chỉ triển khai phần mềm.",
})



# ============================================================
# V10 OCR FALLBACK MODE
# ============================================================

def _require_ocr_dependencies():
    """Import OCR dependencies lazily so normal text-layer mode still works."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        return pytesseract, Image
    except Exception as e:
        raise RuntimeError(
            "OCR mode requires Python packages: pytesseract pillow.\n"
            "Install:\n"
            "  pip install pytesseract pillow\n\n"
            "Windows also needs the Tesseract OCR application installed.\n"
            "Typical path:\n"
            "  C:/Program Files/Tesseract-OCR/tesseract.exe\n"
            "Then set:\n"
            "  $env:TESSERACT_CMD='C:/Program Files/Tesseract-OCR/tesseract.exe'\n"
            "or pass --tesseract-cmd \"C:/Program Files/Tesseract-OCR/tesseract.exe\""
        ) from e


def _configure_tesseract_cmd(tesseract_cmd: Optional[str] = None) -> None:
    if not tesseract_cmd:
        tesseract_cmd = os.getenv("TESSERACT_CMD") or os.getenv("TESSERACT_EXE")
    if not tesseract_cmd:
        return
    pytesseract, _ = _require_ocr_dependencies()
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def page_to_pil_image_v10(page: fitz.Page, dpi: int = 220):
    """Render a PDF page to PIL RGB image and return (image, zoom)."""
    _, Image = _require_ocr_dependencies()
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img, zoom


def median_rgb_v10(pixels: List[Tuple[int, int, int]]) -> RGB:
    if not pixels:
        return (1.0, 1.0, 1.0)
    rs = sorted(p[0] for p in pixels)
    gs = sorted(p[1] for p in pixels)
    bs = sorted(p[2] for p in pixels)
    m = len(pixels) // 2
    return (rs[m] / 255.0, gs[m] / 255.0, bs[m] / 255.0)


def sample_bg_color_v10(img, px_bbox: Tuple[int, int, int, int], pad: int = 4) -> RGB:
    """Sample border pixels around OCR bbox for a simple cover fill."""
    w, h = img.size
    x0, y0, x1, y1 = px_bbox
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return (1.0, 1.0, 1.0)

    sx0 = max(0, x0 - pad)
    sx1 = min(w, x1 + pad)
    sy0 = max(0, y0 - pad)
    sy1 = min(h, y1 + pad)

    pixels = []
    pix = img.load()
    # top/bottom border
    for x in range(sx0, sx1, max(1, (sx1 - sx0) // 80 or 1)):
        for y in [sy0, max(sy0, y0 - 1), min(h - 1, y1), sy1 - 1]:
            if 0 <= x < w and 0 <= y < h:
                pixels.append(pix[x, y])
    # left/right border
    for y in range(sy0, sy1, max(1, (sy1 - sy0) // 40 or 1)):
        for x in [sx0, max(sx0, x0 - 1), min(w - 1, x1), sx1 - 1]:
            if 0 <= x < w and 0 <= y < h:
                pixels.append(pix[x, y])

    return median_rgb_v10(pixels)


def looks_like_remaining_english_v10(text: str, min_chars: int = 3) -> bool:
    """Heuristic: keep OCR strings that are likely untranslated English."""
    s = normalize_special_chars(text or "").strip()
    if len(s) < min_chars:
        return False

    # Ignore pure numbers/prices/page numbers.
    if not re.search(r"[A-Za-z]", s):
        return False
    if re.fullmatch(r"[\d\s.,$%:/+-]+", s):
        return False

    # Skip mostly Vietnamese text.
    vietnamese_marks = set("ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    if any(ch in vietnamese_marks for ch in s):
        # Still allow if it also has a lot of English uppercase product text,
        # but by default OCR should target English leftovers.
        ascii_words = re.findall(r"\b[A-Za-z][A-Za-z-]{2,}\b", s)
        if len(ascii_words) <= 1:
            return False

    words = re.findall(r"\b[A-Za-z][A-Za-z-]{2,}\b", s)
    if not words:
        return False

    allowed_single_terms = {"CXVIEW", "GPT", "Box", "AI", "Video", "Analytics", "ROI", "POC", "CCTV", "VMS", "RTSP", "ONVIF", "POS", "API"}
    if len(words) == 1 and words[0] in allowed_single_terms:
        return False

    return True


def _ocr_word_conf_v10(value) -> float:
    try:
        return float(value)
    except Exception:
        return -1.0


def ocr_page_to_blocks_v10(
    page: fitz.Page,
    page_index: int,
    dpi: int = 220,
    lang: str = "eng",
    min_conf: float = 45.0,
    min_chars: int = 3,
):
    """OCR a rendered page and group words into line-level OCR blocks."""
    pytesseract, _ = _require_ocr_dependencies()
    img, zoom = page_to_pil_image_v10(page, dpi=dpi)

    # Tesseract page segmentation:
    #   6 = assume a single uniform block; not good for slides.
    #   11 = sparse text; better for decks/diagrams.
    psm = os.getenv("PDF_TRANSLATOR_OCR_PSM", "11")
    config = f"--psm {psm}"

    data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DICT)

    grouped: Dict[Tuple[int, int, int, int], List[int]] = {}
    n = len(data.get("text", []))
    for i in range(n):
        word = str(data["text"][i] or "").strip()
        conf = _ocr_word_conf_v10(data.get("conf", ["-1"])[i])
        if not word or conf < min_conf:
            continue

        key = (
            int(data.get("block_num", [0])[i]),
            int(data.get("par_num", [0])[i]),
            int(data.get("line_num", [0])[i]),
            int(data.get("word_num", [0])[i]) // 999999,  # keeps line grouping stable
        )
        # Use real line key without word grouping.
        key = key[:3] + (0,)
        grouped.setdefault(key, []).append(i)

    blocks = []
    for key, idxs in grouped.items():
        idxs = sorted(idxs, key=lambda j: int(data["left"][j]))
        words = [str(data["text"][j] or "").strip() for j in idxs if str(data["text"][j] or "").strip()]
        raw_text = " ".join(words)
        raw_text = re.sub(r"\s+", " ", raw_text).strip()
        if not looks_like_remaining_english_v10(raw_text, min_chars=min_chars):
            continue

        left = min(int(data["left"][j]) for j in idxs)
        top = min(int(data["top"][j]) for j in idxs)
        right = max(int(data["left"][j]) + int(data["width"][j]) for j in idxs)
        bottom = max(int(data["top"][j]) + int(data["height"][j]) for j in idxs)
        avg_conf = sum(_ocr_word_conf_v10(data.get("conf", ["-1"])[j]) for j in idxs) / max(1, len(idxs))

        # Expand bbox slightly to cover antialiasing.
        px_pad = max(2, int(dpi / 110))
        px_bbox = (left - px_pad, top - px_pad, right + px_pad, bottom + px_pad)
        pdf_bbox = (
            max(0.0, px_bbox[0] / zoom),
            max(0.0, px_bbox[1] / zoom),
            min(float(page.rect.width), px_bbox[2] / zoom),
            min(float(page.rect.height), px_bbox[3] / zoom),
        )

        # Role/font size estimate.
        height_pt = max(1.0, pdf_bbox[3] - pdf_bbox[1])
        role = "label"
        if height_pt >= 17:
            role = "title"
        elif len(raw_text) > 60:
            role = "body"

        blocks.append({
            "id": f"ocr_p{page_index}_{len(blocks)}",
            "page_index": page_index,
            "text": raw_text,
            "bbox": pdf_bbox,
            "px_bbox": px_bbox,
            "confidence": avg_conf,
            "role": role,
            "font_size": max(5.0, min(18.0, height_pt * 0.72)),
            "bg_color": sample_bg_color_v10(img, px_bbox),
        })

    return blocks


def ocr_exact_translation_v10(text: str, target_lang: str = "vi") -> Optional[str]:
    if not target_lang.lower().startswith("vi"):
        return None
    key = normalize_translation_key(text)
    if key in COMPACT_TRANSLATION_MEMORY_VI:
        return COMPACT_TRANSLATION_MEMORY_VI[key]
    if key in EXACT_TRANSLATION_MEMORY_VI:
        return EXACT_TRANSLATION_MEMORY_VI[key]
    return None


def translate_ocr_blocks_v10(
    blocks: List[Dict],
    translator: Translator,
    source_lang: str,
    target_lang: str,
    glossary: Optional[Dict[str, str]] = None,
    batch_size: int = 20,
) -> List[Dict]:
    """Translate OCR line blocks. Uses exact memory first, then LLM."""
    pending = []
    for b in blocks:
        exact = ocr_exact_translation_v10(b["text"], target_lang)
        if exact:
            b["translated"] = exact
            continue
        pending.append(b)

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        items = []
        for b in batch:
            w = max(1.0, b["bbox"][2] - b["bbox"][0])
            h = max(1.0, b["bbox"][3] - b["bbox"][1])
            max_chars = max(8, int(w / max(3.2, b["font_size"] * 0.42)))
            items.append({
                "id": b["id"],
                "role": b.get("role", "label"),
                "max_chars": max_chars,
                "max_lines": 1 if b.get("role") in {"label", "title"} else 2,
                "box_width_pt": round(w, 1),
                "box_height_pt": round(h, 1),
                "font_size_pt": round(float(b["font_size"]), 1),
                "text": b["text"],
                "instruction": (
                    "OCR fallback text. Translate the entire visible English string into compact Vietnamese. "
                    "Return only the translated string. Keep brand names/numbers if needed."
                ),
            })
        try:
            translated = translator.translate_batch(items, source_lang, target_lang, glossary)
            by_id = {x["id"]: x.get("translated", "") for x in translated}
            for b in batch:
                b["translated"] = sanitize_text_v9(by_id.get(b["id"], b["text"]))
        except Exception as e:
            print(f"OCR translation batch failed: {e}", file=sys.stderr)
            for b in batch:
                b["translated"] = b["text"]

    return blocks



def safe_font_path_v10(path: Optional[str]) -> Optional[str]:
    """Return a usable font path or None. Prevent OCR pass from crashing on missing font args."""
    if not path:
        return None
    try:
        p = Path(path)
        if p.exists():
            return str(p)
    except Exception:
        pass
    return None


def first_existing_font_v10(*paths: Optional[str]) -> Optional[str]:
    for p in paths:
        ok = safe_font_path_v10(p)
        if ok:
            return ok

    # Windows fallback list with Vietnamese support reasonably likely.
    fallback_names = [
        "arial.ttf",
        "arialbd.ttf",
        "seguiemj.ttf",
        "segoeui.ttf",
        "seguisb.ttf",
        "seguisym.ttf",
    ]
    for name in fallback_names:
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            return str(p)
    return None



def draw_ocr_block_v10(
    page: fitz.Page,
    block: Dict,
    regular_font: Optional[str] = None,
    bold_font: Optional[str] = None,
    title_font: Optional[str] = None,
    condensed_font: Optional[str] = None,
    condensed_bold_font: Optional[str] = None,
):
    """Cover OCR text bbox and draw translated text."""
    src = block.get("text", "")
    translated = block.get("translated", src)
    if not translated or text_is_same(src, translated):
        return False

    rect = fitz.Rect(*block["bbox"])
    if rect.width <= 1 or rect.height <= 1:
        return False

    role = block.get("role", "label")
    bg = block.get("bg_color", (1, 1, 1))

    # Cover original raster/text with sampled background.
    pad = 0.8
    cover = fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad)
    page.draw_rect(cover, color=None, fill=bg, overlay=True)

    # Choose font safely. OCR pass should never crash just because an optional
    # font path does not exist. This can happen when --font-condensed-* points
    # to files that are not present in the local fonts folder.
    regular_font = safe_font_path_v10(regular_font)
    bold_font = safe_font_path_v10(bold_font)
    title_font = safe_font_path_v10(title_font)
    condensed_font = safe_font_path_v10(condensed_font)
    condensed_bold_font = safe_font_path_v10(condensed_bold_font)

    fontfile = first_existing_font_v10(regular_font, bold_font)
    fontname = "FOCRRegular"

    if role == "title":
        fontfile = first_existing_font_v10(title_font, condensed_bold_font, bold_font, regular_font)
        fontname = "FOCRTitle"
    elif role == "label":
        fontfile = first_existing_font_v10(condensed_bold_font, condensed_font, bold_font, regular_font)
        fontname = "FOCRLabel"
    else:
        fontfile = first_existing_font_v10(regular_font, bold_font)

    if not fontfile:
        # Let PyMuPDF use built-in Helvetica as the absolute last fallback.
        fontname = "helv"

    # Estimate text color from original context: OCR often targets dark/white text.
    # For simplicity, use dark text on light backgrounds, white on dark backgrounds.
    lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
    color = (1, 1, 1) if lum < 0.35 else (0.08, 0.08, 0.12)

    # If page area is a purple button, white is often better.
    if role == "label" and lum < 0.55:
        color = (1, 1, 1)

    fontsize = float(block.get("font_size", max(5, rect.height * 0.72)))
    min_size = max(4.0, fontsize * 0.65)
    text_to_draw = translated

    # Use insert_textbox so long OCR labels can shrink/fit.
    rc = -1
    size = fontsize
    while size >= min_size:
        rc = page.insert_textbox(
            rect,
            text_to_draw,
            fontsize=size,
            fontname=fontname,
            fontfile=fontfile,
            color=color,
            align=fitz.TEXT_ALIGN_LEFT,
            overlay=True,
        )
        if rc >= 0:
            break
        # The failed attempt may still not draw in PyMuPDF; it usually does not.
        size -= 0.25

    # Last resort: draw at x/y baseline.
    if rc < 0:
        page.insert_text(
            fitz.Point(rect.x0, rect.y0 + max(4.5, min_size)),
            text_to_draw,
            fontsize=min_size,
            fontname=fontname,
            fontfile=fontfile,
            color=color,
            overlay=True,
        )

    return True


def apply_ocr_fallback_v10(
    pdf_path: str,
    output_pdf: str,
    translator: Translator,
    source_lang: str = "auto",
    target_lang: str = "vi",
    glossary: Optional[Dict[str, str]] = None,
    dpi: int = 220,
    lang: str = "eng",
    min_conf: float = 45.0,
    min_chars: int = 3,
    batch_size: int = 20,
    regular_font: Optional[str] = None,
    bold_font: Optional[str] = None,
    title_font: Optional[str] = None,
    condensed_font: Optional[str] = None,
    condensed_bold_font: Optional[str] = None,
    exclude_rects_by_page: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None,
    strict_english: bool = True,
    max_chars: int = 70,
    max_overlap: float = 0.20,
):
    """OCR fallback pass applied after normal text-layer rendering."""
    pdf = fitz.open(pdf_path)

    total_detected = 0
    total_translated = 0
    total_drawn = 0

    for page_index, page in enumerate(pdf):
        blocks = ocr_page_to_blocks_v10(
            page=page,
            page_index=page_index,
            dpi=dpi,
            lang=lang,
            min_conf=min_conf,
            min_chars=min_chars,
        )
        total_detected += len(blocks)

        blocks = filter_ocr_blocks_v12(
            blocks,
            page_index=page_index,
            exclude_rects_by_page=exclude_rects_by_page,
            strict_english=strict_english,
            max_chars=max_chars,
            max_coverage=max_overlap,
        )

        if not blocks:
            continue

        blocks = translate_ocr_blocks_v10(
            blocks,
            translator=translator,
            source_lang=source_lang,
            target_lang=target_lang,
            glossary=glossary,
            batch_size=batch_size,
        )

        for b in blocks:
            if b.get("translated") and not text_is_same(b["text"], b["translated"]):
                total_translated += 1
                if draw_ocr_block_v10(
                    page,
                    b,
                    regular_font=regular_font,
                    bold_font=bold_font,
                    title_font=title_font,
                    condensed_font=condensed_font,
                    condensed_bold_font=condensed_bold_font,
                ):
                    total_drawn += 1

    pdf.save(output_pdf, garbage=4, deflate=True)
    pdf.close()

    print("OCR fallback summary:")
    print(f"  detected_blocks={total_detected}")
    print(f"  translated_blocks={total_translated}")
    print(f"  drawn_blocks={total_drawn}")



# ============================================================
# V11 PADDLEOCR FALLBACK MODE
# ============================================================

_PADDLE_OCR_ENGINE_CACHE: Dict[Tuple[str, bool], object] = {}


def _require_paddleocr_dependencies_v11():
    """Import PaddleOCR dependencies lazily so non-OCR mode still works.

    V11.2: disable MKLDNN/oneDNN before Paddle import. Some Windows CPU
    environments hit fused_conv2d / OneDnnContext errors during detection.
    """
    # These must be set before importing paddle / paddleocr to be most effective.
    os.environ.setdefault("FLAGS_use_mkldnn", "False")
    os.environ.setdefault("FLAGS_use_onednn", "False")
    os.environ.setdefault("FLAGS_enable_mkldnn", "False")
    os.environ.setdefault("ONEDNN_VERBOSE", "0")
    os.environ.setdefault("OMP_NUM_THREADS", os.getenv("PDF_TRANSLATOR_PADDLE_CPU_THREADS", "4"))
    os.environ.setdefault("MKL_NUM_THREADS", os.getenv("PDF_TRANSLATOR_PADDLE_CPU_THREADS", "4"))

    try:
        from paddleocr import PaddleOCR  # type: ignore
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        # Extra runtime flag setting when Paddle is already importable.
        try:
            import paddle  # type: ignore
            paddle.set_flags({"FLAGS_use_mkldnn": False})
        except Exception:
            pass

        return PaddleOCR, Image, np
    except Exception as e:
        raise RuntimeError(
            "PaddleOCR mode requires PaddlePaddle + PaddleOCR packages.\n"
            "Recommended CPU install inside your venv:\n"
            "  python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/\n"
            "  python -m pip install \"paddleocr[all]\"\n\n"
            "If package resolution is too heavy, try:\n"
            "  python -m pip install paddleocr\n"
        ) from e


def normalize_paddle_lang_v11(lang: str) -> str:
    """Map Tesseract-style language names to PaddleOCR names."""
    value = (lang or "en").lower().strip()
    mapping = {
        "eng": "en",
        "en": "en",
        "english": "en",
        "vie": "vi",
        "vi": "vi",
        "vietnamese": "vi",
        "kor": "korean",
        "ko": "korean",
        "jpn": "japan",
        "ja": "japan",
        "chi_sim": "ch",
        "zh": "ch",
    }
    return mapping.get(value, value)


def get_paddle_ocr_engine_v11(lang: str = "en", use_textline_orientation: bool = False):
    """Create/cache PaddleOCR engine.

    V11.2 tries several constructor signatures and explicitly disables MKLDNN.
    This avoids common Windows CPU oneDNN fused_conv2d crashes.
    """
    PaddleOCR, _, _ = _require_paddleocr_dependencies_v11()
    plang = normalize_paddle_lang_v11(lang)
    key = (plang, bool(use_textline_orientation))
    if key in _PADDLE_OCR_ENGINE_CACHE:
        return _PADDLE_OCR_ENGINE_CACHE[key]

    os.environ.setdefault("FLAGS_logtostderr", "0")
    os.environ.setdefault("GLOG_minloglevel", "2")
    print(f"      Initializing PaddleOCR engine lang={plang} textline_orientation={use_textline_orientation} mkldnn=disabled")

    cpu_threads = int(os.getenv("PDF_TRANSLATOR_PADDLE_CPU_THREADS", "4"))

    constructor_attempts = [
        # PaddleOCR 3.x style + explicit MKLDNN disable.
        dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=use_textline_orientation,
            lang=plang,
            engine="paddle",
            enable_mkldnn=False,
            cpu_threads=cpu_threads,
        ),
        # PaddleOCR 3.x style without engine kw for versions that reject it.
        dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=use_textline_orientation,
            lang=plang,
            enable_mkldnn=False,
            cpu_threads=cpu_threads,
        ),
        # PaddleOCR 2.x style.
        dict(
            use_angle_cls=use_textline_orientation,
            lang=plang,
            show_log=False,
            use_gpu=False,
            enable_mkldnn=False,
            cpu_threads=cpu_threads,
        ),
        # Minimal 2.x fallback.
        dict(
            use_angle_cls=use_textline_orientation,
            lang=plang,
            use_gpu=False,
            enable_mkldnn=False,
        ),
        # Absolute minimal fallback.
        dict(lang=plang),
    ]

    last_error = None
    for kwargs in constructor_attempts:
        try:
            engine = PaddleOCR(**kwargs)
            _PADDLE_OCR_ENGINE_CACHE[key] = engine
            return engine
        except TypeError as e:
            last_error = e
            continue

    raise RuntimeError(f"Could not initialize PaddleOCR engine. Last error: {last_error}")

def page_to_paddle_image_v11(page: fitz.Page, dpi: int = 220):
    """Render PDF page to PIL image for PaddleOCR."""
    _, Image, _ = _require_paddleocr_dependencies_v11()
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img, zoom


def save_paddle_temp_image_v11(img, page_index: int, temp_dir: Path) -> str:
    path = temp_dir / f"paddle_ocr_page_{page_index:04d}.png"
    img.save(path)
    return str(path)


def _to_plain_list_v11(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _bbox_from_any_box_v11(box) -> Optional[Tuple[float, float, float, float]]:
    """Accept PaddleOCR rec_boxes [x0,y0,x1,y1] or polygons [[x,y]...]."""
    box = _to_plain_list_v11(box)
    if box is None:
        return None

    try:
        # rec_boxes format: [x0, y0, x1, y1]
        if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x0, y0, x1, y1 = [float(v) for v in box]
            return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

        # Polygon/list of points format.
        pts = []
        if isinstance(box, (list, tuple)):
            for p in box:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (min(xs), min(ys), max(xs), max(ys))
    except Exception:
        return None

    return None


def _extract_res_dict_v11(res):
    """Normalize PaddleOCR 3.x result object/dict to a dictionary."""
    if res is None:
        return None

    if isinstance(res, dict):
        if "res" in res and isinstance(res["res"], dict):
            return res["res"]
        return res

    # PaddleOCR 3.x result object often exposes .res.
    if hasattr(res, "res"):
        try:
            r = getattr(res, "res")
            if isinstance(r, dict):
                return r
        except Exception:
            pass

    # Some result objects can be converted to dict via json-like attrs.
    for attr in ["json", "to_dict", "dict"]:
        if hasattr(res, attr):
            try:
                obj = getattr(res, attr)
                data = obj() if callable(obj) else obj
                if isinstance(data, dict):
                    if "res" in data and isinstance(data["res"], dict):
                        return data["res"]
                    return data
            except Exception:
                pass

    return None


def _flatten_paddle_legacy_v11(result) -> List[Tuple[str, float, object]]:
    """Parse PaddleOCR 2.x result shape recursively.

    Common shape:
      [ [ [box, (text, score)], ... ] ]
    """
    items: List[Tuple[str, float, object]] = []

    def rec(node):
        if node is None:
            return
        if isinstance(node, tuple):
            node = list(node)

        if isinstance(node, list):
            # candidate item: [box, (text, score)] or [box, [text, score]]
            if len(node) >= 2:
                box = node[0]
                txt_score = node[1]
                if isinstance(txt_score, tuple):
                    txt_score = list(txt_score)
                if isinstance(txt_score, list) and len(txt_score) >= 2 and isinstance(txt_score[0], str):
                    try:
                        score = float(txt_score[1])
                    except Exception:
                        score = 1.0
                    items.append((txt_score[0], score, box))
                    return

            for child in node:
                rec(child)

    rec(result)
    return items


def run_paddle_predict_v11(engine, image_path: str):
    """Run PaddleOCR with 3.x predict() first, fallback to 2.x ocr()."""
    try:
        import paddle  # type: ignore
        paddle.set_flags({"FLAGS_use_mkldnn": False})
    except Exception:
        pass

    try:
        if hasattr(engine, "predict"):
            try:
                return engine.predict(image_path)
            except TypeError:
                try:
                    return engine.predict(input=image_path)
                except TypeError:
                    pass

        if hasattr(engine, "ocr"):
            try:
                return engine.ocr(image_path, cls=False)
            except TypeError:
                try:
                    return engine.ocr(image_path)
                except TypeError:
                    pass
    except RuntimeError as e:
        msg = str(e)
        if "OneDnnContext" in msg or "fused_conv2d" in msg or "onednn" in msg.lower() or "mkldnn" in msg.lower():
            raise RuntimeError(
                "PaddleOCR failed in Windows CPU inference due to oneDNN/MKLDNN fused_conv2d.\n"
                "This script already disables MKLDNN in code. If it still fails, your installed "
                "paddlepaddle/paddleocr versions are likely incompatible. Try reinstalling with:\n"
                "  python -m pip uninstall -y paddlepaddle paddleocr\n"
                "  python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/\n"
                "  python -m pip install paddleocr==2.10.0\n"
                "Or temporarily run: --ocr-engine tesseract"
            ) from e
        raise

    raise RuntimeError("Unsupported PaddleOCR engine API: no usable predict() or ocr().")


def parse_paddle_result_items_v11(result) -> List[Tuple[str, float, object]]:
    """Return list of (text, score, box). Handles PaddleOCR 3.x + 2.x."""
    items: List[Tuple[str, float, object]] = []

    # PaddleOCR 3.x: result is iterable of result objects/dicts with rec_texts, rec_scores, rec_boxes/rec_polys.
    if isinstance(result, (list, tuple)):
        for res in result:
            d = _extract_res_dict_v11(res)
            if d and ("rec_texts" in d or "rec_boxes" in d or "rec_polys" in d):
                texts = list(_to_plain_list_v11(d.get("rec_texts")) or [])
                scores = list(_to_plain_list_v11(d.get("rec_scores")) or [1.0] * len(texts))
                boxes = _to_plain_list_v11(d.get("rec_boxes")) or _to_plain_list_v11(d.get("rec_polys")) or []
                for i, txt in enumerate(texts):
                    if not str(txt or "").strip():
                        continue
                    score = float(scores[i]) if i < len(scores) else 1.0
                    box = boxes[i] if isinstance(boxes, (list, tuple)) and i < len(boxes) else None
                    items.append((str(txt), score, box))
                continue

        if items:
            return items

    # PaddleOCR 2.x fallback.
    return _flatten_paddle_legacy_v11(result)


def ocr_page_to_blocks_paddle_v11(
    page: fitz.Page,
    page_index: int,
    engine,
    dpi: int = 220,
    min_conf: float = 0.45,
    min_chars: int = 3,
    temp_dir: Optional[Path] = None,
):
    """OCR a page with PaddleOCR and return line-level OCR blocks."""
    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir())

    img, zoom = page_to_paddle_image_v11(page, dpi=dpi)
    image_path = save_paddle_temp_image_v11(img, page_index, temp_dir)

    raw = run_paddle_predict_v11(engine, image_path)
    items = parse_paddle_result_items_v11(raw)

    blocks = []
    for raw_text, score, box in items:
        # PaddleOCR score is usually 0..1. Accept either 0..1 or 0..100 user arg.
        threshold = min_conf
        if threshold > 1:
            threshold = threshold / 100.0

        if score < threshold:
            continue

        clean_text = re.sub(r"\s+", " ", normalize_special_chars(raw_text or "")).strip()
        if not looks_like_remaining_english_v10(clean_text, min_chars=min_chars):
            continue

        bbox_px = _bbox_from_any_box_v11(box)
        if not bbox_px:
            continue

        left, top, right, bottom = bbox_px
        px_pad = max(2, int(dpi / 110))
        px_bbox = (
            int(left - px_pad),
            int(top - px_pad),
            int(right + px_pad),
            int(bottom + px_pad),
        )

        pdf_bbox = (
            max(0.0, px_bbox[0] / zoom),
            max(0.0, px_bbox[1] / zoom),
            min(float(page.rect.width), px_bbox[2] / zoom),
            min(float(page.rect.height), px_bbox[3] / zoom),
        )

        height_pt = max(1.0, pdf_bbox[3] - pdf_bbox[1])
        role = "label"
        if height_pt >= 17:
            role = "title"
        elif len(clean_text) > 60:
            role = "body"

        blocks.append({
            "id": f"paddleocr_p{page_index}_{len(blocks)}",
            "page_index": page_index,
            "text": clean_text,
            "bbox": pdf_bbox,
            "px_bbox": px_bbox,
            "confidence": float(score),
            "role": role,
            "font_size": max(5.0, min(18.0, height_pt * 0.72)),
            "bg_color": sample_bg_color_v10(img, px_bbox),
            "ocr_engine": "paddle",
        })

    return blocks


def apply_ocr_fallback_paddle_v11(
    pdf_path: str,
    output_pdf: str,
    translator: Translator,
    source_lang: str = "auto",
    target_lang: str = "vi",
    glossary: Optional[Dict[str, str]] = None,
    dpi: int = 220,
    lang: str = "en",
    min_conf: float = 0.45,
    min_chars: int = 3,
    batch_size: int = 20,
    regular_font: Optional[str] = None,
    bold_font: Optional[str] = None,
    title_font: Optional[str] = None,
    condensed_font: Optional[str] = None,
    condensed_bold_font: Optional[str] = None,
    use_textline_orientation: bool = False,
    exclude_rects_by_page: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None,
    strict_english: bool = True,
    max_chars: int = 70,
    max_overlap: float = 0.20,
):
    """PaddleOCR fallback pass after normal text-layer rendering."""
    engine = get_paddle_ocr_engine_v11(lang=lang, use_textline_orientation=use_textline_orientation)
    pdf = fitz.open(pdf_path)

    total_detected = 0
    total_translated = 0
    total_drawn = 0

    with tempfile.TemporaryDirectory(prefix="pdf_translate_paddleocr_") as td:
        temp_dir = Path(td)

        for page_index, page in enumerate(pdf):
            blocks = ocr_page_to_blocks_paddle_v11(
                page=page,
                page_index=page_index,
                engine=engine,
                dpi=dpi,
                min_conf=min_conf,
                min_chars=min_chars,
                temp_dir=temp_dir,
            )
            total_detected += len(blocks)

            blocks = filter_ocr_blocks_v12(
                blocks,
                page_index=page_index,
                exclude_rects_by_page=exclude_rects_by_page,
                strict_english=strict_english,
                max_chars=max_chars,
                max_coverage=max_overlap,
            )

            if not blocks:
                continue

            blocks = translate_ocr_blocks_v10(
                blocks,
                translator=translator,
                source_lang=source_lang,
                target_lang=target_lang,
                glossary=glossary,
                batch_size=batch_size,
            )

            for b in blocks:
                if b.get("translated") and not text_is_same(b["text"], b["translated"]):
                    total_translated += 1
                    if draw_ocr_block_v10(
                        page,
                        b,
                        regular_font=regular_font,
                        bold_font=bold_font,
                        title_font=title_font,
                        condensed_font=condensed_font,
                        condensed_bold_font=condensed_bold_font,
                    ):
                        total_drawn += 1

    pdf.save(output_pdf, garbage=4, deflate=True)
    pdf.close()

    print("PaddleOCR fallback summary:")
    print(f"  detected_blocks={total_detected}")
    print(f"  translated_blocks={total_translated}")
    print(f"  drawn_blocks={total_drawn}")


# Final active single-file bindings.
parse_pdf_to_ir = parse_pdf_to_ir_v6
render_translated_pdf = render_translated_pdf_v6


if __name__ == "__main__":
    main_v6()
