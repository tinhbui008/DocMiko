#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py

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
    python pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py input.pdf output.pdf `
      --font "fonts/NotoSans-Regular.ttf" `
      --font-bold "fonts/NotoSans-Bold.ttf" `
      --font-title "fonts/NotoSansCondensed-Bold.ttf" `
      --export-ir "ir_single.json"
"""

# VERSION_MARKER = 'merged_v16_full_ocr_rebuild'
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
            "- Use the glossary exactly. For this domain, translate camera as 'camera' not 'mÃ¡y áº£nh'; site as 'Ä‘á»‹a Ä‘iá»ƒm' not 'trang web'; billing as 'thanh toÃ¡n' not 'hÃ³a Ä‘Æ¡n'; Edge Intelligence as 'trÃ­ tuá»‡ biÃªn'."
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
- For Detection labels, use compact patterns like 'PhÃ¡t hiá»‡n ...', 'Nháº­n dáº¡ng ...', 'GiÃ¡m sÃ¡t ...', 'Äáº¿m ...'.
- Never output awkward mixed phrases like 'leo climbing', 'Äáº¥u tranh' for fighting, 'váº­t thá»ƒ khÃ´ng cÃ³ ngÆ°á»i', 'Linhh hoáº¡t hÃ³a Ä‘Æ¡n', or 'CXVIEW GPT Há»™p'.
- For role=title: concise marketing headline, no explanatory wording.
- For role=cta: concise call-to-action.
- For role=label: very short Vietnamese label, preferably 1-4 words, no full sentence.
- For small boxes where max_chars <= 30: use the shortest natural equivalent; keep English term if Vietnamese would be too long.
- For role=body: use natural Vietnamese suitable for a formal flyer/document, but avoid unnecessary length.
- Do not translate hidden/logo/decorative text; those should not appear here.
- Avoid awkward literal translations such as translating product word "Box" into "Há»™p" when it is part of the product name.
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
- Fix literal/wrong phrases: 'leo climbing', 'Äáº¥u tranh', 'váº­t thá»ƒ khÃ´ng cÃ³ ngÆ°á»i', 'thao tÃºng camera', 'Linhh hoáº¡t hÃ³a Ä‘Æ¡n', 'GPT Há»™p'.
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
    text = text.replace("â€œ", '"').replace("â€", '"').replace("â€˜", "'").replace("â€™", "'")
    text = text.replace("â€“", "-").replace("â€”", "-")
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
    return bool(re.fullmatch(r"[-â€“â€”]?\s*\d{1,4}\s*[-â€“â€”]?", text.strip()))


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

    if re.match(r"^\s*(fig\.|figure|table|chart|áº£nh|hÃ¬nh|báº£ng)\s+\d+", text, flags=re.I):
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

    if re.match(r"^(\d+\.|[â€¢\-*])\s+", curr_text):
        return False

    px0, py0, px1, py1 = prev.bbox
    cx0, cy0, cx1, cy1 = curr.bbox

    same_left = abs(px0 - cx0) < 10
    similar_width = abs(bbox_width(prev.bbox) - bbox_width(curr.bbox)) < page.width * 0.18
    vertical_gap = cy0 - py1
    close_gap = 0 <= vertical_gap <= max(prev.font_size, curr.font_size) * 1.35
    similar_size = abs(prev.font_size - curr.font_size) <= 1.0
    prev_ends_sentence = bool(re.search(r"[.!?ã€‚ï¼ï¼Ÿ:]$", prev_text))

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
    "site": "Ä‘á»‹a Ä‘iá»ƒm",
    "sites": "Ä‘á»‹a Ä‘iá»ƒm",
    "edge intelligence": "trÃ­ tuá»‡ biÃªn",
    "edge device": "thiáº¿t bá»‹ biÃªn",
    "on premise": "táº¡i chá»—",
    "on-premise": "táº¡i chá»—",
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
    "billing flexibility": "Linh hoáº¡t thanh toÃ¡n",
    "concurrent AI stream ratio": "Tá»· lá»‡ luá»“ng AI Ä‘á»“ng thá»i",
}

# Normalized exact phrase memory for small labels and stable UI/table text.
# This dramatically improves slide/flyer PDFs where each label has a tiny bbox.
EXACT_TRANSLATION_MEMORY_VI: Dict[str, str] = {
    # Page 1 / overview
    "client challenges. cxview solutions. business impacts.": "ThÃ¡ch thá»©c khÃ¡ch hÃ ng. Giáº£i phÃ¡p CXVIEW. TÃ¡c Ä‘á»™ng kinh doanh.",
    "from conservative cctv infrastructure to real-time intelligence, operational automation and business-performance reporting.": "Tá»« háº¡ táº§ng CCTV truyá»n thá»‘ng Ä‘áº¿n trÃ­ tuá»‡ thá»i gian thá»±c, tá»± Ä‘á»™ng hÃ³a váº­n hÃ nh vÃ  bÃ¡o cÃ¡o hiá»‡u quáº£ kinh doanh.",
    "about cxview": "GIá»šI THIá»†U CXVIEW",
    "cxview gpt box & ai video analytics": "CXVIEW GPT Box & AI Video Analytics",
    "cxview gpt box": "CXVIEW GPT Box",
    "& ai video analytics": "& AI Video Analytics",

    # Page 2 / pricing table
    "package": "GÃ³i",
    "tier": "Cáº¥p",
    "cameras": "Camera",
    "concurrent ai streams active on ai models existed": "Luá»“ng AI Ä‘á»“ng thá»i\ntrÃªn mÃ´ hÃ¬nh AI hiá»‡n cÃ³",
    "concurrent ai streams": "Luá»“ng AI Ä‘á»“ng thá»i",
    "active on ai models existed": "trÃªn mÃ´ hÃ¬nh AI hiá»‡n cÃ³",
    "monthly (usd)": "HÃ ng thÃ¡ng (USD)",
    "24-month (usd)": "24 thÃ¡ng (USD)",
    "36-month (usd)": "36 thÃ¡ng (USD)",
    "concurrent ai stream ratio": "Tá»· lá»‡ luá»“ng AI Ä‘á»“ng thá»i",
    "billing flexibility": "Linh hoáº¡t thanh toÃ¡n",
    "cxview smart ai video analytics solution": "CXVIEW SMART AI VIDEO ANALYTICS SOLUTION",

    # Page 3 / solution labels
    "our solutions": "GIáº¢I PHÃP Cá»¦A CHÃšNG TÃ”I",
    "vehicle plate recognition": "Nháº­n dáº¡ng biá»ƒn sá»‘ xe",
    "fence and wall climbing detection": "PhÃ¡t hiá»‡n leo rÃ o/tÆ°á»ng",
    "intrusion detection": "PhÃ¡t hiá»‡n xÃ¢m nháº­p",
    "camera tampering detection": "PhÃ¡t hiá»‡n can thiá»‡p camera",
    "ppe and uniform detection": "PhÃ¡t hiá»‡n PPE & Ä‘á»“ng phá»¥c",
    "forklift and vehicle safety detection": "An toÃ n xe nÃ¢ng & phÆ°Æ¡ng tiá»‡n",
    "falls and slips detection": "PhÃ¡t hiá»‡n tÃ© ngÃ£/trÆ°á»£t",
    "fighting detection": "PhÃ¡t hiá»‡n Ä‘Ã¡nh nhau",
    "unusual crowd detection": "PhÃ¡t hiá»‡n Ä‘Ã¡m Ä‘Ã´ng báº¥t thÆ°á»ng",
    "smoke and fire detection": "PhÃ¡t hiá»‡n khÃ³i/lá»­a",
    "ai camera in security": "AI CAMERA\nAN NINH",
    "ai camera in operation": "AI CAMERA\nVáº¬N HÃ€NH",
    "ai camera in safety": "AI CAMERA\nAN TOÃ€N",
    "smart workstation monitoring": "GiÃ¡m sÃ¡t tráº¡m lÃ m viá»‡c",
    "automated product counting": "Äáº¿m sáº£n pháº©m tá»± Ä‘á»™ng",
    "product quality inspection": "Kiá»ƒm tra cháº¥t lÆ°á»£ng sáº£n pháº©m",
    "heat maps & route maps analysis": "PhÃ¢n tÃ­ch báº£n Ä‘á»“ nhiá»‡t/lá»™ trÃ¬nh",
    "heat maps and route maps analysis": "PhÃ¢n tÃ­ch báº£n Ä‘á»“ nhiá»‡t/lá»™ trÃ¬nh",
    "dwell time report": "BÃ¡o cÃ¡o thá»i gian lÆ°u láº¡i",
    "traffic counting": "Äáº¿m lÆ°u lÆ°á»£ng",
    "customer demographic analysis": "PhÃ¢n tÃ­ch nhÃ¢n kháº©u há»c",
    "customer engagement detection": "PhÃ¡t hiá»‡n tÆ°Æ¡ng tÃ¡c khÃ¡ch hÃ ng",
    "table cleaning detection": "PhÃ¡t hiá»‡n dá»n bÃ n",
    "patrol automatic report": "BÃ¡o cÃ¡o tuáº§n tra tá»± Ä‘á»™ng",
    "unattended object detection": "PhÃ¡t hiá»‡n váº­t thá»ƒ bá» quÃªn",

    # Page 4 / process
    "our process": "QUY TRÃŒNH",
    "strategic understanding": "Tháº¥u hiá»ƒu chiáº¿n lÆ°á»£c",
    "tailored solution architecture": "Kiáº¿n trÃºc giáº£i phÃ¡p tÃ¹y chá»‰nh",
    "pilot (poc) & validation": "Pilot (POC) & xÃ¡c thá»±c",
    "scale & sustain": "Má»Ÿ rá»™ng & duy trÃ¬",
    "book your live demo": "Äáº·t lá»‹ch demo trá»±c tiáº¿p",
    "near-zero latency": "Äá»™ trá»… gáº§n báº±ng 0",
    "data sovereignty": "Chá»§ quyá»n dá»¯ liá»‡u",
    "cost efficiency": "Tá»‘i Æ°u chi phÃ­",
    "bandwidth savings": "Tiáº¿t kiá»‡m bÄƒng thÃ´ng",

    # Page 5 / platform
    "platform & technology": "Ná»€N Táº¢NG & CÃ”NG NGHá»†",
    "how cxview delivers the transformation": "CXVIEW triá»ƒn khai chuyá»ƒn Ä‘á»•i nhÆ° tháº¿ nÃ o",
    "why a transformation partner is essential": "VÃ¬ sao cáº§n má»™t Ä‘á»‘i tÃ¡c chuyá»ƒn Ä‘á»•i",
    "core deployment benefits": "Lá»£i Ã­ch triá»ƒn khai cá»‘t lÃµi",
    "customer on - premise ai vision infrastructure": "Háº¡ táº§ng AI Vision táº¡i chá»— cá»§a khÃ¡ch hÃ ng",
    "customer on-premise ai vision infrastructure": "Háº¡ táº§ng AI Vision táº¡i chá»— cá»§a khÃ¡ch hÃ ng",
    "no camera replacement required when streams are available": "KhÃ´ng cáº§n thay camera khi cÃ³ sáºµn luá»“ng video",
    "on-premise processing supports latency, bandwidth control and data sovereignty": "Xá»­ lÃ½ táº¡i chá»— há»— trá»£ Ä‘á»™ trá»… tháº¥p, kiá»ƒm soÃ¡t bÄƒng thÃ´ng vÃ  chá»§ quyá»n dá»¯ liá»‡u",
    "automated reports reduce manual debate with employees, vendors and site teams": "BÃ¡o cÃ¡o tá»± Ä‘á»™ng giáº£m tranh luáº­n thá»§ cÃ´ng vá»›i nhÃ¢n viÃªn, nhÃ  cung cáº¥p vÃ  Ä‘á»™i ngÅ© táº¡i Ä‘á»‹a Ä‘iá»ƒm",
    "management receives objective before/after performance data for every site": "Ban quáº£n lÃ½ nháº­n dá»¯ liá»‡u hiá»‡u suáº¥t trÆ°á»›c/sau khÃ¡ch quan cho tá»«ng Ä‘á»‹a Ä‘iá»ƒm",
}

BAD_PHRASE_REPLACEMENTS_VI: List[Tuple[str, str]] = [
    ("Linhh hoáº¡t hÃ³a Ä‘Æ¡n", "Linh hoáº¡t thanh toÃ¡n"),
    ("Linhh hoáº¡t Thanh toÃ¡n", "Linh hoáº¡t thanh toÃ¡n"),
    ("Linh hoáº¡t hÃ³a Ä‘Æ¡n", "Linh hoáº¡t thanh toÃ¡n"),
    ("hÃ³a Ä‘Æ¡n hÃ ng thÃ¡ng", "thanh toÃ¡n hÃ ng thÃ¡ng"),
    ("chu ká»³ hÃ³a Ä‘Æ¡n", "chu ká»³ thanh toÃ¡n"),
    ("máº«u AI", "mÃ´ hÃ¬nh AI"),
    ("cÃ¡c máº«u AI", "cÃ¡c mÃ´ hÃ¬nh AI"),
    ("mÃ¡y áº£nh", "camera"),
    ("MÃ¡y áº£nh", "Camera"),
    ("thao tÃºng Camera", "can thiá»‡p camera"),
    ("thao tÃºng camera", "can thiá»‡p camera"),
    ("Nháº­n dáº¡ng Biá»ƒn xe", "Nháº­n dáº¡ng biá»ƒn sá»‘ xe"),
    ("Biá»ƒn xe", "biá»ƒn sá»‘ xe"),
    ("váº­t thá»ƒ khÃ´ng cÃ³ ngÆ°á»i", "váº­t thá»ƒ bá» quÃªn"),
    ("Äá»‘i tÆ°á»£ng Bá» quÃªn", "váº­t thá»ƒ bá» quÃªn"),
    ("leo climbing", "leo"),
    ("PhÃ¡t hiá»‡n Äáº¥u tranh", "PhÃ¡t hiá»‡n Ä‘Ã¡nh nhau"),
    ("PhÃ¡t hiá»‡n Ä‘áº¥u tranh", "PhÃ¡t hiá»‡n Ä‘Ã¡nh nhau"),
    ("Ä‘Ã¡nh tranh", "Ä‘Ã¡nh nhau"),
    ("LÃ m sáº¡ch bÃ n", "dá»n bÃ n"),
    ("Táº§m nhÃ¬n TrÃªn chá»—", "AI Vision táº¡i chá»—"),
    ("TrÃªn chá»—", "táº¡i chá»—"),
    ("trÃªn chá»—", "táº¡i chá»—"),
    ("thiáº¿t bá»‹ cáº¡nh", "thiáº¿t bá»‹ biÃªn"),
    ("trÃ­ tuá»‡ Edge", "trÃ­ tuá»‡ biÃªn"),
    ("Ä‘á»™ng cÆ¡ quyáº¿t Ä‘á»‹nh", "há»‡ thá»‘ng ra quyáº¿t Ä‘á»‹nh"),
    ("cÆ¡ sá»Ÿ háº¡ táº§ng báº£o thá»§ CCTV", "háº¡ táº§ng CCTV truyá»n thá»‘ng"),
    ("báº£o thá»§ CCTV", "CCTV truyá»n thá»‘ng"),
    ("nhÃ³m trang web", "Ä‘á»™i ngÅ© táº¡i Ä‘á»‹a Ä‘iá»ƒm"),
    ("má»i trang web", "má»i Ä‘á»‹a Ä‘iá»ƒm"),
    ("trang web", "Ä‘á»‹a Ä‘iá»ƒm"),
    ("trang web.", "Ä‘á»‹a Ä‘iá»ƒm."),
    ("site teams", "Ä‘á»™i ngÅ© táº¡i Ä‘á»‹a Ä‘iá»ƒm"),
    ("thÃºc bÃ¡ch", "cháº·t cháº½"),
    ("cáº¥p bÃ¡ch hoáº¡t Ä‘á»™ng", "tÃ­nh cáº¥p thiáº¿t trong váº­n hÃ nh"),
]

BANNED_TRANSLATION_FRAGMENTS_VI = [
    "Linhh", "leo climbing", "Äáº¥u tranh", "Ä‘áº¥u tranh", "thao tÃºng Camera", "thao tÃºng camera",
    "váº­t thá»ƒ khÃ´ng cÃ³ ngÆ°á»i", "Biá»ƒn xe", "mÃ¡y áº£nh", "MÃ¡y áº£nh", "trang web", "TrÃªn chá»—", "trÃªn chá»—",
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
    t = re.sub(r"([â€¢âœ“Ã¼])\s*", r"\1 ", t)
    t = re.sub(r"\s+", " ", t) if "\n" not in t else re.sub(r"[ \t]+", " ", t)

    # Keep protected product names clean if the model inserted odd spacing/casing.
    t = re.sub(r"CXVIEW\s+GPT\s+Box", "CXVIEW GPT Box", t, flags=re.I)
    t = re.sub(r"AI\s+Video\s+Analytics", "AI Video Analytics", t, flags=re.I)
    t = re.sub(r"\bGPT\s+Há»™p\b", "GPT Box", t, flags=re.I)
    t = re.sub(r"CXVIEW\s+GPT\s+Há»™p", "CXVIEW GPT Box", t, flags=re.I)

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
        probe = "ÄƒÃ¢ÃªÃ´Æ¡Æ°Ä‘Ä‚Ã‚ÃŠÃ”Æ Æ¯ÄÃ¡Ã áº£Ã£áº¡áº¥áº§áº©áº«áº­áº¯áº±áº³áºµáº·Ã©Ã¨áº»áº½áº¹áº¿á»á»ƒá»…á»‡Ã­Ã¬á»‰Ä©á»‹Ã³Ã²á»Ãµá»á»‘á»“á»•á»—á»™á»›á»á»Ÿá»¡á»£ÃºÃ¹á»§Å©á»¥á»©á»«á»­á»¯á»±Ã½á»³á»·á»¹á»µ"
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
                if f.has_glyph(ord("âœ“")):
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
        if draw_line.startswith("âœ“ "):
            leading_check = True
            draw_line = draw_line[2:].lstrip()
        elif draw_line == "âœ“":
            leading_check = True
            draw_line = ""

        full_line_for_measure = ("âœ“ " + draw_line) if leading_check else draw_line
        if layout.align == "center":
            w = measure_font.text_length(full_line_for_measure.replace("âœ“", "â€¢"), fontsize=layout.fontsize)
            x = x0 + max(0, (layout.rect.width - w) / 2)
        elif layout.align == "right":
            w = measure_font.text_length(full_line_for_measure.replace("âœ“", "â€¢"), fontsize=layout.fontsize)
            x = x1 - w
        else:
            x = x0

        if leading_check:
            if symbol_file and symbol_font:
                page.insert_text(
                    point=fitz.Point(x, y),
                    text="âœ“",
                    fontsize=layout.fontsize,
                    fontname="FSymbolVN",
                    fontfile=symbol_file,
                    color=color,
                    overlay=True,
                )
                x += symbol_font.text_length("âœ“ ", fontsize=layout.fontsize) + 1.0
            else:
                # Last-resort symbol that most sans fonts support.
                page.insert_text(
                    point=fitz.Point(x, y),
                    text="â€¢",
                    fontsize=layout.fontsize,
                    fontname=fontname,
                    fontfile=fontfile,
                    color=color,
                    overlay=True,
                )
                x += measure_font.text_length("â€¢ ", fontsize=layout.fontsize) + 1.0

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
    t = t.replace("\uf0fc", "âœ“")
    # Many PDFs using Wingdings expose checkmarks as Ã¼. Treat it as a checkmark
    # only at bullet/check positions so normal words are not affected.
    t = re.sub(r"(?m)^\s*Ã¼\s*", "âœ“ ", t)
    t = re.sub(r"(?m)([\n\r])\s*Ã¼\s*", r"\1âœ“ ", t)
    # Normalize common bullet lookalikes.
    t = t.replace("Â·", "â€¢")
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
        keep_with_prev = prev_text in {"â€¢", "âœ“", "Ã¼", "-", "â€“"}
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
    "our solutions": "GIáº¢I PHÃP",
    "vehicle plate recognition": "Nháº­n dáº¡ng biá»ƒn sá»‘",
    "fence and wall climbing detection": "Leo rÃ o/tÆ°á»ng",
    "intrusion detection": "XÃ¢m nháº­p",
    "camera tampering detection": "Can thiá»‡p camera",
    "ppe and uniform detection": "PPE & Ä‘á»“ng phá»¥c",
    "forklift and vehicle safety detection": "An toÃ n xe nÃ¢ng",
    "falls and slips detection": "TÃ© ngÃ£/trÆ°á»£t",
    "fighting detection": "ÄÃ¡nh nhau",
    "unusual crowd detection": "ÄÃ¡m Ä‘Ã´ng báº¥t thÆ°á»ng",
    "smoke and fire detection": "KhÃ³i/lá»­a",
    "smart workstation monitoring": "GiÃ¡m sÃ¡t tráº¡m",
    "automated product counting": "Äáº¿m sáº£n pháº©m",
    "product quality inspection": "Kiá»ƒm tra cháº¥t lÆ°á»£ng",
    "heat maps & route maps analysis": "Báº£n Ä‘á»“ nhiá»‡t/lá»™ trÃ¬nh",
    "heat maps and route maps analysis": "Báº£n Ä‘á»“ nhiá»‡t/lá»™ trÃ¬nh",
    "dwell time report": "Thá»i gian lÆ°u láº¡i",
    "traffic counting": "Äáº¿m lÆ°u lÆ°á»£ng",
    "customer demographic analysis": "NhÃ¢n kháº©u há»c",
    "customer engagement detection": "TÆ°Æ¡ng tÃ¡c khÃ¡ch hÃ ng",
    "table cleaning detection": "Dá»n bÃ n",
    "patrol automatic report": "BÃ¡o cÃ¡o tuáº§n tra",
    "unattended object detection": "Váº­t thá»ƒ bá» quÃªn",
    "package": "GÃ³i",
    "tier": "Cáº¥p",
    "cameras": "Camera",
    "concurrent ai streams active on ai models existed": "Luá»“ng AI Ä‘á»“ng thá»i\ntrÃªn mÃ´ hÃ¬nh hiá»‡n cÃ³",
    "concurrent ai streams": "Luá»“ng AI Ä‘á»“ng thá»i",
    "active on ai models existed": "trÃªn mÃ´ hÃ¬nh hiá»‡n cÃ³",
    "monthly (usd)": "HÃ ng thÃ¡ng\n(USD)",
    "24-month (usd)": "24 thÃ¡ng\n(USD)",
    "36-month (usd)": "36 thÃ¡ng\n(USD)",
    "billing flexibility": "Thanh toÃ¡n linh hoáº¡t",
    "concurrent ai stream ratio": "Tá»· lá»‡ luá»“ng AI Ä‘á»“ng thá»i",
    "book your live demo": "Äáº·t lá»‹ch demo",
    "customer on - premise ai vision infrastructure": "Háº¡ táº§ng AI Vision táº¡i chá»—",
    "customer on-premise ai vision infrastructure": "Háº¡ táº§ng AI Vision táº¡i chá»—",
    "no camera replacement required when streams are available": "KhÃ´ng cáº§n thay camera khi cÃ³ luá»“ng video",
    "on-premise processing supports latency, bandwidth control and data sovereignty": "Xá»­ lÃ½ táº¡i chá»—: Ä‘á»™ trá»… tháº¥p, tiáº¿t kiá»‡m bÄƒng thÃ´ng, chá»§ quyá»n dá»¯ liá»‡u",
    "automated reports reduce manual debate with employees, vendors and site teams": "BÃ¡o cÃ¡o tá»± Ä‘á»™ng giáº£m tranh luáº­n vá»›i nhÃ¢n viÃªn, nhÃ  cung cáº¥p vÃ  Ä‘á»™i ngÅ© táº¡i Ä‘á»‹a Ä‘iá»ƒm",
    "management receives objective before/after performance data for every site": "Ban quáº£n lÃ½ cÃ³ dá»¯ liá»‡u trÆ°á»›c/sau khÃ¡ch quan cho tá»«ng Ä‘á»‹a Ä‘iá»ƒm",
    "Ã¼no camera replacement required when streams are available": "âœ“ KhÃ´ng cáº§n thay camera khi cÃ³ luá»“ng video",
    "Ã¼on-premise processing supports latency, bandwidth control and data sovereignty": "âœ“ Xá»­ lÃ½ táº¡i chá»—: Ä‘á»™ trá»… tháº¥p, tiáº¿t kiá»‡m bÄƒng thÃ´ng, chá»§ quyá»n dá»¯ liá»‡u",
    "Ã¼automated reports reduce manual debate with employees, vendors and site teams": "âœ“ BÃ¡o cÃ¡o tá»± Ä‘á»™ng giáº£m tranh luáº­n vá»›i nhÃ¢n viÃªn, nhÃ  cung cáº¥p vÃ  Ä‘á»™i ngÅ© táº¡i Ä‘á»‹a Ä‘iá»ƒm",
    "Ã¼management receives objective before/after performance data for every site": "âœ“ Ban quáº£n lÃ½ cÃ³ dá»¯ liá»‡u trÆ°á»›c/sau khÃ¡ch quan cho tá»«ng Ä‘á»‹a Ä‘iá»ƒm",
})

BAD_PHRASE_REPLACEMENTS_VI.extend([
    ("TÃ¡c Ä‘á»™ng kinh doanh", "TÃ¡c Ä‘á»™ng kinh doanh"),
    ("thá»i gian biá»ƒu tham vá»ng", "lá»™ trÃ¬nh triá»ƒn khai nhanh"),
    ("sá»± tÃ­nh cáº¥p thiáº¿t", "tÃ­nh cáº¥p thiáº¿t"),
    ("thÃ´ng tin hÃ nh Ä‘á»™ng", "thÃ´ng tin cÃ³ thá»ƒ hÃ nh Ä‘á»™ng"),
    ("theo dÃµi báº£ng Ä‘iá»u khiá»ƒn", "dashboard"),
    ("AI mÃ´ hÃ¬nh", "mÃ´ hÃ¬nh AI"),
    ("CXVIEW AI mÃ´ hÃ¬nh", "mÃ´ hÃ¬nh CXVIEW AI"),
])
BANNED_TRANSLATION_FRAGMENTS_VI.extend(["thá»i gian biá»ƒu tham vá»ng", "sá»± tÃ­nh cáº¥p thiáº¿t", "AI mÃ´ hÃ¬nh"])


def normalize_translation_key(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("\n", " ")
    text = text.replace("&", " & ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    # Match Wingdings checkmark extraction exposed as Ã¼.
    text = re.sub(r"^âœ“\s*", "Ã¼", text)
    return text


def postprocess_translation(text: str, original_text: str, role: str = "body", target_lang: str = "vi") -> str:
    if not text or not target_lang.lower().startswith("vi"):
        return text
    t = normalize_special_chars(text.strip())

    # Preserve leading bullets/checkmarks from the source when the model drops them.
    original_clean = normalize_special_chars(original_text.strip())
    if original_clean.startswith("â€¢") and not t.startswith("â€¢"):
        t = "â€¢ " + t.lstrip("â€¢ ").strip()
    if original_clean.startswith("âœ“") and not t.startswith("âœ“"):
        t = "âœ“ " + t.lstrip("âœ“ ").strip()

    for bad, good in BAD_PHRASE_REPLACEMENTS_VI:
        t = t.replace(bad, good)

    # Product and acronym cleanup.
    t = re.sub(r"CXVIEW\s+GPT\s+Box", "CXVIEW GPT Box", t, flags=re.I)
    t = re.sub(r"CXVIEW\s+GPT\s+Há»™p", "CXVIEW GPT Box", t, flags=re.I)
    t = re.sub(r"\bGPT\s+Há»™p\b", "GPT Box", t, flags=re.I)
    t = re.sub(r"AI\s+Video\s+Analytics", "AI Video Analytics", t, flags=re.I)
    t = re.sub(r"\b24\s*-\s*thÃ¡ng\b", "24 thÃ¡ng", t, flags=re.I)
    t = re.sub(r"\b36\s*-\s*thÃ¡ng\b", "36 thÃ¡ng", t, flags=re.I)

    # Normalize spacing without destroying deliberate line breaks.
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([â€¢âœ“])\s*", r"\1 ", t)
    if "\n" in t:
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r" *\n *", "\n", t)
    else:
        t = re.sub(r"\s+", " ", t)

    # Tiny labels: no trailing punctuation and avoid verbose prefix where memory failed.
    if role in {"label", "title", "cta"}:
        t = t.strip().rstrip(".")
        t = re.sub(r"^PhÃ¡t hiá»‡n\s+(can thiá»‡p camera|xÃ¢m nháº­p|khÃ³i/lá»­a|Ä‘Ã¡nh nhau|váº­t thá»ƒ bá» quÃªn)$", r"\1", t, flags=re.I)

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
        sample = "ÄƒÃ¢Ä‘ÃªÃ´Æ¡Æ°ÃÃ€áº¢Ãƒáº áº¿á»‡á»™á»£á»­á»¯Äâœ“â€¢"
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
                if f.has_glyph(ord("âœ“")):
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
        if draw_line.startswith("âœ“ "):
            leading_check = True
            draw_line = draw_line[2:].lstrip()
        elif draw_line == "âœ“":
            leading_check = True
            draw_line = ""

        full_line_for_measure = ("âœ“ " + draw_line) if leading_check else draw_line
        if layout.align == "center":
            w = measure_font.text_length(full_line_for_measure.replace("âœ“", "â€¢"), fontsize=layout.fontsize)
            x = x0 + max(0, (layout.rect.width - w) / 2)
        elif layout.align == "right":
            w = measure_font.text_length(full_line_for_measure.replace("âœ“", "â€¢"), fontsize=layout.fontsize)
            x = x1 - w
        else:
            x = x0

        if leading_check:
            if symbol_file and symbol_font:
                page.insert_text(
                    point=fitz.Point(x, y),
                    text="âœ“",
                    fontsize=layout.fontsize,
                    fontname="FSymbolVN",
                    fontfile=symbol_file,
                    color=color,
                    overlay=True,
                )
                x += symbol_font.text_length("âœ“ ", fontsize=layout.fontsize) + 1.0
            else:
                # Last-resort symbol that most sans fonts support.
                page.insert_text(
                    point=fitz.Point(x, y),
                    text="â€¢",
                    fontsize=layout.fontsize,
                    fontname=fontname,
                    fontfile=fontfile,
                    color=color,
                    overlay=True,
                )
                x += measure_font.text_length("â€¢ ", fontsize=layout.fontsize) + 1.0

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
    "our solutions": "GIáº¢I PHÃP",
    "our process": "QUY TRÃŒNH",
    "platform & technology": "Ná»€N Táº¢NG & CÃ”NG NGHá»†",
    "about cxview": "GIá»šI THIá»†U CXVIEW",

    # pricing table
    "package": "GÃ³i",
    "tier": "Cáº¥p",
    "cameras": "Camera",
    "concurrent ai streams": "Luá»“ng AI Ä‘á»“ng thá»i",
    "active on ai models existed": "trÃªn mÃ´ hÃ¬nh AI hiá»‡n cÃ³",
    "concurrent ai streams active on ai models existed": "Luá»“ng AI Ä‘á»“ng thá»i\ntrÃªn mÃ´ hÃ¬nh AI hiá»‡n cÃ³",
    "monthly (usd)": "HÃ ng thÃ¡ng\n(USD)",
    "24-month (usd)": "24 thÃ¡ng\n(USD)",
    "36-month (usd)": "36 thÃ¡ng\n(USD)",
    "concurrent ai stream ratio": "Tá»· lá»‡ luá»“ng AI Ä‘á»“ng thá»i",
    "billing flexibility": "Linh hoáº¡t thanh toÃ¡n",

    # page 3 categories
    "ai camera in security": "AI CAMERA\nAN NINH",
    "ai camera in operation": "AI CAMERA\nVáº¬N HÃ€NH",
    "ai camera in safety": "AI CAMERA\nAN TOÃ€N",

    # page 3 solution labels - ultra short versions
    "camera tampering detection": "Can thiá»‡p camera",
    "vehicle plate recognition": "Nháº­n dáº¡ng biá»ƒn sá»‘",
    "patrol automatic report": "BÃ¡o cÃ¡o tuáº§n tra",
    "unattended object detection": "Váº­t thá»ƒ bá» quÃªn",
    "fence and wall climbing detection": "Leo rÃ o/tÆ°á»ng",
    "intrusion detection": "XÃ¢m nháº­p",
    "product quality inspection": "Kiá»ƒm tra cháº¥t lÆ°á»£ng",
    "customer demographic analysis": "NhÃ¢n kháº©u há»c",
    "customer engagement detection": "TÆ°Æ¡ng tÃ¡c khÃ¡ch hÃ ng",
    "smart workstation monitoring": "GiÃ¡m sÃ¡t tráº¡m",
    "heat maps & route maps analysis": "Báº£n Ä‘á»“ nhiá»‡t/lá»™ trÃ¬nh",
    "heat maps and route maps analysis": "Báº£n Ä‘á»“ nhiá»‡t/lá»™ trÃ¬nh",
    "traffic counting": "Äáº¿m lÆ°u lÆ°á»£ng",
    "automated product counting": "Äáº¿m sáº£n pháº©m",
    "dwell time report": "Thá»i gian lÆ°u láº¡i",
    "table cleaning detection": "Dá»n bÃ n",
    "falls and slips detection": "TÃ© ngÃ£/trÆ°á»£t",
    "unusual crowd detection": "ÄÃ¡m Ä‘Ã´ng báº¥t thÆ°á»ng",
    "ppe and uniform detection": "PPE & Ä‘á»“ng phá»¥c",
    "forklift and vehicle safety detection": "An toÃ n xe nÃ¢ng",
    "smoke and fire detection": "KhÃ³i/lá»­a",
    "fighting detection": "ÄÃ¡nh nhau",

    # page 4 process
    "strategic understanding": "Tháº¥u hiá»ƒu chiáº¿n lÆ°á»£c",
    "tailored solution architecture": "Kiáº¿n trÃºc giáº£i phÃ¡p tÃ¹y chá»‰nh",
    "pilot (poc) & validation": "Pilot (POC) & xÃ¡c thá»±c",
    "scale & sustain": "Má»Ÿ rá»™ng & duy trÃ¬",
    "book your live demo": "Äáº·t lá»‹ch\ndemo",
    "near-zero latency": "Äá»™ trá»… gáº§n 0",
    "data sovereignty": "Chá»§ quyá»n dá»¯ liá»‡u",
    "cost efficiency": "Tá»‘i Æ°u chi phÃ­",
    "bandwidth savings": "Tiáº¿t kiá»‡m bÄƒng thÃ´ng",

    # page 5
    "how cxview delivers the transformation": "CXVIEW triá»ƒn khai chuyá»ƒn Ä‘á»•i nhÆ° tháº¿ nÃ o",
    "why a transformation partner is essential": "VÃ¬ sao cáº§n Ä‘á»‘i tÃ¡c chuyá»ƒn Ä‘á»•i",
    "core deployment benefits": "Lá»£i Ã­ch triá»ƒn khai cá»‘t lÃµi",
    "customer on - premise ai vision infrastructure": "Háº¡ táº§ng AI Vision táº¡i chá»—",
    "customer on-premise ai vision infrastructure": "Háº¡ táº§ng AI Vision táº¡i chá»—",
    "no camera replacement required when streams are available": "KhÃ´ng cáº§n thay camera khi cÃ³ luá»“ng video",
    "on-premise processing supports latency, bandwidth control and data sovereignty": "Xá»­ lÃ½ táº¡i chá»—: Ä‘á»™ trá»… tháº¥p, tiáº¿t kiá»‡m bÄƒng thÃ´ng, chá»§ quyá»n dá»¯ liá»‡u",
    "automated reports reduce manual debate with employees, vendors and site teams": "BÃ¡o cÃ¡o tá»± Ä‘á»™ng giáº£m tranh luáº­n vá»›i nhÃ¢n viÃªn, nhÃ  cung cáº¥p vÃ  Ä‘á»™i ngÅ© táº¡i Ä‘á»‹a Ä‘iá»ƒm",
    "management receives objective before/after performance data for every site": "Ban quáº£n lÃ½ cÃ³ dá»¯ liá»‡u trÆ°á»›c/sau khÃ¡ch quan cho tá»«ng Ä‘á»‹a Ä‘iá»ƒm",
}


# V_USERFIX_HEADER_MEMORY
EXACT_TRANSLATION_MEMORY_VI.update({
    "how cxview delivers the transformation": "CXVIEW triá»ƒn khai chuyá»ƒn Ä‘á»•i nhÆ° tháº¿ nÃ o",
    "understand your goals, challenges and objectives.": "Hiá»ƒu má»¥c tiÃªu, thÃ¡ch thá»©c vÃ  má»¥c tiÃªu cá»§a báº¡n.",
    "assess current systems & environment.": "ÄÃ¡nh giÃ¡ há»‡ thá»‘ng & mÃ´i trÆ°á»ng hiá»‡n táº¡i.",
})
COMPACT_TRANSLATION_MEMORY_VI.update({
    "how cxview delivers the transformation": "CXVIEW triá»ƒn khai chuyá»ƒn Ä‘á»•i nhÆ° tháº¿ nÃ o",
    "understand your goals, challenges and objectives.": "Hiá»ƒu má»¥c tiÃªu & thÃ¡ch thá»©c cá»§a báº¡n.",
    "assess current systems & environment.": "ÄÃ¡nh giÃ¡ há»‡ thá»‘ng & mÃ´i trÆ°á»ng hiá»‡n táº¡i.",
})



# V7 all-text translation memory for extractable headers / marketing copy.
EXACT_TRANSLATION_MEMORY_VI.update({
    "client challenges. cxview solutions. business impacts.": "ThÃ¡ch thá»©c khÃ¡ch hÃ ng. Giáº£i phÃ¡p CXVIEW. TÃ¡c Ä‘á»™ng kinh doanh.",
    "how cxview delivers the transformation": "CXVIEW triá»ƒn khai chuyá»ƒn Ä‘á»•i nhÆ° tháº¿ nÃ o",
    "cxview smart ai video analytics solution": "GIáº¢I PHÃP AI VIDEO ANALYTICS THÃ”NG MINH CXVIEW",
    "cxview gpt box & ai video analytics": "CXVIEW GPT Box & AI Video Analytics",
    "& ai video analytics": "& AI Video Analytics",
    "the new era of physical edge ai": "Ká»· nguyÃªn má»›i cá»§a AI biÃªn váº­t lÃ½",
    "near-zero latency": "Äá»™ trá»… gáº§n báº±ng 0",
    "data sovereignty": "Chá»§ quyá»n dá»¯ liá»‡u",
    "cost efficiency": "Tá»‘i Æ°u chi phÃ­",
    "bandwidth savings": "Tiáº¿t kiá»‡m bÄƒng thÃ´ng",
})

COMPACT_TRANSLATION_MEMORY_VI.update({
    "client challenges. cxview solutions. business impacts.": "ThÃ¡ch thá»©c khÃ¡ch hÃ ng. Giáº£i phÃ¡p CXVIEW. TÃ¡c Ä‘á»™ng kinh doanh.",
    "how cxview delivers the transformation": "CXVIEW triá»ƒn khai chuyá»ƒn Ä‘á»•i nhÆ° tháº¿ nÃ o",
    "cxview smart ai video analytics solution": "GIáº¢I PHÃP AI VIDEO ANALYTICS THÃ”NG MINH CXVIEW",
    "the new era of physical edge ai": "Ká»· nguyÃªn má»›i cá»§a AI biÃªn váº­t lÃ½",
    "near-zero latency": "Äá»™ trá»… gáº§n 0",
    "data sovereignty": "Chá»§ quyá»n dá»¯ liá»‡u",
    "cost efficiency": "Tá»‘i Æ°u chi phÃ­",
    "bandwidth savings": "Tiáº¿t kiá»‡m bÄƒng thÃ´ng",
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
    ("ï¬", "fi"),
    ("ï¬‚", "fl"),
    ("â€“", "-"),
    ("â€”", "-"),
    ("â€œ", '"'),
    ("â€", '"'),
    ("â€˜", "'"),
    ("â€™", "'"),
    ("minimum 24 thÃ¡ng", "tá»‘i thiá»ƒu 24 thÃ¡ng"),
    ("minimum 24-thÃ¡ng", "tá»‘i thiá»ƒu 24 thÃ¡ng"),
    ("CXVIEW mÃ´ hÃ¬nh AI", "mÃ´ hÃ¬nh AI cá»§a CXVIEW"),
    ("CXVIEW AI mÃ´ hÃ¬nh", "mÃ´ hÃ¬nh AI cá»§a CXVIEW"),
    ("mÃ¡y áº£nh", "camera"),
    ("MÃ¡y áº£nh", "Camera"),
    ("hÃ³a Ä‘Æ¡n", "thanh toÃ¡n"),
    ("trang web", "Ä‘á»‹a Ä‘iá»ƒm"),
    ("nhÃ³m site", "Ä‘á»™i ngÅ© táº¡i Ä‘á»‹a Ä‘iá»ƒm"),
    ("site teams", "Ä‘á»™i ngÅ© táº¡i Ä‘á»‹a Ä‘iá»ƒm"),
    ("thiáº¿t bá»‹ cáº¡nh", "thiáº¿t bá»‹ biÃªn"),
    ("trÃ­ tuá»‡ Edge", "trÃ­ tuá»‡ biÃªn"),
    ("Äáº¥u tranh", "Ä‘Ã¡nh nhau"),
    ("Ä‘áº¥u tranh", "Ä‘Ã¡nh nhau"),
    ("leo climbing", "leo"),
    ("váº­t thá»ƒ khÃ´ng cÃ³ ngÆ°á»i", "váº­t thá»ƒ bá» quÃªn"),
    ("thao tÃºng camera", "can thiá»‡p camera"),
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

    # Convert checkmark glyphs extracted as Ã¼ only when they behave like list markers.
    t = re.sub(r"(^|\n|\s)[Ã¼âœ“]\s*", lambda m: m.group(1) + "âœ“ ", t)

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
    t = re.sub(r"GIáº¢I PHÃP\s+OUR\s+SOLUTIONS", "GIáº¢I PHÃP", t, flags=re.I)
    t = re.sub(r"QUY TRÃŒNH\s+OUR\s+PROCESS", "QUY TRÃŒNH", t, flags=re.I)
    t = re.sub(r"Ná»€N Táº¢NG\s*&\s*CÃ”NG NGHá»†\s+PLATFORM\s*&\s*TECHNOLOGY", "Ná»€N Táº¢NG & CÃ”NG NGHá»†", t, flags=re.I)

    # Stable spacing, preserving explicit newlines.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in t.splitlines()]
    t = "\n".join(line for line in lines if line != "")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    # Tiny labels should not end with sentence punctuation.
    if role in {"label", "title", "cta", "table", "table_cell", "header"}:
        t = t.rstrip(" .ã€‚")
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
        ("PhÃ¡t hiá»‡n ", ""),
        ("BÃ¡o cÃ¡o ", ""),
        ("GiÃ¡m sÃ¡t ", ""),
        ("PhÃ¢n tÃ­ch ", ""),
        ("Kiá»ƒm tra ", ""),
        ("Nháº­n dáº¡ng ", ""),
        ("tá»± Ä‘á»™ng", ""),
        ("khÃ¡ch hÃ ng", ""),
        ("thá»i gian ", ""),
        ("sáº£n pháº©m", "SP"),
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
        if draw_line.startswith("âœ“ "):
            leading_check = True
            draw_line = draw_line[2:].lstrip()
        elif draw_line == "âœ“":
            leading_check = True
            draw_line = ""

        full_for_measure = ("âœ“ " + draw_line) if leading_check else draw_line
        measure_line = full_for_measure.replace("âœ“", "â€¢")

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
                    text="âœ“",
                    fontsize=fontsize,
                    fontname="FSymbolVN",
                    fontfile=symbol_file,
                    color=line_color,
                    overlay=True,
                )
                x += symbol_font.text_length("âœ“ ", fontsize=fontsize) + 1.0
            else:
                page.insert_text(
                    point=fitz.Point(x, baseline),
                    text="â€¢",
                    fontsize=fontsize,
                    fontname=fontname,
                    fontfile=fontfile,
                    color=line_color,
                    overlay=True,
                )
                x += measure_font.text_length("â€¢ ", fontsize=fontsize) + 1.0

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
    marks = set("ÄƒÃ¢Ä‘ÃªÃ´Æ¡Æ°Ä‚Ã‚ÄÃŠÃ”Æ Æ¯Ã¡Ã áº£Ã£áº¡áº¥áº§áº©áº«áº­áº¯áº±áº³áºµáº·Ã©Ã¨áº»áº½áº¹áº¿á»á»ƒá»…á»‡Ã­Ã¬á»‰Ä©á»‹Ã³Ã²á»Ãµá»á»‘á»“á»•á»—á»™á»›á»á»Ÿá»¡á»£ÃºÃ¹á»§Å©á»¥á»©á»«á»­á»¯á»±Ã½á»³á»·á»¹á»µ")
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



# ============================================================
# V13 QUALITY OCR CONTROLS
# ============================================================

def parse_page_set_v13(spec: str) -> Optional[set]:
    """Parse 1-based pages like '1,3-5' into 0-based page indexes."""
    spec = (spec or "").strip()
    if not spec:
        return None
    pages = set()
    for part in re.split(r"[,; ]+", spec):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = int(a)
                end = int(b)
            except Exception:
                continue
            for p in range(min(start, end), max(start, end) + 1):
                if p > 0:
                    pages.add(p - 1)
        else:
            try:
                p = int(part)
                if p > 0:
                    pages.add(p - 1)
            except Exception:
                continue
    return pages


def ocr_page_allowed_v13(page_index: int) -> bool:
    pages = parse_page_set_v13(os.getenv("PDF_TRANSLATOR_OCR_PAGES", ""))
    if pages is None:
        return True
    return page_index in pages


def ocr_min_width_v13() -> float:
    return float(os.getenv("PDF_TRANSLATOR_OCR_MIN_WIDTH_PT", "10"))


def ocr_min_height_v13() -> float:
    return float(os.getenv("PDF_TRANSLATOR_OCR_MIN_HEIGHT_PT", "5.8"))


def ocr_max_height_v13() -> float:
    return float(os.getenv("PDF_TRANSLATOR_OCR_MAX_HEIGHT_PT", "26"))


def ocr_bbox_size_ok_v13(block: Dict) -> Tuple[bool, str]:
    bbox = tuple(float(x) for x in block.get("bbox", (0, 0, 0, 0)))
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    if w < ocr_min_width_v13():
        return False, "too_narrow"
    if h < ocr_min_height_v13():
        return False, "too_short"
    if h > ocr_max_height_v13():
        return False, "too_tall"
    return True, "size_ok"


def draw_centered_textbox_v13(page: fitz.Page, rect: fitz.Rect, text: str, fontfile: Optional[str], fontname: str, fontsize: float, color: RGB):
    """Center text in a small bbox, shrinking until it fits."""
    size = fontsize
    while size >= 4.0:
        rc = page.insert_textbox(
            rect,
            text,
            fontsize=size,
            fontname=fontname,
            fontfile=fontfile,
            color=color,
            align=fitz.TEXT_ALIGN_CENTER,
            overlay=True,
        )
        if rc >= 0:
            return True
        size -= 0.3
    page.insert_text(
        fitz.Point(rect.x0 + 2, rect.y0 + max(4.5, fontsize * 0.85)),
        text,
        fontsize=max(4.0, size),
        fontname=fontname,
        fontfile=fontfile,
        color=color,
        overlay=True,
    )
    return False



# ============================================================
# V15 REGION PATCH MAP
# ============================================================

def parse_color_v15(value, default=(0, 0, 0)) -> RGB:
    """Parse color from '#RRGGBB', [0..1], [0..255], or None."""
    if value is None:
        return default
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("#") and len(s) == 7:
            try:
                return (int(s[1:3], 16) / 255.0, int(s[3:5], 16) / 255.0, int(s[5:7], 16) / 255.0)
            except Exception:
                return default
        named = {
            "white": (1, 1, 1),
            "black": (0, 0, 0),
            "purple": (0.45, 0.23, 0.64),
            "dark": (0.08, 0.08, 0.12),
        }
        return named.get(s.lower(), default)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        vals = [float(value[0]), float(value[1]), float(value[2])]
        if max(vals) > 1.0:
            vals = [v / 255.0 for v in vals]
        return (max(0, min(1, vals[0])), max(0, min(1, vals[1])), max(0, min(1, vals[2])))
    return default


def draw_patch_text_v15(
    page: fitz.Page,
    rect: fitz.Rect,
    text_value: str,
    fontfile: Optional[str],
    fontname: str,
    fontsize: float,
    color: RGB,
    align: str = "center",
    line_height: float = 1.05,
):
    """Draw patch text with shrinking and simple alignment."""
    if not text_value:
        return False

    align_map = {
        "left": fitz.TEXT_ALIGN_LEFT,
        "center": fitz.TEXT_ALIGN_CENTER,
        "right": fitz.TEXT_ALIGN_RIGHT,
    }
    pdf_align = align_map.get(str(align).lower(), fitz.TEXT_ALIGN_CENTER)

    size = float(fontsize or 8)
    min_size = max(3.5, size * 0.62)

    while size >= min_size:
        try:
            rc = page.insert_textbox(
                rect,
                text_value,
                fontsize=size,
                fontname=fontname,
                fontfile=fontfile,
                color=color,
                align=pdf_align,
                overlay=True,
            )
            if rc >= 0:
                return True
        except Exception:
            pass
        size -= 0.25

    try:
        page.insert_text(
            fitz.Point(rect.x0 + 1.0, rect.y0 + max(4.0, min_size)),
            text_value,
            fontsize=min_size,
            fontname=fontname,
            fontfile=fontfile,
            color=color,
            overlay=True,
        )
        return False
    except Exception:
        return False


def apply_region_patch_map_v15(
    pdf_path: str,
    output_pdf: str,
    patch_map_path: str,
    font_regular: Optional[str] = None,
    font_bold: Optional[str] = None,
):
    """Apply JSON region patch map.

    JSON shape:
    {
      "patches": [
        {
          "page": 4,                       # 1-based
          "bbox": [648,129,777,152.5],     # PDF points
          "text": "Äá»˜ TRá»„ Gáº¦N 0",
          "fill": "#733AA3",               # null means no cover
          "color": "#FFFFFF",
          "font_size": 7.5,
          "weight": "bold",
          "align": "center"
        }
      ]
    }
    """
    with open(patch_map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    patches = data.get("patches", data if isinstance(data, list) else [])
    if not isinstance(patches, list):
        raise ValueError("patch map must be a list or an object with a 'patches' list")

    pdf = fitz.open(pdf_path)
    font_regular = first_existing_font_v10(font_regular, font_bold)
    font_bold = first_existing_font_v10(font_bold, font_regular)

    applied = 0
    for p in patches:
        if not isinstance(p, dict):
            continue

        page_num = int(p.get("page", 1))
        page_index = page_num - 1
        if page_index < 0 or page_index >= len(pdf):
            continue

        bbox = p.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        page = pdf[page_index]
        rect = fitz.Rect(*[float(x) for x in bbox])

        fill_raw = p.get("fill", None)
        if fill_raw is not None:
            fill = parse_color_v15(fill_raw, default=(1, 1, 1))
            page.draw_rect(rect, color=None, fill=fill, overlay=True)

        text_value = str(p.get("text", ""))
        color = parse_color_v15(p.get("color", "#000000"), default=(0, 0, 0))
        fontsize = float(p.get("font_size", 8.0))
        weight = str(p.get("weight", "regular")).lower()
        align = str(p.get("align", "center")).lower()

        fontfile = font_bold if weight in {"bold", "semibold", "black"} else font_regular
        fontname = "FRegionPatchBold" if weight in {"bold", "semibold", "black"} else "FRegionPatchRegular"

        draw_patch_text_v15(
            page=page,
            rect=rect,
            text_value=text_value,
            fontfile=fontfile,
            fontname=fontname,
            fontsize=fontsize,
            color=color,
            align=align,
        )
        applied += 1

    pdf.save(output_pdf, garbage=4, deflate=True)
    pdf.close()
    print(f"Region patch map applied: patches={applied} output={output_pdf}")


def apply_cxview_manual_patches_v13(
    pdf_path: str,
    output_pdf: str,
    font_regular: Optional[str] = None,
    font_bold: Optional[str] = None,
):
    """Targeted patch for the current CXVIEW deck.

    V14 change: this patch is deliberately narrow. It does NOT touch the main
    title/body text that the text-layer renderer already handled. It only patches
    small raster/vector labels that the text layer did not translate cleanly.

    For page 4, the four purple feature buttons are image/vector-like labels:
      NEAR-ZERO LATENCY / DATA SOVEREIGNTY / COST EFFICIENCY / BANDWIDTH SAVINGS
    Earlier versions used broad OCR and degraded typography. This function treats
    those as fixed UI badges and redraws them using exact source coordinates.
    """
    pdf = fitz.open(pdf_path)
    font_regular = first_existing_font_v10(font_regular, font_bold)
    font_bold = first_existing_font_v10(font_bold, font_regular)

    patches = [
        # Page 4 feature buttons. Coords measured from the source PDF at 841.89 x 595.28 pt.
        # Full purple button rectangles include icon area; text is centered slightly right.
        (3, (648, 129, 777, 152.5), "Äá»˜ TRá»„ Gáº¦N 0", (0.45, 0.23, 0.64), (1, 1, 1), 7.6, "bold"),
        (3, (648, 156, 777, 179.5), "CHá»¦ QUYá»€N Dá»® LIá»†U", (0.45, 0.23, 0.64), (1, 1, 1), 7.0, "bold"),
        (3, (648, 183, 777, 206.5), "Tá»I Æ¯U CHI PHÃ", (0.45, 0.23, 0.64), (1, 1, 1), 7.5, "bold"),
        (3, (648, 210, 777, 233.5), "TIáº¾T KIá»†M BÄ‚NG THÃ”NG", (0.45, 0.23, 0.64), (1, 1, 1), 6.7, "bold"),
    ]

    for page_index, box, label, fill, color, fontsize, weight in patches:
        if page_index >= len(pdf):
            continue
        page = pdf[page_index]
        rect = fitz.Rect(*box)

        # Repaint the whole badge area, not only text, so old English is fully removed.
        page.draw_rect(rect, color=None, fill=fill, overlay=True)

        fontfile = font_bold if weight == "bold" else font_regular
        fontname = "FCXVPatchBold" if weight == "bold" else "FCXVPatchRegular"

        # Button icons on the left are part of the original graphic. Since repainting
        # removes them, draw text centered. This is cleaner than leaving double text.
        draw_centered_textbox_v13(page, rect, label, fontfile, fontname, fontsize, color)

    pdf.save(output_pdf, garbage=4, deflate=True)
    pdf.close()
    print(f"CXVIEW targeted patches applied: {output_pdf}")


def filter_ocr_blocks_v12(
    blocks: List[Dict],
    page_index: int,
    exclude_rects_by_page: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None,
    strict_english: bool = True,
    max_chars: int = 70,
    max_coverage: float = 0.20,
) -> List[Dict]:
    if not ocr_page_allowed_v13(page_index):
        print(f"      OCR filter page {page_index + 1}: skipped by --ocr-pages")
        return []

    exclude_rects = (exclude_rects_by_page or {}).get(page_index, [])
    kept: List[Dict] = []
    stats: Dict[str, int] = {}

    for b in blocks:
        size_ok, size_reason = ocr_bbox_size_ok_v13(b)
        if not size_ok:
            stats[size_reason] = stats.get(size_reason, 0) + 1
            continue

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



# ============================================================
# V16 FULL OCR REBUILD MODE
# ============================================================

def should_keep_full_ocr_text_v16(text_value: str, min_chars: int = 2) -> bool:
    s = re.sub(r"\s+", " ", normalize_special_chars(text_value or "")).strip()
    if len(s) < min_chars:
        return False
    # Skip meaningless punctuation.
    if not re.search(r"[A-Za-z0-9]", s):
        return False
    # Skip lone page-like artifacts.
    if re.fullmatch(r"[\.\-_=:/\\|â€¢Â·]+", s):
        return False
    return True


def ocr_page_to_all_line_blocks_tesseract_v16(
    page: fitz.Page,
    page_index: int,
    dpi: int = 240,
    lang: str = "eng",
    min_conf: float = 40.0,
    min_chars: int = 2,
):
    """OCR all visible text lines on source page.

    Unlike fallback OCR, this does not filter for English leftovers. It treats
    OCR as the primary source of text for a full-page rebuild.
    """
    pytesseract, _ = _require_ocr_dependencies()
    img, zoom = page_to_pil_image_v10(page, dpi=dpi)
    psm = os.getenv("PDF_TRANSLATOR_FULL_OCR_PSM", "11")
    config = f"--psm {psm}"

    data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DICT)

    grouped: Dict[Tuple[int, int, int], List[int]] = {}
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
        )
        grouped.setdefault(key, []).append(i)

    blocks = []
    for key, idxs in grouped.items():
        idxs = sorted(idxs, key=lambda j: int(data["left"][j]))
        words = [str(data["text"][j] or "").strip() for j in idxs if str(data["text"][j] or "").strip()]
        raw_text = re.sub(r"\s+", " ", " ".join(words)).strip()
        if not should_keep_full_ocr_text_v16(raw_text, min_chars=min_chars):
            continue

        left = min(int(data["left"][j]) for j in idxs)
        top = min(int(data["top"][j]) for j in idxs)
        right = max(int(data["left"][j]) + int(data["width"][j]) for j in idxs)
        bottom = max(int(data["top"][j]) + int(data["height"][j]) for j in idxs)
        avg_conf = sum(_ocr_word_conf_v10(data.get("conf", ["-1"])[j]) for j in idxs) / max(1, len(idxs))

        px_pad = max(2, int(dpi / 130))
        px_bbox = (left - px_pad, top - px_pad, right + px_pad, bottom + px_pad)
        pdf_bbox = (
            max(0.0, px_bbox[0] / zoom),
            max(0.0, px_bbox[1] / zoom),
            min(float(page.rect.width), px_bbox[2] / zoom),
            min(float(page.rect.height), px_bbox[3] / zoom),
        )

        height_pt = max(1.0, pdf_bbox[3] - pdf_bbox[1])
        role = "label"
        if height_pt >= 24:
            role = "title"
        elif len(raw_text) > 80 or height_pt > 15:
            role = "body"

        blocks.append({
            "id": f"fullocr_p{page_index}_{len(blocks)}",
            "page_index": page_index,
            "text": raw_text,
            "bbox": pdf_bbox,
            "px_bbox": px_bbox,
            "confidence": avg_conf,
            "role": role,
            "font_size": max(4.5, min(30.0, height_pt * 0.76)),
            "bg_color": sample_bg_color_v10(img, px_bbox),
            "ocr_engine": "tesseract_full",
        })

    return img, zoom, blocks


def ocr_page_to_all_line_blocks_paddle_v16(
    page: fitz.Page,
    page_index: int,
    engine,
    dpi: int = 240,
    min_conf: float = 0.40,
    min_chars: int = 2,
    temp_dir: Optional[Path] = None,
):
    """Full OCR line blocks via PaddleOCR."""
    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir())

    img, zoom = page_to_paddle_image_v11(page, dpi=dpi)
    image_path = save_paddle_temp_image_v11(img, page_index, temp_dir)

    raw = run_paddle_predict_v11(engine, image_path)
    items = parse_paddle_result_items_v11(raw)

    blocks = []
    for raw_text, score, box in items:
        threshold = min_conf / 100.0 if min_conf > 1 else min_conf
        if score < threshold:
            continue

        clean_text = re.sub(r"\s+", " ", normalize_special_chars(raw_text or "")).strip()
        if not should_keep_full_ocr_text_v16(clean_text, min_chars=min_chars):
            continue

        bbox_px = _bbox_from_any_box_v11(box)
        if not bbox_px:
            continue

        left, top, right, bottom = bbox_px
        px_pad = max(2, int(dpi / 130))
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
        if height_pt >= 24:
            role = "title"
        elif len(clean_text) > 80 or height_pt > 15:
            role = "body"

        blocks.append({
            "id": f"fullpaddle_p{page_index}_{len(blocks)}",
            "page_index": page_index,
            "text": clean_text,
            "bbox": pdf_bbox,
            "px_bbox": px_bbox,
            "confidence": float(score),
            "role": role,
            "font_size": max(4.5, min(30.0, height_pt * 0.76)),
            "bg_color": sample_bg_color_v10(img, px_bbox),
            "ocr_engine": "paddle_full",
        })

    return img, zoom, blocks


def cover_ocr_text_on_background_v16(img, blocks: List[Dict], pad_px: int = 1):
    """Return a copy of page image with OCR text regions covered by sampled bg."""
    from PIL import ImageDraw  # type: ignore
    out = img.copy()
    draw = ImageDraw.Draw(out)

    w, h = out.size
    # Cover larger/background text first, then small labels.
    sorted_blocks = sorted(blocks, key=lambda b: rect_area_v12(tuple(float(x) for x in b.get("bbox", (0, 0, 0, 0)))), reverse=True)

    for b in sorted_blocks:
        px = b.get("px_bbox")
        if not px:
            continue
        x0, y0, x1, y1 = [int(v) for v in px]
        x0 = max(0, min(w - 1, x0 - pad_px))
        y0 = max(0, min(h - 1, y0 - pad_px))
        x1 = max(0, min(w, x1 + pad_px))
        y1 = max(0, min(h, y1 + pad_px))
        if x1 <= x0 or y1 <= y0:
            continue

        rgb = b.get("bg_color", (1, 1, 1))
        fill = tuple(max(0, min(255, int(float(c) * 255))) for c in rgb)
        draw.rectangle([x0, y0, x1, y1], fill=fill)

    return out


def choose_ocr_text_color_v16(bg: RGB, role: str = "label") -> RGB:
    lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
    if lum < 0.42:
        return (1, 1, 1)
    return (0.08, 0.08, 0.12)


def draw_full_ocr_text_block_v16(
    page: fitz.Page,
    block: Dict,
    regular_font: Optional[str],
    bold_font: Optional[str],
    title_font: Optional[str],
):
    translated = str(block.get("translated", block.get("text", "")) or "").strip()
    if not translated:
        return False

    rect = fitz.Rect(*block["bbox"])
    if rect.width <= 1 or rect.height <= 1:
        return False

    role = str(block.get("role", "label"))
    bg = block.get("bg_color", (1, 1, 1))
    color = choose_ocr_text_color_v16(bg, role=role)

    fontfile = first_existing_font_v10(regular_font, bold_font)
    fontname = "FFullOCRRegular"
    if role == "title":
        fontfile = first_existing_font_v10(title_font, bold_font, regular_font)
        fontname = "FFullOCRTitle"
    elif role == "label":
        # Many slide labels are semibold/bold.
        fontfile = first_existing_font_v10(bold_font, regular_font)
        fontname = "FFullOCRBold"

    fontsize = float(block.get("font_size", max(5.0, rect.height * 0.72)))
    # Vietnamese often needs a tiny shrink.
    if len(translated) > len(str(block.get("text", ""))) * 1.15:
        fontsize *= 0.90

    align = fitz.TEXT_ALIGN_LEFT
    if role in {"title", "label"} and rect.width < 190:
        align = fitz.TEXT_ALIGN_CENTER

    min_size = max(3.6, fontsize * 0.58)
    size = fontsize
    while size >= min_size:
        try:
            rc = page.insert_textbox(
                rect,
                translated,
                fontsize=size,
                fontname=fontname,
                fontfile=fontfile,
                color=color,
                align=align,
                overlay=True,
            )
            if rc >= 0:
                return True
        except Exception:
            pass
        size -= 0.25

    try:
        page.insert_text(
            fitz.Point(rect.x0, rect.y0 + max(4.0, min_size)),
            translated,
            fontsize=min_size,
            fontname=fontname,
            fontfile=fontfile,
            color=color,
            overlay=True,
        )
        return False
    except Exception:
        return False


def translate_full_ocr_blocks_v16(
    blocks: List[Dict],
    translator: Translator,
    source_lang: str,
    target_lang: str,
    glossary: Optional[Dict[str, str]],
    batch_size: int,
) -> List[Dict]:
    """Translate full OCR blocks with cache/memory support."""
    # Reuse OCR translator but adjust instruction/role for full rebuild.
    for b in blocks:
        exact = ocr_exact_translation_v10(b["text"], target_lang)
        if exact:
            b["translated"] = exact

    pending = [b for b in blocks if not b.get("translated")]
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        items = []
        for b in batch:
            w = max(1.0, b["bbox"][2] - b["bbox"][0])
            max_chars = max(8, int(w / max(3.2, float(b.get("font_size", 8)) * 0.42)))
            items.append({
                "id": b["id"],
                "role": b.get("role", "label"),
                "max_chars": max_chars,
                "max_lines": 1 if b.get("role") in {"label", "title"} else 3,
                "text": b["text"],
                "instruction": (
                    "Full OCR rebuild. Translate this visible English text into compact Vietnamese. "
                    "Keep brand names, model names, URLs, numbers, and currency when appropriate. "
                    "Return only the translated string."
                ),
            })
        translated = translator.translate_batch(items, source_lang, target_lang, glossary)
        by_id = {x["id"]: x.get("translated", "") for x in translated}
        for b in batch:
            b["translated"] = sanitize_text_v9(by_id.get(b["id"], b["text"]))

    return blocks



# â”€â”€ Claude Vision OCR helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _claude_vision_page_to_b64(page, dpi: int = 200) -> str:
    import base64
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def _claude_vision_call(b64_image, api_key, base_url, model,
                         anthropic_version="2023-06-01", max_tokens=4096, timeout=120):
    import urllib.request as _ur, json as _json
    prompt = (
        "You are a precise OCR engine for PDF slides. "
        "Analyze this PDF page image and extract ALL visible text including "
        "text inside colored buttons, badges, diagrams, and raster images. "
        "Return ONLY a JSON array. Each element: "
        '{"text":"...","x0":0.0,"y0":0.0,"x1":1.0,"y1":1.0,"role":"title|body|label","bold":false,"font_size_approx":12.0} '
        "where x0/y0/x1/y1 are fractions (0.0-1.0) of image dimensions. "
        "Skip QR codes. No markdown, no explanation, ONLY the JSON array."
    )
    body = {
        "model": model, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_image}},
            {"type": "text", "text": prompt},
        ]}],
    }
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": anthropic_version,
               "content-type": "application/json", "accept": "application/json"}
    data = _json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = _ur.Request(url, data=data, headers=headers, method="POST")
    with _ur.urlopen(req, timeout=timeout) as resp:
        result = _json.loads(resp.read().decode("utf-8"))
    for block in result.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", ""))
    raise ValueError(f"No text in Claude response: {result}")


def _claude_vision_parse_blocks(raw_response, page, page_index, min_chars=2):
    import re as _re, json as _json
    text = raw_response.strip()
    text = _re.sub(r"^```(?:json)?\s*", "", text, flags=_re.I)
    text = _re.sub(r"\s*```$", "", text).strip()
    try:
        s, e = text.find("["), text.rfind("]")
        json_str = text[s:e+1] if s >= 0 and e > s else text
        try:
            items = _json.loads(json_str)
        except Exception:
            # Response was truncated — repair by closing open structures
            repaired = json_str.rstrip().rstrip(",")
            # Close any open string
            if repaired.count('"') % 2 == 1:
                repaired += '"' 
            # Close open objects and array
            open_braces = repaired.count("{") - repaired.count("}")
            repaired += "}" * max(0, open_braces)
            repaired += "]"
            try:
                items = _json.loads(repaired)
                print(f"      Claude Vision: JSON repaired ({len(items)} items)")
            except Exception as ex2:
                print(f"      Claude Vision JSON parse error: {ex2}")
                return []
    except Exception as ex:
        print(f"      Claude Vision JSON parse error: {ex}")
        return []
    pw = float(page.rect.width); ph = float(page.rect.height); pcx = pw / 2.0
    blocks = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = _re.sub(r"\s+", " ", normalize_special_chars(str(item.get("text","") or ""))).strip()
        if not should_keep_full_ocr_text_v16(raw, min_chars=min_chars):
            continue
        try:
            x0f,y0f,x1f,y1f = (max(0.,min(1.,float(item.get(k,d)))) for k,d in [("x0",0),("y0",0),("x1",1),("y1",1)])
        except Exception:
            continue
        if x1f<=x0f or y1f<=y0f:
            continue
        bbox = (x0f*pw, y0f*ph, x1f*pw, y1f*ph)
        ht = max(1., bbox[3]-bbox[1]); wt = max(1., bbox[2]-bbox[0])
        cr = str(item.get("role","")).lower()
        role = cr if cr in {"title","body","label","header","footer"} else ("title" if ht>=20 else ("body" if len(raw)>100 else "label"))
        fs = float(item.get("font_size_approx",0))
        if fs < 3.: fs = _estimate_font_size_v17(ht, raw, wt)
        bcx = (bbox[0]+bbox[2])/2.
        align = "center" if abs(bcx-pcx)<pw*0.10 and wt<pw*0.5 else ("right" if bbox[2]>pw*0.75 and bbox[0]>pw*0.45 else "left")
        blocks.append({
            "id": f"claudevision_p{page_index}_{len(blocks)}",
            "page_index": page_index, "text": raw, "bbox": bbox, "px_bbox": None,
            "confidence": 0.95, "role": role, "align": align, "font_size": fs,
            "bold": bool(item.get("bold",False)), "bg_color": (1.,1.,1.),
            "ocr_engine": "claude_vision",
        })
    return blocks


def ocr_page_to_all_line_blocks_claude_vision_v16(page, page_index, api_key, base_url,
        model, dpi=200, min_chars=2, anthropic_version="2023-06-01", timeout=120):
    from PIL import Image as _PI
    zoom = dpi / 72.
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = _PI.frombytes("RGB", [pix.width, pix.height], pix.samples)
    b64 = _claude_vision_page_to_b64(page, dpi=dpi)
    print(f"        Claude Vision API call (page {page_index+1}) ...")
    try:
        raw = _claude_vision_call(b64, api_key, base_url, model,
                                   anthropic_version=anthropic_version, timeout=timeout)
    except Exception as e:
        print(f"        Claude Vision error: {e}")
        return img, zoom, []
    blocks = _claude_vision_parse_blocks(raw, page, page_index, min_chars=min_chars)
    pw = float(page.rect.width); ph = float(page.rect.height)
    for b in blocks:
        px = (max(0,int(b["bbox"][0]*zoom)-2), max(0,int(b["bbox"][1]*zoom)-2),
              int(b["bbox"][2]*zoom)+2, int(b["bbox"][3]*zoom)+2)
        b["px_bbox"] = px
        b["bg_color"] = sample_bg_color_v10(img, px)
    print(f"        Claude Vision: {len(blocks)} blocks extracted")
    return img, zoom, blocks

def full_ocr_rebuild_pdf_v16(
    input_pdf: str,
    output_pdf: str,
    translator: Translator,
    source_lang: str = "auto",
    target_lang: str = "vi",
    glossary: Optional[Dict[str, str]] = None,
    ocr_engine: str = "tesseract",
    ocr_lang: str = "eng",
    dpi: int = 240,
    min_conf: float = 40.0,
    min_chars: int = 2,
    batch_size: int = 20,
    regular_font: Optional[str] = None,
    bold_font: Optional[str] = None,
    title_font: Optional[str] = None,
    tesseract_cmd: Optional[str] = None,
    claude_vision_model: str = "claude-haiku-4-5",
    claude_vision_dpi: int = 200,
):
    """Rebuild PDF from OCR rather than PDF text layer.

    Output pages are:
      cleaned raster background image + translated OCR text layer.

    This avoids using source PDF text objects entirely. It still necessarily
    covers/removes text regions on a raster background because OCR only reads
    text; it cannot reconstruct clean backgrounds by itself.
    """
    if ocr_engine == "tesseract":
        _configure_tesseract_cmd(tesseract_cmd)

    claude_api_key = ""; claude_base_url = "https://api.anthropic.com"; claude_version = "2023-06-01"
    if ocr_engine == "claude":
        claude_api_key  = os.getenv("LLM_API_KEY", "").strip()
        claude_base_url = os.getenv("LLM_BASE_URL", "https://api.anthropic.com").strip().rstrip("/")
        claude_version  = os.getenv("ANTHROPIC_VERSION", "2023-06-01").strip()
        if not claude_api_key:
            raise ValueError("Claude Vision requires LLM_API_KEY in .env")
        dpi = claude_vision_dpi
        print(f"      Claude Vision OCR: model={claude_vision_model} dpi={dpi}")

    src = fitz.open(input_pdf)
    out_pdf = fitz.open()

    total_blocks = 0
    total_drawn = 0

    paddle_engine = None
    temp_dir_obj = None
    if ocr_engine == "paddle":
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="pdf_translate_full_paddleocr_")
        paddle_engine = get_paddle_ocr_engine_v11(lang=ocr_lang, use_textline_orientation=False)

    try:
        for page_index, src_page in enumerate(src):
            page_rect = src_page.rect
            print(f"      Full OCR page {page_index + 1}/{len(src)}")

            if ocr_engine == "claude":
                img, zoom, blocks = ocr_page_to_all_line_blocks_claude_vision_v16(
                    page=src_page,
                    page_index=page_index,
                    api_key=claude_api_key,
                    base_url=claude_base_url,
                    model=claude_vision_model,
                    dpi=dpi,
                    min_chars=min_chars,
                    anthropic_version=claude_version,
                )
            elif ocr_engine == "paddle":
                img, zoom, blocks = ocr_page_to_all_line_blocks_paddle_v16(
                    page=src_page,
                    page_index=page_index,
                    engine=paddle_engine,
                    dpi=dpi,
                    min_conf=min_conf,
                    min_chars=min_chars,
                    temp_dir=Path(temp_dir_obj.name),
                )
            else:
                tess_lang = "eng" if ocr_lang == "en" else ocr_lang
                img, zoom, blocks = ocr_page_to_all_line_blocks_tesseract_v16(
                    page=src_page,
                    page_index=page_index,
                    dpi=dpi,
                    lang=tess_lang,
                    min_conf=min_conf,
                    min_chars=min_chars,
                )

            total_blocks += len(blocks)
            if blocks:
                blocks = translate_full_ocr_blocks_v16(
                    blocks,
                    translator=translator,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    glossary=glossary,
                    batch_size=batch_size,
                )

            cleaned = cover_ocr_text_on_background_v16(img, blocks, pad_px=max(1, int(dpi / 160)))
            tmp_img = Path(tempfile.gettempdir()) / f"full_ocr_rebuild_{os.getpid()}_{page_index}.png"
            cleaned.save(tmp_img)

            new_page = out_pdf.new_page(width=page_rect.width, height=page_rect.height)
            new_page.insert_image(fitz.Rect(0, 0, page_rect.width, page_rect.height), filename=str(tmp_img))

            for b in blocks:
                if draw_full_ocr_text_block_v16(
                    new_page,
                    b,
                    regular_font=regular_font,
                    bold_font=bold_font,
                    title_font=title_font,
                ):
                    total_drawn += 1

            try:
                tmp_img.unlink(missing_ok=True)
            except Exception:
                pass
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

    out_pdf.save(output_pdf, garbage=4, deflate=True)
    out_pdf.close()
    src.close()

    print("Full OCR rebuild summary:")
    print(f"  ocr_blocks={total_blocks}")
    print(f"  drawn_blocks={total_drawn}")
    print(f"  output={output_pdf}")


# ============================================================
# V26.3 REGION PATCH ENGINE (inline)
# ============================================================

def _v263_first_existing(*paths):
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    return None


def _v263_rgb01(value, default=(1.0, 1.0, 1.0)):
    if value is None:
        return default
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in {"none", "sample"}:
            return default
        if s.startswith("#") and len(s) == 7:
            try:
                return (int(s[1:3],16)/255.0, int(s[3:5],16)/255.0, int(s[5:7],16)/255.0)
            except Exception:
                return default
        named = {"white":(1,1,1),"black":(0,0,0),"purple":(0.427,0.227,0.6),"dark":(0.08,0.08,0.12)}
        return named.get(s.lower(), default)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        vals = [float(value[0]), float(value[1]), float(value[2])]
        if max(vals) > 1.0:
            vals = [v/255.0 for v in vals]
        return tuple(max(0, min(1, v)) for v in vals[:3])
    return default


def _v263_luminance(rgb) -> float:
    return 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]


def _v263_load_patch_map(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    patches = data.get("patches", data if isinstance(data, list) else [])
    return [p for p in patches if isinstance(p, dict) and p.get("enabled", True)]


def _v263_load_region_map(path: Optional[str]) -> Dict[int, List]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("regions", data if isinstance(data, list) else [])
    regions: Dict[int, List] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        page = int(item.get("page", 0))
        bbox = item.get("bbox")
        if page <= 0 or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        regions.setdefault(page, []).append(fitz.Rect(*[float(x) for x in bbox]))
    return regions


def _v263_auto_image_rects(source_pdf: str, min_area: float = 500.0) -> Dict[int, List]:
    out: Dict[int, List] = {}
    if not source_pdf or not Path(source_pdf).exists():
        return out
    doc = fitz.open(source_pdf)
    try:
        for pi, page in enumerate(doc):
            rects = []
            for img in page.get_images(full=True):
                try:
                    for r in page.get_image_rects(img[0]):
                        rr = fitz.Rect(r)
                        if rr.get_area() >= min_area:
                            rects.append(rr)
                except Exception:
                    pass
            out[pi + 1] = rects
    finally:
        doc.close()
    return out


def _v263_rect_allowed(rect: fitz.Rect, regions: List, threshold: float = 0.25) -> bool:
    if not regions:
        return False
    cx = (rect.x0 + rect.x1) / 2.0
    cy = (rect.y0 + rect.y1) / 2.0
    area = max(0.001, rect.get_area())
    for region in regions:
        if region.contains(fitz.Point(cx, cy)):
            return True
        inter = rect & region
        if not inter.is_empty and inter.get_area() / area >= threshold:
            return True
    return False


def _v263_sample_bg(page: fitz.Page, rect: fitz.Rect) -> tuple:
    try:
        zoom = 140 / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        data_bytes = pix.samples
        n = pix.n
        w, h = pix.width, pix.height
        x0 = int(rect.x0 * zoom); y0 = int(rect.y0 * zoom)
        x1 = int(rect.x1 * zoom); y1 = int(rect.y1 * zoom)
        pad = max(3, int(4 * zoom))
        pts = [
            (x0-pad, y0-pad), (x1+pad, y0-pad),
            (x0-pad, y1+pad), (x1+pad, y1+pad),
            ((x0+x1)//2, y0-pad), ((x0+x1)//2, y1+pad),
            (x0-pad, (y0+y1)//2), (x1+pad, (y0+y1)//2),
        ]
        pixels = []
        for sx, sy in pts:
            if 0 <= sx < w and 0 <= sy < h:
                idx = (sy * w + sx) * n
                if idx + 2 < len(data_bytes):
                    pixels.append((data_bytes[idx], data_bytes[idx+1], data_bytes[idx+2]))
        if not pixels:
            return (1.0, 1.0, 1.0)
        rs = sorted(p[0] for p in pixels)
        gs = sorted(p[1] for p in pixels)
        bs = sorted(p[2] for p in pixels)
        mid = len(pixels) // 2
        return (rs[mid]/255.0, gs[mid]/255.0, bs[mid]/255.0)
    except Exception:
        return (1.0, 1.0, 1.0)


def _v263_draw_pill(page: fitz.Page, rect: fitz.Rect, fill: tuple, radius=None):
    if radius is None:
        radius = min(rect.height/2.0, rect.width/2.0)
    radius = max(0.0, min(float(radius), rect.height/2.0, rect.width/2.0))
    if radius <= 0.2:
        page.draw_rect(rect, color=None, fill=fill, overlay=True)
        return
    r2 = 2 * radius
    mid = fitz.Rect(rect.x0+radius, rect.y0, rect.x1-radius, rect.y1)
    if mid.x1 > mid.x0:
        page.draw_rect(mid, color=None, fill=fill, overlay=True)
    ctr = fitz.Rect(rect.x0, rect.y0+radius, rect.x1, rect.y1-radius)
    if ctr.y1 > ctr.y0:
        page.draw_rect(ctr, color=None, fill=fill, overlay=True)
    for ox, oy in [(0,0),(rect.width-r2,0),(0,rect.height-r2),(rect.width-r2,rect.height-r2)]:
        page.draw_oval(
            fitz.Rect(rect.x0+ox, rect.y0+oy, rect.x0+ox+r2, rect.y0+oy+r2),
            color=None, fill=fill, overlay=True,
        )


def _v263_draw_text(page: fitz.Page, rect: fitz.Rect, text: str,
                    fontfile: Optional[str], fontname: str,
                    fontsize: float, color: tuple, align: str):
    align_map = {
        "left": fitz.TEXT_ALIGN_LEFT,
        "center": fitz.TEXT_ALIGN_CENTER,
        "right": fitz.TEXT_ALIGN_RIGHT,
    }
    pdf_align = align_map.get(str(align).lower(), fitz.TEXT_ALIGN_CENTER)
    size = float(fontsize)
    min_size = max(3.2, size * 0.55)
    while size >= min_size:
        try:
            rc = page.insert_textbox(
                rect, text, fontsize=size, fontname=fontname,
                fontfile=fontfile, color=color, align=pdf_align, overlay=True,
            )
            if rc >= 0:
                return
        except Exception:
            pass
        size -= 0.2
    try:
        page.insert_text(
            fitz.Point(rect.x0+0.5, rect.y0+max(3.8, min_size)),
            text, fontsize=min_size, fontname=fontname,
            fontfile=fontfile, color=color, overlay=True,
        )
    except Exception:
        pass


def apply_v263_patch(
    pdf_path: str,
    output_pdf: str,
    patch_map_path: str,
    source_pdf: Optional[str] = None,
    image_region_map: Optional[str] = None,
    font_regular: Optional[str] = None,
    font_bold: Optional[str] = None,
    font_title: Optional[str] = None,
    debug_regions: bool = False,
) -> int:
    """Apply V26.3 image-region-only patches inline. Returns patches applied."""
    patches = _v263_load_patch_map(patch_map_path)
    if not patches:
        print(f"      V26.3: no enabled patches in {patch_map_path}")
        return 0

    auto_regions = _v263_auto_image_rects(source_pdf or pdf_path)
    manual_regions = _v263_load_region_map(image_region_map)

    allowed: Dict[int, List] = {}
    for m in (auto_regions, manual_regions):
        for page_num, rects in m.items():
            allowed.setdefault(page_num, []).extend(rects)

    pdf = fitz.open(pdf_path)

    reg   = _v263_first_existing(font_regular, "fonts/NotoSans-Regular.ttf")
    bold  = _v263_first_existing(font_bold, "fonts/NotoSans-Bold.ttf", reg)
    title = _v263_first_existing(font_title, bold, reg)

    applied = 0
    skipped = 0

    for page_index, page in enumerate(pdf):
        page_num = page_index + 1
        regions = allowed.get(page_num, [])

        if debug_regions:
            for r in regions:
                page.draw_rect(r, color=(0, 0.45, 1), width=0.8, overlay=True)

        for p in patches:
            if int(p.get("page", 0)) != page_num:
                continue

            text_val = str(p.get("translation") or p.get("text") or "").strip()
            bbox = p.get("bbox")
            if not text_val or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                skipped += 1
                continue

            raw_rect = fitz.Rect(*[float(x) for x in bbox])
            if regions and not _v263_rect_allowed(raw_rect, regions):
                skipped += 1
                continue

            pad_x = float(p.get("pad_x", 1.5))
            pad_y = float(p.get("pad_y", 1.2))
            rect = fitz.Rect(
                raw_rect.x0 - pad_x, raw_rect.y0 - pad_y,
                raw_rect.x1 + pad_x, raw_rect.y1 + pad_y,
            )

            bg = _v263_sample_bg(page, rect)
            fill_val = p.get("fill")
            if fill_val is not None and not (isinstance(fill_val, str) and fill_val.lower() == "none"):
                if isinstance(fill_val, str) and fill_val.lower() == "sample":
                    fill = bg
                else:
                    fill = _v263_rgb01(fill_val, default=bg)
                shape = str(p.get("shape", "rect")).lower()
                if shape in {"pill", "round", "rounded"}:
                    _v263_draw_pill(page, rect, fill=fill, radius=p.get("radius"))
                else:
                    page.draw_rect(rect, color=None, fill=fill, overlay=True)
                bg = fill

            explicit_color = p.get("color")
            if explicit_color is None:
                color = (1.0,1.0,1.0) if _v263_luminance(bg) < 0.46 else (0.08,0.08,0.12)
            else:
                color = _v263_rgb01(explicit_color, default=(0.0,0.0,0.0))

            role   = str(p.get("role", "label")).lower()
            weight = str(p.get("weight", "auto")).lower()
            if weight == "auto":
                weight = "bold" if role in {"title","label"} else "regular"

            if role == "title":
                fontfile = title or bold or reg
                fontname = "FV263Title"
            elif weight in {"bold","semibold","black"}:
                fontfile = bold or reg
                fontname = "FV263Bold"
            else:
                fontfile = reg or bold
                fontname = "FV263Regular"

            if not fontfile:
                fontname = "helv"

            fontsize = float(p.get("font_size", max(4.0, rect.height * 0.72)))
            align    = str(p.get("align", "center"))
            _v263_draw_text(page, rect, text_val, fontfile, fontname, fontsize, color, align)
            applied += 1

    # PyMuPDF cannot save to the same path as the opened file.
    # Use a temp file in the SAME directory as output_pdf to avoid cross-drive
    # issues on Windows, then replace atomically.
    import shutil as _shutil
    out_path = Path(output_pdf).resolve()
    same_path = Path(pdf_path).resolve() == out_path

    if same_path:
        # Create temp file next to the output file (same drive/directory).
        tmp_path = str(out_path.parent / (out_path.stem + "_v263tmp.pdf"))
        try:
            pdf.save(tmp_path, garbage=4, deflate=True)
            pdf.close()
            _shutil.move(tmp_path, str(out_path))
        except Exception:
            try:
                pdf.close()
            except Exception:
                pass
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
            raise
    else:
        pdf.save(output_pdf, garbage=4, deflate=True)
        pdf.close()
    return applied



# ── Claude Vision Region Patch helpers ────────────────────────────────────

def _cvr_render_region_b64(page, region, dpi=220):
    import base64
    zoom = dpi / 72.0
    clip = fitz.Rect(max(0,region.x0),max(0,region.y0),
                     min(page.rect.width,region.x1),min(page.rect.height,region.y1))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom,zoom), clip=clip, alpha=False)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8"), zoom

def _cvr_call_claude(b64_image, region, api_key, base_url, model,
                     anthropic_version="2023-06-01", timeout=60):
    prompt = (
        "Extract all visible text from this PDF region. "
        "Return ONLY a JSON array with this schema: "
        '[{"text":"...","x0":0.0,"y0":0.0,"x1":1.0,"y1":1.0}]. '
        "Coordinates must be fractions 0.0-1.0 of this image. "
        "Include button/badge text. Skip QR codes. No markdown."
    )
    body = {"model": model, "max_tokens": 2048,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_image}},
                {"type": "text", "text": prompt}]}]}
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": anthropic_version,
               "content-type": "application/json", "accept": "application/json"}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"        Claude Vision error: {e}"); return []
    raw = ""
    for block in result.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            raw = str(block.get("text", "")); break
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        s, e = raw.find("["), raw.rfind("]")
        items = json.loads(raw[s:e+1]) if s>=0 and e>s else json.loads(raw)
    except Exception as ex:
        try:
            rep = raw.rstrip().rstrip(",")
            rep += "}" * max(0, rep.count("{")-rep.count("}")) + "]"
            s = rep.find("[")
            items = json.loads(rep[s:]) if s>=0 else []
        except Exception:
            print(f"        Claude Vision JSON error: {ex}"); return []
    rw, rh = region.width, region.height
    results = []
    for item in items:
        if not isinstance(item, dict): continue
        text = re.sub(r"\s+", " ", str(item.get("text","") or "")).strip()
        if not text or len(text) < 2: continue
        try:
            x0f=max(0.,min(1.,float(item.get("x0",0.)))); y0f=max(0.,min(1.,float(item.get("y0",0.))))
            x1f=max(0.,min(1.,float(item.get("x1",1.)))); y1f=max(0.,min(1.,float(item.get("y1",1.))))
        except: continue
        if x1f<=x0f or y1f<=y0f: continue
        results.append({"text": text, "bbox": (
            region.x0+x0f*rw, region.y0+y0f*rh,
            region.x0+x1f*rw, region.y0+y1f*rh)})
    return results

def _cvr_translate_blocks(blocks, translator, source_lang, target_lang, glossary, batch_size=20):
    pending = []
    for i, b in enumerate(blocks):
        class _P:
            def __init__(self, blk):
                self.original_text=blk["text"]; self.role="label"
                self.font_size=max(4.,(blk["bbox"][3]-blk["bbox"][1])*.72)
                self.bbox=blk["bbox"]; self.lines=[type("L",(),{"text":blk["text"]})()]
        ov = compact_override_for_block(_P(b), target_lang)
        if ov: b["translated"] = sanitize_text_v9(ov, b["text"], "label")
        else: pending.append((i, b))
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start+batch_size]
        items = [{"id": str(idx), "role": "label", "text": b["text"],
                  "max_chars": 60, "max_lines": 3,
                  "instruction": "Translate to target language. Keep brand names."}
                 for idx, b in batch]
        res = translator.translate_batch(items, source_lang, target_lang, glossary)
        by_id = {x["id"]: x.get("translated","") for x in res}
        for idx, b in batch:
            b["translated"] = sanitize_text_v9(by_id.get(str(idx), b["text"]), b["text"], "label")
    return blocks

def apply_claude_vision_region_patch(
    source_pdf, translated_pdf, output_pdf, image_region_map, translator,
    source_lang="auto", target_lang="vi", glossary=None,
    api_key="", base_url="https://api.anthropic.com", model="claude-haiku-4-5",
    anthropic_version="2023-06-01", dpi=220,
    font_regular=None, font_bold=None, font_title=None, batch_size=20, debug=False,
):
    import shutil as _sh
    manual = _v263_load_region_map(image_region_map)
    auto   = _v263_auto_image_rects(source_pdf, min_area=800.0)
    all_r: Dict[int, List] = {}
    for m in (auto, manual):
        for pn, rects in m.items(): all_r.setdefault(pn,[]).extend(rects)
    if not all_r:
        print("      CV Region Patch: no regions"); return 0
    pdf = fitz.open(translated_pdf)
    reg  = _v263_first_existing(font_regular, "fonts/NotoSans-Regular.ttf")
    bold = _v263_first_existing(font_bold, "fonts/NotoSans-Bold.ttf", reg)
    total = 0
    for page_index, page in enumerate(pdf):
        pn = page_index + 1
        regions = all_r.get(pn, [])
        if not regions: continue
        print(f"      CV Region page {pn}: {len(regions)} regions")
        all_blocks = []
        for region in regions:
            if region.width < 20 or region.height < 10: continue
            if debug: page.draw_rect(region, color=(0,.6,0), width=1., overlay=True)
            b64, _ = _cvr_render_region_b64(page, region, dpi=dpi)
            ext = _cvr_call_claude(b64, region, api_key, base_url, model, anthropic_version)
            print(f"        [{region.x0:.0f},{region.y0:.0f}]: {len(ext)} blocks")
            all_blocks.extend(ext)
        if not all_blocks: continue
        all_blocks = _cvr_translate_blocks(all_blocks, translator, source_lang, target_lang, glossary, batch_size)
        for b in all_blocks:
            src_t = b.get("text",""); vi_t = b.get("translated","")
            if not vi_t or text_is_same(src_t, vi_t): continue
            rect = fitz.Rect(*b["bbox"])
            if rect.width<=2 or rect.height<=2: continue
            bg = _v263_sample_bg(page, rect)
            lum = _v263_luminance(bg); sat = max(bg)-min(bg)
            page.draw_rect(rect, color=None, fill=bg, overlay=True)
            color = (1.,1.,1.) if (lum<.50 or sat>.15) else (.06,.06,.10)
            fontfile = bold or reg; fontname = "FCVRBold"
            if not fontfile: fontname = "helv"
            ht = max(1., rect.height); wt = max(1., rect.width)
            fs = _estimate_font_size_v17(ht, vi_t, wt)
            size = fs; min_s = max(3.5, fs*.55)
            while size >= min_s:
                try:
                    rc = page.insert_textbox(rect, vi_t, fontsize=size, fontname=fontname,
                                             fontfile=fontfile, color=color,
                                             align=fitz.TEXT_ALIGN_CENTER, overlay=True)
                    if rc >= 0: break
                except: pass
                size -= .2
            total += 1
    out_path = Path(output_pdf).resolve(); tr_path = Path(translated_pdf).resolve()
    if tr_path == out_path:
        tmp = str(out_path.parent/(out_path.stem+"_cvrpatch_tmp.pdf"))
        pdf.save(tmp, garbage=4, deflate=True); pdf.close(); _sh.move(tmp, str(out_path))
    else:
        pdf.save(output_pdf, garbage=4, deflate=True); pdf.close()
    print(f"      CV Region Patch: {total} patches → {output_pdf}")
    return total

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
    p.add_argument("--ocr-pages", default="", help="Limit OCR to 1-based pages/ranges, e.g. '4' or '4-5'")
    p.add_argument("--ocr-min-height-pt", type=float, default=5.8, help="Skip OCR bbox shorter than this PDF-point height")
    p.add_argument("--ocr-min-width-pt", type=float, default=10.0, help="Skip OCR bbox narrower than this PDF-point width")
    p.add_argument("--ocr-max-height-pt", type=float, default=26.0, help="Skip OCR bbox taller than this PDF-point height")
    p.add_argument("--cxview-manual-patches", action="store_true", help="Apply optional targeted patches for the CXVIEW sample deck")
    p.add_argument("--patch-only-from", default=None, help="Skip OCR/translation; apply targeted manual patches to an existing rendered PDF")
    p.add_argument("--patch-map", default=None, help="Apply generic JSON region patch map; can be used with --patch-only-from")

    # V26.3 inline patch â€” runs at the END of the pipeline automatically.
    p.add_argument("--v263-patch-map", default=None,
                   help="V26.3 patch map JSON applied AFTER translation (no separate script needed)")
    p.add_argument("--v263-image-region-map", default=None,
                   help="V26.3 manual image/graphic region map JSON (auto-detect is always on)")
    p.add_argument("--cv-region-patch", action="store_true",
                   help="Use Claude Vision to read/translate text inside image regions")
    p.add_argument("--cv-region-map", default=None,
                   help="Image region map JSON for Claude Vision region patch")
    p.add_argument("--cv-model", default="claude-haiku-4-5",
                   help="Claude model for --cv-region-patch")
    p.add_argument("--cv-dpi", type=int, default=220,
                   help="DPI for region rendering in --cv-region-patch")
    p.add_argument("--cv-debug", action="store_true",
                   help="Draw green outlines around processed image regions")
    p.add_argument("--v263-debug-regions", action="store_true",
                   help="Draw blue outlines on V26.3 image regions for debugging")

    # Full OCR rebuild: ignores PDF text layer and reconstructs pages from OCR.
    p.add_argument("--full-ocr-rebuild", action="store_true", help="OCR all visible text and rebuild pages from cleaned raster background + translated OCR text")
    p.add_argument("--full-ocr-engine", default="tesseract", choices=["tesseract", "paddle", "claude"], help="Engine for --full-ocr-rebuild. 'claude' uses Claude Vision API (best quality)")
    p.add_argument("--full-ocr-lang", default="eng", help="OCR language for --full-ocr-rebuild")
    p.add_argument("--full-ocr-dpi", type=int, default=240, help="Render DPI for --full-ocr-rebuild")
    p.add_argument("--claude-vision-model", default="claude-haiku-4-5",
                   help="Claude model for --full-ocr-engine claude (default: claude-haiku-4-5)")
    p.add_argument("--claude-vision-dpi", type=int, default=200,
                   help="DPI for Claude Vision page rendering (default: 200)")
    p.add_argument("--full-ocr-min-conf", type=float, default=40.0, help="Minimum OCR confidence for --full-ocr-rebuild")
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

    if args.full_ocr_rebuild:
        print(f"[Full OCR rebuild] Input: {args.input_pdf}")
        full_ocr_rebuild_pdf_v16(
            input_pdf=args.input_pdf,
            output_pdf=args.output_pdf,
            translator=translator,
            source_lang=args.source,
            target_lang=args.target,
            glossary=glossary,
            ocr_engine=args.full_ocr_engine,
            ocr_lang=args.full_ocr_lang,
            dpi=args.full_ocr_dpi,
            min_conf=args.full_ocr_min_conf,
            min_chars=args.ocr_min_chars,
            batch_size=args.batch_size,
            regular_font=args.font,
            bold_font=args.font_bold,
            title_font=args.font_title,
            tesseract_cmd=args.tesseract_cmd,
            claude_vision_model=getattr(args, "claude_vision_model", "claude-haiku-4-5"),
            claude_vision_dpi=getattr(args, "claude_vision_dpi", 200),
        )
        if args.preview_dir:
            print(f"[Full OCR rebuild] Render preview PNGs: {args.preview_dir}")
            render_pdf_pages(args.output_pdf, args.preview_dir)
        print(f"Done: {args.output_pdf}")
        return

    if args.patch_only_from:
        print(f"[Patch-only] Apply patches on: {args.patch_only_from}")
        if not Path(args.patch_only_from).exists():
            raise FileNotFoundError(args.patch_only_from)

        if args.patch_map:
            apply_region_patch_map_v15(
                pdf_path=args.patch_only_from,
                output_pdf=args.output_pdf,
                patch_map_path=args.patch_map,
                font_regular=args.font,
                font_bold=args.font_bold or args.font_title,
            )
        else:
            apply_cxview_manual_patches_v13(
                pdf_path=args.patch_only_from,
                output_pdf=args.output_pdf,
                font_regular=args.font,
                font_bold=args.font_bold or args.font_title,
            )

        if args.preview_dir:
            print(f"[Patch-only] Render preview PNGs: {args.preview_dir}")
            render_pdf_pages(args.output_pdf, args.preview_dir)
        print(f"Done: {args.output_pdf}")
        return

    os.environ["PDF_TRANSLATOR_OCR_PAGES"] = args.ocr_pages or ""
    os.environ["PDF_TRANSLATOR_OCR_MIN_HEIGHT_PT"] = str(args.ocr_min_height_pt)
    os.environ["PDF_TRANSLATOR_OCR_MIN_WIDTH_PT"] = str(args.ocr_min_width_pt)
    os.environ["PDF_TRANSLATOR_OCR_MAX_HEIGHT_PT"] = str(args.ocr_max_height_pt)

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

        if args.cxview_manual_patches:
            patched_output = str(Path(final_ocr_output).with_suffix(".cxview_patch.pdf"))
            apply_cxview_manual_patches_v13(
                pdf_path=final_ocr_output,
                output_pdf=patched_output,
                font_regular=args.font,
                font_bold=args.font_bold or args.font_title,
            )
            Path(patched_output).replace(final_ocr_output)

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
            if args.cxview_manual_patches:
                patch_input = final_ocr_output
                patched_output = str(Path(patch_input).with_suffix(".cxview_patch.pdf"))
                apply_cxview_manual_patches_v13(
                    pdf_path=patch_input,
                    output_pdf=patched_output,
                    font_regular=args.font,
                    font_bold=args.font_bold or args.font_title,
                )
                Path(patched_output).replace(patch_input)
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

    # Claude Vision Region Patch hook.
    if getattr(args, "cv_region_patch", False):
        _cv_out = args.output_pdf if not getattr(args,"ocr_output",None) else args.ocr_output
        _cv_key = os.getenv("LLM_API_KEY","").strip()
        _cv_url = os.getenv("LLM_BASE_URL","https://api.anthropic.com").strip().rstrip("/")
        _cv_ver = os.getenv("ANTHROPIC_VERSION","2023-06-01").strip()
        if not _cv_key:
            print("WARNING: --cv-region-patch requires LLM_API_KEY — skipping")
        else:
            print(f"[CV Region Patch] model={getattr(args,'cv_model','claude-haiku-4-5')}")
            apply_claude_vision_region_patch(
                source_pdf=args.input_pdf, translated_pdf=_cv_out, output_pdf=_cv_out,
                image_region_map=getattr(args,"cv_region_map",None), translator=translator,
                source_lang=args.source, target_lang=args.target, glossary=glossary,
                api_key=_cv_key, base_url=_cv_url,
                model=getattr(args,"cv_model","claude-haiku-4-5"),
                anthropic_version=_cv_ver,
                dpi=getattr(args,"cv_dpi",220),
                font_regular=args.font, font_bold=args.font_bold, font_title=args.font_title,
                batch_size=args.batch_size, debug=getattr(args,"cv_debug",False),
            )
        # V26.3 inline region patch â€” runs after the full pipeline with no extra script.
    _v263_out = args.output_pdf if not getattr(args, "ocr_output", None) else args.ocr_output
    if getattr(args, "v263_patch_map", None) and Path(args.v263_patch_map).exists():
        print(f"[V26.3] Applying region patches from {args.v263_patch_map}")
        n_patches = apply_v263_patch(
            pdf_path=_v263_out,
            output_pdf=_v263_out,
            patch_map_path=args.v263_patch_map,
            source_pdf=args.input_pdf,
            image_region_map=getattr(args, "v263_image_region_map", None),
            font_regular=args.font,
            font_bold=args.font_bold,
            font_title=args.font_title,
            debug_regions=getattr(args, "v263_debug_regions", False),
        )
        print(f"[V26.3] Applied {n_patches} patches â†’ {_v263_out}")

    print(f"Done: {_v263_out}")



# ============================================================
# V8 forced all-text memory overrides
# ============================================================
# These are intentionally added late so they override earlier entries that kept
# product phrases in English. Glossary can be added later for preferred terms.
EXACT_TRANSLATION_MEMORY_VI.update({
    "cxview gpt box": "Há»™p CXVIEW GPT",
    "& ai video analytics": "& PhÃ¢n tÃ­ch Video AI",
    "cxview gpt box & ai video analytics": "Há»™p CXVIEW GPT & PhÃ¢n tÃ­ch Video AI",
    "client challenges. cxview solutions. business impacts.": "ThÃ¡ch thá»©c KH. Giáº£i phÃ¡p CXVIEW. TÃ¡c Ä‘á»™ng KD.",
    "how cxview delivers the transformation": "CXVIEW triá»ƒn khai chuyá»ƒn Ä‘á»•i",
})

COMPACT_TRANSLATION_MEMORY_VI.update({
    "cxview gpt box": "Há»™p CXVIEW GPT",
    "& ai video analytics": "& PhÃ¢n tÃ­ch Video AI",
    "cxview gpt box & ai video analytics": "Há»™p CXVIEW GPT & PhÃ¢n tÃ­ch Video AI",
    "client challenges. cxview solutions. business impacts.": "ThÃ¡ch thá»©c KH. Giáº£i phÃ¡p CXVIEW. TÃ¡c Ä‘á»™ng KD.",
    "how cxview delivers the transformation": "CXVIEW triá»ƒn khai chuyá»ƒn Ä‘á»•i",
})



# ============================================================
# V9 line-aware marketing/callout translation overrides
# ============================================================
# These are late overrides so they win over previous memories.
EXACT_TRANSLATION_MEMORY_VI.update({
    "we promise results within 30 days and roi in 60 days.": "ChÃºng tÃ´i cam káº¿t mang láº¡i káº¿t quáº£\ntrong 30 ngÃ y vÃ  ROI trong 60 ngÃ y.",
    "cxview delivers measurable business impact on an aggressive timeline that respects your operational urgency, going beyond mere software deployment.": "CXVIEW mang láº¡i tÃ¡c Ä‘á»™ng kinh doanh cÃ³ thá»ƒ Ä‘o lÆ°á»ng\nvá»›i tiáº¿n Ä‘á»™ triá»ƒn khai nhanh, phÃ¹ há»£p nhu cáº§u váº­n hÃ nh\ncáº¥p thiáº¿t cá»§a báº¡n, vÆ°á»£t xa viá»‡c chá»‰ triá»ƒn khai pháº§n má»m.",
})

COMPACT_TRANSLATION_MEMORY_VI.update({
    "we promise results within 30 days and roi in 60 days.": "ChÃºng tÃ´i cam káº¿t mang láº¡i káº¿t quáº£\ntrong 30 ngÃ y vÃ  ROI trong 60 ngÃ y.",
    "cxview delivers measurable business impact on an aggressive timeline that respects your operational urgency, going beyond mere software deployment.": "CXVIEW mang láº¡i tÃ¡c Ä‘á»™ng kinh doanh cÃ³ thá»ƒ Ä‘o lÆ°á»ng\nvá»›i tiáº¿n Ä‘á»™ triá»ƒn khai nhanh, phÃ¹ há»£p nhu cáº§u váº­n hÃ nh\ncáº¥p thiáº¿t cá»§a báº¡n, vÆ°á»£t xa viá»‡c chá»‰ triá»ƒn khai pháº§n má»m.",
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
    vietnamese_marks = set("ÄƒÃ¢Ä‘ÃªÃ´Æ¡Æ°Ä‚Ã‚ÄÃŠÃ”Æ Æ¯Ã¡Ã áº£Ã£áº¡áº¥áº§áº©áº«áº­áº¯áº±áº³áºµáº·Ã©Ã¨áº»áº½áº¹áº¿á»á»ƒá»…á»‡Ã­Ã¬á»‰Ä©á»‹Ã³Ã²á»Ãµá»á»‘á»“á»•á»—á»™á»›á»á»Ÿá»¡á»£ÃºÃ¹á»§Å©á»¥á»©á»«á»­á»¯á»±Ã½á»³á»·á»¹á»µ")
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


