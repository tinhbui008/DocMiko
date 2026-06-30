#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LONGMAN V1 Auto Patch Planner

Purpose
- Convert RapidOCR image-region OCR report into a renderer-compatible patch map.
- Designed for slide/image-heavy PDFs like "LONGMAN BFFB Renewable.pdf".
- No Anthropic/API call. Translation can be supplied by a local translation map.

Typical local flow
1) OCR discovery already done:
   python ocr_image_region_scanner_v28_1_4_rapidocr_numpyfix.py ... --output-json longman_ocr_remaining_english.json

2) Create patch map + full-page allowed regions:
   python auto_patch_planner_longman_v1.py ^
     --ocr-report longman_ocr_remaining_english.json ^
     --pdf "LONGMAN BFFB Renewable.pdf" ^
     --output-patch-map longman_patch_map_v1.json ^
     --output-image-region-map longman_image_regions_render_v1.json ^
     --report-json longman_planner_report_v1.json ^
     --translation-mode source

3) Render with v27 renderer:
   python pdf_image_region_only_patch_v27_style_presets.py ^
     "LONGMAN BFFB Renewable.pdf" ^
     "longman_output_v1_layout_test.pdf" ^
     --patch-map longman_patch_map_v1.json ^
     --image-region-map longman_image_regions_render_v1.json ^
     --no-auto-image-rects ^
     --font "fonts/NotoSans-Regular.ttf" ^
     --font-bold "fonts/NotoSans-Bold.ttf" ^
     --font-title "fonts/NotoSans-Bold.ttf" ^
     --report-json longman_render_report_v1.json

Notes
- translation-mode=source is for layout test only. It patches OCR text back onto the page.
- For actual Vietnamese translation, pass --translation-map longman_translation_map.json.
- Translation map supports either:
  {"English text": "Vietnamese text"}
  or {"translations": [{"source": "...", "translation": "..."}]}
  or CSV with columns source,translation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None  # type: ignore

BBox = Tuple[float, float, float, float]

BRAND_KEEP_EXACT = {
    "LONGMAN",
    "LONGMAN GROUP",
    "BFFB",
    "BOFA",
    "BOFA HOLDING GROUP",
    "MES",
    "PV",
    "BESS",
    "EPC",
    "O&M",
    "PPA",
    "ESG",
    "AI",
    "IoT",
    "MW",
    "MWh",
    "kW",
    "kWh",
}

# Small safety dictionary only. It is intentionally not a full translator.
# Use --translation-map for production translations.
BUILTIN_TRANSLATIONS = {
    "Create Social Wealth Bring Happiness to Humanity": "Tạo dựng thịnh vượng xã hội, mang hạnh phúc đến nhân loại",
    "BOFA HOLDING GROUP": "BOFA HOLDING GROUP",
    "INCORPORATION WITH": "HỢP TÁC VỚI",
    "LONGMAN GROUP": "LONGMAN GROUP",
    "Renewable Energy": "Năng lượng tái tạo",
    "Solar Energy": "Năng lượng mặt trời",
    "Energy Storage": "Lưu trữ năng lượng",
    "About Us": "Về chúng tôi",
    "Contact Us": "Liên hệ",
    "Project Overview": "Tổng quan dự án",
    "Business Model": "Mô hình kinh doanh",
    "Solution": "Giải pháp",
    "Solutions": "Giải pháp",
    "Advantages": "Lợi thế",
    "Benefits": "Lợi ích",
    "Our Services": "Dịch vụ của chúng tôi",
    "Service": "Dịch vụ",
    "Services": "Dịch vụ",
    "Investment": "Đầu tư",
    "Operation": "Vận hành",
    "Maintenance": "Bảo trì",
    "Engineering": "Kỹ thuật",
    "Construction": "Xây dựng",
    "Development": "Phát triển",
    "Overview": "Tổng quan",
    "Technology": "Công nghệ",
    "Partner": "Đối tác",
    "Partners": "Đối tác",
    "Customer": "Khách hàng",
    "Customers": "Khách hàng",
}

NOISE_PATTERNS = [
    re.compile(r"^[\W_]+$"),
    re.compile(r"^[A-Za-z]{1}$"),
    re.compile(r"^(l|I|i|\||/|\\|_|-|—|–)$"),
]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def norm_text(s: Any) -> str:
    s = str(s or "").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def bbox_tuple(b: Any) -> Optional[BBox]:
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(x) for x in b]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def bbox_w(b: BBox) -> float:
    return max(0.0, b[2] - b[0])


def bbox_h(b: BBox) -> float:
    return max(0.0, b[3] - b[1])


def bbox_area(b: BBox) -> float:
    return bbox_w(b) * bbox_h(b)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def expand_bbox(b: BBox, page_size: Optional[Tuple[float, float]], pad_x: float, pad_y: float) -> List[float]:
    x0, y0, x1, y1 = b
    x0 -= pad_x
    y0 -= pad_y
    x1 += pad_x
    y1 += pad_y
    if page_size:
        w, h = page_size
        x0 = clamp(x0, 0, w)
        x1 = clamp(x1, 0, w)
        y0 = clamp(y0, 0, h)
        y1 = clamp(y1, 0, h)
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def is_mostly_numeric(text: str) -> bool:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    numish = sum(1 for c in chars if c.isdigit() or c in "$€£¥%,.+-/():")
    return numish / max(1, len(chars)) >= 0.65


def is_all_caps_label(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 2:
        return False
    upper = sum(1 for c in letters if c.upper() == c)
    return upper / len(letters) >= 0.82


def looks_like_noise(text: str, conf: float, min_text_length: int) -> bool:
    if not text:
        return True
    if len(text) < min_text_length and not is_mostly_numeric(text):
        return True
    for pat in NOISE_PATTERNS:
        if pat.match(text):
            return True
    # Low-confidence OCR often returns random short uppercase chunks.
    if conf < 0.35 and len(text) <= 4 and text.upper() not in BRAND_KEEP_EXACT and not is_mostly_numeric(text):
        return True
    # A chunk with almost no vowels and many consonants is often OCR garbage, unless brand/numeric.
    letters = re.sub(r"[^A-Za-z]", "", text)
    if conf < 0.55 and len(letters) >= 6 and text.upper() not in BRAND_KEEP_EXACT:
        vowels = len(re.findall(r"[aeiouAEIOU]", letters))
        if vowels / max(1, len(letters)) < 0.12:
            return True
    return False


def load_translation_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Translation map not found: {path}")
    out: Dict[str, str] = {}
    if p.suffix.lower() == ".csv":
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = norm_text(row.get("source") or row.get("Source") or row.get("english") or row.get("English"))
                tr = norm_text(row.get("translation") or row.get("Translation") or row.get("vietnamese") or row.get("Vietnamese"))
                if src and tr:
                    out[norm_key(src)] = tr
        return out

    data = load_json(str(p))
    if isinstance(data, dict) and "translations" in data and isinstance(data["translations"], list):
        for item in data["translations"]:
            if not isinstance(item, dict):
                continue
            src = norm_text(item.get("source") or item.get("text") or item.get("english"))
            tr = norm_text(item.get("translation") or item.get("target") or item.get("vietnamese"))
            if src and tr:
                out[norm_key(src)] = tr
        return out

    if isinstance(data, dict):
        for k, v in data.items():
            src = norm_text(k)
            tr = norm_text(v)
            if src and tr:
                out[norm_key(src)] = tr
        return out

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            src = norm_text(item.get("source") or item.get("text") or item.get("english"))
            tr = norm_text(item.get("translation") or item.get("target") or item.get("vietnamese"))
            if src and tr:
                out[norm_key(src)] = tr
        return out

    raise ValueError("Unsupported translation map format")


def translate_text(text: str, translation_map: Dict[str, str], mode: str) -> Optional[Tuple[str, str]]:
    """Return (translation, method) or None to skip."""
    text = norm_text(text)
    if not text:
        return None

    key = norm_key(text)
    if key in translation_map:
        return translation_map[key], "translation_map"

    # Keep exact brand terms. This avoids translating company/product marks.
    if text.upper() in BRAND_KEEP_EXACT:
        return text, "brand_keep"

    if mode == "map-only":
        return None
    if mode == "blank":
        return "", "blank"
    if mode == "dictionary":
        if text in BUILTIN_TRANSLATIONS:
            return BUILTIN_TRANSLATIONS[text], "builtin_dictionary"
        if key in {norm_key(k) for k in BUILTIN_TRANSLATIONS.keys()}:
            for k, v in BUILTIN_TRANSLATIONS.items():
                if norm_key(k) == key:
                    return v, "builtin_dictionary"
        return text, "source_fallback"
    # source mode
    if text in BUILTIN_TRANSLATIONS:
        return BUILTIN_TRANSLATIONS[text], "builtin_dictionary"
    return text, "source"


def get_page_sizes(pdf_path: Optional[str]) -> Dict[int, Tuple[float, float]]:
    sizes: Dict[int, Tuple[float, float]] = {}
    if not pdf_path:
        return sizes
    if fitz is None:
        print("WARNING: PyMuPDF/fitz is not available; page sizes will be inferred from OCR bbox only.")
        return sizes
    p = Path(pdf_path)
    if not p.exists():
        print(f"WARNING: PDF not found for page sizes: {pdf_path}")
        return sizes
    doc = fitz.open(str(p))
    try:
        for i, page in enumerate(doc):
            sizes[i + 1] = (float(page.rect.width), float(page.rect.height))
    finally:
        doc.close()
    return sizes


def infer_page_sizes_from_items(items: Iterable[Dict[str, Any]]) -> Dict[int, Tuple[float, float]]:
    maxes: Dict[int, Tuple[float, float]] = {}
    for it in items:
        try:
            page = int(it.get("page", 0))
        except Exception:
            continue
        b = bbox_tuple(it.get("bbox"))
        if page <= 0 or not b:
            continue
        mx, my = maxes.get(page, (0.0, 0.0))
        maxes[page] = (max(mx, b[2]), max(my, b[3]))
    # Add some margin; if actual PDF is 960x540, this will be close enough only when PDF not supplied.
    return {p: (math.ceil(x + 20), math.ceil(y + 20)) for p, (x, y) in maxes.items()}


def classify_patch(text: str, bbox: BBox, page_size: Optional[Tuple[float, float]], conf: float) -> Dict[str, Any]:
    w = bbox_w(bbox)
    h = bbox_h(bbox)
    y0 = bbox[1]
    page_h = page_size[1] if page_size else 540.0
    area = bbox_area(bbox)
    text_len = len(text)

    all_caps = is_all_caps_label(text)
    numeric = is_mostly_numeric(text)
    near_top = y0 < page_h * 0.22

    # Estimate font from OCR bbox height. RapidOCR boxes include line height; use conservative size.
    base_size = clamp(h * 0.60, 4.0, 24.0)
    min_size = clamp(base_size * 0.58, 3.0, 12.0)

    style: Dict[str, Any] = {
        "fill": "sample",
        "shape": "rect",
        "align": "center" if (all_caps or numeric or text_len <= 28) else "left",
        "pad_x": 2.2,
        "pad_y": 1.2,
        "line_height": 1.0,
    }

    # Logo / brand / big title
    if text.upper() in BRAND_KEEP_EXACT or (all_caps and h >= 18 and area >= 700):
        style.update({
            "role": "title" if h >= 24 else "label",
            "weight": "bold",
            "font": "title" if h >= 24 else "bold",
            "font_size": round(clamp(h * 0.64, 8.0, 28.0), 2),
            "min_font_size": round(clamp(h * 0.42, 5.0, 14.0), 2),
            "align": "center",
        })
        return style

    # Slide title: big, near top, or very large line.
    if h >= 20 or (near_top and (all_caps or text_len >= 18) and h >= 12):
        style.update({
            "role": "title",
            "weight": "bold",
            "font": "title",
            "font_size": round(clamp(h * 0.62, 9.0, 24.0), 2),
            "min_font_size": round(clamp(h * 0.40, 5.5, 13.0), 2),
            "align": "center" if w > 180 or all_caps else "left",
        })
        return style

    # Section label / short uppercase label.
    if all_caps or text_len <= 16 or numeric:
        style.update({
            "role": "label",
            "weight": "bold" if all_caps or not numeric else "regular",
            "font": "bold" if all_caps else "regular",
            "font_size": round(clamp(h * 0.62, 4.2, 15.0), 2),
            "min_font_size": round(clamp(h * 0.42, 3.0, 9.0), 2),
            "align": "center" if w < 180 else "left",
        })
        return style

    # Body text.
    style.update({
        "role": "body",
        "weight": "regular",
        "font": "regular",
        "font_size": round(base_size, 2),
        "min_font_size": round(min_size, 2),
        "align": "left",
        "pad_x": 2.8,
        "pad_y": 1.6,
    })
    return style


def build_patches(
    ocr_report: Dict[str, Any],
    page_sizes: Dict[int, Tuple[float, float]],
    translation_map: Dict[str, str],
    translation_mode: str,
    min_confidence: float,
    min_text_length: int,
    max_text_length: int,
    filter_noise: bool,
    include_low_confidence_brand: bool,
    bbox_pad_x: float,
    bbox_pad_y: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    patches: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    raw_items = ocr_report.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("OCR report must contain an 'items' list")

    seen = set()

    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        text = norm_text(item.get("text") or item.get("source"))
        bbox = bbox_tuple(item.get("bbox"))
        try:
            page = int(item.get("page", 0))
        except Exception:
            page = 0
        try:
            conf = float(item.get("confidence", item.get("conf", 0.0)) or 0.0)
        except Exception:
            conf = 0.0

        reason = None
        if page <= 0:
            reason = "invalid_page"
        elif bbox is None:
            reason = "invalid_bbox"
        elif not text:
            reason = "empty_text"
        elif len(text) > max_text_length:
            reason = "too_long_for_v1_item_patch"
        elif conf < min_confidence and not (include_low_confidence_brand and text.upper() in BRAND_KEEP_EXACT):
            reason = "low_confidence"
        elif filter_noise and looks_like_noise(text, conf, min_text_length):
            reason = "noise_filter"

        if reason:
            rejected.append({"index": idx, "page": page, "text": text, "confidence": conf, "bbox": list(bbox) if bbox else None, "reason": reason})
            continue

        # Deduplicate repeated OCR candidates with same text and very close bbox.
        dedupe_key = (page, norm_key(text), round(bbox[0], 1), round(bbox[1], 1), round(bbox[2], 1), round(bbox[3], 1))
        if dedupe_key in seen:
            rejected.append({"index": idx, "page": page, "text": text, "confidence": conf, "bbox": list(bbox), "reason": "duplicate"})
            continue
        seen.add(dedupe_key)

        tr = translate_text(text, translation_map, translation_mode)
        if tr is None:
            rejected.append({"index": idx, "page": page, "text": text, "confidence": conf, "bbox": list(bbox), "reason": "no_translation_map_match"})
            continue
        translation, method = tr
        if not translation:
            rejected.append({"index": idx, "page": page, "text": text, "confidence": conf, "bbox": list(bbox), "reason": "blank_translation"})
            continue

        page_size = page_sizes.get(page)
        # Small bbox expansion before renderer-level padding. Helps Vietnamese fit without making text too tiny.
        h = bbox_h(bbox)
        w = bbox_w(bbox)
        pad_x = max(bbox_pad_x, min(12.0, w * 0.08))
        pad_y = max(bbox_pad_y, min(5.0, h * 0.18))
        out_bbox = expand_bbox(bbox, page_size, pad_x=pad_x, pad_y=pad_y)
        style = classify_patch(text, bbox, page_size, conf)

        patches.append({
            "page": page,
            "source": text,
            "translation": translation,
            "bbox": out_bbox,
            "planner_method": "longman_v1_ocr_item",
            "planner_confidence": round(conf, 3),
            "review_status": "auto_approved" if method in {"translation_map", "builtin_dictionary", "brand_keep"} else "layout_test_source",
            "translation_method": method,
            **style,
        })

    patches.sort(key=lambda p: (int(p.get("page", 0)), p.get("bbox", [0, 0, 0, 0])[1], p.get("bbox", [0, 0, 0, 0])[0]))
    return patches, rejected


def build_full_page_region_map(page_sizes: Dict[int, Tuple[float, float]], fallback_pages: Iterable[int]) -> Dict[str, Any]:
    sizes = dict(page_sizes)
    for p in fallback_pages:
        sizes.setdefault(int(p), (960.0, 540.0))
    regions = []
    for p in sorted(sizes):
        w, h = sizes[p]
        regions.append({
            "page": int(p),
            "name": f"longman_p{int(p)}_fullpage_allowed",
            "bbox": [0, 0, round(float(w), 2), round(float(h), 2)],
            "type": "full_page_slide_image",
            "reason": "LONGMAN image-only slide page; allow OCR patches anywhere on the page.",
        })
    return {
        "version": "longman_v1_fullpage_image_regions",
        "note": "Full-page allowed regions for slide/image-heavy LONGMAN PDF. Refine later if needed.",
        "regions": regions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="LONGMAN V1 OCR report to v27 renderer patch map. No API calls.")
    ap.add_argument("--ocr-report", required=True, help="RapidOCR JSON report, e.g. longman_ocr_remaining_english.json")
    ap.add_argument("--pdf", default=None, help="Source PDF for page sizes, e.g. LONGMAN BFFB Renewable.pdf")
    ap.add_argument("--output-patch-map", default="longman_patch_map_v1.json")
    ap.add_argument("--output-image-region-map", default="longman_image_regions_render_v1.json")
    ap.add_argument("--report-json", default="longman_planner_report_v1.json")
    ap.add_argument("--translation-map", default=None, help="Optional JSON/CSV map: source -> translation")
    ap.add_argument("--translation-mode", default="source", choices=["source", "dictionary", "map-only", "blank"], help="source=layout test; dictionary=small built-in dictionary + source fallback; map-only=only mapped translations")
    ap.add_argument("--target-language", default="vi")
    ap.add_argument("--min-confidence", type=float, default=0.35)
    ap.add_argument("--min-text-length", type=int, default=2)
    ap.add_argument("--max-text-length", type=int, default=180)
    ap.add_argument("--no-filter-noise", action="store_true")
    ap.add_argument("--include-low-confidence-brand", action="store_true", default=True)
    ap.add_argument("--bbox-pad-x", type=float, default=1.5)
    ap.add_argument("--bbox-pad-y", type=float, default=0.8)
    args = ap.parse_args()

    ocr = load_json(args.ocr_report)
    raw_items = ocr.get("items", []) if isinstance(ocr, dict) else []
    if not isinstance(raw_items, list):
        raise ValueError("Invalid OCR report: missing items list")

    page_sizes = get_page_sizes(args.pdf)
    if not page_sizes:
        page_sizes = infer_page_sizes_from_items(raw_items)

    translation_map = load_translation_map(args.translation_map)

    patches, rejected = build_patches(
        ocr_report=ocr,
        page_sizes=page_sizes,
        translation_map=translation_map,
        translation_mode=args.translation_mode,
        min_confidence=args.min_confidence,
        min_text_length=args.min_text_length,
        max_text_length=args.max_text_length,
        filter_noise=not args.no_filter_noise,
        include_low_confidence_brand=args.include_low_confidence_brand,
        bbox_pad_x=args.bbox_pad_x,
        bbox_pad_y=args.bbox_pad_y,
    )

    patch_map = {
        "version": "auto_patch_planner_longman_v1_patch_map",
        "base_recommended": args.pdf or "LONGMAN BFFB Renewable.pdf",
        "planner": {
            "name": "auto_patch_planner_longman_v1",
            "plugin": "longman_generic_slide_ocr",
            "mode": "ocr_report + item_patch + inline_style + optional_translation_map",
            "source_ocr_report": args.ocr_report,
            "source_pdf": args.pdf,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "api_calls": 0,
            "translation_mode": args.translation_mode,
            "translation_map": args.translation_map,
            "target_language": args.target_language,
            "review_required_before_production": True,
        },
        "note": "Auto-generated LONGMAN draft patch map. translation-mode=source is for layout test only; use --translation-map for production translation.",
        "patches": patches,
    }

    pages_from_items = sorted({int(i.get("page", 0)) for i in raw_items if isinstance(i, dict) and int(i.get("page", 0) or 0) > 0})
    region_map = build_full_page_region_map(page_sizes, pages_from_items)

    save_json(args.output_patch_map, patch_map)
    save_json(args.output_image_region_map, region_map)

    by_page = Counter(int(p.get("page", 0)) for p in patches)
    rejected_by_reason = Counter(str(r.get("reason")) for r in rejected)
    translation_methods = Counter(str(p.get("translation_method")) for p in patches)
    report = {
        "version": "auto_patch_planner_longman_v1_report",
        "ocr_report": args.ocr_report,
        "pdf": args.pdf,
        "output_patch_map": args.output_patch_map,
        "output_image_region_map": args.output_image_region_map,
        "api_calls": 0,
        "ocr_items_total": len(raw_items),
        "patches_total": len(patches),
        "rejected_total": len(rejected),
        "pages_total": len(page_sizes) or len(pages_from_items),
        "items_by_page": {str(k): v for k, v in sorted(Counter(int(i.get("page", 0)) for i in raw_items if isinstance(i, dict)).items()) if k > 0},
        "patches_by_page": {str(k): v for k, v in sorted(by_page.items()) if k > 0},
        "rejected_by_reason": dict(rejected_by_reason),
        "translation_methods": dict(translation_methods),
        "settings": {
            "translation_mode": args.translation_mode,
            "min_confidence": args.min_confidence,
            "min_text_length": args.min_text_length,
            "max_text_length": args.max_text_length,
            "filter_noise": not args.no_filter_noise,
            "bbox_pad_x": args.bbox_pad_x,
            "bbox_pad_y": args.bbox_pad_y,
        },
        "sample_patches": patches[:30],
        "sample_rejected": rejected[:30],
    }
    save_json(args.report_json, report)

    print("LONGMAN V1 Auto Patch Planner summary:")
    print(f"  ocr_report={args.ocr_report}")
    print(f"  pdf={args.pdf or ''}")
    print(f"  output_patch_map={args.output_patch_map}")
    print(f"  output_image_region_map={args.output_image_region_map}")
    print(f"  report_json={args.report_json}")
    print(f"  ocr_items_total={len(raw_items)}")
    print(f"  patches_total={len(patches)}")
    print(f"  rejected_total={len(rejected)}")
    print(f"  pages_total={len(page_sizes) or len(pages_from_items)}")
    print(f"  translation_mode={args.translation_mode}")
    print(f"  translation_methods={dict(translation_methods)}")
    print(f"  rejected_by_reason={dict(rejected_by_reason)}")
    print("  note=No API call. Render with pdf_image_region_only_patch_v27_style_presets.py using the generated patch map and image region map.")


if __name__ == "__main__":
    main()
