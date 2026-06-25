#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V28 Auto Patch Planner for image/graphic-region PDF translation patches.

Purpose:
- Read an OCR report (for text that remains inside image/graphic regions).
- Filter only candidates inside approved image/graphic regions.
- Infer compact Vietnamese translations and visual style preset names.
- Output a v27-compatible patch map that can be reviewed and rendered by
  pdf_image_region_only_patch_v27_style_presets.py.

Design notes:
- This planner does NOT edit PDFs.
- This planner does NOT call an LLM/API.
- It produces a draft/approved patch map from OCR candidates + deterministic rules.
- Renderer remains v27 Style Preset Engine.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except Exception as e:  # pragma: no cover
    raise SystemExit("PyMuPDF is required: pip install pymupdf") from e

BBox = List[float]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def rect_from_bbox(bbox: Any) -> Optional[fitz.Rect]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        r = fitz.Rect(*[float(x) for x in bbox])
        if r.is_empty or r.width <= 0 or r.height <= 0:
            return None
        return r
    except Exception:
        return None


def load_manual_regions(path: Optional[str]) -> Dict[int, List[fitz.Rect]]:
    if not path:
        return {}
    data = load_json(path)
    raw = data.get("regions", data if isinstance(data, list) else [])
    out: Dict[int, List[fitz.Rect]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        page = int(item.get("page", 0) or 0)
        r = rect_from_bbox(item.get("bbox"))
        if page > 0 and r is not None:
            out.setdefault(page, []).append(r)
    return out


def rect_inside_regions(rect: fitz.Rect, regions: List[fitz.Rect], min_intersection: float = 0.25) -> bool:
    if not regions:
        return False
    center = fitz.Point((rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0)
    area = max(0.001, rect.get_area())
    for reg in regions:
        if reg.contains(center):
            return True
        inter = rect & reg
        if not inter.is_empty and inter.get_area() / area >= min_intersection:
            return True
    return False


def norm_text(text: str) -> str:
    t = str(text or "").strip()
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t)
    replacements = {
        "Al": "AI",
        "A1": "AI",
        "O5": "05",
        "o5": "05",
        "cavers": "cameras",
        "caver": "camera",
        "siver": "Silver",
        "Siver": "Silver",
        "thirc": "thức",
    }
    for a, b in replacements.items():
        t = t.replace(a, b)
    return t.strip()


def key_text(text: str) -> str:
    t = norm_text(text).lower()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9+]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


TRANSLATION_MEMORY: Dict[str, str] = {
    "basic": "Cơ bản",
    "basic +": "Cơ bản +",
    "basic plus": "Cơ bản +",
    "silver": "Bạc",
    "silver +": "Bạc +",
    "silver plus": "Bạc +",
    "gold": "Vàng",
    "tailored": "Tùy chỉnh",
    "cxview gpt box": "Hộp CXVIEW GPT",
    "the new era of physical edge ai": "Kỷ nguyên mới của AI biên vật lý",
    "near zero latency": "ĐỘ TRỄ GẦN 0",
    "near-zero latency": "ĐỘ TRỄ GẦN 0",
    "data sovereignty": "CHỦ QUYỀN DỮ LIỆU",
    "cost efficiency": "TỐI ƯU CHI PHÍ",
    "bandwidth savings": "TIẾT KIỆM BĂNG THÔNG",
    "upto 05 cameras": "tối đa 05 cam",
    "up to 05 cameras": "tối đa 05 cam",
    "upto 15 cameras": "tối đa 15 cam",
    "up to 15 cameras": "tối đa 15 cam",
    "upto 30 cameras": "tối đa 30 cam",
    "up to 30 cameras": "tối đa 30 cam",
    "authorized personnel": "Nhân sự ủy quyền",
    "dashboard statistical info monitoring": "Giám sát thống kê",
    "functional units": "Đơn vị chức năng",
    "system gpt box cctv": "Hệ thống GPT Box, CCTV",
}


def memory_translate(source: str, suggested: str = "") -> str:
    # Prefer curated memory because OCR suggested translations may be too long.
    k = key_text(source)
    if k in TRANSLATION_MEMORY:
        return TRANSLATION_MEMORY[k]
    if suggested and suggested.strip():
        s = suggested.strip()
        # Keep compact wording for known long suggestions.
        s = s.replace("camera", "cam") if s.lower().startswith("tối đa") else s
        s = s.replace("Giám sát thông tin thống kê", "Giám sát thống kê")
        return s
    return ""


def patch(page: int, source: str, translation: str, bbox: BBox, style: str, **extra: Any) -> Dict[str, Any]:
    p: Dict[str, Any] = {
        "page": int(page),
        "source": source,
        "translation": translation,
        "bbox": [round(float(x), 2) for x in bbox],
        "style": style,
    }
    p.update({k: v for k, v in extra.items() if v is not None})
    return p


def sorted_page_items(items: List[Dict[str, Any]], page: int) -> List[Dict[str, Any]]:
    return sorted([x for x in items if int(x.get("page", 0) or 0) == page], key=lambda x: (float(x.get("bbox", [0,0,0,0])[1]), float(x.get("bbox", [0,0,0,0])[0])))


def plan_pricing_badges(items: List[Dict[str, Any]], regions: Dict[int, List[fitz.Rect]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    planned: List[Dict[str, Any]] = []
    notes: List[Dict[str, Any]] = []
    badge_items = []
    for it in sorted_page_items(items, 2):
        r = rect_from_bbox(it.get("bbox"))
        if r is None or not rect_inside_regions(r, regions.get(2, [])):
            continue
        k = key_text(it.get("text", ""))
        if k in {"basic", "silver", "gold"}:
            badge_items.append(it)

    # Pricing table uses repeated same text. Row position determines plus variant.
    expected = [
        ("Basic", "Cơ bản", "pricing_badge_basic", None),
        ("Basic +", "Cơ bản +", "pricing_badge_basic", 5.9),
        ("Silver", "Bạc", "pricing_badge_silver", None),
        ("Silver +", "Bạc +", "pricing_badge_silver", 5.9),
        ("Gold", "Vàng", "pricing_badge_gold", None),
    ]
    # Use OCR y centers if found, otherwise known row centers.
    default_centers = [200.15, 236.0, 273.9, 311.15, 350.45]
    for idx, (source, trans, style, fs) in enumerate(expected):
        det = badge_items[idx] if idx < len(badge_items) else None
        if det and rect_from_bbox(det.get("bbox")):
            r = rect_from_bbox(det.get("bbox"))
            yc = (r.y0 + r.y1) / 2.0  # type: ignore
            method = "ocr_row_inference"
            confidence = float(det.get("confidence", 70) or 70) / 100.0
        else:
            yc = default_centers[idx]
            method = "template_fallback"
            confidence = 0.62
        bbox = [224.5, yc - 8.65, 281.8, yc + 8.65]
        planned.append(patch(2, source, trans, bbox, style, font_size=fs, planner_method=method, planner_confidence=round(confidence, 3), review_status="auto_approved"))
    notes.append({"page": 2, "rule": "pricing_badges", "ocr_matches": len(badge_items), "planned": 5})
    return planned, notes


def plan_page4_product_and_buttons(items: List[Dict[str, Any]], regions: Dict[int, List[fitz.Rect]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    planned: List[Dict[str, Any]] = []
    notes: List[Dict[str, Any]] = []
    page_items = []
    for it in sorted_page_items(items, 4):
        r = rect_from_bbox(it.get("bbox"))
        if r is not None and rect_inside_regions(r, regions.get(4, [])):
            page_items.append({**it, "_key": key_text(it.get("text", ""))})

    seen = {x["_key"] for x in page_items}
    has_product_area = bool(page_items) or bool(regions.get(4))
    if not has_product_area:
        notes.append({"page": 4, "rule": "product_buttons", "planned": 0, "reason": "no approved region or OCR candidate"})
        return planned, notes

    # Product title/subtitle are detected by OCR. Feature buttons often appear as image text and only some OCR engines catch them,
    # so the four button slots are inferred as a template once the product graphic region is detected.
    templates = [
        ("CXVIEW GPT BOX", "Hộp CXVIEW GPT", [322.0, 111.0, 538.0, 135.2], "product_title_purple", None, "ocr_or_template"),
        ("The new era of physical edge AI", "Kỷ nguyên mới của AI biên vật lý", [326.0, 136.3, 533.0, 154.6], "product_subtitle_dark", None, "ocr_or_template"),
        ("NEAR-ZERO LATENCY", "ĐỘ TRỄ GẦN 0", [674.0, 138.0, 768.0, 152.5], "purple_feature_button", None, "template_inferred"),
        ("DATA SOVEREIGNTY", "CHỦ QUYỀN DỮ LIỆU", [674.0, 164.5, 768.0, 179.0], "purple_feature_button", None, "template_inferred"),
        ("COST EFFICIENCY", "TỐI ƯU CHI PHÍ", [674.0, 191.0, 768.0, 205.5], "purple_feature_button", None, "template_inferred"),
        ("BANDWIDTH SAVINGS", "TIẾT KIỆM BĂNG THÔNG", [674.0, 217.0, 768.0, 231.5], "purple_feature_button", 6.05, "ocr_or_template"),
    ]
    for src, trans, bbox, style, fs, method in templates:
        k = key_text(src)
        detected = k in seen or ("physical edge ai" in k and any("physical edge ai" in s for s in seen))
        pmethod = "ocr_detected" if detected else method
        conf = 0.94 if detected else 0.72
        planned.append(patch(4, src, trans, bbox, style, font_size=fs, planner_method=pmethod, planner_confidence=conf, review_status="auto_approved" if conf >= 0.7 else "needs_review"))
    notes.append({"page": 4, "rule": "product_title_and_feature_buttons", "ocr_matches": len(page_items), "planned": len(templates)})
    return planned, notes


def plan_page5_diagram(items: List[Dict[str, Any]], regions: Dict[int, List[fitz.Rect]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    planned: List[Dict[str, Any]] = []
    notes: List[Dict[str, Any]] = []
    page_items = []
    for it in sorted_page_items(items, 5):
        r = rect_from_bbox(it.get("bbox"))
        if r is not None and rect_inside_regions(r, regions.get(5, [])):
            page_items.append({**it, "_key": key_text(it.get("text", ""))})

    keys = {x["_key"] for x in page_items}
    # Stable diagram slots. OCR provides enough signal to choose these, but exact bboxes are normalized
    # to full text slots so the renderer covers old English cleanly.
    templates = [
        ("Basic", "Cơ bản", [309.5, 260.8, 333.0, 270.2], "diagram_tier_label", None, "basic"),
        ("Silver", "Bạc", [309.5, 274.5, 333.0, 284.2], "diagram_tier_label", None, "silver"),
        ("Gold", "Vàng", [310.0, 288.4, 333.0, 298.2], "diagram_tier_label", None, "gold"),
        ("Tailored", "Tùy chỉnh", [306.0, 301.8, 336.0, 312.0], "diagram_tier_label", 4.3, "tailored"),
        ("upto 05 cameras", "tối đa 05 cam", [336.0, 260.5, 376.0, 270.3], "diagram_small_label", None, "upto 05 cameras"),
        ("upto 15 cameras", "tối đa 15 cam", [336.0, 274.5, 376.0, 284.0], "diagram_small_label", None, "upto 15 cameras"),
        ("upto 30 cameras", "tối đa 30 cam", [336.0, 288.5, 376.0, 300.0], "diagram_small_label", None, "upto 30 cameras"),
        ("Authorized Personnel", "Nhân sự ủy quyền", [590.0, 308.0, 653.0, 318.0], "diagram_caption", None, "authorized personnel"),
        ("Dashboard Statistical info monitoring", "Giám sát thống kê", [424.0, 311.0, 526.0, 322.0], "diagram_caption", None, "dashboard statistical info monitoring"),
        ("Functional Units", "Đơn vị chức năng", [600.0, 315.0, 647.0, 325.0], "diagram_caption", None, "functional units"),
        ("System GPT Box, CCTV", "Hệ thống GPT Box, CCTV", [441.5, 319.0, 508.0, 330.0], "diagram_caption", None, "system gpt box cctv"),
    ]
    for src, trans, bbox, style, fs, key_hint in templates:
        detected = any(key_hint in k or k in key_hint for k in keys)
        # Low-confidence OCR for Silver/upto30 still counts if row/text pattern exists nearby.
        if src == "Silver" and any(k in {"silver", "siver"} for k in keys):
            detected = True
        if src == "upto 30 cameras" and any("upto 30" in k or "up to 30" in k for k in keys):
            detected = True
        conf = 0.91 if detected else 0.68
        planned.append(patch(5, src, trans, bbox, style, font_size=fs, planner_method="ocr_slot_normalized" if detected else "template_fallback", planner_confidence=round(conf, 3), review_status="auto_approved" if conf >= 0.7 else "needs_review"))
    notes.append({"page": 5, "rule": "center_ai_vision_diagram", "ocr_matches": len(page_items), "planned": len(templates)})
    return planned, notes


def generic_candidates_needing_review(items: List[Dict[str, Any]], regions: Dict[int, List[fitz.Rect]], already_planned: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    planned_boxes = [(int(p["page"]), fitz.Rect(*p["bbox"])) for p in already_planned]
    out = []
    for it in items:
        page = int(it.get("page", 0) or 0)
        r = rect_from_bbox(it.get("bbox"))
        if page <= 0 or r is None or not rect_inside_regions(r, regions.get(page, [])):
            continue
        # Ignore if overlaps an already planned patch heavily.
        matched = False
        for p_page, pr in planned_boxes:
            if p_page != page:
                continue
            inter = r & pr
            if not inter.is_empty and inter.get_area() / max(0.001, r.get_area()) > 0.35:
                matched = True
                break
        if matched:
            continue
        txt = norm_text(it.get("text", ""))
        # Skip noise labels that are not useful standalone.
        if key_text(txt) in {"group", "cctv", "nvr"}:
            continue
        trans = memory_translate(txt, str(it.get("suggested_translation", "")))
        out.append({
            "page": page,
            "source": txt,
            "suggested_translation": trans,
            "bbox": [round(float(x), 2) for x in list(r)],
            "confidence": it.get("confidence"),
            "reason": "inside approved image/graphic region but no high-confidence planner rule matched",
            "review_status": "needs_review",
        })
    return out


def dedupe_patches(patches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for p in patches:
        key = (int(p.get("page", 0)), str(p.get("source", "")).lower(), str(p.get("translation", "")).lower(), tuple(round(float(x), 1) for x in p.get("bbox", [])))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def plan_patches(
    ocr_report_path: str,
    image_region_map: str,
    style_presets_path: Optional[str] = None,
    base_recommended: str = "ocr_report_dummy.pdf",
    plugin: str = "cxview",
    include_needs_review: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    report = load_json(ocr_report_path)
    items = report.get("items", report if isinstance(report, list) else [])
    if not isinstance(items, list):
        raise ValueError("OCR report must be a list or contain an 'items' list")
    regions = load_manual_regions(image_region_map)

    planned: List[Dict[str, Any]] = []
    rule_notes: List[Dict[str, Any]] = []

    if plugin.lower() == "cxview":
        for planner in [plan_pricing_badges, plan_page4_product_and_buttons, plan_page5_diagram]:
            p, n = planner(items, regions)
            planned.extend(p)
            rule_notes.extend(n)
    else:
        raise ValueError(f"Unknown plugin: {plugin}. Supported: cxview")

    planned = dedupe_patches(planned)
    needs_review = generic_candidates_needing_review(items, regions, planned)

    if include_needs_review:
        # Keep review candidates disabled so renderer will not apply them until approved.
        for c in needs_review:
            trans = c.get("suggested_translation") or ""
            if not trans:
                continue
            planned.append({
                "page": c["page"],
                "source": c["source"],
                "translation": trans,
                "bbox": c["bbox"],
                "style": "diagram_small_label",
                "enabled": False,
                "planner_method": "generic_needs_review",
                "planner_confidence": 0.45,
                "review_status": "needs_review",
            })

    patch_map = {
        "version": "v28_auto_patch_planner_patch_map",
        "base_recommended": base_recommended,
        "style_presets": Path(style_presets_path).name if style_presets_path else "style_presets_v27.json",
        "planner": {
            "name": "v28_auto_patch_planner",
            "plugin": plugin,
            "mode": "ocr_report + approved_image_regions + deterministic_rules + layout_templates",
            "source_ocr_report": str(ocr_report_path),
            "image_region_map": str(image_region_map),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "api_calls": 0,
            "review_required_before_production": True,
        },
        "note": "Auto-generated draft patch map. Renderer-compatible with v27 Style Preset Engine. Review before production.",
        "patches": planned,
    }

    auto_approved = sum(1 for p in planned if p.get("enabled", True) and p.get("review_status") == "auto_approved")
    inferred = sum(1 for p in planned if "template" in str(p.get("planner_method", "")))
    low_conf = [p for p in planned if float(p.get("planner_confidence", 1.0) or 1.0) < 0.7]
    planner_report = {
        "version": "v28_auto_patch_planner_report",
        "ocr_report": ocr_report_path,
        "image_region_map": image_region_map,
        "style_presets": style_presets_path,
        "plugin": plugin,
        "ocr_items_total": len(items),
        "manual_region_pages": len(regions),
        "planned_patches_total": len(planned),
        "auto_approved_enabled": auto_approved,
        "template_inferred_count": inferred,
        "low_confidence_patch_count": len(low_conf),
        "needs_review_candidates_count": len(needs_review),
        "rule_notes": rule_notes,
        "needs_review_candidates": needs_review,
        "planned_patches": planned,
    }
    return patch_map, planner_report


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="V28 Auto Patch Planner: OCR report -> v27-compatible style-preset patch map")
    ap.add_argument("--ocr-report", required=True, help="OCR report JSON, usually ocr_remaining_english.json")
    ap.add_argument("--image-region-map", required=True, help="Approved image/graphic region map JSON")
    ap.add_argument("--style-presets", default="style_presets_v27.json", help="Style presets referenced by generated patch map")
    ap.add_argument("--output-patch-map", required=True, help="Output generated patch map JSON")
    ap.add_argument("--report-json", default=None, help="Output planner report JSON")
    ap.add_argument("--base-recommended", default="ocr_report_dummy.pdf", help="Clean translated PDF to use as renderer base")
    ap.add_argument("--plugin", default="cxview", choices=["cxview"], help="Rule/plugin profile")
    ap.add_argument("--include-needs-review", action="store_true", help="Append disabled generic candidates for human review")
    args = ap.parse_args(argv)

    patch_map, report = plan_patches(
        ocr_report_path=args.ocr_report,
        image_region_map=args.image_region_map,
        style_presets_path=args.style_presets,
        base_recommended=args.base_recommended,
        plugin=args.plugin,
        include_needs_review=args.include_needs_review,
    )
    write_json(args.output_patch_map, patch_map)
    if args.report_json:
        write_json(args.report_json, report)

    print("V28 Auto Patch Planner summary:")
    print(f"  ocr_report={args.ocr_report}")
    print(f"  image_region_map={args.image_region_map}")
    print(f"  style_presets={args.style_presets}")
    print(f"  output_patch_map={args.output_patch_map}")
    if args.report_json:
        print(f"  report_json={args.report_json}")
    print(f"  ocr_items_total={report['ocr_items_total']}")
    print(f"  manual_region_pages={report['manual_region_pages']}")
    print(f"  planned_patches_total={report['planned_patches_total']}")
    print(f"  auto_approved_enabled={report['auto_approved_enabled']}")
    print(f"  template_inferred_count={report['template_inferred_count']}")
    print(f"  low_confidence_patch_count={report['low_confidence_patch_count']}")
    print(f"  needs_review_candidates_count={report['needs_review_candidates_count']}")
    print("  note=No API call. This creates a draft patch map; render it with v27 Style Preset Engine.")


if __name__ == "__main__":
    main()
