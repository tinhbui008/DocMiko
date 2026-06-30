#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V29.3.1 QA Fit Cleanup for translated OCR block patch maps.

Purpose:
- No OCR / no LLM / no API calls.
- Reuse the translated V29.3 patch map and fix layout-risk blocks.
- Compact overlong Vietnamese translations.
- Add safe line breaks for map labels and project cards.
- Mark risky blocks and write an audit report.

Inputs:
  --patch-map longman_patch_map_v29_3_real_translate.json
  --render-report longman_render_report_v29_3_real_translate.json  (optional but useful)

Output:
  longman_patch_map_v29_3_1_qa_fit_cleanup.json
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def nkey(s: Any) -> str:
    return norm(s).lower()


def bbox_size(p: Dict[str, Any]) -> Tuple[float, float]:
    b = p.get("bbox") or [0, 0, 0, 0]
    try:
        return max(0.0, float(b[2]) - float(b[0])), max(0.0, float(b[3]) - float(b[1]))
    except Exception:
        return 0.0, 0.0


def load_patch_map(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    data = load_json(path)
    if isinstance(data, list):
        root = {"version": "v29_3_1_input_list", "patches": data}
        return root, root["patches"]
    if isinstance(data, dict):
        patches = data.get("patches")
        if isinstance(patches, list):
            root = copy.deepcopy(data)
            return root, root["patches"]
    raise ValueError("Patch map must be a list or a dict containing 'patches'.")


def load_render_fail_keys(report_path: Optional[str]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    if not report_path:
        return {}
    data = load_json(report_path)
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for p in data.get("patches", []):
        if not isinstance(p, dict):
            continue
        if p.get("fit_ok") is False:
            out[(int(p.get("page", 0)), norm(p.get("source")))] = p
    return out


def set_text(p: Dict[str, Any], new_text: str, actions: List[str], reason: str) -> None:
    old = p.get("translation", "")
    # Compare raw stripped strings, not normalized strings, because line breaks are layout-critical.
    if str(old or "").strip() != str(new_text or "").strip():
        p["translation_before_v29_3_1"] = old
        p["translation"] = new_text
        actions.append(reason)


def set_style(
    p: Dict[str, Any],
    actions: List[str],
    *,
    font_size: Optional[float] = None,
    min_font_size: Optional[float] = None,
    align: Optional[str] = None,
    weight: Optional[str] = None,
    pad_x: Optional[float] = None,
    pad_y: Optional[float] = None,
) -> None:
    if font_size is not None:
        old = p.get("font_size")
        if old != font_size:
            p["font_size"] = font_size
            actions.append(f"font_size={font_size}")
    if min_font_size is not None:
        old = p.get("min_font_size")
        if old != min_font_size:
            p["min_font_size"] = min_font_size
            actions.append(f"min_font_size={min_font_size}")
    if align is not None:
        old = p.get("align")
        if old != align:
            p["align"] = align
            actions.append(f"align={align}")
    if weight is not None:
        old = p.get("weight")
        if old != weight:
            p["weight"] = weight
            actions.append(f"weight={weight}")
    if pad_x is not None:
        old = p.get("pad_x")
        if old != pad_x:
            p["pad_x"] = pad_x
            actions.append(f"pad_x={pad_x}")
    if pad_y is not None:
        old = p.get("pad_y")
        if old != pad_y:
            p["pad_y"] = pad_y
            actions.append(f"pad_y={pad_y}")


def wrap_words(text: str, width: int = 24, max_lines: Optional[int] = None) -> str:
    words = norm(text).split()
    lines: List[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if max_lines and len(lines) > max_lines:
        kept = lines[:max_lines]
        tail = " ".join(lines[max_lines-1:])
        # Compact last line with ellipsis only as last resort.
        kept[-1] = tail[: max(6, width - 1)].rstrip() + "…"
        lines = kept
    return "\n".join(lines)


def add_heading_break(text: str, headings: List[str]) -> str:
    t = norm(text)
    for h in headings:
        if t.startswith(h + " "):
            return h + "\n" + t[len(h):].strip()
    return text


def generic_readability_breaks(p: Dict[str, Any], actions: List[str]) -> None:
    """Improve large paragraph blocks without changing meaning."""
    page = int(p.get("page", 0))
    tr = norm(p.get("translation"))
    if not tr:
        return

    if page == 2:
        # Split page 2's two large text blocks into readable title/paragraph lines.
        replacements = [
            (
                "Thời gian thành lập và điểm khởi đầu phát triển BOFA được thành lập",
                "Thời gian thành lập và phát triển\nĐiểm khởi đầu\nBOFA được thành lập",
            ),
            (
                "Các lĩnh vực kinh doanh cốt lõi Danh mục của công ty",
                "Các lĩnh vực kinh doanh cốt lõi\nDanh mục của công ty",
            ),
            (
                "Bốn Nhóm Công ty Con Cốt lõi Tập đoàn",
                "Bốn nhóm công ty con cốt lõi\nTập đoàn",
            ),
        ]
        new = tr
        for a, b in replacements:
            new = new.replace(a, b)
        if new != tr:
            set_text(p, new, actions, "readability_breaks_page2")

    if page == 3:
        new = tr
        for h in [
            "Bố cục các Công ty Con tại Trung Quốc",
            "Phân bổ các Công ty Con ở Nước ngoài",
            "Ưu điểm của Cơ cấu Vận hành Toàn cầu",
        ]:
            new = add_heading_break(new, [h])
        if new != tr:
            set_text(p, new, actions, "readability_breaks_page3")

    if page == 5:
        new = tr
        for h in [
            "Thị trường châu Á: Bố cục dự án trọng điểm",
            "Thị trường châu Phi: Mở rộng dự án năng lượng",
            "Thị trường Nam Mỹ: Tăng cường hợp tác kỹ thuật",
        ]:
            new = add_heading_break(new, [h])
        if new != tr:
            set_text(p, new, actions, "readability_breaks_page5")


def fix_known_risky_block(p: Dict[str, Any], render_fail: bool, actions: List[str]) -> None:
    page = int(p.get("page", 0))
    src = norm(p.get("source"))
    tr = norm(p.get("translation"))

    # Page 6 map labels: they must be compact, multi-line labels.
    if page == 6 and src == "Africa Nigeria Angola Algeria Senegal Botswana":
        set_text(p, "Châu Phi\nNigeria Angola Algeria\nSenegal Botswana", actions, "fix_map_africa_label")
        set_style(p, actions, font_size=11.8, min_font_size=7.2, align="left", weight="bold", pad_x=0.0, pad_y=0.0)
        p["qa_risk"] = "map_label_fit"
        return
    if page == 6 and src.startswith("Asia China Malaysia Laos"):
        set_text(p, "Châu Á\nTrung Quốc Malaysia Lào\nPhilippines Uzbekistan\nViệt Nam Ấn Độ\nIndonesia Pakistan", actions, "fix_map_asia_label")
        set_style(p, actions, font_size=10.4, min_font_size=6.8, align="left", weight="regular", pad_x=0.0, pad_y=0.0)
        p["qa_risk"] = "map_label_fit"
        return
    if page == 6 and src == "South America Brazil Venezuela":
        set_text(p, "Nam Mỹ\nBrazil Venezuela", actions, "fix_map_south_america_label")
        set_style(p, actions, font_size=10.8, min_font_size=7.0, align="left", weight="bold", pad_x=0.0, pad_y=0.0)
        p["qa_risk"] = "map_label_fit"
        return
    if page == 6 and src == "Oceania Australia Papua New Guinea":
        set_text(p, "Châu Đại Dương\nÚc Papua New Guinea", actions, "fix_map_oceania_label")
        set_style(p, actions, font_size=7.8, min_font_size=5.8, align="left", weight="regular", pad_x=0.0, pad_y=0.0)
        p["qa_risk"] = "map_label_fit"
        return

    # Project card OCR ordering / incomplete-source fixes.
    if page == 9 and "Power Station Project Indonesia 3×330MW" in src:
        set_text(p, "Dự án Nhà máy Điện\n3×330MW tại Indonesia", actions, "fix_page9_indonesia_reorder")
        set_style(p, actions, font_size=12.0, min_font_size=7.2, align="center", weight="bold")
        p["qa_risk"] = "ocr_order_repaired"
        return
    if page == 9 and src.startswith("Laos 3×600MW"):
        set_text(p, "Dự án Nhiệt điện Than\nMiệng Mỏ 3×600MW tại Lào", actions, "compact_page9_laos_card")
        set_style(p, actions, font_size=14.0, min_font_size=7.8, align="center", weight="bold")
        return

    if page == 10 and src.startswith("4x420t/h High-Pressure Gas"):
        set_text(p, "Dự án lò hơi đốt khí áp suất cao\nvà lò hơi dầu-khí đồng đốt\n4×420t/h của PetroChina", actions, "compact_page10_boiler_card")
        set_style(p, actions, font_size=10.0, min_font_size=6.4, align="left", weight="bold")
        p["qa_risk"] = "long_blue_card_compacted"
        return

    if page == 11 and src == "Fired Boiler Island Project Algeria 3×200t/h Gas-":
        set_text(p, "Dự án Lò hơi Đốt khí\n3×200t/h tại Algeria", actions, "fix_page11_algeria_incomplete_ocr")
        set_style(p, actions, font_size=13.5, min_font_size=7.4, align="center", weight="bold")
        p["qa_risk"] = "incomplete_ocr_repaired"
        return
    if page == 11 and src.startswith("Combined Cycle Power Malaysia"):
        set_text(p, "Dự án Nhà máy Điện\nChu trình Hỗn hợp\n190MW tại Malaysia", actions, "compact_page11_malaysia_card")
        set_style(p, actions, font_size=13.6, min_font_size=7.2, align="center", weight="bold")
        return

    if page == 12 and src.startswith("50MW Boiler Island Project of Jiangsu"):
        set_text(p, "Dự án Lò hơi 50MW\nJiangsu Shiyou Chemical\nTrung Quốc", actions, "fix_page12_jiangsu_overflow")
        set_style(p, actions, font_size=11.6, min_font_size=6.8, align="center", weight="bold")
        p["qa_risk"] = "overflow_fixed"
        return
    if page == 12 and src.startswith("Project of Baodian"):
        set_text(p, "Dự án Lò hơi Đảo 50MW\nNhà máy Điện Baodian\nYankuang, Trung Quốc", actions, "compact_page12_baodian_card")
        set_style(p, actions, font_size=12.2, min_font_size=6.8, align="center", weight="bold")
        return
    if page == 12 and src.startswith("West Africa 75MW"):
        set_text(p, "Dự án Nhà máy Điện\nChu trình Hỗn hợp khí-hơi\n75MW tại Tây Phi", actions, "compact_page12_west_africa_card")
        set_style(p, actions, font_size=11.2, min_font_size=6.8, align="center", weight="regular")
        return

    # Page 16 contact card: use title/content line breaks.
    if page == 16 and src.startswith("Thank you BOFA FORTUNE"):
        set_text(p, "Cảm ơn\nBOFA FORTUNE BEST HOLDING GROUP\nĐịa chỉ: Courtyard 29A, Xisi Beitou Tiao, Trung Quốc", actions, "readability_page16_contact")
        set_style(p, actions, font_size=18.0, min_font_size=9.0, align="left", weight="regular")
        return

    # For any remaining fit failure, apply conservative compact wrapping and lower max font.
    if render_fail:
        w, h = bbox_size(p)
        width_chars = 22 if w < 190 else 30
        max_lines = max(2, int(h // 12))
        compact = wrap_words(tr, width=width_chars, max_lines=max_lines + 1)
        set_text(p, compact, actions, "generic_fit_failure_wrap")
        old_font = float(p.get("font_size", 14.0) or 14.0)
        set_style(p, actions, font_size=min(old_font, 12.0), min_font_size=5.8, align=str(p.get("align", "center")))
        p["qa_risk"] = p.get("qa_risk", "fit_failure_generic")


def mark_risks(p: Dict[str, Any]) -> List[str]:
    risks: List[str] = []
    page = int(p.get("page", 0))
    src = norm(p.get("source"))
    tr = norm(p.get("translation"))
    w, h = bbox_size(p)
    if page == 6:
        risks.append("map_label_or_global_market_page")
    if 9 <= page <= 15:
        risks.append("project_card_page")
    if len(tr) > max(45, int(w / 3.0)) and h < 75:
        risks.append("long_text_in_small_box")
    if src.endswith("Gas-") or tr.endswith("Khí-"):
        risks.append("incomplete_ocr_tail")
    if "Project Indonesia" in src or "Điện Indonesia" in tr:
        risks.append("possible_ocr_order_issue")
    return sorted(set(risks))


def cleanup_patch_map(input_patch_map: str, output_patch_map: str, render_report: Optional[str], report_json: str) -> None:
    root, patches = load_patch_map(input_patch_map)
    fail_keys = load_render_fail_keys(render_report)

    edits: List[Dict[str, Any]] = []
    risk_items: List[Dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for idx, p in enumerate(patches):
        if not isinstance(p, dict):
            continue
        before = copy.deepcopy(p)
        page = int(p.get("page", 0))
        src = norm(p.get("source"))
        render_fail = (page, src) in fail_keys
        actions: List[str] = []

        # Ensure fields old renderer needs if input came from a render report.
        if "font_size" not in p and "font_size_requested" in p:
            p["font_size"] = p.get("font_size_requested")
        if "fill" not in p:
            p["fill"] = "sample"
        if "shape" not in p:
            p["shape"] = "rect"
        if "pad_x" not in p:
            p["pad_x"] = 0.0
        if "pad_y" not in p:
            p["pad_y"] = 0.0
        if "align" not in p:
            p["align"] = "center"
        if "weight" not in p:
            p["weight"] = "bold" if page in {1, 4, 7, 8} or 9 <= page <= 15 else "regular"
        if "role" not in p:
            p["role"] = "label"

        generic_readability_breaks(p, actions)
        fix_known_risky_block(p, render_fail, actions)

        risks = mark_risks(p)
        if risks:
            p["qa_risks_v29_3_1"] = risks
            risk_items.append({
                "index": idx,
                "page": page,
                "source": src[:160],
                "translation": norm(p.get("translation"))[:180],
                "risks": risks,
                "render_fail_before": bool(render_fail),
            })

        if actions:
            p["qa_fit_cleanup_v29_3_1"] = actions
            for a in actions:
                counters[a.split("=")[0]] += 1
            edits.append({
                "index": idx,
                "page": page,
                "source": src[:180],
                "translation_before": norm(before.get("translation"))[:220],
                "translation_after": norm(p.get("translation"))[:220],
                "actions": actions,
                "render_fail_before": bool(render_fail),
            })

    root["version"] = "v29_3_1_qa_fit_cleanup_patch_map"
    root["created_at"] = now_iso()
    root["planner"] = "qa_patch_map_v29_3_1_fit_cleanup.py"
    root["base_patch_map"] = input_patch_map
    if render_report:
        root["base_render_report"] = render_report
    root["note"] = "No OCR/LLM/API call. V29.3.1 compacts translated risky blocks and fixes fit/overflow candidates."

    save_json(output_patch_map, root)

    report = {
        "version": "v29_3_1_qa_fit_cleanup_report",
        "created_at": now_iso(),
        "input_patch_map": input_patch_map,
        "render_report": render_report,
        "output_patch_map": output_patch_map,
        "patches_total": len(patches),
        "edits_total": len(edits),
        "risk_items_total": len(risk_items),
        "edits_by_action": dict(counters),
        "edits": edits,
        "risk_items_sample": risk_items[:120],
        "fit_failures_from_render_report": [
            {"page": k[0], "source": k[1], "translation": norm(v.get("translation"))}
            for k, v in sorted(fail_keys.items())
        ],
        "next_step": "Render with pdf_image_region_only_patch_v27_1_fit_safe.py or existing v27 renderer and inspect fit_fallbacks/visual pages 6, 9, 11, 12.",
    }
    save_json(report_json, report)

    print("V29.3.1 QA Fit Cleanup summary:")
    print(f"  input_patch_map={input_patch_map}")
    print(f"  render_report={render_report or ''}")
    print(f"  output_patch_map={output_patch_map}")
    print(f"  report_json={report_json}")
    print(f"  patches_total={len(patches)}")
    print(f"  edits_total={len(edits)}")
    print(f"  risk_items_total={len(risk_items)}")
    print(f"  fit_failures_from_render_report={len(fail_keys)}")
    print("  note=No OCR/LLM/API call. Use cache-safe translated patch map cleanup only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-map", required=True, help="Input V29.3 translated patch map JSON")
    ap.add_argument("--render-report", default=None, help="Optional V29.3 render report JSON to locate fit_ok=false blocks")
    ap.add_argument("--output-patch-map", required=True, help="Output V29.3.1 QA-cleaned patch map JSON")
    ap.add_argument("--report-json", required=True, help="Output QA cleanup report JSON")
    args = ap.parse_args()
    cleanup_patch_map(args.patch_map, args.output_patch_map, args.render_report, args.report_json)


if __name__ == "__main__":
    main()
