#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V27.1 Fit-Safe Style Preset Engine for Image/Graphic-Region PDF Patching

What changed from v26.3:
- Patch maps can reference visual presets via "style".
- Shared typography/fill/shape/color/padding lives in style_presets_v27.json.
- Patch-level values override preset values.
- Backward compatible with v26/v26.3 patch maps.
- Still safe by default: only patches approved bboxes inside image/graphic regions.
- No OCR and no LLM/API call at runtime.
- V27.1: fallback never draws a long single line outside the patch box; it wraps/clips safely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import fitz  # PyMuPDF

RGB = Tuple[float, float, float]


def rgb01(value: Any, default: RGB = (1.0, 1.0, 1.0)) -> RGB:
    if value is None:
        return default
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in {"none", "sample"}:
            return default
        if s.startswith("#") and len(s) == 7:
            try:
                return (
                    int(s[1:3], 16) / 255.0,
                    int(s[3:5], 16) / 255.0,
                    int(s[5:7], 16) / 255.0,
                )
            except Exception:
                return default
        named = {
            "white": (1, 1, 1),
            "black": (0, 0, 0),
            "purple": (0.427, 0.227, 0.6),
            "dark": (0.08, 0.08, 0.12),
        }
        return named.get(s.lower(), default)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        vals = [float(value[0]), float(value[1]), float(value[2])]
        if max(vals) > 1.0:
            vals = [v / 255.0 for v in vals]
        return tuple(max(0, min(1, v)) for v in vals[:3])  # type: ignore
    return default


def luminance(rgb: RGB) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def first_existing(*paths: Optional[str]) -> Optional[str]:
    for p in paths:
        if p and Path(p).exists():
            return str(Path(p))
    return None


def default_font_paths(font=None, bold=None, title=None):
    regular = first_existing(
        font,
        "fonts/NotoSans-Regular.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    bold_path = first_existing(
        bold,
        "fonts/NotoSans-Bold.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        regular,
    )
    title_path = first_existing(title, bold_path, regular)
    return regular, bold_path, title_path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_patch_map(path: str) -> List[Dict[str, Any]]:
    data = load_json(path)
    patches = data.get("patches", data if isinstance(data, list) else [])
    if not isinstance(patches, list):
        raise ValueError("Patch map must be a list or contain a 'patches' list")
    return [p for p in patches if isinstance(p, dict) and p.get("enabled", True)]


def load_style_presets(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    data = load_json(path)
    presets = data.get("presets", data if isinstance(data, dict) else {})
    if not isinstance(presets, dict):
        raise ValueError("Style preset JSON must be a dict or contain a 'presets' dict")
    return {str(k): v for k, v in presets.items() if isinstance(v, dict)}


def apply_style_presets(patch: Dict[str, Any], presets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Merge style presets into a patch. Patch-level values always win."""
    style = patch.get("style") or patch.get("preset")
    merged: Dict[str, Any] = {}

    style_names: List[str] = []
    if isinstance(style, str) and style.strip():
        style_names = [style.strip()]
    elif isinstance(style, list):
        style_names = [str(x).strip() for x in style if str(x).strip()]

    for name in style_names:
        preset = presets.get(name)
        if preset:
            merged.update(preset)
        else:
            print(f"WARNING: unknown style preset '{name}', using patch fields only")

    merged.update(patch)
    if style_names:
        merged["_resolved_style"] = "+".join(style_names)
    else:
        merged["_resolved_style"] = "inline"
    return merged


def load_manual_regions(path: Optional[str]) -> Dict[int, List[fitz.Rect]]:
    if not path:
        return {}
    data = load_json(path)
    raw_regions = data.get("regions", data if isinstance(data, list) else [])
    regions: Dict[int, List[fitz.Rect]] = {}
    for item in raw_regions:
        if not isinstance(item, dict):
            continue
        page = int(item.get("page", 0))
        bbox = item.get("bbox")
        if page <= 0 or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        regions.setdefault(page, []).append(fitz.Rect(*[float(x) for x in bbox]))
    return regions


def auto_image_rects(source_pdf: str, min_area: float = 1000.0) -> Dict[int, List[fitz.Rect]]:
    out: Dict[int, List[fitz.Rect]] = {}
    if not source_pdf or not Path(source_pdf).exists():
        return out
    doc = fitz.open(source_pdf)
    try:
        for page_index, page in enumerate(doc):
            page_num = page_index + 1
            rects: List[fitz.Rect] = []
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    for r in page.get_image_rects(xref):
                        rr = fitz.Rect(r)
                        if rr.get_area() >= min_area:
                            rects.append(rr)
                except Exception:
                    pass
            out[page_num] = merge_nearby_rects(rects)
    finally:
        doc.close()
    return out


def merge_nearby_rects(rects: List[fitz.Rect], gap: float = 2.0) -> List[fitz.Rect]:
    merged: List[fitz.Rect] = []
    for r in rects:
        cur = fitz.Rect(r)
        changed = True
        while changed:
            changed = False
            new_merged: List[fitz.Rect] = []
            for m in merged:
                mm = fitz.Rect(m)
                test = fitz.Rect(cur)
                test.x0 -= gap
                test.y0 -= gap
                test.x1 += gap
                test.y1 += gap
                if test.intersects(mm):
                    cur |= mm
                    changed = True
                else:
                    new_merged.append(mm)
            merged = new_merged
        merged.append(cur)
    return merged


def combine_regions(*maps: Dict[int, List[fitz.Rect]]) -> Dict[int, List[fitz.Rect]]:
    out: Dict[int, List[fitz.Rect]] = {}
    for m in maps:
        for page, rects in m.items():
            out.setdefault(page, []).extend(rects)
    return {page: merge_nearby_rects(rects, gap=0.5) for page, rects in out.items()}


def rect_allowed(rect: fitz.Rect, regions: List[fitz.Rect], mode: str = "center", threshold: float = 0.30) -> bool:
    if not regions:
        return False
    cx = (rect.x0 + rect.x1) / 2.0
    cy = (rect.y0 + rect.y1) / 2.0
    area = max(0.001, rect.get_area())
    for region in regions:
        if mode == "center" and region.contains(fitz.Point(cx, cy)):
            return True
        inter = rect & region
        if not inter.is_empty and inter.get_area() / area >= threshold:
            return True
    return False


def expand_rect(bbox, pad_x: float = 1.5, pad_y: float = 1.2) -> fitz.Rect:
    r = fitz.Rect(*[float(x) for x in bbox])
    return fitz.Rect(r.x0 - pad_x, r.y0 - pad_y, r.x1 + pad_x, r.y1 + pad_y)


def sample_bg(page: fitz.Page, rect: fitz.Rect, dpi: int = 140) -> RGB:
    try:
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        from PIL import Image  # type: ignore
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        x0 = int(rect.x0 * zoom)
        y0 = int(rect.y0 * zoom)
        x1 = int(rect.x1 * zoom)
        y1 = int(rect.y1 * zoom)
        w, h = img.size
        pad = max(2, int(4 * zoom))
        pts = [
            (x0 - pad, y0 - pad),
            (x1 + pad, y0 - pad),
            (x0 - pad, y1 + pad),
            (x1 + pad, y1 + pad),
            ((x0 + x1) // 2, y0 - pad),
            ((x0 + x1) // 2, y1 + pad),
            (x0 - pad, (y0 + y1) // 2),
            (x1 + pad, (y0 + y1) // 2),
        ]
        vals = []
        for x, y in pts:
            if 0 <= x < w and 0 <= y < h:
                vals.append(img.getpixel((x, y)))
        if not vals:
            return (1, 1, 1)
        rs = sorted(v[0] for v in vals)
        gs = sorted(v[1] for v in vals)
        bs = sorted(v[2] for v in vals)
        mid = len(vals) // 2
        return (rs[mid] / 255.0, gs[mid] / 255.0, bs[mid] / 255.0)
    except Exception:
        return (1, 1, 1)


def draw_round_rect(page: fitz.Page, rect: fitz.Rect, fill: RGB, radius: Optional[float] = None, overlay: bool = True):
    if radius is None:
        radius = min(rect.height / 2.0, rect.width / 2.0)
    radius = max(0.0, min(float(radius), rect.height / 2.0, rect.width / 2.0))
    if radius <= 0.2:
        page.draw_rect(rect, color=None, fill=fill, overlay=overlay)
        return
    mid = fitz.Rect(rect.x0 + radius, rect.y0, rect.x1 - radius, rect.y1)
    if mid.x1 > mid.x0:
        page.draw_rect(mid, color=None, fill=fill, overlay=overlay)
    center = fitz.Rect(rect.x0, rect.y0 + radius, rect.x1, rect.y1 - radius)
    if center.y1 > center.y0:
        page.draw_rect(center, color=None, fill=fill, overlay=overlay)
    page.draw_oval(fitz.Rect(rect.x0, rect.y0, rect.x0 + 2 * radius, rect.y0 + 2 * radius), color=None, fill=fill, overlay=overlay)
    page.draw_oval(fitz.Rect(rect.x1 - 2 * radius, rect.y0, rect.x1, rect.y0 + 2 * radius), color=None, fill=fill, overlay=overlay)
    page.draw_oval(fitz.Rect(rect.x0, rect.y1 - 2 * radius, rect.x0 + 2 * radius, rect.y1), color=None, fill=fill, overlay=overlay)
    page.draw_oval(fitz.Rect(rect.x1 - 2 * radius, rect.y1 - 2 * radius, rect.x1, rect.y1), color=None, fill=fill, overlay=overlay)


def _wrap_for_rect(text: str, rect: fitz.Rect, fontsize: float) -> str:
    """Approximate wrap used only for fit-safe fallback. Keeps drawing inside rect."""
    import re
    words = re.sub(r"\s+", " ", str(text or "").strip()).split()
    if not words:
        return ""
    # NotoSans average char width is roughly 0.50-0.56 * font size. Use conservative 0.54.
    max_chars = max(4, int(rect.width / max(1.0, fontsize * 0.54)))
    lines = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    line_height = max(1.0, fontsize * 1.18)
    max_lines = max(1, int(rect.height / line_height))
    if len(lines) > max_lines:
        # Clip safely instead of drawing one overflowing line.
        lines = lines[:max_lines]
        if lines:
            ell = "…"
            if len(lines[-1]) >= max_chars:
                lines[-1] = lines[-1][: max(1, max_chars - 1)].rstrip() + ell
            else:
                lines[-1] = (lines[-1] + ell)[:max_chars]
    return "\n".join(lines)


def _draw_manual_lines(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontfile: Optional[str],
    fontname: str,
    fontsize: float,
    color: RGB,
    align: str,
) -> None:
    # Draw safely within rect. Long lines were pre-wrapped by _wrap_for_rect.
    lines = str(text or "").splitlines() or [str(text or "")]
    line_height = fontsize * 1.18
    y = rect.y0 + fontsize
    for line in lines:
        if y > rect.y1 + 0.1:
            break
        # PyMuPDF text_length may fail for custom font in rare cases; approximate fallback.
        try:
            tw = fitz.get_text_length(line, fontname=fontname, fontsize=fontsize)
        except Exception:
            tw = len(line) * fontsize * 0.54
        if str(align).lower() == "center":
            x = rect.x0 + max(0.0, (rect.width - tw) / 2.0)
        elif str(align).lower() == "right":
            x = rect.x1 - tw
        else:
            x = rect.x0
        x = max(rect.x0, min(x, rect.x1 - 1.0))
        page.insert_text(
            fitz.Point(x, y),
            line,
            fontsize=fontsize,
            fontname=fontname,
            fontfile=fontfile,
            color=color,
            overlay=True,
        )
        y += line_height


def draw_textbox_fit(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontfile: Optional[str],
    fontname: str,
    fontsize: float,
    color: RGB,
    align: str,
    min_font_size: Optional[float] = None,
) -> Tuple[bool, float]:
    align_map = {
        "left": fitz.TEXT_ALIGN_LEFT,
        "center": fitz.TEXT_ALIGN_CENTER,
        "right": fitz.TEXT_ALIGN_RIGHT,
    }
    pdf_align = align_map.get(str(align).lower(), fitz.TEXT_ALIGN_CENTER)

    size = float(fontsize)
    min_size = float(min_font_size) if min_font_size is not None else max(3.2, size * 0.55)
    min_size = max(2.5, min_size)

    # First try the original textbox engine. It handles real font metrics best.
    while size >= min_size:
        rc = page.insert_textbox(
            rect,
            text,
            fontsize=size,
            fontname=fontname,
            fontfile=fontfile,
            color=color,
            align=pdf_align,
            overlay=True,
        )
        if rc >= 0:
            return True, size
        size -= 0.2

    # V27.1 fit-safe fallback: never draw one long overflowing line.
    # Try progressively smaller wrapped text; if still too long, clipped ellipsis stays inside rect.
    size = min_size
    wrapped = _wrap_for_rect(text, rect, size)
    rc = page.insert_textbox(
        rect,
        wrapped,
        fontsize=size,
        fontname=fontname,
        fontfile=fontfile,
        color=color,
        align=pdf_align,
        overlay=True,
    )
    if rc >= 0:
        return False, size

    _draw_manual_lines(page, rect, wrapped, fontfile, fontname, size, color, align)
    return False, size


def resolve_font(patch: Dict[str, Any], regular_font: Optional[str], bold_font: Optional[str], title_font: Optional[str]):
    font_kind = str(patch.get("font", "auto")).lower()
    role = str(patch.get("role", "label")).lower()
    weight = str(patch.get("weight", "auto")).lower()
    if weight == "auto":
        weight = "bold" if role in {"title", "label"} else "regular"

    if font_kind == "title" or role == "title":
        return title_font or bold_font or regular_font, "FV27Title"
    if font_kind == "regular" or weight == "regular":
        return regular_font or bold_font, "FV27Regular"
    if font_kind in {"bold", "semibold", "black"} or weight in {"bold", "semibold", "black"}:
        return bold_font or regular_font, "FV27Bold"
    return (bold_font or regular_font, "FV27Bold") if role in {"title", "label"} else (regular_font or bold_font, "FV27Regular")


def patch_image_regions_with_styles(
    clean_pdf: str,
    output_pdf: str,
    patch_map_path: str,
    style_presets_path: Optional[str] = None,
    source_pdf: Optional[str] = None,
    image_region_map: Optional[str] = None,
    font: Optional[str] = None,
    font_bold: Optional[str] = None,
    font_title: Optional[str] = None,
    use_auto_image_rects: bool = True,
    dry_run: bool = False,
    debug_regions: bool = False,
    report_json: Optional[str] = None,
):
    raw_patches = load_patch_map(patch_map_path)
    presets = load_style_presets(style_presets_path)
    patches = [apply_style_presets(p, presets) for p in raw_patches]

    auto_regions = auto_image_rects(source_pdf, min_area=1000.0) if (use_auto_image_rects and source_pdf) else {}
    manual_regions = load_manual_regions(image_region_map)
    allowed_regions = combine_regions(auto_regions, manual_regions)

    pdf = fitz.open(clean_pdf)
    regular_font, bold_font, title_font = default_font_paths(font, font_bold, font_title)

    applied = 0
    skipped_outside_region = 0
    skipped_invalid = 0
    fit_fallbacks = 0
    applied_report: List[Dict[str, Any]] = []

    for page_index, page in enumerate(pdf):
        page_num = page_index + 1
        regions = allowed_regions.get(page_num, [])

        if debug_regions:
            for r in regions:
                page.draw_rect(r, color=(0, 0.45, 1), width=0.8, overlay=True)

        for p in patches:
            if int(p.get("page", 0)) != page_num:
                continue

            text_value = str(p.get("translation") or p.get("text") or "").strip()
            bbox = p.get("bbox")
            if not text_value or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                skipped_invalid += 1
                continue

            raw_rect = fitz.Rect(*[float(x) for x in bbox])
            if not rect_allowed(raw_rect, regions, mode="center", threshold=0.25):
                skipped_outside_region += 1
                continue

            pad_x = float(p.get("pad_x", 1.5))
            pad_y = float(p.get("pad_y", 1.2))
            rect = expand_rect(bbox, pad_x=pad_x, pad_y=pad_y)

            if dry_run:
                page.draw_rect(rect, color=(1, 0, 0), width=0.7, overlay=True)
                applied += 1
                applied_report.append({"page": page_num, "source": p.get("source"), "translation": text_value, "style": p.get("_resolved_style"), "dry_run": True})
                continue

            fill_mode = str(p.get("fill", "sample")).lower()
            bg = sample_bg(page, rect)
            if fill_mode and fill_mode != "none":
                fill = bg if fill_mode == "sample" else rgb01(p.get("fill"), default=bg)
                shape = str(p.get("shape", "rect")).lower()
                if shape in {"pill", "round", "rounded"}:
                    radius = p.get("radius", None)
                    draw_round_rect(page, rect, fill=fill, radius=float(radius) if radius is not None else None, overlay=True)
                else:
                    page.draw_rect(rect, color=None, fill=fill, overlay=True)
                bg = fill

            explicit_color = p.get("color")
            if explicit_color is None:
                color = (1, 1, 1) if luminance(bg) < 0.46 else (0.08, 0.08, 0.12)
            else:
                color = rgb01(explicit_color, default=(0, 0, 0))

            fontfile, fontname = resolve_font(p, regular_font, bold_font, title_font)
            if not fontfile:
                fontname = "helv"

            fontsize = float(p.get("font_size", max(4.0, rect.height * 0.72)))
            min_font_size = p.get("min_font_size", None)
            align = str(p.get("align", "center"))
            ok, final_size = draw_textbox_fit(page, rect, text_value, fontfile, fontname, fontsize, color, align, float(min_font_size) if min_font_size is not None else None)
            if not ok:
                fit_fallbacks += 1
            applied += 1
            applied_report.append({
                "page": page_num,
                "source": p.get("source"),
                "translation": text_value,
                "style": p.get("_resolved_style"),
                "bbox": list(raw_rect),
                "font_size_requested": fontsize,
                "font_size_final": round(final_size, 2),
                "fit_ok": ok,
            })

    pdf.save(output_pdf, garbage=4, deflate=True)
    pdf.close()

    if report_json:
        report = {
            "version": "v27_style_presets_report",
            "clean_pdf": clean_pdf,
            "output_pdf": output_pdf,
            "patch_map": patch_map_path,
            "style_presets": style_presets_path,
            "source_pdf": source_pdf,
            "image_region_map": image_region_map,
            "auto_image_pages": sum(1 for v in auto_regions.values() if v),
            "manual_region_pages": sum(1 for v in manual_regions.values() if v),
            "applied": applied,
            "fit_fallbacks": fit_fallbacks,
            "skipped_outside_image_or_graphic_region": skipped_outside_region,
            "skipped_invalid": skipped_invalid,
            "patches": applied_report,
        }
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print("V27.1 fit-safe style-preset image/graphic-region patch summary:")
    print(f"  clean_pdf={clean_pdf}")
    print(f"  output_pdf={output_pdf}")
    print(f"  patch_map={patch_map_path}")
    print(f"  style_presets={style_presets_path or ''}")
    print(f"  source_pdf={source_pdf or ''}")
    print(f"  image_region_map={image_region_map or ''}")
    print(f"  styles_loaded={len(presets)}")
    print(f"  auto_image_pages={sum(1 for v in auto_regions.values() if v)}")
    print(f"  manual_region_pages={sum(1 for v in manual_regions.values() if v)}")
    print(f"  applied={applied}")
    print(f"  fit_fallbacks={fit_fallbacks}")
    print(f"  skipped_outside_image_or_graphic_region={skipped_outside_region}")
    print(f"  skipped_invalid={skipped_invalid}")
    print("  note=Patch style comes from presets first, then patch-level overrides. Fit-safe fallback prevents long overflow outside patch boxes.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clean_pdf", help="Clean translated PDF from text-layer pipeline")
    ap.add_argument("output_pdf", help="Output patched PDF")
    ap.add_argument("--patch-map", required=True, help="Approved patch map JSON")
    ap.add_argument("--style-presets", default=None, help="Style preset JSON. Optional for old inline patch maps.")
    ap.add_argument("--source-pdf", default=None, help="Original/source PDF for automatic raster image detection")
    ap.add_argument("--image-region-map", default=None, help="Manual image/graphic region map JSON")
    ap.add_argument("--font", default=None)
    ap.add_argument("--font-bold", default=None)
    ap.add_argument("--font-title", default=None)
    ap.add_argument("--no-auto-image-rects", action="store_true", help="Disable automatic page.get_images() regions")
    ap.add_argument("--dry-run", action="store_true", help="Draw red patch boxes only")
    ap.add_argument("--debug-regions", action="store_true", help="Draw blue image/graphic allowed regions")
    ap.add_argument("--report-json", default=None, help="Write patch application report JSON")
    args = ap.parse_args()

    patch_image_regions_with_styles(
        clean_pdf=args.clean_pdf,
        output_pdf=args.output_pdf,
        patch_map_path=args.patch_map,
        style_presets_path=args.style_presets,
        source_pdf=args.source_pdf,
        image_region_map=args.image_region_map,
        font=args.font,
        font_bold=args.font_bold,
        font_title=args.font_title,
        use_auto_image_rects=not args.no_auto_image_rects,
        dry_run=args.dry_run,
        debug_regions=args.debug_regions,
        report_json=args.report_json,
    )


if __name__ == "__main__":
    main()
