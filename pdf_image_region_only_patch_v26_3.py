#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V26.3 Image/Graphic-Region-Only Patch

Goal:
- Keep the normal text-layer PDF output untouched.
- Only patch OCR leftovers inside approved image/graphic regions.
- Do NOT OCR or patch the whole page.
- Do NOT call any LLM API.

This is the safer "hybrid" direction:
  text layer outside image/graphic areas = preserved
  OCR/patch candidates inside image/graphic areas = patched only if explicitly approved
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

RGB = Tuple[float, float, float]


def rgb01(value, default: RGB = (1.0, 1.0, 1.0)) -> RGB:
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
            return p
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


def load_patch_map(path: str) -> List[Dict]:
    data = load_json(path)
    patches = data.get("patches", data if isinstance(data, list) else [])
    if not isinstance(patches, list):
        raise ValueError("Patch map must be a list or contain a 'patches' list")
    return [p for p in patches if isinstance(p, dict) and p.get("enabled", True)]


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
    """Detect raster image rectangles from the source PDF.

    Note: many slide graphics are vector groups, not raster images. For those,
    pass --image-region-map to define approved graphic regions.
    """
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
            new_merged = []
            for m in merged:
                mm = fitz.Rect(m)
                test = fitz.Rect(cur)
                test.x0 -= gap; test.y0 -= gap; test.x1 += gap; test.y1 += gap
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



def draw_round_rect_v263(page: fitz.Page, rect: fitz.Rect, fill: RGB, radius: Optional[float] = None, overlay: bool = True):
    """Draw a simple rounded rectangle/pill using one center rect + side circles.

    PyMuPDF's primitive support varies by version; this avoids requiring
    page.draw_rect(..., radius=...) which may not exist.
    """
    if radius is None:
        radius = min(rect.height / 2.0, rect.width / 2.0)
    radius = max(0.0, min(float(radius), rect.height / 2.0, rect.width / 2.0))

    if radius <= 0.2:
        page.draw_rect(rect, color=None, fill=fill, overlay=overlay)
        return

    # Main horizontal bar and vertical middle.
    mid = fitz.Rect(rect.x0 + radius, rect.y0, rect.x1 - radius, rect.y1)
    if mid.x1 > mid.x0:
        page.draw_rect(mid, color=None, fill=fill, overlay=overlay)

    center = fitz.Rect(rect.x0, rect.y0 + radius, rect.x1, rect.y1 - radius)
    if center.y1 > center.y0:
        page.draw_rect(center, color=None, fill=fill, overlay=overlay)

    # Four corner circles approximate rounded corners.
    page.draw_oval(fitz.Rect(rect.x0, rect.y0, rect.x0 + 2*radius, rect.y0 + 2*radius), color=None, fill=fill, overlay=overlay)
    page.draw_oval(fitz.Rect(rect.x1 - 2*radius, rect.y0, rect.x1, rect.y0 + 2*radius), color=None, fill=fill, overlay=overlay)
    page.draw_oval(fitz.Rect(rect.x0, rect.y1 - 2*radius, rect.x0 + 2*radius, rect.y1), color=None, fill=fill, overlay=overlay)
    page.draw_oval(fitz.Rect(rect.x1 - 2*radius, rect.y1 - 2*radius, rect.x1, rect.y1), color=None, fill=fill, overlay=overlay)


def draw_textbox_fit(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontfile: Optional[str],
    fontname: str,
    fontsize: float,
    color: RGB,
    align: str,
):
    align_map = {
        "left": fitz.TEXT_ALIGN_LEFT,
        "center": fitz.TEXT_ALIGN_CENTER,
        "right": fitz.TEXT_ALIGN_RIGHT,
    }
    pdf_align = align_map.get(str(align).lower(), fitz.TEXT_ALIGN_CENTER)

    size = float(fontsize)
    min_size = max(3.2, size * 0.55)
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
            return True
        size -= 0.2

    page.insert_text(
        fitz.Point(rect.x0 + 0.5, rect.y0 + max(3.8, min_size)),
        text,
        fontsize=min_size,
        fontname=fontname,
        fontfile=fontfile,
        color=color,
        overlay=True,
    )
    return False


def patch_image_regions_only(
    clean_pdf: str,
    output_pdf: str,
    patch_map_path: str,
    source_pdf: Optional[str] = None,
    image_region_map: Optional[str] = None,
    font: Optional[str] = None,
    font_bold: Optional[str] = None,
    font_title: Optional[str] = None,
    use_auto_image_rects: bool = True,
    dry_run: bool = False,
    debug_regions: bool = False,
):
    patches = load_patch_map(patch_map_path)

    auto_regions = auto_image_rects(source_pdf, min_area=1000.0) if (use_auto_image_rects and source_pdf) else {}
    manual_regions = load_manual_regions(image_region_map)
    allowed_regions = combine_regions(auto_regions, manual_regions)

    pdf = fitz.open(clean_pdf)
    regular_font, bold_font, title_font = default_font_paths(font, font_bold, font_title)

    applied = 0
    skipped_outside_region = 0
    skipped_invalid = 0

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
                continue

            fill_mode = str(p.get("fill", "sample")).lower()
            bg = sample_bg(page, rect)
            if fill_mode and fill_mode != "none":
                fill = bg if fill_mode == "sample" else rgb01(p.get("fill"), default=bg)
                shape = str(p.get("shape", "rect")).lower()
                if shape in {"pill", "round", "rounded"}:
                    radius = p.get("radius", None)
                    draw_round_rect_v263(page, rect, fill=fill, radius=float(radius) if radius is not None else None, overlay=True)
                else:
                    page.draw_rect(rect, color=None, fill=fill, overlay=True)
                bg = fill

            explicit_color = p.get("color")
            if explicit_color is None:
                color = (1, 1, 1) if luminance(bg) < 0.46 else (0.08, 0.08, 0.12)
            else:
                color = rgb01(explicit_color, default=(0, 0, 0))

            role = str(p.get("role", "label")).lower()
            weight = str(p.get("weight", "auto")).lower()
            if weight == "auto":
                weight = "bold" if role in {"title", "label"} else "regular"

            if role == "title":
                fontfile = title_font or bold_font or regular_font
                fontname = "FV26Title"
            elif weight in {"bold", "semibold", "black"}:
                fontfile = bold_font or regular_font
                fontname = "FV26Bold"
            else:
                fontfile = regular_font or bold_font
                fontname = "FV26Regular"

            if not fontfile:
                fontname = "helv"

            fontsize = float(p.get("font_size", max(4.0, rect.height * 0.72)))
            align = str(p.get("align", "center"))
            draw_textbox_fit(page, rect, text_value, fontfile, fontname, fontsize, color, align)
            applied += 1

    pdf.save(output_pdf, garbage=4, deflate=True)
    pdf.close()

    print("V26.3 image/graphic-region-only patch summary:")
    print(f"  clean_pdf={clean_pdf}")
    print(f"  output_pdf={output_pdf}")
    print(f"  patch_map={patch_map_path}")
    print(f"  source_pdf={source_pdf or ''}")
    print(f"  image_region_map={image_region_map or ''}")
    print(f"  auto_image_pages={sum(1 for v in auto_regions.values() if v)}")
    print(f"  manual_region_pages={sum(1 for v in manual_regions.values() if v)}")
    print(f"  applied={applied}")
    print(f"  skipped_outside_image_or_graphic_region={skipped_outside_region}")
    print(f"  skipped_invalid={skipped_invalid}")
    print("  note=Only approved patches whose bbox is inside image/graphic regions were applied. Text-layer outside those regions was preserved.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clean_pdf", help="Clean translated PDF from text-layer pipeline")
    ap.add_argument("output_pdf", help="Output patched PDF")
    ap.add_argument("--patch-map", required=True, help="Approved patch map JSON")
    ap.add_argument("--source-pdf", default=None, help="Original/source PDF for automatic raster image detection")
    ap.add_argument("--image-region-map", default=None, help="Manual image/graphic region map JSON")
    ap.add_argument("--font", default=None)
    ap.add_argument("--font-bold", default=None)
    ap.add_argument("--font-title", default=None)
    ap.add_argument("--no-auto-image-rects", action="store_true", help="Disable automatic page.get_images() regions")
    ap.add_argument("--dry-run", action="store_true", help="Draw red patch boxes only")
    ap.add_argument("--debug-regions", action="store_true", help="Draw blue image/graphic allowed regions")
    args = ap.parse_args()

    patch_image_regions_only(
        clean_pdf=args.clean_pdf,
        output_pdf=args.output_pdf,
        patch_map_path=args.patch_map,
        source_pdf=args.source_pdf,
        image_region_map=args.image_region_map,
        font=args.font,
        font_bold=args.font_bold,
        font_title=args.font_title,
        use_auto_image_rects=not args.no_auto_image_rects,
        dry_run=args.dry_run,
        debug_regions=args.debug_regions,
    )


if __name__ == "__main__":
    main()
