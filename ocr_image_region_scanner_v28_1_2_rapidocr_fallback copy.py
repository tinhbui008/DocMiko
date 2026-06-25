#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V28.1.2 OCR Image-Region Scanner for PDF translation patches.

Purpose:
- OCR only approved image/graphic regions, not the whole page.
- Use multiple preprocessing variants to improve recall on small embedded labels.
- Convert OCR boxes back to PDF coordinate space.
- Merge duplicate detections across variants.
- Output a planner-compatible OCR report for auto_patch_planner_v28.py.

No LLM/API call is made.

Recommended flow:
  python ocr_image_region_scanner_v28_1.py source_or_clean.pdf \
    --image-region-map cxview_image_graphic_regions_v26.json \
    --output-json ocr_remaining_english_v28_1.json \
    --debug-dir ocr_debug_v28_1 \
    --dpi 320 --lang en

Then:
  python auto_patch_planner_v28.py --ocr-report ocr_remaining_english_v28_1.json ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise SystemExit("PyMuPDF is required: pip install pymupdf") from e

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
except Exception as e:
    raise SystemExit("Pillow is required: pip install pillow") from e

try:
    import numpy as np
except Exception as e:
    raise SystemExit("numpy is required: pip install numpy") from e

try:
    import cv2  # optional but recommended
except Exception:
    cv2 = None

BBox = Tuple[float, float, float, float]


@dataclass
class OCRItem:
    page: int
    region_id: str
    text: str
    normalized_text: str
    bbox: List[float]
    confidence: float
    variant: str
    detection_count: int
    source: str = "ocr_v28_1_2"
    english_like: bool = True
    review_status: str = "candidate"


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


def load_regions(path: str, pages: Optional[Iterable[int]] = None) -> Dict[int, List[Tuple[str, fitz.Rect]]]:
    data = load_json(path)
    raw = data.get("regions", data if isinstance(data, list) else [])
    page_filter = set(int(p) for p in pages) if pages else None
    out: Dict[int, List[Tuple[str, fitz.Rect]]] = {}
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        page = int(item.get("page", 0) or 0)
        if page_filter is not None and page not in page_filter:
            continue
        r = rect_from_bbox(item.get("bbox"))
        if page > 0 and r is not None:
            rid = str(item.get("id") or f"p{page}_region_{idx+1}")
            out.setdefault(page, []).append((rid, r))
    return out


def pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    return img


def render_page(pdf: fitz.Document, page_index0: int, dpi: int) -> Image.Image:
    page = pdf[page_index0]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pixmap_to_pil(pix)


def crop_region(page_img: Image.Image, rect: fitz.Rect, dpi: int, pad_px: int = 4) -> Tuple[Image.Image, Tuple[int, int]]:
    zoom = dpi / 72.0
    x0 = max(0, int(math.floor(rect.x0 * zoom)) - pad_px)
    y0 = max(0, int(math.floor(rect.y0 * zoom)) - pad_px)
    x1 = min(page_img.width, int(math.ceil(rect.x1 * zoom)) + pad_px)
    y1 = min(page_img.height, int(math.ceil(rect.y1 * zoom)) + pad_px)
    return page_img.crop((x0, y0, x1, y1)), (x0, y0)


def preprocess_variants(img: Image.Image) -> Dict[str, Image.Image]:
    """Return OCR-friendly variants. Keep names stable because reports use them."""
    variants: Dict[str, Image.Image] = {}
    rgb = img.convert("RGB")
    variants["rgb"] = rgb

    # Slight upscale helps small labels in diagrams.
    variants["rgb_up2"] = rgb.resize((max(1, rgb.width * 2), max(1, rgb.height * 2)), Image.Resampling.LANCZOS)

    gray = rgb.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(1.7)
    gray = gray.filter(ImageFilter.SHARPEN)
    variants["gray_contrast"] = gray.convert("RGB")
    variants["gray_contrast_up2"] = gray.resize((max(1, gray.width * 2), max(1, gray.height * 2)), Image.Resampling.LANCZOS).convert("RGB")

    if cv2 is not None:
        arr = np.array(rgb)
        gray_cv = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray_cv = cv2.GaussianBlur(gray_cv, (3, 3), 0)
        # Adaptive threshold is useful for purple/gray badges and tiny captions.
        thr = cv2.adaptiveThreshold(gray_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
        variants["adaptive_thresh"] = Image.fromarray(thr).convert("RGB")
        inv = 255 - thr
        variants["adaptive_thresh_inv"] = Image.fromarray(inv).convert("RGB")
        # CLAHE improves low-contrast text without hard thresholding.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_cv)
        variants["clahe"] = Image.fromarray(clahe).convert("RGB")
    return variants


def init_paddle(lang: str = "en", use_angle_cls: bool = True):
    """Initialize PaddleOCR with CPU-safe defaults.

    Some Paddle/PaddleOCR builds on Windows CPU can crash inside oneDNN/MKLDNN
    fused_conv2d with errors like:
        NotFoundError: OneDnnContext does not have the input Filter
    For this scanner we prefer stable OCR over oneDNN speed, so MKLDNN is
    disabled both by environment flag and by constructor kwargs when supported.
    """
    # Must be set before Paddle initializes inference internals.
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        raise SystemExit(
            "PaddleOCR is not installed. Install locally, for example:\n"
            "  pip install paddleocr paddlepaddle\n"
            "Then rerun this script."
        ) from e

    # PaddleOCR has changed parameters across versions. Try CPU-safe signatures.
    candidate_kwargs = [
        {"use_angle_cls": use_angle_cls, "lang": lang, "show_log": False, "use_gpu": False, "enable_mkldnn": False},
        {"use_angle_cls": use_angle_cls, "lang": lang, "show_log": False, "enable_mkldnn": False},
        {"use_angle_cls": use_angle_cls, "lang": lang, "show_log": False, "use_mkldnn": False},
        {"use_angle_cls": use_angle_cls, "lang": lang, "show_log": False},
        {"use_angle_cls": use_angle_cls, "lang": lang},
        {"lang": lang},
    ]

    last_type_error = None
    for kwargs in candidate_kwargs:
        try:
            return PaddleOCR(**kwargs)
        except TypeError as e:
            last_type_error = e
            continue

    raise last_type_error or RuntimeError("Could not initialize PaddleOCR")




def init_rapidocr():
    """Initialize RapidOCR / ONNXRuntime fallback.

    This avoids Paddle CPU oneDNN/MKLDNN fused_conv2d issues on some Windows builds.
    Install one of these if missing:
      pip install rapidocr onnxruntime
    or, for older API:
      pip install rapidocr-onnxruntime
    """
    last_error = None
    for mod_name in ("rapidocr", "rapidocr_onnxruntime"):
        try:
            mod = __import__(mod_name, fromlist=["RapidOCR"])
            RapidOCR = getattr(mod, "RapidOCR")
            return RapidOCR()
        except Exception as e:
            last_error = e
            continue
    raise SystemExit(
        "RapidOCR fallback is not installed. Install it in your venv:\n"
        "  pip install rapidocr onnxruntime\n"
        "If that fails on your Python version, try:\n"
        "  pip install rapidocr-onnxruntime\n"
        f"Last import error: {last_error}"
    )


def run_rapidocr(engine, image_path: str, box_thresh: float = 0.35, text_score: float = 0.20):
    """Run RapidOCR across old/new APIs and return raw OCR result list."""
    try:
        raw = engine(image_path, box_thresh=box_thresh, text_score=text_score)
    except TypeError:
        raw = engine(image_path)
    # Common older API: (ocr_result, infer_elapse)
    if isinstance(raw, tuple) and len(raw) >= 1:
        raw = raw[0]
    # Some newer APIs may return an object with .boxes/.txts/etc.; keep raw for normalizer.
    return raw


def init_ocr_engine(engine_name: str, lang: str = "en", use_angle_cls: bool = True):
    engine_name = (engine_name or "auto").strip().lower()
    if engine_name == "rapidocr":
        return "rapidocr", init_rapidocr()
    if engine_name == "paddle":
        return "paddle", init_paddle(lang=lang, use_angle_cls=use_angle_cls)
    # auto: prefer RapidOCR because Paddle can crash on Windows CPU oneDNN/MKLDNN.
    try:
        return "rapidocr", init_rapidocr()
    except SystemExit as rapid_error:
        print(f"WARNING: RapidOCR unavailable, falling back to PaddleOCR: {rapid_error}", file=sys.stderr)
        return "paddle", init_paddle(lang=lang, use_angle_cls=use_angle_cls)


def normalize_ocr_result(raw: Any) -> List[Tuple[List[List[float]], str, float]]:
    """Normalize PaddleOCR result into [(quad, text, confidence 0-1), ...]."""
    out: List[Tuple[List[List[float]], str, float]] = []
    if raw is None:
        return out

    # Old PaddleOCR: [ [ [box], (text, score) ], ... ] or [page_result]
    candidates = raw
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        # Could be page wrapper or a single line; inspect nested shape.
        if raw[0] and isinstance(raw[0][0], (list, tuple)) and len(raw[0][0]) == 2 and isinstance(raw[0][0][1], (tuple, list)):
            candidates = raw[0]

    # New PaddleOCR sometimes returns dicts with rec_texts / rec_scores / rec_boxes.
    if isinstance(raw, dict):
        texts = raw.get("rec_texts") or raw.get("texts") or []
        scores = raw.get("rec_scores") or raw.get("scores") or []
        boxes = raw.get("rec_boxes") or raw.get("boxes") or []
        for box, text, score in zip(boxes, texts, scores):
            quad = box_to_quad(box)
            if quad and text:
                out.append((quad, str(text), float(score)))
        return out

    if not isinstance(candidates, list):
        return out

    for item in candidates:
        try:
            if isinstance(item, dict):
                text = item.get("text") or item.get("rec_text") or ""
                score = item.get("score") or item.get("confidence") or item.get("rec_score") or 0
                quad = box_to_quad(item.get("box") or item.get("bbox") or item.get("points"))
                if quad and text:
                    out.append((quad, str(text), float(score)))
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                quad = box_to_quad(item[0])
                # Paddle style: [box, (text, score)]
                # RapidOCR style: [box, text, score]
                if len(item) >= 3 and not isinstance(item[1], (list, tuple, dict)):
                    text = str(item[1])
                    score = float(item[2] or 0.0)
                else:
                    text_score = item[1]
                    if isinstance(text_score, (list, tuple)) and len(text_score) >= 2:
                        text = str(text_score[0])
                        score = float(text_score[1])
                    else:
                        text = str(text_score)
                        score = 0.0
                if quad and text:
                    out.append((quad, text, score))
        except Exception:
            continue
    return out


def box_to_quad(box: Any) -> Optional[List[List[float]]]:
    if box is None:
        return None
    try:
        # Rect [x0,y0,x1,y1]
        if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
            x0, y0, x1, y1 = [float(x) for x in box]
            return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        # Quad [[x,y]...]
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            pts = []
            for p in box[:4]:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append([float(p[0]), float(p[1])])
            if len(pts) == 4:
                return pts
    except Exception:
        return None
    return None


def quad_to_bbox_pdf(quad: List[List[float]], crop_offset: Tuple[int, int], dpi: int, variant_scale: float) -> List[float]:
    # variant_scale = 2.0 for upscaled variants; OCR quad is in variant pixels.
    xs = [p[0] / variant_scale + crop_offset[0] for p in quad]
    ys = [p[1] / variant_scale + crop_offset[1] for p in quad]
    zoom = dpi / 72.0
    x0, y0, x1, y1 = min(xs) / zoom, min(ys) / zoom, max(xs) / zoom, max(ys) / zoom
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def normalize_text(text: str) -> str:
    t = str(text or "").strip()
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t)
    fix = {
        "Al": "AI", "A1": "AI", "A|": "AI", "O5": "05", "o5": "05",
        "cavers": "cameras", "caver": "camera", "Siver": "Silver", "siver": "Silver",
        "Basc": "Basic", "BASIC": "Basic", "Goid": "Gold", "G0ld": "Gold",
        "Latencv": "Latency", "Sovereiqnty": "Sovereignty", "Bandwith": "Bandwidth",
    }
    for a, b in fix.items():
        t = t.replace(a, b)
    return t.strip()


def english_like(text: str, include_short_labels: bool = True) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    lower = t.lower()
    keep_phrases = [
        "basic", "silver", "gold", "tailored", "cxview", "gpt", "physical", "edge", "latency",
        "sovereignty", "efficiency", "bandwidth", "authorized", "functional", "dashboard", "system",
        "camera", "cameras", "upto", "up to", "nvr", "cctv",
    ]
    if any(p in lower for p in keep_phrases):
        return True
    words = re.findall(r"\b[A-Za-z][A-Za-z\-]{2,}\b", t)
    if include_short_labels and re.fullmatch(r"[A-Za-z]{2,}\s*\+?", t):
        return True
    return bool(words)


def text_key(text: str) -> str:
    t = normalize_text(text).lower()
    t = re.sub(r"[^a-z0-9+]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def rect_iou(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def center_dist(a: List[float], b: List[float]) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return math.hypot(acx - bcx, acy - bcy)


def merge_detections(items: List[OCRItem]) -> List[OCRItem]:
    # Prefer higher-confidence text and merge boxes from repeated variants.
    sorted_items = sorted(items, key=lambda x: (-x.confidence, x.page, x.bbox[1], x.bbox[0]))
    groups: List[List[OCRItem]] = []
    for item in sorted_items:
        k = text_key(item.normalized_text)
        placed = False
        for g in groups:
            g0 = g[0]
            same_text = k == text_key(g0.normalized_text) or (k and k in text_key(g0.normalized_text)) or (text_key(g0.normalized_text) in k)
            close = center_dist(item.bbox, g0.bbox) <= 9.0 or rect_iou(item.bbox, g0.bbox) >= 0.35
            if item.page == g0.page and item.region_id == g0.region_id and same_text and close:
                g.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])

    out: List[OCRItem] = []
    for g in groups:
        best = max(g, key=lambda x: x.confidence)
        # Union box is safer for covering source text; but do not let a bad variant balloon too much.
        xs0 = [x.bbox[0] for x in g]
        ys0 = [x.bbox[1] for x in g]
        xs1 = [x.bbox[2] for x in g]
        ys1 = [x.bbox[3] for x in g]
        union = [round(min(xs0), 2), round(min(ys0), 2), round(max(xs1), 2), round(max(ys1), 2)]
        if (union[2] - union[0]) > max(1.0, (best.bbox[2] - best.bbox[0]) * 2.2):
            union = best.bbox
        best.bbox = union
        best.detection_count = len(g)
        # Confidence bonus for repeat detections.
        best.confidence = round(min(0.99, best.confidence + 0.03 * (len(g) - 1)), 3)
        out.append(best)
    return sorted(out, key=lambda x: (x.page, x.bbox[1], x.bbox[0]))


def annotate_debug(crop: Image.Image, detections: List[OCRItem], crop_offset: Tuple[int, int], dpi: int, out_path: str) -> None:
    zoom = dpi / 72.0
    img = crop.convert("RGB")
    draw = ImageDraw.Draw(img)
    ox, oy = crop_offset
    for item in detections:
        x0, y0, x1, y1 = item.bbox
        px0, py0 = x0 * zoom - ox, y0 * zoom - oy
        px1, py1 = x1 * zoom - ox, y1 * zoom - oy
        draw.rectangle([px0, py0, px1, py1], outline="red", width=2)
        draw.text((px0, max(0, py0 - 10)), item.normalized_text[:36], fill="red")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def scan_regions(
    pdf_path: str,
    region_map: str,
    output_json: str,
    dpi: int = 320,
    lang: str = "en",
    min_confidence: float = 0.35,
    pages: Optional[List[int]] = None,
    debug_dir: Optional[str] = None,
    use_angle_cls: bool = True,
    engine_name: str = "auto",
) -> Dict[str, Any]:
    regions = load_regions(region_map, pages=pages)
    if not regions:
        raise ValueError(f"No regions found in {region_map}")

    active_engine_name, ocr = init_ocr_engine(engine_name, lang=lang, use_angle_cls=use_angle_cls)
    pdf = fitz.open(pdf_path)
    all_items: List[OCRItem] = []
    debug_index: Dict[str, Any] = {}

    for page_num in sorted(regions):
        page_index0 = page_num - 1
        if page_index0 < 0 or page_index0 >= len(pdf):
            continue
        page_img = render_page(pdf, page_index0, dpi=dpi)
        for ridx, (region_id, rect) in enumerate(regions[page_num], start=1):
            crop, offset = crop_region(page_img, rect, dpi=dpi, pad_px=5)
            variants = preprocess_variants(crop)
            region_items: List[OCRItem] = []
            if debug_dir:
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
                crop.save(str(Path(debug_dir) / f"p{page_num:02d}_{region_id}_crop.png"))

            for variant_name, vimg in variants.items():
                # Save temp image because PaddleOCR APIs are most stable with file paths.
                cache_name = hashlib.sha1((pdf_path + region_id + variant_name + str(dpi)).encode("utf-8")).hexdigest()[:12]
                tmp_dir = Path(debug_dir or ".ocr_tmp_v28_1")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / f"ocr_{cache_name}.png"
                vimg.save(tmp_path)
                try:
                    if active_engine_name == "rapidocr":
                        raw = run_rapidocr(ocr, str(tmp_path), box_thresh=0.30, text_score=max(0.10, min_confidence * 0.55))
                    else:
                        try:
                            raw = ocr.ocr(str(tmp_path), cls=use_angle_cls)
                        except TypeError:
                            raw = ocr.ocr(str(tmp_path))
                except Exception as e:
                    print(f"WARNING: OCR failed page={page_num} region={region_id} variant={variant_name} engine={active_engine_name}: {e}", file=sys.stderr)
                    continue
                scale = 2.0 if variant_name.endswith("up2") else 1.0
                for quad, text, score in normalize_ocr_result(raw):
                    conf = float(score)
                    if conf > 1.0:
                        conf = conf / 100.0
                    if conf < min_confidence:
                        continue
                    nt = normalize_text(text)
                    if not english_like(nt):
                        continue
                    bbox = quad_to_bbox_pdf(quad, offset, dpi=dpi, variant_scale=scale)
                    # Clip to region + small tolerance.
                    if bbox[2] < rect.x0 - 2 or bbox[0] > rect.x1 + 2 or bbox[3] < rect.y0 - 2 or bbox[1] > rect.y1 + 2:
                        continue
                    region_items.append(OCRItem(
                        page=page_num,
                        region_id=region_id,
                        text=str(text).strip(),
                        normalized_text=nt,
                        bbox=bbox,
                        confidence=round(conf, 3),
                        variant=variant_name,
                        detection_count=1,
                        english_like=True,
                        source=f"{active_engine_name}_v28_1_2",
                    ))

            merged = merge_detections(region_items)
            all_items.extend(merged)
            if debug_dir:
                annotate_debug(crop, merged, offset, dpi, str(Path(debug_dir) / f"p{page_num:02d}_{region_id}_debug.png"))
                debug_index[f"p{page_num}_{region_id}"] = {
                    "region_bbox": [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)],
                    "raw_candidates": len(region_items),
                    "merged_candidates": len(merged),
                    "debug_image": f"p{page_num:02d}_{region_id}_debug.png",
                }

    pdf.close()
    report = {
        "version": "v28_1_ocr_image_region_report",
        "input_pdf": pdf_path,
        "image_region_map": region_map,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": active_engine_name,
        "engine_requested": engine_name,
        "lang": lang,
        "dpi": dpi,
        "min_confidence": min_confidence,
        "pages_scanned": sorted(list(regions.keys())),
        "regions_total": sum(len(v) for v in regions.values()),
        "items_total": len(all_items),
        "items": [asdict(x) for x in all_items],
        "debug": debug_index,
        "note": "OCR was restricted to approved image/graphic regions. Boxes are in PDF point coordinates and are planner-compatible.",
    }
    write_json(output_json, report)
    return report


def parse_pages(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="V28.1 OCR scanner: approved image/graphic regions -> planner-compatible OCR JSON")
    ap.add_argument("pdf", help="PDF to OCR, usually source PDF or clean translated PDF with remaining image text")
    ap.add_argument("--image-region-map", required=True, help="Approved image/graphic region map JSON")
    ap.add_argument("--output-json", required=True, help="Output OCR report JSON")
    ap.add_argument("--dpi", type=int, default=320, help="Render DPI for OCR crops")
    ap.add_argument("--lang", default="en", help="PaddleOCR language, usually en for remaining English image text")
    ap.add_argument("--min-confidence", type=float, default=0.35)
    ap.add_argument("--pages", default=None, help="Optional page filter, e.g. 2,4-5")
    ap.add_argument("--debug-dir", default=None, help="Save crop/debug images")
    ap.add_argument("--engine", default="auto", choices=["auto", "rapidocr", "paddle"], help="OCR engine. auto prefers RapidOCR and falls back to PaddleOCR")
    ap.add_argument("--no-angle-cls", action="store_true", help="Disable PaddleOCR angle classifier")
    args = ap.parse_args(argv)

    report = scan_regions(
        pdf_path=args.pdf,
        region_map=args.image_region_map,
        output_json=args.output_json,
        dpi=args.dpi,
        lang=args.lang,
        min_confidence=args.min_confidence,
        pages=parse_pages(args.pages),
        debug_dir=args.debug_dir,
        use_angle_cls=not args.no_angle_cls,
        engine_name=args.engine,
    )
    print("V28.1 OCR Image-Region Scanner summary:")
    print(f"  pdf={args.pdf}")
    print(f"  image_region_map={args.image_region_map}")
    print(f"  output_json={args.output_json}")
    print(f"  dpi={args.dpi}")
    print(f"  engine={report.get('engine')}")
    print(f"  regions_total={report['regions_total']}")
    print(f"  items_total={report['items_total']}")
    if args.debug_dir:
        print(f"  debug_dir={args.debug_dir}")
    print("  note=No API call. Use this OCR report as --ocr-report for auto_patch_planner_v28.py")


if __name__ == "__main__":
    main()
