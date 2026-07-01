"""
Precise text detection for Track B.

Vision-LLM (Claude) is great at reading and translating text but its bounding
boxes are only approximate — not good enough to *erase* the original text
cleanly. RapidOCR (ONNX) gives pixel-tight boxes for every text line, which we
use to build the erase mask and to place the translated text exactly where the
original was. This is what makes the translation *replace* the baked-in text
instead of being overlaid on top of it.

RapidOCR is loaded lazily and cached (first call initializes the ONNX models).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def detect_text_boxes(image: Image.Image) -> list[dict]:
    """
    Detect every text line in `image`.
    Returns a list of {"bbox": [x0, y0, x1, y1] (pixels), "text": str, "conf": float}.
    """
    arr = np.array(image.convert("RGB"))
    result, _ = _engine()(arr)
    boxes: list[dict] = []
    for poly, text, conf in result or []:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        boxes.append(
            {
                "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                "text": text,
                "conf": float(conf),
            }
        )
    return boxes


def merge_lines_to_blocks(
    boxes: list[dict],
    gap_ratio: float = 0.6,
    x_tol_ratio: float = 1.5,
    color_tol: float = 70.0,
) -> list[dict]:
    """
    Merge detector line-boxes that belong to the same paragraph into one block,
    so a sentence RapidOCR split across several lines is translated as a whole
    (coherent translation) instead of as disconnected fragments.

    Heuristic (works for single-column / stacked-card layouts): a line joins the
    previous block if it sits directly below it, with a similar left edge,
    similar line height, and horizontal overlap. Headings (different height or a
    larger gap) start a new block.

    Returns blocks: {"bbox":[x0,y0,x1,y1], "text": joined, "conf", "line_h"}.
    """
    def _color_close(a, b, tol):
        if a is None or b is None:
            return True  # no colour info -> don't block the merge
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5 <= tol

    if not boxes:
        return []
    bs = sorted(boxes, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    blocks: list[dict] = []

    for b in bs:
        x0, y0, x1, y1 = b["bbox"]
        h = y1 - y0
        if blocks:
            g = blocks[-1]
            lx0, ly0, lx1, ly1 = g["_last"]
            lh = ly1 - ly0
            vgap = y0 - ly1
            same_left = abs(x0 - lx0) <= x_tol_ratio * lh
            same_h = 0.6 <= (h / lh if lh else 1) <= 1.6
            overlap = min(x1, lx1) - max(x0, lx0) > 0.3 * min(x1 - x0, lx1 - lx0)
            same_color = _color_close(b.get("_rgb"), g.get("_rgb"), color_tol)
            if (-0.3 * lh <= vgap <= gap_ratio * lh and same_left and same_h
                    and overlap and same_color):
                g["texts"].append(b["text"])
                g["bbox"] = [min(g["bbox"][0], x0), min(g["bbox"][1], y0),
                             max(g["bbox"][2], x1), max(g["bbox"][3], y1)]
                g["_last"] = b["bbox"]
                continue
        blocks.append({"texts": [b["text"]], "bbox": [x0, y0, x1, y1],
                       "line_h": h, "_last": b["bbox"], "conf": b.get("conf", 1.0),
                       "_rgb": b.get("_rgb")})

    out = []
    for g in blocks:
        out.append({
            "bbox": g["bbox"],
            "text": " ".join(g["texts"]),
            "conf": g["conf"],
            "line_h": g["line_h"],
        })
    return out


def _iou_x_overlap(a: list, b: list) -> float:
    """Fraction of box `b`'s area that lies inside box `a` (containment of b in a)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    barea = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return inter / barea


def assign_precise_boxes(
    blocks: list[dict], det_boxes: list[dict], img_w: int, img_h: int
) -> list[dict]:
    """
    For each Claude block, find the detector boxes it covers and replace the
    block's approximate `bbox_norm` with the tight union of those detector
    boxes. Blocks with no detector match keep their approximate box (flagged
    via `_matched=False`) so callers can decide whether to trust them.

    Mutates and returns `blocks`.
    """
    for block in blocks:
        bx0, by0, bx1, by1 = block["bbox_norm"]
        approx = [bx0 * img_w, by0 * img_h, bx1 * img_w, by1 * img_h]
        matched = [d for d in det_boxes if _iou_x_overlap(approx, d["bbox"]) >= 0.5]
        if not matched:
            block["_matched"] = False
            continue
        xs0 = min(d["bbox"][0] for d in matched)
        ys0 = min(d["bbox"][1] for d in matched)
        xs1 = max(d["bbox"][2] for d in matched)
        ys1 = max(d["bbox"][3] for d in matched)
        block["bbox_norm"] = [xs0 / img_w, ys0 / img_h, xs1 / img_w, ys1 / img_h]
        block["_matched"] = True
    return blocks
