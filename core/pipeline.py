"""
End-to-end orchestrator.

Opens a PDF once, classifies each page with the router, and dispatches:
  TEXT_LAYER  -> Track A  (extract -> translate -> write in place)
  IMAGE_BASED -> Track B  (render -> OCR+translate -> inpaint -> re-render -> recompose)

Then saves a single translated PDF.

Design notes
------------
* `translate_fn` and `ocr_fn` are injectable. Defaults call the real Claude
  API; tests and quality runs can pass deterministic/offline stand-ins so the
  whole pipeline can be exercised without network access or an API key.
* Track B pages that OCR returns nothing for are left untouched rather than
  blanked, so a bad OCR call never destroys a page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import fitz

from core.router import classify_page, PageType
from core.track_a.extractor import extract_spans
from core.track_a.writer import write_page
from core.track_b.pageio import render_page_image, apply_image_to_page
from core.track_b.inpainter import inpaint_blocks
from core.track_b.renderer import render_blocks
from core.track_b.detector import assign_precise_boxes

# translate_fn(spans, target_lang) -> spans (each gets a 'translation' field)
TranslateFn = Callable[[list[dict], str], list[dict]]
# ocr_fn(image, target_lang, page_index) -> list of OCR block dicts
OcrFn = Callable[..., list[dict]]
# detect_fn(image) -> list of {"bbox":[x0,y0,x1,y1] px, "text", "conf"}
DetectFn = Callable[..., list[dict]]


@dataclass
class PageResult:
    page_index: int
    track: str            # "A" | "B" | "skip"
    spans: int = 0        # Track A: spans translated
    blocks: int = 0       # Track B: OCR blocks rendered
    note: str = ""


@dataclass
class PipelineResult:
    input_path: str
    output_path: str
    pages: list[PageResult] = field(default_factory=list)

    def summary(self) -> str:
        a = sum(1 for p in self.pages if p.track == "A")
        b = sum(1 for p in self.pages if p.track == "B")
        skip = sum(1 for p in self.pages if p.track == "skip")
        return f"{len(self.pages)} pages -> Track A: {a}, Track B: {b}, skipped: {skip}"


def _default_translate_fn(spans: list[dict], target_lang: str) -> list[dict]:
    from core.track_a.translator import translate_spans
    return translate_spans(spans, target_lang=target_lang)


def _default_ocr_fn(image, target_lang: str, page_index: int) -> list[dict]:
    from core.track_b.ocr import ocr_image
    return ocr_image(image, target_lang=target_lang, page_index=page_index)


def _default_detect_fn(image) -> list[dict]:
    from core.track_b.detector import detect_text_boxes
    return detect_text_boxes(image)


def translate_pdf(
    input_path: str,
    output_path: str,
    target_lang: str = "Vietnamese",
    enable_track_b: bool = True,
    translate_fn: Optional[TranslateFn] = None,
    ocr_fn: Optional[OcrFn] = None,
    detect_fn: Optional[DetectFn] = None,
    inpaint_strategy: str = "telea",
    dpi: int = 200,
    min_char_count: int = 20,
    strip_source: bool = False,
    on_progress: Optional[Callable[[PageResult], None]] = None,
) -> PipelineResult:
    """
    Translate a whole PDF and write the result to `output_path`.
    Returns a PipelineResult describing what happened per page.
    """
    from core.config import get_provider

    translate_fn = translate_fn or _default_translate_fn
    # Local text providers (Ollama) have no vision model: drive Track B from
    # RapidOCR + text translation. Its boxes are already precise, so no separate
    # detect pass is needed.
    local = get_provider() in ("ollama", "openai")
    if ocr_fn is None:
        if local:
            from core.track_b.local_ocr import ocr_translate_local
            ocr_fn = ocr_translate_local
        else:
            ocr_fn = _default_ocr_fn
    if detect_fn is None and not local:
        detect_fn = _default_detect_fn

    doc = fitz.open(input_path)
    result = PipelineResult(input_path=input_path, output_path=output_path)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        ptype = classify_page(page, min_char_count=min_char_count)

        if ptype == PageType.TEXT_LAYER:
            pr = _run_track_a(doc, page_idx, target_lang, translate_fn, strip_source)
        elif enable_track_b:
            pr = _run_track_b(
                doc, page_idx, target_lang, ocr_fn, detect_fn, inpaint_strategy, dpi
            )
        else:
            pr = PageResult(page_idx, "skip", note="image page, Track B disabled")

        result.pages.append(pr)
        if on_progress:
            on_progress(pr)

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    return result


def _run_track_a(
    doc: "fitz.Document",
    page_idx: int,
    target_lang: str,
    translate_fn: TranslateFn,
    strip_source: bool = False,
) -> PageResult:
    spans = extract_spans(doc.name, page_indices=[page_idx])
    if not spans:
        return PageResult(page_idx, "skip", note="no extractable spans")
    spans = translate_fn(spans, target_lang)
    write_page(doc[page_idx], spans, strip_source=strip_source)
    return PageResult(page_idx, "A", spans=len(spans))


def _run_track_b(
    doc: "fitz.Document",
    page_idx: int,
    target_lang: str,
    ocr_fn: OcrFn,
    detect_fn: DetectFn,
    inpaint_strategy: str,
    dpi: int,
) -> PageResult:
    page = doc[page_idx]
    image = render_page_image(page, dpi=dpi)

    blocks = ocr_fn(image, target_lang, page_idx)
    if not blocks:
        return PageResult(page_idx, "skip", note="OCR returned no blocks")

    # Only touch text we are actually replacing: a real translation that differs
    # from the source. Everything else (logos, brand names Claude left as-is) is
    # NOT erased, so it stays intact.
    targets = [
        b for b in blocks
        if (b.get("translation") or "").strip()
        and b.get("translation") != b.get("text")
    ]
    if not targets:
        return PageResult(page_idx, "skip", note="nothing to translate")

    # Replace Claude's approximate boxes with pixel-tight detector boxes so the
    # original text is erased exactly and completely (not overlaid).
    if detect_fn is not None:
        det = detect_fn(image)
        assign_precise_boxes(targets, det, image.width, image.height)

    cleaned = inpaint_blocks(image, targets, strategy=inpaint_strategy)
    rendered = render_blocks(cleaned, targets, use_translation=True)
    apply_image_to_page(page, rendered)
    return PageResult(page_idx, "B", blocks=len(targets))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Translate a PDF end-to-end.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--lang", default="Vietnamese")
    ap.add_argument("--no-track-b", action="store_true", help="skip image pages")
    ap.add_argument("--inpaint", default="telea", choices=["solid", "telea", "lama"])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--strip-source", action="store_true",
                    help="delete underlying original text (page-level PDFs)")
    args = ap.parse_args()

    res = translate_pdf(
        args.input,
        args.output,
        target_lang=args.lang,
        enable_track_b=not args.no_track_b,
        inpaint_strategy=args.inpaint,
        dpi=args.dpi,
        strip_source=args.strip_source,
        on_progress=lambda pr: print(
            f"  page {pr.page_index}: Track {pr.track} "
            f"({pr.spans or pr.blocks} items) {pr.note}"
        ),
    )
    print(res.summary())
    print("saved ->", res.output_path)
