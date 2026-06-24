"""
V6 Patch - Weight-aware font detection.
Uses pdf_generator_single_anthropic_fixed.py as base.
"""

import os
import re
import copy
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import fitz

from pdf_generator_single_anthropic_fixed import (
    DocumentIR, PageIR, TextBlock, TextLine, TextSpan, LayoutResult,
    is_translatable_block, text_is_same,
    postprocess_translation_v9, expand_bbox, wrap_text_to_width,
    int_color_to_rgb, sanitize_text_v9, normalize_special_chars,
    rect_union, bbox_width, bbox_height, median, most_common,
    clean_font_name, safe_font_resource_name,
    redaction_rects_for_block_v9, add_no_fill_redactions, apply_no_fill_redactions,
    choose_layout_for_block, compute_layout_v9,
    get_symbol_font, is_small_box,
    extract_image_rects, estimate_alignment, classify_block_role,
    sort_blocks_reading_order, make_block_from_lines,
    split_line_into_text_islands, should_split_raw_text_block,
    normalize_text, rebuild_document_paragraphs, extract_embedded_fonts,
    export_ir_json, render_pdf_pages, load_json_map,
    translate_ir_v9, load_dotenv_file,
)
import pdf_generator_single_anthropic_fixed as base

# ============================================================
# Weight constants
# ============================================================

WEIGHT_THIN = 100
WEIGHT_LIGHT = 300
WEIGHT_REGULAR = 400
WEIGHT_MEDIUM = 500
WEIGHT_SEMIBOLD = 600
WEIGHT_BOLD = 700
WEIGHT_BLACK = 900


def detect_font_weight(font_name: str, flags: int) -> int:
    name = str(font_name).lower()
    if any(k in name for k in ["black", "heavy", "extrabold"]):
        return WEIGHT_BLACK
    if "bold" in name and "semi" not in name and "demi" not in name:
        return WEIGHT_BOLD
    if any(k in name for k in ["semibold", "demibold", "semi-bold", "demi-bold"]):
        return WEIGHT_SEMIBOLD
    if "medium" in name:
        return WEIGHT_MEDIUM
    if any(k in name for k in ["light", "thin"]):
        return WEIGHT_LIGHT
    if flags & 16:
        return WEIGHT_BOLD
    return WEIGHT_REGULAR


def detect_span_style_v6(span: dict) -> Tuple[bool, bool, int]:
    font = str(span.get("font", ""))
    flags = int(span.get("flags", 0) or 0)
    weight = detect_font_weight(font, flags)
    is_bold = weight >= WEIGHT_SEMIBOLD
    is_italic = bool(flags & 2) or any(k in font.lower() for k in ["italic", "oblique"])
    return is_bold, is_italic, weight


# ============================================================
# Parser
# ============================================================

def _dominant_weight_of_block(block: TextBlock) -> int:
    weights = []
    for line in block.lines:
        for span in line.spans:
            w = getattr(span, "weight", WEIGHT_REGULAR)
            char_count = len(span.text.strip())
            weights.extend([w] * max(1, char_count))
    if not weights:
        return WEIGHT_REGULAR
    return most_common(weights)


def parse_pdf_to_ir_v6(source_pdf: str) -> DocumentIR:
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

            visual_lines = []
            for raw_line in raw_block.get("lines", []):
                spans = []
                for raw_span in raw_line.get("spans", []):
                    chars = raw_span.get("chars", [])
                    if chars:
                        text = "".join(ch.get("c", "") for ch in chars)
                    else:
                        text = raw_span.get("text", "")
                    text = normalize_special_chars(text)
                    if not text or not text.strip():
                        continue

                    is_bold, is_italic, weight = detect_span_style_v6(raw_span)
                    bbox = tuple(float(v) for v in raw_span.get("bbox", (0, 0, 0, 0)))

                    span = TextSpan(
                        text=text,
                        bbox=bbox,
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
                line_bbox = tuple(
                    float(v) for v in raw_line.get("bbox", rect_union(s.bbox for s in spans))
                )
                line = TextLine(bbox=line_bbox, spans=spans)
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
                bbox=block_bbox,
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


# ============================================================
# FontResolverV6
# ============================================================

class FontResolverV6:
    """Standalone weight-aware font resolver."""

    def __init__(
        self,
        regular_font=None,
        bold_font=None,
        title_font=None,
        condensed_font=None,
        condensed_bold_font=None,
        condensed_semibold_font=None,
        medium_font=None,
        semibold_font=None,
        black_font=None,
        symbol_font=None,
        embedded_fonts=None,
        prefer_original_fonts=True,
    ):
        def resolve(p):
            if p and Path(p).exists():
                return str(Path(p))
            return None

        def win(name):
            p = Path("C:/Windows/Fonts") / name
            return str(p) if p.exists() else None

        self.regular_font = resolve(regular_font) or win("arial.ttf")
        self.bold_font = resolve(bold_font) or win("arialbd.ttf") or self.regular_font
        self.title_font = resolve(title_font) or self.bold_font
        self.condensed_regular_font = resolve(condensed_font) or self.regular_font
        self.condensed_bold_font = resolve(condensed_bold_font) or self.title_font
        self.condensed_semibold_font = resolve(condensed_semibold_font) or self.condensed_bold_font
        self.medium_font = resolve(medium_font) or self.regular_font
        self.semibold_font = resolve(semibold_font) or self.bold_font
        self.black_font = resolve(black_font) or self.bold_font
        self.symbol_font = resolve(symbol_font) or win("seguisym.ttf") or self.regular_font
        self.embedded_fonts = embedded_fonts or {}
        self.prefer_original_fonts = prefer_original_fonts
        self._vi_cache: Dict[str, bool] = {}
        self._font_cache: Dict[str, fitz.Font] = {}

        print("FontResolverV6:")
        print(f"  regular={self.regular_font}")
        print(f"  medium={self.medium_font}")
        print(f"  semibold={self.semibold_font}")
        print(f"  bold={self.bold_font}")
        print(f"  black={self.black_font}")
        print(f"  condensed={self.condensed_regular_font}")
        print(f"  condensed_semibold={self.condensed_semibold_font}")
        print(f"  condensed_bold={self.condensed_bold_font}")

    def _vi_ok(self, path: Optional[str]) -> bool:
        if not path:
            return False
        if path not in self._vi_cache:
            try:
                font = fitz.Font(fontfile=path)
                probe = "ăâđêôơưĂÂĐáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
                self._vi_cache[path] = all(font.has_glyph(ord(c)) for c in probe)
            except Exception:
                self._vi_cache[path] = False
        return self._vi_cache[path]

    def _weight_to_font(self, weight: int, condensed: bool = False) -> Optional[str]:
        if condensed:
            if weight >= WEIGHT_BOLD:
                return self.condensed_bold_font
            if weight >= WEIGHT_MEDIUM:
                return self.condensed_semibold_font or self.condensed_regular_font
            return self.condensed_regular_font
        if weight >= WEIGHT_BLACK:
            return self.black_font
        if weight >= WEIGHT_BOLD:
            return self.bold_font
        if weight >= WEIGHT_SEMIBOLD:
            return self.semibold_font
        if weight >= WEIGHT_MEDIUM:
            return self.medium_font
        return self.regular_font

    def fontfile_for(self, block, text=None) -> Optional[str]:
        weight = getattr(block, "dominant_weight", WEIGHT_REGULAR)
        name = clean_font_name(block.main_font).lower()
        condensed = "condensed" in name or "narrow" in name or block.role in {"label"}
        chosen = self._weight_to_font(weight, condensed=condensed)
        if self._vi_ok(chosen):
            return chosen
        for f in [self.regular_font, self.bold_font, self.title_font]:
            if self._vi_ok(f):
                return f
        return self.regular_font

    def fontname_for(self, block, text=None) -> str:
        weight = getattr(block, "dominant_weight", WEIGHT_REGULAR)
        name = clean_font_name(block.main_font).lower()
        condensed = "condensed" in name or "narrow" in name or block.role in {"label"}
        if condensed:
            if weight >= WEIGHT_BOLD:
                return "FCondensedBoldVN"
            if weight >= WEIGHT_MEDIUM:
                return "FCondensedSemiBoldVN"
            return "FCondensedRegularVN"
        if weight >= WEIGHT_BLACK:
            return "FBlackVN"
        if weight >= WEIGHT_BOLD:
            return "FBoldVN"
        if weight >= WEIGHT_SEMIBOLD:
            return "FSemiBoldVN"
        if weight >= WEIGHT_MEDIUM:
            return "FMediumVN"
        return "FRegularVN"

    def fitz_font_for(self, block, text=None) -> fitz.Font:
        fontfile = self.fontfile_for(block, text)
        key = self.fontname_for(block, text) + "|" + (fontfile or "helv")
        if key not in self._font_cache:
            self._font_cache[key] = (
                fitz.Font(fontfile=fontfile) if fontfile else fitz.Font("helv")
            )
        return self._font_cache[key]

    def font_size_scale_for(self, block, text=None) -> float:
        return 1.0


# ============================================================
# draw_layout_v6
# ============================================================

def draw_layout_v6(page, layout, block, resolver, color):
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

        full_for_measure = ("✓ " + draw_line) if leading_check else draw_line
        if layout.align == "center":
            w = measure_font.text_length(
                full_for_measure.replace("✓", "•"), fontsize=layout.fontsize
            )
            x = x0 + max(0, (layout.rect.width - w) / 2)
        elif layout.align == "right":
            w = measure_font.text_length(
                full_for_measure.replace("✓", "•"), fontsize=layout.fontsize
            )
            x = x1 - w
        else:
            x = x0

        if leading_check and symbol_file and symbol_font:
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


# ============================================================
# render_translated_pdf_v6
# ============================================================

def render_translated_pdf_v6(
    input_pdf,
    translated_ir,
    output_pdf,
    regular_font=None,
    bold_font=None,
    title_font=None,
    condensed_font=None,
    condensed_bold_font=None,
    condensed_semibold_font=None,
    medium_font=None,
    semibold_font=None,
    black_font=None,
    symbol_font=None,
    embedded_fonts=None,
    prefer_original_fonts=True,
    cover_text=True,
    sampled_background=True,
    translate_headers_footers=False,
    force_render=False,
):
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

    rendered = redacted = skip_role = skip_notrans = skip_same = skip_unfit = drawn_unfit = 0
    weight_dist: Dict[int, int] = {}

    for page_ir in translated_ir.pages:
        page = pdf[page_ir.page_index]
        draw_jobs = []
        redaction_rects = []

        for block in page_ir.blocks:
            if not is_translatable_block(block, translate_headers_footers):
                skip_role += 1
                continue
            if not (block.translated_text and block.translated_text.strip()):
                skip_notrans += 1
                continue
            if text_is_same(block.original_text, block.translated_text) and not force_render:
                skip_same += 1
                continue

            raw_text = postprocess_translation_v9(
                block.translated_text.strip(), block.original_text, block.role, "vi"
            )
            if not raw_text:
                skip_notrans += 1
                continue

            rect = fitz.Rect(*expand_bbox(block.bbox, 0.18, page_ir.width, page_ir.height))
            setattr(block, "_font_size_scale", 1.0)
            font = resolver.fitz_font_for(block, raw_text)
            layout, chosen_text, unfit = choose_layout_for_block(block, raw_text, rect, font)

            if layout is None or not chosen_text:
                skip_unfit += 1
                continue
            if unfit and not (force_unfit or force_render or (render_small_unfit and is_small_box(block))):
                skip_unfit += 1
                continue

            if chosen_text != raw_text:
                font = resolver.fitz_font_for(block, chosen_text)
                layout = compute_layout_v9(chosen_text, block, rect, font)

            w = getattr(block, "dominant_weight", 400)
            weight_dist[w] = weight_dist.get(w, 0) + 1

            if cover_text:
                redaction_rects.extend(redaction_rects_for_block_v9(block, page_ir))

            draw_jobs.append((layout, block, int_color_to_rgb(block.color)))
            if unfit:
                drawn_unfit += 1

        if cover_text and redaction_rects:
            add_no_fill_redactions(page, redaction_rects)
            apply_no_fill_redactions(page)
            redacted += len(redaction_rects)

        for layout, block, color in draw_jobs:
            draw_layout_v6(page, layout, block, resolver, color)
            rendered += 1

    pdf.save(output_pdf, garbage=4, deflate=True)
    pdf.close()

    print("Render summary v6:")
    print(f"  rendered={rendered}, redacted={redacted}")
    print(f"  weight_distribution={dict(sorted(weight_dist.items()))}")
    print(f"  drawn_unfit={drawn_unfit}")
    print(f"  skipped: role={skip_role}, same={skip_same}, notrans={skip_notrans}, unfit={skip_unfit}")


# ============================================================
# CLI
# ============================================================

def build_arg_parser_v6():
    import argparse
    p = argparse.ArgumentParser(description="V6 weight-aware PDF translator")
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
    return p


def main_v6(argv=None):
    args = build_arg_parser_v6().parse_args(argv)
    if not Path(args.input_pdf).exists():
        raise FileNotFoundError(args.input_pdf)

    load_dotenv_file(args.env_file)
    base._prepare_anthropic_env()
    base._print_provider_info_once()

    if args.translation_map:
        translator = base.JsonMapTranslator(args.translation_map)
        print("Translator: JSON map")
    elif not args.no_llm and os.getenv("LLM_API_KEY") and os.getenv("LLM_MODEL"):
        translator = base.OpenAICompatibleTranslator(
            temperature=args.llm_temperature,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
        )
    else:
        translator = base.DummyTranslator()
        print("WARNING: DummyTranslator.")

    glossary = load_json_map(args.glossary)

    print(f"[1/5] Parse PDF -> IR: {args.input_pdf}")
    ir = parse_pdf_to_ir_v6(args.input_pdf)

    print("[2/5] Rebuild paragraph blocks")
    ir = rebuild_document_paragraphs(ir)

    embedded_fonts = {}
    if not args.no_extract_fonts:
        print(f"      Extract embedded fonts -> {args.embedded_font_dir}")
        embedded_fonts = extract_embedded_fonts(args.input_pdf, args.embedded_font_dir)
        print(f"      Extracted font records: {len(embedded_fonts)}")

    if args.export_ir:
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
        tpath = str(
            Path(args.export_ir).with_name(
                Path(args.export_ir).stem + "_translated.json"
            )
        )
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

    if args.preview_dir:
        print(f"[5/5] Preview PNGs: {args.preview_dir}")
        render_pdf_pages(args.output_pdf, args.preview_dir)

    print(f"Done: {args.output_pdf}")


if __name__ == "__main__":
    main_v6()
