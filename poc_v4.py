"""
POC v4: PDF Translation with Quality Improvements
- Better white rectangle coverage (no original text bleeding through)
- All OCR lines covered (no missing translations)
- Brand name protection (FLORIDACOMMERCE, POWERED BY, etc.)
- Accurate font sizing with binary search
- Smarter paragraph grouping
- Two-pass rendering: cover all → translate paragraphs
"""
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

import pypdfium2 as pdfium
from paddleocr import PaddleOCR
from PIL import Image
from openai import OpenAI
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import white, black
import numpy as np

# ===== Config =====
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

INPUT_PDF = "test_vietnamese.pdf"
OUTPUT_JSON = "output_v4.json"
OUTPUT_PDF = "output_v4_translated.pdf"

# Fonts
FONT_REGULAR = "fonts/NotoSans-Regular.ttf"
FONT_BOLD = "fonts/NotoSans-Bold.ttf"
if not Path(FONT_REGULAR).exists():
    FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"
    FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"

pdfmetrics.registerFont(TTFont('VN', FONT_REGULAR))
pdfmetrics.registerFont(TTFont('VN-Bold', FONT_BOLD))

# Brand names — KHÔNG dịch (case-insensitive)
BRAND_WHITELIST = {
    'floridacommerce', 'powered by', 'powered', 'florida commerce',
    # Add more brands tại đây khi cần
}

# Skip patterns (regex) — không dịch
SKIP_PATTERNS = [
    r'^https?://',           # URLs
    r'^[\w\.-]+@[\w\.-]+',   # Emails
    r'^\$?\d[\d,\.]*$',      # Numbers / money
    r'^[A-Z]{2,}$',          # All caps acronyms (<=3 chars)
]

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

print("Loading PaddleOCR...")
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)
except Exception:
    ocr = PaddleOCR(use_angle_cls=True, lang='latin', show_log=False)
print("PaddleOCR loaded\n")


# ============================================================
# OCR
# ============================================================

def pdf_to_images(pdf_path):
    pdf = pdfium.PdfDocument(pdf_path)
    return [page.render(scale=2).to_pil() for page in pdf], pdf


def detect_lines(image):
    img_array = np.array(image)
    result = ocr.ocr(img_array, cls=True)
    lines = []
    if result and result[0]:
        for line in result[0]:
            polygon = line[0]
            text, confidence = line[1]
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            lines.append({
                'box': [x1, y1, x2, y2],
                'text': text.strip(),
                'confidence': confidence,
                'height': y2 - y1,
                'center_y': (y1 + y2) / 2,
                'left_x': x1,
            })
    lines.sort(key=lambda b: (b['center_y'], b['left_x']))
    return lines


# ============================================================
# BRAND PROTECTION
# ============================================================

def should_skip_translation(text):
    """Check if text is a brand/skip pattern that shouldn't be translated."""
    text_lower = text.lower().strip()

    # Check brand whitelist
    if text_lower in BRAND_WHITELIST:
        return True
    for brand in BRAND_WHITELIST:
        if brand in text_lower and len(text_lower) <= len(brand) + 5:
            return True

    # Check skip patterns
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, text):
            return True

    return False


# ============================================================
# PARAGRAPH GROUPING (improved)
# ============================================================

def group_into_paragraphs(lines, vertical_gap_factor=1.3,
                          horizontal_align_threshold=40,
                          height_diff_ratio=0.3):
    """
    More lenient paragraph grouping:
    - vertical_gap_factor 2.0 (was 1.5) — allow bigger gaps
    - horizontal_align 80 (was 50) — allow indented continuations
    - height_diff 0.5 (was 0.4) — better title separation
    """
    if not lines:
        return []

    paragraphs = []
    current_para = [lines[0]]

    for i in range(1, len(lines)):
        prev = current_para[-1]
        curr = lines[i]

        # Vertical gap
        vertical_gap = curr['box'][1] - prev['box'][3]
        avg_height = (prev['height'] + curr['height']) / 2
        gap_ok = (vertical_gap < vertical_gap_factor * avg_height
                  and vertical_gap >= -avg_height)

        # Horizontal alignment (with some tolerance for indents)
        horizontal_diff = abs(prev['left_x'] - curr['left_x'])
        align_ok = horizontal_diff < horizontal_align_threshold

        # Similar height
        h1, h2 = prev['height'], curr['height']
        height_ratio = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 0
        height_ok = height_ratio > (1 - height_diff_ratio)

        if gap_ok and align_ok and height_ok:
            current_para.append(curr)
        else:
            paragraphs.append(_build_paragraph(current_para))
            current_para = [curr]

    if current_para:
        paragraphs.append(_build_paragraph(current_para))

    return paragraphs


def _build_paragraph(lines):
    text = ' '.join(line['text'] for line in lines)
    x1 = min(line['box'][0] for line in lines)
    y1 = min(line['box'][1] for line in lines)
    x2 = max(line['box'][2] for line in lines)
    y2 = max(line['box'][3] for line in lines)
    avg_height = sum(line['height'] for line in lines) / len(lines)
    return {
        'lines': lines,
        'text': text,
        'box': [x1, y1, x2, y2],
        'num_lines': len(lines),
        'avg_line_height': avg_height,
    }


# ============================================================
# TRANSLATION
# ============================================================

STRICT_PROMPT = """You are a professional English-to-Vietnamese translator.

RULES:
1. Translate to natural Vietnamese with proper diacritics
2. Output ONLY translation - NO explanations, NO comments
3. Preserve: numbers, $, %, URLs, emails, brand names, code
4. Keep technical terms in English if no Vietnamese equivalent
5. Brand names like "FloridaCommerce", "POWERED BY" stay as-is
6. Translate concisely — Vietnamese should not be much longer than English

INPUT: JSON array of texts
OUTPUT: JSON array of translations, SAME ORDER, SAME LENGTH

Example:
Input: ["Hello world", "FloridaCommerce", "Click here"]
Output: ["Xin chào thế giới", "FloridaCommerce", "Nhấn vào đây"]
"""


def translate_batch(paragraphs, batch_size=8):
    """Translate paragraphs, skipping brands."""
    results = []

    # Separate: brands (skip) vs translatable
    to_translate_indices = []
    to_translate_texts = []

    for i, para in enumerate(paragraphs):
        if should_skip_translation(para['text']):
            results.append(para['text'])  # keep original
            print(f"   P{i+1}: [BRAND/SKIP] {para['text'][:50]}")
        else:
            results.append(None)  # placeholder
            to_translate_indices.append(i)
            to_translate_texts.append(para['text'])

    # Batch translate
    print(f"   Translating {len(to_translate_texts)} paragraphs "
          f"({len(paragraphs) - len(to_translate_texts)} skipped)")

    translations = []
    for batch_start in range(0, len(to_translate_texts), batch_size):
        batch = to_translate_texts[batch_start:batch_start + batch_size]
        try:
            batch_result = _call_llm(batch)
            if len(batch_result) != len(batch):
                raise ValueError(f"Length mismatch: {len(batch_result)} vs {len(batch)}")
            translations.extend(batch_result)
        except Exception as e:
            print(f"Batch failed ({e}), trying individual...")
            for text in batch:
                try:
                    single = _call_llm([text])
                    translations.append(single[0] if single else text)
                except Exception:
                    translations.append(f"[FAIL] {text}")

    # Fill placeholders
    for idx, translation in zip(to_translate_indices, translations):
        results[idx] = translation

    return results


def _call_llm(texts):
    user_msg = json.dumps(texts, ensure_ascii=False)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": STRICT_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    raw = response.choices[0].message.content.strip()

    # Clean markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(l for l in lines if not l.strip().startswith("```"))

    return json.loads(raw)


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def image_to_pdf_box(image_box, image_size, pdf_size, padding_ratio=0.0):
    """Convert image bbox to PDF bbox with optional padding."""
    img_w, img_h = image_size
    pdf_w, pdf_h = pdf_size
    sx = pdf_w / img_w
    sy = pdf_h / img_h

    x1, y1, x2, y2 = image_box
    w = x2 - x1
    h = y2 - y1

    # Add padding (expand bbox)
    pad_x = w * padding_ratio
    pad_y = h * padding_ratio
    x1 -= pad_x
    x2 += pad_x
    y1 -= pad_y
    y2 += pad_y

    pdf_x1 = x1 * sx
    pdf_x2 = x2 * sx
    pdf_y1 = pdf_h - (y2 * sy)
    pdf_y2 = pdf_h - (y1 * sy)
    return [pdf_x1, pdf_y1, pdf_x2, pdf_y2]


# ============================================================
# TEXT RENDERING (with binary search font sizing)
# ============================================================

def find_best_font_size(text, font_name, box_w, box_h, c,
                        min_size=6, max_size=24):
    """Binary search optimal font size."""
    best = min_size
    lo, hi = min_size, max_size

    while lo <= hi:
        mid = (lo + hi) // 2
        line_h = mid * 1.25
        max_lines = max(1, int(box_h / line_h))

        # Wrap to find lines needed
        lines = wrap_text(text, c, font_name, mid, box_w)

        if len(lines) <= max_lines:
            best = mid
            lo = mid + 1  # try bigger
        else:
            hi = mid - 1

    return best


def wrap_text(text, c, font_name, font_size, max_width):
    words = text.split()
    if not words:
        return []
    lines = []
    current = []
    for word in words:
        test = ' '.join(current + [word])
        if c.stringWidth(test, font_name, font_size) <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(' '.join(current))
            current = [word]
    if current:
        lines.append(' '.join(current))
    return lines


def render_text_in_box(c, text, pdf_box, font_name='VN'):
    """Render text fitting in box, no overflow."""
    x1, y1, x2, y2 = pdf_box
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return

    font_size = find_best_font_size(text, font_name, box_w, box_h, c)
    lines = wrap_text(text, c, font_name, font_size, box_w)

    if not lines:
        return

    c.setFillColor(black)
    c.setFont(font_name, font_size)

    line_h = font_size * 1.25
    y_cursor = y2 - font_size  # start from top

    for line in lines:
        if y_cursor < y1:
            break
        c.drawString(x1, y_cursor, line)
        y_cursor -= line_h


def cover_with_white(c, pdf_box, padding=2):
    """Draw white rectangle to cover original text."""
    x1, y1, x2, y2 = pdf_box
    c.setFillColor(white)
    c.rect(x1 - padding, y1 - padding,
           (x2 - x1) + 2 * padding,
           (y2 - y1) + 2 * padding,
           fill=1, stroke=0)


# ============================================================
# PAGE PROCESSING
# ============================================================

def is_title(para, image_height):
    """Detect title: bigger height ratio + few lines."""
    height_ratio = para['avg_line_height'] / image_height
    return height_ratio > 0.025 and para['num_lines'] <= 2


def process_page(image, page_idx, total_pages):
    """OCR + paragraph grouping + translation for one page."""
    print(f"\n{'='*60}")
    print(f"PAGE {page_idx + 1}/{total_pages}")
    print(f"{'='*60}")

    # Step 1: OCR
    print("Step 1: OCR detection...")
    lines = detect_lines(image)
    print(f"   Found {len(lines)} lines")

    # Step 2: Group paragraphs
    print("Step 2: Group paragraphs...")
    paragraphs = group_into_paragraphs(lines)
    print(f"   {len(lines)} lines → {len(paragraphs)} paragraphs")
    for i, p in enumerate(paragraphs):
        preview = p['text'][:60].replace('\n', ' ')
        print(f"   P{i+1} ({p['num_lines']}L, h={p['avg_line_height']:.0f}): {preview}")

    # Step 3: Translate
    print("Step 3: Translate...")
    translations = translate_batch(paragraphs)

    # Step 4: Pair up
    for para, trans in zip(paragraphs, translations):
        para['translated'] = trans

    return {
        'page': page_idx + 1,
        'image_size': image.size,
        'all_lines': lines,            # for white-cover coverage
        'paragraphs': paragraphs,
    }


# ============================================================
# PDF RECONSTRUCTION (two-pass)
# ============================================================

def reconstruct_pdf(pages_data, src_pdf, output_path):
    """Two-pass rendering:
    Pass 1: Cover ALL OCR lines with white (no English bleeding through)
    Pass 2: Render translated paragraphs
    """
    first_page = src_pdf[0]
    page_w = first_page.get_width()
    page_h = first_page.get_height()

    c = canvas.Canvas(output_path, pagesize=(page_w, page_h))

    for page_data in pages_data:
        page_idx = page_data['page'] - 1
        src_page = src_pdf[page_idx]
        pw = src_page.get_width()
        ph = src_page.get_height()
        c.setPageSize((pw, ph))

        # Render background image
        page_image = src_page.render(scale=2).to_pil()
        bg_path = f"_temp_bg_{page_data['page']}.png"
        page_image.save(bg_path)
        c.drawImage(bg_path, 0, 0, width=pw, height=ph)

        image_size = tuple(page_data['image_size'])

        # ===== PASS 1: Cover ALL lines with white =====
        # This ensures no English text bleeds through
        for line in page_data['all_lines']:
            # Bigger padding for cover (15%)
            pdf_box = image_to_pdf_box(
                line['box'], image_size, (pw, ph), padding_ratio=0.15
            )
            cover_with_white(c, pdf_box, padding=1)

       # ===== PASS 1.5: Track translated line boxes =====
        # Build a set of all line boxes that ARE inside a translated paragraph
        translated_line_boxes = set()
        for para in page_data['paragraphs']:
            if para.get('translated') and not para['translated'].startswith('[FAIL]'):
                for line in para.get('lines', []):
                    box_key = tuple(line['box'])
                    translated_line_boxes.add(box_key)

        # ===== PASS 1.6: Translate ORPHAN lines individually =====
        # Lines that were detected by OCR but NOT in any translated paragraph
        orphan_lines = []
        for line in page_data['all_lines']:
            box_key = tuple(line['box'])
            if box_key not in translated_line_boxes:
                if not should_skip_translation(line['text']):
                    orphan_lines.append(line)

        if orphan_lines:
            print(f"   Found {len(orphan_lines)} orphan lines, translating...")
            orphan_texts = [l['text'] for l in orphan_lines]
            try:
                orphan_translations = _call_llm(orphan_texts)
                for line, trans in zip(orphan_lines, orphan_translations):
                    line['translated'] = trans
            except Exception as e:
                print(f"Orphan translation failed: {e}")
                for line in orphan_lines:
                    line['translated'] = line['text']  # fallback

            # Render orphan lines
            for line in orphan_lines:
                if 'translated' not in line:
                    continue
                pdf_box = image_to_pdf_box(
                    line['box'], image_size, (pw, ph), padding_ratio=0.0
                )
                render_text_in_box(c, line['translated'], pdf_box, font_name='VN')

        # ===== PASS 2: Render translations =====
        for para in page_data['paragraphs']:
            translation = para.get('translated', '')

        # Cleanup
        Path(bg_path).unlink(missing_ok=True)
        c.showPage()
        print(f"Page {page_data['page']} rendered")

    c.save()
    print(f"\nPDF saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    if not Path(INPUT_PDF).exists():
        print(f"{INPUT_PDF} not found")
        return

    # OCR + Translate all pages
    images, src_pdf = pdf_to_images(INPUT_PDF)
    print(f"PDF: {len(images)} pages, size: {images[0].size}")

    pages_data = []
    for i, image in enumerate(images):
        page_data = process_page(image, i, len(images))
        pages_data.append(page_data)

    # Save metadata
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json_data = {
            'source': INPUT_PDF,
            'pages': [
                {
                    'page': p['page'],
                    'image_size': p['image_size'],
                    'paragraphs': [
                        {
                            'box': para['box'],
                            'original': para['text'],
                            'translated': para.get('translated', ''),
                            'num_lines': para['num_lines'],
                        }
                        for para in p['paragraphs']
                    ]
                }
                for p in pages_data
            ]
        }
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Metadata: {OUTPUT_JSON}")

    # Reconstruct PDF
    print(f"\n{'='*60}")
    print("RECONSTRUCTING PDF")
    print(f"{'='*60}")
    reconstruct_pdf(pages_data, src_pdf, OUTPUT_PDF)
    src_pdf.close()


if __name__ == "__main__":
    main()