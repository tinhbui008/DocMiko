"""
Phase 3: Generate PDF from scratch using extracted metadata + translations.

Input:  translated_pdf.json
Output: final_translated.pdf (pixel-close to original)
"""
import json
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, white

INPUT_JSON = "translated_pdf.json"
OUTPUT_PDF = "final_translated.pdf"


# ============================================================
# FONT MAPPING — Map PDF source fonts → Vietnamese-capable fonts
# ============================================================

# Vietnamese-capable fonts (must support diacritics)
FONT_FILES = {
    'regular': 'fonts/NotoSans-Regular.ttf',
    'bold': 'fonts/NotoSans-Bold.ttf',
}

# Fallback to Windows fonts
if not Path(FONT_FILES['regular']).exists():
    FONT_FILES = {
        'regular': 'C:/Windows/Fonts/arial.ttf',
        'bold': 'C:/Windows/Fonts/arialbd.ttf',
    }

# Register fonts
pdfmetrics.registerFont(TTFont('VN-Regular', FONT_FILES['regular']))
pdfmetrics.registerFont(TTFont('VN-Bold', FONT_FILES['bold']))


def map_font(fontname: str, is_bold: bool) -> str:
    """
    Map original PDF font to Vietnamese-capable equivalent.

    Strategy:
    - Detect bold/regular from fontname or is_bold flag
    - Return our registered VN font
    """
    fontname_lower = fontname.lower()

    # Check fontname for bold/black indicators
    is_bold_font = (
        is_bold
        or 'bold' in fontname_lower
        or 'black' in fontname_lower
        or 'semibold' in fontname_lower  # Semibold mapped to Bold for visibility
    )

    return 'VN-Bold' if is_bold_font else 'VN-Regular'


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def pdfplumber_to_reportlab(bbox, page_height):
    """
    pdfplumber: origin TOP-LEFT, Y increases DOWN
    reportlab: origin BOTTOM-LEFT, Y increases UP

    Convert bbox from pdfplumber → reportlab.
    """
    x0, top, x1, bottom = bbox
    rl_y_bottom = page_height - bottom
    rl_y_top = page_height - top
    return [x0, rl_y_bottom, x1, rl_y_top]


# ============================================================
# TEXT FITTING (binary search for best font size)
# ============================================================

def wrap_text(text, c, font_name, font_size, max_width):
    """Wrap text into lines that fit max_width."""
    words = text.split()
    if not words:
        return []

    lines = []
    current = []

    for word in words:
        test = ' '.join(current + [word])
        width = c.stringWidth(test, font_name, font_size)

        if width <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(' '.join(current))
                current = [word]
            else:
                # Single word too long — force include
                lines.append(word)
                current = []

    if current:
        lines.append(' '.join(current))

    return lines


def find_optimal_font_size(text, original_size, font_name, box_w, box_h, c,
                           min_ratio=0.5):
    """
    Find largest font size that fits text in box.
    Starts from original size, shrinks if needed.
    """
    min_size = max(6, int(original_size * min_ratio))
    line_spacing = 1.15  # less than 1.2 for tight fit

    # Try original size first
    for size in range(int(original_size), min_size - 1, -1):
        line_h = size * line_spacing
        max_lines = max(1, int(box_h / line_h))

        lines = wrap_text(text, c, font_name, size, box_w)

        if len(lines) <= max_lines:
            return size, lines

    # Force smallest size
    lines = wrap_text(text, c, font_name, min_size, box_w)
    return min_size, lines


# ============================================================
# RENDERING
# ============================================================

def render_paragraph(c, para, page_height):
    """Render a translated paragraph with exact style."""
    translation = para.get('translated', para['text'])

    # Map font
    font_name = map_font(para['fontname'], para['is_bold'])
    original_size = para['size']
    color = HexColor(para['color'])

    # Convert bbox to reportlab coords
    bbox = pdfplumber_to_reportlab(para['bbox'], page_height)
    x0, y_bottom, x1, y_top = bbox
    box_w = x1 - x0
    box_h = y_top - y_bottom

    if box_w <= 0 or box_h <= 0:
        return

    # Find optimal font size
    font_size, lines = find_optimal_font_size(
        translation, original_size, font_name, box_w, box_h, c
    )

    # Set style
    c.setFillColor(color)
    c.setFont(font_name, font_size)

    # Render lines, top-down
    line_spacing = 1.15
    y_cursor = y_top - font_size  # baseline of first line

    for line in lines:
        if y_cursor < y_bottom - font_size:  # would overflow below box
            break
        c.drawString(x0, y_cursor, line)
        y_cursor -= font_size * line_spacing


def render_image(c, img_data, page_height):
    """Place extracted image at original position."""
    img_path = img_data['path']

    if not Path(img_path).exists():
        print(f"   ⚠️ Image not found: {img_path}")
        return

    # Convert bbox to reportlab
    bbox = pdfplumber_to_reportlab(img_data['bbox'], page_height)
    x0, y_bottom, x1, y_top = bbox

    width = x1 - x0
    height = y_top - y_bottom

    try:
        c.drawImage(img_path, x0, y_bottom, width=width, height=height,
                    preserveAspectRatio=False, mask='auto')
    except Exception as e:
        print(f"   ⚠️ Failed to draw image {img_path}: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    if not Path(INPUT_JSON).exists():
        print(f"❌ Not found: {INPUT_JSON}")
        return

    print(f"📥 Loading: {INPUT_JSON}")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        doc = json.load(f)

    print(f"📊 Document: {len(doc['pages'])} pages\n")

    # Setup canvas with first page size
    first_page = doc['pages'][0]
    page_w, page_h = first_page['size']

    c = canvas.Canvas(OUTPUT_PDF, pagesize=(page_w, page_h))

    for page_data in doc['pages']:
        page_num = page_data['page']
        pw, ph = page_data['size']
        c.setPageSize((pw, ph))

        print(f"--- Rendering Page {page_num} ---")

        # === Step 1: Draw images FIRST (background layer) ===
        # Sort images by size (large first → background)
        images = sorted(
            page_data.get('images', []),
            key=lambda img: -(img['width'] * img['height'])
        )
        for img in images:
            print(f"   Image: {Path(img['path']).name}")
            render_image(c, img, ph)

        # === Step 2: Draw text on top ===
        for i, para in enumerate(page_data['paragraphs']):
            translation = para.get('translated', '')
            if not translation:
                continue
            preview = translation[:50].replace('\n', ' ')
            print(f"   P{i+1} [{para['size']:.0f}pt {para['color']}]: {preview}...")
            render_paragraph(c, para, ph)

        c.showPage()
        print(f"   ✅ Page {page_num} done\n")

    c.save()
    print(f"{'='*60}")
    print(f"✅ Generated: {OUTPUT_PDF}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()