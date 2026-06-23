"""
POC v3: PDF Reconstruction
- Load translated metadata from output_v2.json
- Render Vietnamese text into PDF (overlay on original)
- Output: PDF với text Việt giữ nguyên layout
"""
import os
import json
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import white, black

# ============================================================
# CONFIG
# ============================================================

INPUT_PDF = "test_vietnamese.pdf"
INPUT_JSON = "output_v2.json"
OUTPUT_PDF = "output_translated.pdf"

# Font Vietnamese
FONT_REGULAR = "fonts/NotoSans-Regular.ttf"
FONT_BOLD = "fonts/NotoSans-Bold.ttf"

# Fallback Windows system font
if not Path(FONT_REGULAR).exists():
    FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"
    FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
    print(f"Noto Sans not found, using system fonts")

# Register fonts với ReportLab
pdfmetrics.registerFont(TTFont('VN', FONT_REGULAR))
pdfmetrics.registerFont(TTFont('VN-Bold', FONT_BOLD))


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def image_to_pdf_coords(image_box, image_size, pdf_size):
    """
    Convert image bbox [x1, y1, x2, y2] to PDF bbox.

    Image coords: (0,0) top-left, Y goes down
    PDF coords: (0,0) bottom-left, Y goes up
    """
    img_w, img_h = image_size
    pdf_w, pdf_h = pdf_size

    # Scale factors
    sx = pdf_w / img_w
    sy = pdf_h / img_h

    x1, y1, x2, y2 = image_box

    # Convert to PDF coords
    pdf_x1 = x1 * sx
    pdf_x2 = x2 * sx
    pdf_y1 = pdf_h - (y2 * sy)  # bottom of box in PDF
    pdf_y2 = pdf_h - (y1 * sy)  # top of box in PDF

    return [pdf_x1, pdf_y1, pdf_x2, pdf_y2]


# ============================================================
# TEXT FITTING
# ============================================================

def calculate_font_size(text, box_width, box_height,
                        initial_size=12, min_size=6):
    """
    Find best font size that fits text in box.
    Simple heuristic: estimate char width = font_size × 0.5
    """
    if not text:
        return initial_size

    estimated_char_width = 0.5  # approximate width/height ratio
    chars_per_line = max(1, int(box_width / (initial_size * estimated_char_width)))
    lines_needed = max(1, len(text) / chars_per_line)
    total_height = lines_needed * initial_size * 1.2

    if total_height <= box_height:
        return initial_size

    # Shrink to fit
    for size in range(initial_size, min_size - 1, -1):
        chars_per_line = max(1, int(box_width / (size * estimated_char_width)))
        lines_needed = max(1, len(text) / chars_per_line)
        if lines_needed * size * 1.2 <= box_height:
            return size

    return min_size


# ============================================================
# WRAP TEXT INTO LINES
# ============================================================

def wrap_text(text, c: canvas.Canvas, font_name, font_size, max_width):
    """Wrap text to fit within max_width, return list of lines."""
    words = text.split()
    if not words:
        return []

    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        width = c.stringWidth(test_line, font_name, font_size)

        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines


# ============================================================
# RENDER PARAGRAPH
# ============================================================

def render_paragraph(c: canvas.Canvas, text: str, pdf_box: list,
                     font_name='VN', is_bold=False):
    """Render translated text into PDF box."""
    x1, y1, x2, y2 = pdf_box
    box_width = x2 - x1
    box_height = y2 - y1

    font = 'VN-Bold' if is_bold else font_name

    # 1. Draw white rectangle to cover original text
    c.setFillColor(white)
    c.rect(x1 - 1, y1 - 1, box_width + 2, box_height + 2, fill=1, stroke=0)

    # 2. Calculate font size to fit
    font_size = calculate_font_size(text, box_width, box_height)

    # 3. Wrap text
    lines = wrap_text(text, c, font, font_size, box_width)

    if not lines:
        return

    # 4. Draw text lines
    c.setFillColor(black)
    c.setFont(font, font_size)

    line_height = font_size * 1.2
    # Start from top of box
    y_cursor = y2 - font_size

    for line in lines:
        if y_cursor < y1:  # don't overflow below box
            break
        c.drawString(x1, y_cursor, line)
        y_cursor -= line_height


# ============================================================
# IS TITLE? (heuristic — title boxes are wider/taller)
# ============================================================

def is_likely_title(box, image_height, num_lines):
    """Heuristic: titles are usually large height + short text."""
    x1, y1, x2, y2 = box
    height = y2 - y1
    relative_height = height / image_height
    return relative_height > 0.025 and num_lines <= 2


# ============================================================
# MAIN
# ============================================================

def reconstruct_pdf(input_pdf: str, json_path: str, output_pdf: str):
    """Main reconstruction pipeline."""
    # Load translations
    with open(json_path, 'r', encoding='utf-8') as f:
        doc_data = json.load(f)

    print(f"📄 Loading source PDF: {input_pdf}")
    src_pdf = pdfium.PdfDocument(input_pdf)

    # Get first page size to setup canvas
    first_page = src_pdf[0]
    page_w_pts = first_page.get_width()
    page_h_pts = first_page.get_height()

    print(f"📐 PDF page size: {page_w_pts:.0f} × {page_h_pts:.0f} pts")
    print(f"📝 Creating output PDF: {output_pdf}")

    c = canvas.Canvas(output_pdf, pagesize=(page_w_pts, page_h_pts))

    for page_data in doc_data['pages']:
        page_num = page_data['page']
        page_idx = page_num - 1

        print(f"\n--- Processing Page {page_num} ---")

        # Get source page
        src_page = src_pdf[page_idx]
        page_w = src_page.get_width()
        page_h = src_page.get_height()

        # Set page size if different
        c.setPageSize((page_w, page_h))

        # === Step 1: Render source page as image background ===
        print(f"   Rendering background image...")
        page_image = src_page.render(scale=2).to_pil()

        # Save temp background image
        bg_path = f"_temp_bg_page{page_num}.png"
        page_image.save(bg_path)

        # Draw background filling whole page
        c.drawImage(bg_path, 0, 0, width=page_w, height=page_h)

        # === Step 2: Overlay translated paragraphs ===
        print(f"   Overlaying {len(page_data['paragraphs'])} paragraphs...")

        image_size = tuple(page_data['image_size'])

        for i, para in enumerate(page_data['paragraphs']):
            image_box = para['box']
            translation = para['translated']

            if not translation or translation == "[TRANSLATION FAILED]":
                continue

            # Convert coords
            pdf_box = image_to_pdf_coords(image_box, image_size,
                                          (page_w, page_h))

            # Detect title
            is_title = is_likely_title(image_box, image_size[1],
                                       para['num_lines'])

            # Render
            render_paragraph(c, translation, pdf_box, is_bold=is_title)

        # Cleanup temp
        Path(bg_path).unlink(missing_ok=True)

        # Next page
        c.showPage()
        print(f"Page {page_num} done")

    # Save final PDF
    c.save()
    src_pdf.close()

    print(f"\n{'='*60}")
    print(f"PDF translated saved: {output_pdf}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if not Path(INPUT_JSON).exists():
        print(f"Missing {INPUT_JSON}. Run poc_v2.py first!")
        exit(1)

    if not Path(INPUT_PDF).exists():
        print(f"Missing {INPUT_PDF}")
        exit(1)

    reconstruct_pdf(INPUT_PDF, INPUT_JSON, OUTPUT_PDF)