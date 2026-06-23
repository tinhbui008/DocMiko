"""
PDF Native Parser - Phase 1 of Hybrid Approach.
Extract structured data from text-based PDFs.
NO OCR — use PDF native API for 100% accuracy.

Output: structured JSON với font, color, exact positions.
"""
import json
from pathlib import Path
from collections import defaultdict

import pdfplumber
import pypdfium2 as pdfium


# ============================================================
# COLOR UTILS
# ============================================================

def rgb_to_hex(rgb):
    """Convert RGB tuple (0-1 floats) to hex string."""
    if rgb is None:
        return "#000000"
    if isinstance(rgb, (int, float)):
        # Grayscale
        val = int(rgb * 255)
        return f"#{val:02x}{val:02x}{val:02x}"
    if len(rgb) == 3:
        r, g, b = [int(c * 255) for c in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"
    if len(rgb) == 4:  # CMYK
        c, m, y, k = rgb
        r = (1 - c) * (1 - k) * 255
        g = (1 - m) * (1 - k) * 255
        b = (1 - y) * (1 - k) * 255
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    return "#000000"


# ============================================================
# DETECT PDF TYPE
# ============================================================

def is_text_based_pdf(pdf_path: str, threshold: int = 50) -> bool:
    """
    Check if PDF has extractable text (vs scanned).
    Returns True if total text > threshold chars across first 3 pages.
    """
    with pdfplumber.open(pdf_path) as pdf:
        total_chars = 0
        for page in pdf.pages[:3]:
            text = page.extract_text() or ""
            total_chars += len(text.strip())
        return total_chars > threshold


# ============================================================
# EXTRACT TEXT WITH STYLES
# ============================================================

def extract_chars_grouped(page):
    """
    Extract chars from page, grouped by similar style (font + size + color).
    Returns list of char dicts.
    """
    chars = page.chars
    return chars


def chars_to_words(chars, x_tolerance=2.0):
    """Group chars → words based on horizontal proximity."""
    if not chars:
        return []

    # Sort by line first (Y position), then X
    # chars_sorted = sorted(chars, key=lambda c: (-c['top'], c['x0']))
    chars_sorted = sorted(chars, key=lambda c: (c['top'], c['x0']))

    words = []
    current_word = [chars_sorted[0]]

    for char in chars_sorted[1:]:
        prev = current_word[-1]

        # Same line check
        same_line = abs(char['top'] - prev['top']) < 2

        # Close horizontally?
        gap = char['x0'] - prev['x1']
        close = same_line and gap < x_tolerance

        # Same style?
        same_style = (
            char.get('fontname') == prev.get('fontname')
            and abs(char.get('size', 0) - prev.get('size', 0)) < 0.5
        )

        if close and same_style:
            current_word.append(char)
        else:
            words.append(_build_word(current_word))
            current_word = [char]

    if current_word:
        words.append(_build_word(current_word))

    return words


def _build_word(chars):
    """Merge chars into word dict."""
    text = ''.join(c['text'] for c in chars)
    x0 = min(c['x0'] for c in chars)
    x1 = max(c['x1'] for c in chars)
    top = min(c['top'] for c in chars)
    bottom = max(c['bottom'] for c in chars)

    first = chars[0]
    return {
        'text': text,
        'bbox': [x0, top, x1, bottom],
        'fontname': first.get('fontname', ''),
        'size': first.get('size', 12),
        'color': rgb_to_hex(first.get('non_stroking_color')),
        'is_bold': 'Bold' in first.get('fontname', '') or 'Black' in first.get('fontname', ''),
        'is_italic': 'Italic' in first.get('fontname', '') or 'Oblique' in first.get('fontname', ''),
    }


def words_to_lines(words, y_tolerance=3.0):
    """Group words → lines based on Y position."""
    if not words:
        return []

    # Sort by Y (top first) then X
    # words_sorted = sorted(words, key=lambda w: (-w['bbox'][1], w['bbox'][0]))
    words_sorted = sorted(words, key=lambda w: (w['bbox'][1], w['bbox'][0]))

    lines = []
    current_line = [words_sorted[0]]

    for word in words_sorted[1:]:
        prev_top = current_line[-1]['bbox'][1]
        curr_top = word['bbox'][1]

        if abs(curr_top - prev_top) < y_tolerance:
            current_line.append(word)
        else:
            lines.append(_build_line(current_line))
            current_line = [word]

    if current_line:
        lines.append(_build_line(current_line))

    return lines


def _build_line(words):
    """Merge words into line dict."""
    # Sort by X position
    words = sorted(words, key=lambda w: w['bbox'][0])
    text = ' '.join(w['text'] for w in words)

    x0 = min(w['bbox'][0] for w in words)
    x1 = max(w['bbox'][2] for w in words)
    top = min(w['bbox'][1] for w in words)
    bottom = max(w['bbox'][3] for w in words)

    # Dominant style (most chars)
    first = words[0]

    return {
        'text': text,
        'bbox': [x0, top, x1, bottom],
        'fontname': first['fontname'],
        'size': first['size'],
        'color': first['color'],
        'is_bold': first['is_bold'],
        'is_italic': first['is_italic'],
        'num_words': len(words),
    }


def lines_to_paragraphs(lines, vertical_gap_ratio=0.5):
    """Group lines → paragraphs."""
    if not lines:
        return []

    paragraphs = []
    current_para = [lines[0]]

    for i in range(1, len(lines)):
        prev = current_para[-1]
        curr = lines[i]

        # Vertical gap
        gap = prev['bbox'][1] - curr['bbox'][3]
        line_height = prev['bbox'][3] - prev['bbox'][1]

        # Same style?
        same_font = prev['fontname'] == curr['fontname']
        same_size = abs(prev['size'] - curr['size']) < 1
        same_color = prev['color'] == curr['color']

        # Group?
        close = gap < vertical_gap_ratio * line_height
        is_same_para = close and same_font and same_size and same_color

        if is_same_para:
            current_para.append(curr)
        else:
            paragraphs.append(_build_paragraph(current_para))
            current_para = [curr]

    if current_para:
        paragraphs.append(_build_paragraph(current_para))

    return paragraphs


def _build_paragraph(lines):
    text = ' '.join(line['text'] for line in lines)
    x0 = min(line['bbox'][0] for line in lines)
    x1 = max(line['bbox'][2] for line in lines)
    top = min(line['bbox'][1] for line in lines)
    bottom = max(line['bbox'][3] for line in lines)

    first = lines[0]
    return {
        'text': text,
        'bbox': [x0, top, x1, bottom],
        'fontname': first['fontname'],
        'size': first['size'],
        'color': first['color'],
        'is_bold': first['is_bold'],
        'is_italic': first['is_italic'],
        'num_lines': len(lines),
    }


# ============================================================
# EXTRACT IMAGES
# ============================================================

def extract_images(pdf_path: str, output_dir: str = "extracted_images"):
    """Extract images using pdfplumber + pypdfium2 render."""
    Path(output_dir).mkdir(exist_ok=True)
    images_by_page = defaultdict(list)

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1
            page_height = page.height

            # pdfplumber gives image metadata + bbox
            for img_idx, img in enumerate(page.images):
                try:
                    # Get image bbox in PDF coords (top-left origin from pdfplumber)
                    x0 = img['x0']
                    y0 = img['top']
                    x1 = img['x1']
                    y1 = img['bottom']

                    # Render the image region from the page
                    img_path = f"{output_dir}/page{page_num}_img{img_idx}.png"

                    # Use pypdfium2 to render the specific region
                    pdf_doc = pdfium.PdfDocument(pdf_path)
                    pdf_page = pdf_doc[page_idx]

                    # Render whole page at high res
                    rendered = pdf_page.render(scale=2).to_pil()
                    rw, rh = rendered.size
                    pw = float(page.width)
                    ph = float(page.height)
                    sx = rw / pw
                    sy = rh / ph

                    # Crop to image region
                    cropped = rendered.crop((
                        int(x0 * sx),
                        int(y0 * sy),
                        int(x1 * sx),
                        int(y1 * sy)
                    ))
                    cropped.save(img_path)
                    pdf_doc.close()

                    images_by_page[page_num].append({
                        'path': img_path,
                        'bbox': [x0, y0, x1, y1],  # top-left origin
                        'width': x1 - x0,
                        'height': y1 - y0,
                    })
                except Exception as e:
                    print(f"   ⚠️ Could not extract image {img_idx}: {e}")

    return dict(images_by_page)


# ============================================================
# MAIN PARSER
# ============================================================

def parse_pdf(pdf_path: str, output_json: str = "parsed_pdf.json"):
    """
    Main function: parse PDF → structured JSON.
    """
    print(f"Parsing: {pdf_path}")

    # Check if text-based
    if not is_text_based_pdf(pdf_path):
        print("PDF appears scanned - native parsing won't work")
        print("   Fallback to OCR pipeline (poc_v4.py)")
        return None

    print("PDF is text-based, extracting...")

    # Extract images
    print("\nExtracting images...")
    images_data = extract_images(pdf_path)
    total_images = sum(len(imgs) for imgs in images_data.values())
    print(f"   Found {total_images} images across {len(images_data)} pages")

    # Parse text
    document = {
        'source': pdf_path,
        'pages': []
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1
            print(f"\n--- Page {page_num} ---")

            # Page size
            page_size = [float(page.width), float(page.height)]
            print(f"   Size: {page_size[0]:.0f} × {page_size[1]:.0f} pts")

            # Extract chars
            chars = extract_chars_grouped(page)
            print(f"   Chars: {len(chars)}")

            # Group into structures
            words = chars_to_words(chars)
            lines = words_to_lines(words)
            paragraphs = lines_to_paragraphs(lines)

            print(f"   Words: {len(words)}")
            print(f"   Lines: {len(lines)}")
            print(f"   Paragraphs: {len(paragraphs)}")

            # Show paragraph preview
            for i, para in enumerate(paragraphs[:5]):
                preview = para['text'][:60].replace('\n', ' ')
                style_info = (
                    f"size={para['size']:.0f}, "
                    f"color={para['color']}, "
                    f"bold={para['is_bold']}"
                )
                print(f"   P{i+1} ({style_info}):")
                print(f"      {preview}...")

            if len(paragraphs) > 5:
                print(f"   ... and {len(paragraphs) - 5} more paragraphs")

            page_data = {
                'page': page_num,
                'size': page_size,
                'paragraphs': paragraphs,
                'images': images_data.get(page_num, []),
            }
            document['pages'].append(page_data)

    # Save JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {output_json}")
    return document


if __name__ == "__main__":
    pdf_path = "test_vietnamese.pdf"

    if not Path(pdf_path).exists():
        print(f"Not found: {pdf_path}")
        exit(1)

    document = parse_pdf(pdf_path, output_json="parsed_pdf.json")

    if document:
        print(f"\nSummary:")
        print(f"   Total pages: {len(document['pages'])}")
        total_paras = sum(len(p['paragraphs']) for p in document['pages'])
        total_imgs = sum(len(p['images']) for p in document['pages'])
        print(f"   Total paragraphs: {total_paras}")
        print(f"   Total images: {total_imgs}")