"""
POC v2: PDF Translation Pipeline with improvements
- Paragraph grouping (gộp lines thành paragraphs)
- Batch translation (dịch nhiều paragraphs 1 lần)
- Strict prompt (LLM không lan man)
- JSON metadata output (chuẩn bị cho PDF reconstruction)
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

import pypdfium2 as pdfium
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
from openai import OpenAI
import numpy as np

# ===== Load config =====
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# ===== Init PaddleOCR =====
print("Loading PaddleOCR...")
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)
except Exception:
    ocr = PaddleOCR(use_angle_cls=True, lang='latin', show_log=False)
print("PaddleOCR loaded\n")


# ============================================================
# PART 1: PDF & OCR (giống v1)
# ============================================================

def pdf_to_images(pdf_path: str):
    """Convert PDF pages to PIL Images."""
    print(f"Reading PDF: {pdf_path}")
    pdf = pdfium.PdfDocument(pdf_path)
    images = []
    for i, page in enumerate(pdf):
        image = page.render(scale=2).to_pil()
        images.append(image)
        print(f"   Page {i+1}: {image.size}")
    return images


def detect_lines(image: Image.Image):
    """
    Detect text lines using PaddleOCR.
    Returns list of:
        {
            'box': [x1, y1, x2, y2],  # normalized to rectangle
            'polygon': [[x,y], ...],   # original 4 points
            'text': str,
            'confidence': float,
            'height': float,           # line height
            'center_y': float,         # for grouping
        }
    """
    img_array = np.array(image)
    result = ocr.ocr(img_array, cls=True)

    lines = []
    if result and result[0]:
        for line in result[0]:
            polygon = line[0]
            text, confidence = line[1]

            # Convert 4-point polygon to rectangle bbox
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            lines.append({
                'box': [x1, y1, x2, y2],
                'polygon': polygon,
                'text': text,
                'confidence': confidence,
                'height': y2 - y1,
                'center_y': (y1 + y2) / 2,
                'left_x': x1,
            })

    # Sort by reading order: top to bottom, left to right
    lines.sort(key=lambda b: (b['center_y'], b['left_x']))
    return lines


# ============================================================
# PART 2: PARAGRAPH GROUPING (NEW)
# ============================================================
def group_into_paragraphs(lines: list,
                          vertical_gap_factor: float = 1.5,
                          horizontal_align_threshold: float = 50,
                          height_diff_ratio: float = 0.4):
    """
    Group lines into paragraphs.

    Rules - same paragraph requires ALL:
    1. Vertical gap < factor × line_height
    2. Horizontal left edge aligned
    3. Similar line height (within ratio) — separates titles from body
    """
    if not lines:
        return []

    paragraphs = []
    current_para = [lines[0]]

    for i in range(1, len(lines)):
        prev_line = current_para[-1]
        curr_line = lines[i]

        # Rule 1: vertical gap
        prev_bottom = prev_line['box'][3]
        curr_top = curr_line['box'][1]
        vertical_gap = curr_top - prev_bottom
        avg_height = (prev_line['height'] + curr_line['height']) / 2
        gap_ok = (
            vertical_gap < vertical_gap_factor * avg_height
            and vertical_gap >= -avg_height
        )

        # Rule 2: horizontal alignment
        horizontal_diff = abs(prev_line['left_x'] - curr_line['left_x'])
        align_ok = horizontal_diff < horizontal_align_threshold

        # Rule 3: similar font height (CRITICAL — separates titles from body)
        height_ratio = (min(prev_line['height'], curr_line['height']) /
                        max(prev_line['height'], curr_line['height']))
        height_ok = height_ratio > (1 - height_diff_ratio)

        # All rules must pass
        is_same_para = gap_ok and align_ok and height_ok

        if is_same_para:
            current_para.append(curr_line)
        else:
            paragraphs.append(_build_paragraph(current_para))
            current_para = [curr_line]

    if current_para:
        paragraphs.append(_build_paragraph(current_para))

    return paragraphs


def _build_paragraph(lines: list) -> dict:
    """Merge lines into a single paragraph dict."""
    # Merge text với space
    text = ' '.join(line['text'].strip() for line in lines)

    # Compute paragraph bbox (covering all lines)
    x1 = min(line['box'][0] for line in lines)
    y1 = min(line['box'][1] for line in lines)
    x2 = max(line['box'][2] for line in lines)
    y2 = max(line['box'][3] for line in lines)

    return {
        'lines': lines,
        'text': text,
        'box': [x1, y1, x2, y2],
        'num_lines': len(lines),
    }


# ============================================================
# PART 3: BATCH TRANSLATION (NEW)
# ============================================================

STRICT_SYSTEM_PROMPT = """You are a professional English-to-Vietnamese translator.

TRANSLATION RULES:
1. Translate the input texts to Vietnamese with proper diacritics (dấu thanh)
2. Output ONLY the translation - NO explanations, NO comments, NO questions
3. Preserve all numbers, codes, URLs, emails, brand names unchanged
4. For technical terms without Vietnamese equivalents, keep the English term
5. If input is a brand/logo (e.g., "FLORIDACOMMERCE"), keep it as-is
6. Maintain the same paragraph structure

OUTPUT FORMAT:
Input is JSON array of paragraphs. Output JSON array of translations in the SAME ORDER.
Output ONLY the JSON, no markdown, no explanations.

Example input:
["Hello world", "Welcome to FloridaCommerce"]

Example output:
["Xin chào thế giới", "Chào mừng đến với FloridaCommerce"]
"""


def translate_batch(paragraphs: list, target_lang: str = "Vietnamese",
                    batch_size: int = 10) -> list:
    """
    Translate paragraphs in batches for better context + efficiency.

    Args:
        paragraphs: list of paragraph dicts (with 'text' field)
        batch_size: number of paragraphs per LLM call

    Returns:
        list of translation strings (same order as input)
    """
    all_translations = []

    for batch_start in range(0, len(paragraphs), batch_size):
        batch = paragraphs[batch_start:batch_start + batch_size]
        batch_texts = [p['text'] for p in batch]

        print(f"   Batch {batch_start//batch_size + 1}: "
              f"translating {len(batch)} paragraphs...")

        try:
            translations = _call_llm_batch(batch_texts, target_lang)

            # Validate length match
            if len(translations) != len(batch):
                print(f"Translation count mismatch "
                      f"({len(translations)} vs {len(batch)}), "
                      f"falling back to individual translation")
                translations = [_translate_single(t, target_lang) for t in batch_texts]

            all_translations.extend(translations)

        except Exception as e:
            print(f"Batch failed: {e}, falling back to individual")
            for text in batch_texts:
                try:
                    translations = _translate_single(text, target_lang)
                    all_translations.append(translations)
                except Exception:
                    all_translations.append("[TRANSLATION FAILED]")

    return all_translations


def _call_llm_batch(texts: list, target_lang: str) -> list:
    """Send batch to LLM, expect JSON array output."""
    user_msg = json.dumps(texts, ensure_ascii=False)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": STRICT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.2,
        max_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()

    # Remove markdown code fences if LLM added them
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].startswith("```")
                        else lines[1:])

    # Parse JSON
    try:
        translations = json.loads(raw)
        if not isinstance(translations, list):
            raise ValueError("Output is not a JSON array")
        return translations
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output as JSON: {e}\nRaw: {raw[:200]}")


def _translate_single(text: str, target_lang: str) -> str:
    """Fallback: translate one paragraph at a time."""
    if not text.strip():
        return ""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Translate to {target_lang}. "
                    f"Output ONLY the translation. No explanations."
                )
            },
            {"role": "user", "content": text}
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# ============================================================
# PART 4: VISUALIZATION (improved)
# ============================================================

def visualize_paragraphs(image: Image.Image, paragraphs: list, output_path: str):
    """Draw paragraph boxes (not line boxes) for cleaner visualization."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

    for i, para in enumerate(paragraphs):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = [int(c) for c in para['box']]

        # Draw paragraph box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label
        draw.text((x1, y1 - 22), f"P{i+1} ({para['num_lines']} lines)",
                  fill=color)

    img_copy.save(output_path)
    print(f"   Saved: {output_path}")


# ============================================================
# PART 5: MAIN PIPELINE
# ============================================================

def process_pdf(pdf_path: str, output_prefix: str = "output_v2"):
    """End-to-end pipeline."""
    if not Path(pdf_path).exists():
        print(f"ERROR: Not found: {pdf_path}")
        return

    # Step 1: PDF → Images
    images = pdf_to_images(pdf_path)

    # Final output structure
    document_data = {
        'source_pdf': pdf_path,
        'target_language': 'Vietnamese',
        'pages': []
    }

    for page_idx, image in enumerate(images):
        page_num = page_idx + 1
        print(f"\n{'='*60}")
        print(f"PAGE {page_num}")
        print(f"{'='*60}")

        # Step 2: Detect lines
        print("Step 1: Detecting text lines...")
        lines = detect_lines(image)
        print(f"   Found {len(lines)} lines")

        # Step 3: Group into paragraphs
        print("\nStep 2: Grouping into paragraphs...")
        paragraphs = group_into_paragraphs(lines)
        print(f"   Grouped {len(lines)} lines → {len(paragraphs)} paragraphs")

        # Show paragraph preview
        for i, para in enumerate(paragraphs):
            preview = para['text'][:70].replace('\n', ' ')
            print(f"   P{i+1} ({para['num_lines']} lines): {preview}...")

        # Step 4: Visualize paragraphs
        print(f"\nStep 3: Visualizing paragraph layout...")
        viz_path = f"{output_prefix}_layout_page{page_num}.png"
        visualize_paragraphs(image, paragraphs, viz_path)

        # Step 5: Batch translate
        print(f"\nStep 4: Translating {len(paragraphs)} paragraphs in batches...")
        translations = translate_batch(paragraphs, "Vietnamese", batch_size=5)

        # Store results
        page_data = {
            'page': page_num,
            'image_size': image.size,
            'paragraphs': []
        }
        for para, translation in zip(paragraphs, translations):
            page_data['paragraphs'].append({
                'box': para['box'],
                'num_lines': para['num_lines'],
                'original': para['text'],
                'translated': translation,
            })

        document_data['pages'].append(page_data)

        # Preview translations
        print(f"\n   --- Page {page_num} translations preview ---")
        for i, p in enumerate(page_data['paragraphs'][:3]):
            print(f"   P{i+1}:")
            print(f"      EN: {p['original'][:80]}...")
            print(f"      VI: {p['translated'][:80]}...")

    # Save outputs
    print(f"\n{'='*60}")
    print("SAVING OUTPUTS")
    print(f"{'='*60}")

    # JSON (for next stage - PDF reconstruction)
    json_path = f"{output_prefix}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(document_data, f, ensure_ascii=False, indent=2)
    print(f"📄 JSON metadata: {json_path}")

    # TXT (human-readable)
    txt_path = f"{output_prefix}_translated.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        for page_data in document_data['pages']:
            f.write(f"\n{'='*60}\n")
            f.write(f"PAGE {page_data['page']}\n")
            f.write(f"{'='*60}\n\n")
            for i, p in enumerate(page_data['paragraphs']):
                f.write(f"[Paragraph {i+1}] ({p['num_lines']} lines)\n")
                f.write(f"EN: {p['original']}\n")
                f.write(f"VI: {p['translated']}\n\n")
    print(f"Text output: {txt_path}")

    print(f"\nDONE!")
    print(f"Total: {len(document_data['pages'])} pages, "
          f"{sum(len(p['paragraphs']) for p in document_data['pages'])} paragraphs")


if __name__ == "__main__":
    pdf_path = "test_vietnamese.pdf"
    process_pdf(pdf_path, output_prefix="output_v2")