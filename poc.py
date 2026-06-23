"""
POC: PDF Translation Pipeline
Input:  test_vietnamese.pdf (Vietnamese PDF)
Output: 
  - output_layout_page*.png  (visualize bounding boxes)
  - output_translated.txt  (translated text)
"""
import os
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

# ===== Init LLM client =====
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# ===== Init PaddleOCR =====
print("Loading PaddleOCR (first run downloads ~50MB model)...")
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)
except Exception:
    print("'vi' lang not available, falling back to 'latin'")
    ocr = PaddleOCR(use_angle_cls=True, lang='latin', show_log=False)
print("PaddleOCR loaded\n")


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


def detect_and_extract(image: Image.Image):
    """Detect text blocks + extract text using PaddleOCR."""
    img_array = np.array(image)
    result = ocr.ocr(img_array, cls=True)

    blocks = []
    if result and result[0]:
        for line in result[0]:
            box = line[0]
            text, confidence = line[1]
            blocks.append({
                'box': box,
                'text': text,
                'confidence': confidence
            })
    return blocks


def translate_text(text: str, target_lang: str = "Vietnamese") -> str:
    """Translate using LLM."""
    if not text.strip():
        return ""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a professional translator. "
                    f"Translate the following English text to {target_lang}. "
                    f"Use proper Vietnamese diacritics (dấu thanh). "
                    f"Return ONLY the translation, no explanations."
                )
            },
            {"role": "user", "content": text}
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

def visualize_layout(image: Image.Image, blocks: list, output_path: str):
    """Draw bounding boxes on image."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    for i, block in enumerate(blocks):
        box = block['box']
        points = [(int(p[0]), int(p[1])) for p in box]
        draw.polygon(points, outline='red', width=3)
        draw.text((points[0][0], points[0][1] - 20),
                  f"#{i+1}", fill='red')

    img_copy.save(output_path)
    print(f"Saved visualization: {output_path}")


def main():
    pdf_path = "test_vietnamese.pdf"

    if not Path(pdf_path).exists():
        print(f"ERROR: Not found: {pdf_path}")
        print("   Please copy test_vietnamese.pdf to this folder")
        return

    images = pdf_to_images(pdf_path)
    all_translations = []

    for page_idx, image in enumerate(images):
        print(f"\n{'='*60}")
        print(f"PAGE {page_idx + 1}")
        print(f"{'='*60}")

        print("Detecting layout...")
        blocks = detect_and_extract(image)
        print(f"Found {len(blocks)} text blocks")

        viz_path = f"output_layout_page{page_idx+1}.png"
        visualize_layout(image, blocks, viz_path)

        print(f"\nTranslating {len(blocks)} blocks...")
        page_translations = []
        for i, block in enumerate(blocks):
            original = block['text']
            print(f"\n   [{i+1}/{len(blocks)}] VI: {original[:60]}...")

            try:
                translated = translate_text(original, "English")
                print(f"           EN: {translated[:60]}...")
                page_translations.append({
                    'original': original,
                    'translated': translated,
                    'box': block['box']
                })
            except Exception as e:
                print(f"   Translation failed: {e}")
                page_translations.append({
                    'original': original,
                    'translated': '[TRANSLATION FAILED]',
                    'box': block['box']
                })

        all_translations.append({
            'page': page_idx + 1,
            'blocks': page_translations
        })

    output_txt = "output_translated.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        for page_data in all_translations:
            f.write(f"\n{'='*60}\n")
            f.write(f"PAGE {page_data['page']}\n")
            f.write(f"{'='*60}\n\n")
            for i, block in enumerate(page_data['blocks']):
                f.write(f"[Block {i+1}]\n")
                f.write(f"VI: {block['original']}\n")
                f.write(f"EN: {block['translated']}\n\n")

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"{'='*60}")
    print(f"Translations: {output_txt}")
    print(f"Layout images: output_layout_page*.png")


if __name__ == "__main__":
    main()