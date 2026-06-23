"""
Phase 2: Translate paragraphs while preserving ALL style metadata.

Input:  parsed_pdf.json (from pdf_parser.py)
Output: translated_pdf.json (same structure + 'translated' field)
"""
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

INPUT_JSON = "parsed_pdf.json"
OUTPUT_JSON = "translated_pdf.json"
TARGET_LANG = "Vietnamese"

# Brand whitelist - không dịch
BRAND_WHITELIST = {
    'floridacommerce', 'powered by', 'powered', 'by:', 'florida commerce',
}

SKIP_PATTERNS = [
    r'^https?://',
    r'^[\w\.-]+@[\w\.-]+',
    r'^\$?\d[\d,\.]*\s*$',
]


# ============================================================
# TRANSLATION PROMPT
# ============================================================

SYSTEM_PROMPT = f"""You are a professional English-to-{TARGET_LANG} translator.

CRITICAL RULES:
1. Translate to natural, fluent {TARGET_LANG} with proper diacritics
2. Output ONLY translation - NO explanations, NO meta-comments
3. Keep these UNCHANGED:
   - Numbers, prices, percentages ($50,000, 40%, etc.)
   - URLs, emails (LocalGovernmentBridge@Commerce.FL.gov)
   - Brand names (FloridaCommerce, U.S. Small Business Administration)
   - Acronyms (USA, SBA, etc.)
4. Translate concisely - {TARGET_LANG} should not be much longer than English
5. Preserve formatting: capitalize titles, keep punctuation

INPUT FORMAT: JSON array of texts
OUTPUT FORMAT: JSON array of translations - SAME ORDER, SAME LENGTH

Example:
Input: ["Florida Small Business Bridge Loan", "Loans up to $50,000", "FloridaCommerce"]
Output: ["Khoản Vay Bắc Cầu Doanh Nghiệp Nhỏ Florida", "Khoản vay lên đến $50,000", "FloridaCommerce"]
"""


# ============================================================
# UTILITIES
# ============================================================

def should_skip_translation(text: str) -> bool:
    """Check if text should not be translated."""
    text_lower = text.lower().strip()

    if text_lower in BRAND_WHITELIST:
        return True
    for brand in BRAND_WHITELIST:
        if brand in text_lower and len(text_lower) <= len(brand) + 5:
            return True

    for pattern in SKIP_PATTERNS:
        if re.match(pattern, text):
            return True

    return False


def clean_text(text: str) -> str:
    """Clean text before translation - normalize whitespace."""
    # Multiple spaces → single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# TRANSLATION CORE
# ============================================================

def translate_batch(texts: list) -> list:
    """Send batch of texts to LLM, return translations in same order."""
    if not texts:
        return []

    user_msg = json.dumps(texts, ensure_ascii=False)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.2,
        max_tokens=3000,
    )

    raw = response.choices[0].message.content.strip()

    # Clean markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(l for l in lines if not l.strip().startswith("```"))

    try:
        translations = json.loads(raw)
        if not isinstance(translations, list):
            raise ValueError("Output is not JSON array")
        return translations
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}\nRaw output: {raw[:300]}")


def translate_single(text: str) -> str:
    """Fallback: translate one text at a time."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"Translate to {TARGET_LANG}. Output ONLY the translation."
            },
            {"role": "user", "content": text}
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def translate_with_fallback(texts: list, batch_size: int = 8) -> list:
    """Translate with batch + individual fallback."""
    if not texts:
        return []

    all_translations = []

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start:batch_start + batch_size]
        try:
            print(f"Batch {batch_start//batch_size + 1}: "
                  f"translating {len(batch)} texts...")
            translations = translate_batch(batch)

            if len(translations) != len(batch):
                raise ValueError(
                    f"Length mismatch: {len(translations)} vs {len(batch)}"
                )

            all_translations.extend(translations)

        except Exception as e:
            print(f"Batch failed: {e}")
            print(f"Falling back to individual translation...")
            for text in batch:
                try:
                    trans = translate_single(text)
                    all_translations.append(trans)
                except Exception as e2:
                    print(f"Failed: {text[:40]}... ({e2})")
                    all_translations.append(text)  # keep original

    return all_translations


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    if not Path(INPUT_JSON).exists():
        print(f"Not found: {INPUT_JSON}")
        print(f"Run pdf_parser.py first!")
        return

    print(f"Loading: {INPUT_JSON}")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        doc = json.load(f)

    print(f"Document: {len(doc['pages'])} pages\n")

    # Collect ALL paragraphs across pages
    all_paragraphs = []
    for page in doc['pages']:
        for para in page['paragraphs']:
            all_paragraphs.append(para)

    total = len(all_paragraphs)
    print(f"Total paragraphs: {total}")

    # Identify skip vs translate
    to_translate_indices = []
    to_translate_texts = []
    skip_count = 0

    for i, para in enumerate(all_paragraphs):
        cleaned = clean_text(para['text'])

        if should_skip_translation(cleaned):
            para['translated'] = para['text']  # keep original
            skip_count += 1
            print(f"   [SKIP-BRAND] {cleaned[:60]}")
        elif len(cleaned) < 2:
            para['translated'] = para['text']
            skip_count += 1
        else:
            to_translate_indices.append(i)
            to_translate_texts.append(cleaned)

    print(f"\nTo translate: {len(to_translate_texts)}")
    print(f"Skipped (brands): {skip_count}\n")

    # Translate
    print("Translating...")
    translations = translate_with_fallback(to_translate_texts, batch_size=8)

    # Fill back into paragraphs
    for idx, translation in zip(to_translate_indices, translations):
        all_paragraphs[idx]['translated'] = translation

    # Make sure ALL paragraphs have 'translated' field
    for para in all_paragraphs:
        if 'translated' not in para:
            para['translated'] = para['text']

    # Preview
    print(f"\n{'='*60}")
    print("PREVIEW (first 5 paragraphs):")
    print(f"{'='*60}")
    for i, para in enumerate(all_paragraphs[:5]):
        print(f"\nP{i+1} [{para['fontname']}, {para['size']}pt, {para['color']}]")
        print(f"   EN: {para['text'][:80]}...")
        print(f"   VI: {para['translated'][:80]}...")

    # Save
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"{'='*60}")
    print(f"Total: {total} paragraphs ({len(to_translate_texts)} translated, "
          f"{skip_count} skipped)")


if __name__ == "__main__":
    main()