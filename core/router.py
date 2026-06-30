"""
Page classifier: determines whether a PDF page should go to Track A or Track B.

Track A — page has a usable text layer (text can be extracted and replaced in-place)
Track B — page is image-based / scan / flat slide (needs Vision-LLM OCR + inpainting)
"""

from enum import Enum
import fitz  # PyMuPDF


class PageType(Enum):
    TEXT_LAYER = "text_layer"   # Track A
    IMAGE_BASED = "image_based" # Track B


def classify_page(page: fitz.Page, min_char_count: int = 20) -> PageType:
    """
    Classify a single PDF page.

    Heuristic: if the page has at least `min_char_count` extractable characters,
    treat it as a text-layer page (Track A). Otherwise Track B.
    """
    text = page.get_text("text")
    if len(text.strip()) >= min_char_count:
        return PageType.TEXT_LAYER
    return PageType.IMAGE_BASED


def classify_document(pdf_path: str) -> list[dict]:
    """
    Classify every page in a PDF.

    Returns a list of dicts: [{page_index, page_type, char_count}, ...]
    """
    results = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text("text")
        char_count = len(text.strip())
        results.append({
            "page_index": i,
            "page_type": classify_page(page).value,
            "char_count": char_count,
        })
    doc.close()
    return results


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python -m core.router <file.pdf>")
        sys.exit(1)
    pages = classify_document(sys.argv[1])
    print(json.dumps(pages, indent=2, ensure_ascii=False))
