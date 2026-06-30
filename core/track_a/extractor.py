"""
Extract text spans from a PDF text-layer page.

Output schema per span:
{
    "page": int,
    "block": int,
    "line": int,
    "span": int,
    "text": str,
    "font": str,
    "size": float,
    "color": int,       # RGB packed int
    "bbox": [x0, y0, x1, y1],
    "origin": [x, y],
}
"""

import fitz


def extract_spans(pdf_path: str, page_indices: list[int] | None = None) -> list[dict]:
    """
    Extract all text spans from the given pages (or all pages if None).
    """
    doc = fitz.open(pdf_path)
    spans = []
    pages = page_indices if page_indices is not None else range(len(doc))

    for page_idx in pages:
        page = doc[page_idx]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for b_idx, block in enumerate(blocks):
            if block["type"] != 0:  # skip image blocks
                continue
            for l_idx, line in enumerate(block["lines"]):
                for s_idx, span in enumerate(line["spans"]):
                    spans.append({
                        "page": page_idx,
                        "block": b_idx,
                        "line": l_idx,
                        "span": s_idx,
                        "text": span["text"],
                        "font": span["font"],
                        "size": span["size"],
                        "color": span["color"],
                        "bbox": list(span["bbox"]),
                        "origin": list(span["origin"]),
                    })
    doc.close()
    return spans


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python -m core.track_a.extractor <file.pdf>")
        sys.exit(1)
    result = extract_spans(sys.argv[1])
    print(json.dumps(result, indent=2, ensure_ascii=False))
