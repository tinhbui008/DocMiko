"""
Write translated text back into PDF text-layer (in-place replacement).

Strategy: redact the original span bbox, then insert translated text with
the same font/size/color at the same position.
"""

import fitz


def write_translations(
    pdf_path: str,
    spans: list[dict],
    output_path: str,
    fonts_dir: str | None = None,
) -> None:
    """
    Given translated spans (with 'translation' field), produce a new PDF
    with all translated text replacing the originals.
    """
    doc = fitz.open(pdf_path)

    # Group spans by page
    by_page: dict[int, list[dict]] = {}
    for span in spans:
        by_page.setdefault(span["page"], []).append(span)

    for page_idx, page_spans in by_page.items():
        page = doc[page_idx]
        for span in page_spans:
            translation = span.get("translation", span["text"])
            if translation == span["text"]:
                continue  # nothing changed

            bbox = fitz.Rect(span["bbox"])

            # Redact original text
            page.add_redact_annot(bbox, fill=(1, 1, 1))  # white fill
        page.apply_redactions()

        # Re-draw translated text
        for span in page_spans:
            translation = span.get("translation", span["text"])
            if translation == span["text"]:
                continue

            bbox = fitz.Rect(span["bbox"])
            color = _unpack_color(span["color"])
            page.insert_text(
                point=fitz.Point(span["origin"]),
                text=translation,
                fontsize=span["size"],
                color=color,
                # fontname fallback — expand later with font matching
            )

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()


def _unpack_color(packed: int) -> tuple[float, float, float]:
    r = ((packed >> 16) & 0xFF) / 255.0
    g = ((packed >> 8) & 0xFF) / 255.0
    b = (packed & 0xFF) / 255.0
    return (r, g, b)
