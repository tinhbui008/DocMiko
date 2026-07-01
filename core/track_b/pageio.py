"""
Bridge between fitz pages and PIL images for Track B.

  render_page_image(page, dpi)      — rasterize a PDF page to a PIL RGB image
  apply_image_to_page(page, image)  — draw a full-page image over the page,
                                       replacing its visible content

This closes the missing link: Track B produces a translated raster, and this
module writes it back so the whole document can be saved as one PDF.
"""

import io

import fitz
from PIL import Image


def render_page_image(page: "fitz.Page", dpi: int = 200) -> Image.Image:
    """Rasterize a PDF page to a PIL RGB image at the given DPI."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def apply_image_to_page(page: "fitz.Page", image: Image.Image) -> None:
    """
    Replace the page's visible content with `image`.

    The original text/vector content is redacted away first (so no original
    text bleeds through), then the translated raster is drawn to fill the page.
    """
    # Wipe existing content in the page rectangle.
    page.add_redact_annot(page.rect, fill=(1, 1, 1))
    page.apply_redactions()

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    page.insert_image(page.rect, stream=buf.getvalue(), keep_proportion=False)
