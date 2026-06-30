"""
Re-render translated text onto a cleaned (inpainted) image.

Matches font color from OCR block metadata.
Font: NotoSans family (covers Vietnamese + Latin).
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

_FONTS_DIR = Path(__file__).parent.parent.parent / "assets" / "fonts"
_FALLBACK_FONT = _FONTS_DIR / "NotoSans-Regular.ttf"


def _load_font(size_pt: float | None, bold: bool = False) -> ImageFont.FreeTypeFont:
    size = max(8, int(size_pt or 12))
    font_file = _FONTS_DIR / ("NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf")
    if not font_file.exists():
        font_file = _FALLBACK_FONT
    try:
        return ImageFont.truetype(str(font_file), size)
    except Exception:
        return ImageFont.load_default()


def _hex_to_rgb(hex_color: str | None) -> tuple[int, int, int]:
    if not hex_color:
        return (0, 0, 0)
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def render_blocks(
    image: Image.Image,
    blocks: list[dict],
    use_translation: bool = True,
) -> Image.Image:
    """
    Draw translated text onto image for each block.

    blocks: OCR block dicts with bbox_norm, translation, font_color_hex, font_size_pt
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    for block in blocks:
        text = block.get("translation") if use_translation else block.get("text", "")
        if not text:
            continue

        px0, py0, px1, py1 = _bbox_pixels(block, image.width, image.height)
        if px0 >= px1 or py0 >= py1:
            continue

        color = _hex_to_rgb(block.get("font_color_hex"))
        role = block.get("role", "body")
        bold = role in ("heading", "subheading", "button", "label")
        font = _load_font(block.get("font_size_pt"), bold=bold)

        # Wrap text to fit bbox width
        _draw_text_wrapped(draw, text, px0, py0, px1, py1, font, color)

    return img


def _bbox_pixels(block: dict, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = block["bbox_norm"]
    return (int(x0 * img_w), int(y0 * img_h), int(x1 * img_w), int(y1 * img_h))


def _draw_text_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
) -> None:
    max_width = x1 - x0
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    y = y0
    line_height = font.size + 2
    for line in lines:
        if y + line_height > y1:
            break
        draw.text((x0, y), line, font=font, fill=color)
        y += line_height
