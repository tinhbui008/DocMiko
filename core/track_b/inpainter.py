"""
Inpainting: erase original text from an image and fill the background.

Two strategies (selectable):
  "solid"   — fill bbox with sampled background color (fast, low quality)
  "lama"    — LaMa model inpainting (high quality, requires lama-cleaner)

Goal: replace with "lama" once R&D confirms quality gain.
"""

import numpy as np
from PIL import Image, ImageDraw


def inpaint_blocks(
    image: Image.Image,
    blocks: list[dict],
    strategy: str = "solid",
) -> Image.Image:
    """
    Remove text from image for all blocks, return cleaned image.

    blocks: list of OCR block dicts (must have bbox_norm, image_width, image_height)
    """
    if strategy == "solid":
        return _inpaint_solid(image, blocks)
    elif strategy == "lama":
        return _inpaint_lama(image, blocks)
    else:
        raise ValueError(f"Unknown inpainting strategy: {strategy}")


def _bbox_pixels(block: dict, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = block["bbox_norm"]
    return (
        int(x0 * img_w),
        int(y0 * img_h),
        int(x1 * img_w),
        int(y1 * img_h),
    )


def _inpaint_solid(image: Image.Image, blocks: list[dict]) -> Image.Image:
    """Sample a border pixel from each bbox and flood-fill the bbox."""
    img = image.copy().convert("RGB")
    arr = np.array(img)
    draw = ImageDraw.Draw(img)

    for block in blocks:
        px0, py0, px1, py1 = _bbox_pixels(block, image.width, image.height)
        if px0 >= px1 or py0 >= py1:
            continue
        # Sample color from 2px above the bbox (or from bg_color_hex if available)
        sample_y = max(0, py0 - 2)
        sample_x = (px0 + px1) // 2
        bg_color = tuple(arr[sample_y, sample_x])
        draw.rectangle([px0, py0, px1, py1], fill=bg_color)

    return img


def _inpaint_lama(image: Image.Image, blocks: list[dict]) -> Image.Image:
    """
    LaMa inpainting via lama-cleaner HTTP API or local model.
    Requires lama-cleaner running: `lama-cleaner --model=lama --device=cpu --port=8080`
    """
    try:
        import requests
        import io
    except ImportError:
        raise RuntimeError("requests not installed — needed for LaMa inpainting")

    img = image.copy().convert("RGB")
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)

    for block in blocks:
        px0, py0, px1, py1 = _bbox_pixels(block, image.width, image.height)
        if px0 >= px1 or py0 >= py1:
            continue
        draw.rectangle([px0, py0, px1, py1], fill=255)

    # Encode image and mask as PNG
    def to_bytes(im: Image.Image) -> bytes:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    resp = requests.post(
        "http://localhost:8080/inpaint",
        files={
            "image": ("image.png", to_bytes(img), "image/png"),
            "mask": ("mask.png", to_bytes(mask), "image/png"),
        },
        data={"ldmSteps": 25},
        timeout=120,
    )
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")
