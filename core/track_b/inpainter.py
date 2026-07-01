"""
Inpainting: erase original text from an image and reconstruct the background.

Strategies (selectable):
  "solid"  — fill each bbox with a sampled flat color (fast, destroys texture)
  "telea"  — glyph-level mask + OpenCV Telea inpainting (reconstructs background
             texture from surrounding real pixels — the default upgrade over
             the old flat-fill approach; no GPU/model needed)
  "lama"   — LaMa model via lama-cleaner HTTP API (highest quality, external svc)

Why "telea" is the new default target: the old flat-fill (`fill:"sample"`)
paints one color over the whole text box, which wipes out gradients and photo
backgrounds. Telea only removes the glyph pixels themselves and grows the
surrounding background inward, so smooth/gradient/photo backgrounds survive.
"""

import numpy as np
from PIL import Image, ImageDraw


def inpaint_blocks(
    image: Image.Image,
    blocks: list[dict],
    strategy: str = "telea",
) -> Image.Image:
    """Remove text for all blocks and return a cleaned image."""
    if strategy == "solid":
        return _inpaint_solid(image, blocks)
    if strategy == "telea":
        return _inpaint_telea(image, blocks)
    if strategy == "lama":
        return _inpaint_lama(image, blocks)
    raise ValueError(f"Unknown inpainting strategy: {strategy}")


def _bbox_pixels(block: dict, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = block["bbox_norm"]
    return (int(x0 * img_w), int(y0 * img_h), int(x1 * img_w), int(y1 * img_h))


def _glyph_mask(arr: np.ndarray, blocks: list[dict]) -> np.ndarray:
    """
    Build a mask (255 = remove) covering the glyph pixels inside each block.

    Within each bbox the background is estimated as the median gray value;
    pixels whose luminance differs from it by more than an adaptive threshold
    are treated as text strokes. This keeps the mask tight so the inpainter has
    real background to sample from, instead of blanking the whole rectangle.
    """
    import cv2

    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mask = np.zeros((h, w), dtype=np.uint8)

    for block in blocks:
        px0, py0, px1, py1 = _bbox_pixels(block, w, h)
        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(w, px1), min(h, py1)
        if px1 <= px0 or py1 <= py0:
            continue
        region = gray[py0:py1, px0:px1].astype(np.int32)
        bg = np.median(region)
        diff = np.abs(region - bg)
        thr = max(28.0, float(diff.std()))
        local = (diff > thr).astype(np.uint8) * 255
        mask[py0:py1, px0:px1] = np.maximum(mask[py0:py1, px0:px1], local)

    # Grow the mask to swallow anti-aliased glyph edges.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    return mask


def _inpaint_telea(image: Image.Image, blocks: list[dict]) -> Image.Image:
    import cv2

    arr = np.array(image.convert("RGB"))
    mask = _glyph_mask(arr, blocks)
    if mask.sum() == 0:
        return image.convert("RGB")
    out = cv2.inpaint(arr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(out)


def _inpaint_solid(image: Image.Image, blocks: list[dict]) -> Image.Image:
    """Sample a border pixel from each bbox and flood-fill the bbox."""
    img = image.copy().convert("RGB")
    arr = np.array(img)
    draw = ImageDraw.Draw(img)

    for block in blocks:
        px0, py0, px1, py1 = _bbox_pixels(block, image.width, image.height)
        if px0 >= px1 or py0 >= py1:
            continue
        sample_y = max(0, py0 - 2)
        sample_x = (px0 + px1) // 2
        bg_color = tuple(int(v) for v in arr[sample_y, sample_x])
        draw.rectangle([px0, py0, px1, py1], fill=bg_color)

    return img


def _inpaint_lama(image: Image.Image, blocks: list[dict]) -> Image.Image:
    """
    LaMa inpainting via lama-cleaner HTTP API.
    Requires: `lama-cleaner --model=lama --device=cpu --port=8080`
    Uses the same tight glyph mask as Telea.
    """
    import io

    import requests

    arr = np.array(image.convert("RGB"))
    mask_arr = _glyph_mask(arr, blocks)
    img = Image.fromarray(arr)
    mask = Image.fromarray(mask_arr)

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
