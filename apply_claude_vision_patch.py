#!/usr/bin/env python3
"""
Minimal patch to add Claude Vision OCR engine to existing
pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py

Usage:
    python apply_claude_vision_patch.py pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py
"""
import re, sys, shutil
from pathlib import Path

def patch(filepath: str):
    src = Path(filepath)
    if not src.exists():
        print(f"ERROR: {filepath} not found")
        sys.exit(1)

    text = src.read_text(encoding="utf-8")
    changed = 0

    # ── 1. Add 'claude' to choices ────────────────────────────────────────
    old1 = 'choices=["tesseract", "paddle"], help="Engine for --full-ocr-rebuild"'
    new1 = 'choices=["tesseract", "paddle", "claude"], help="Engine for --full-ocr-rebuild. \'claude\' uses Claude Vision API (best quality)"'
    if old1 in text:
        text = text.replace(old1, new1)
        changed += 1
        print("✓ Patched: added 'claude' to --full-ocr-engine choices")
    else:
        print("⚠ Skip: choices line not found (may already be patched)")

    # ── 2. Add --claude-vision-model and --claude-vision-dpi args ─────────
    marker = '    p.add_argument("--full-ocr-min-conf"'
    new_args = """    p.add_argument("--claude-vision-model", default="claude-haiku-4-5",
                   help="Claude model for --full-ocr-engine claude (default: claude-haiku-4-5)")
    p.add_argument("--claude-vision-dpi", type=int, default=200,
                   help="DPI for Claude Vision page rendering (default: 200)")
    """
    if '--claude-vision-model' not in text and marker in text:
        text = text.replace(marker, new_args + marker)
        changed += 1
        print("✓ Patched: added --claude-vision-model and --claude-vision-dpi args")
    else:
        print("⚠ Skip: vision args already present or marker not found")

    # ── 3. Add Claude Vision helper functions + engine dispatch ───────────
    vision_code = '''
# ── Claude Vision OCR helpers ──────────────────────────────────────────────

def _claude_vision_page_to_b64(page, dpi: int = 200) -> str:
    import base64
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def _claude_vision_call(b64_image, api_key, base_url, model,
                         anthropic_version="2023-06-01", max_tokens=4096, timeout=120):
    import urllib.request as _ur, json as _json
    prompt = (
        "You are a precise OCR engine for PDF slides. "
        "Analyze this PDF page image and extract ALL visible text including "
        "text inside colored buttons, badges, diagrams, and raster images. "
        "Return ONLY a JSON array. Each element: "
        \'{"text":"...","x0":0.0,"y0":0.0,"x1":1.0,"y1":1.0,"role":"title|body|label","bold":false,"font_size_approx":12.0} \'
        "where x0/y0/x1/y1 are fractions (0.0-1.0) of image dimensions. "
        "Skip QR codes. No markdown, no explanation, ONLY the JSON array."
    )
    body = {
        "model": model, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_image}},
            {"type": "text", "text": prompt},
        ]}],
    }
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": anthropic_version,
               "content-type": "application/json", "accept": "application/json"}
    data = _json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = _ur.Request(url, data=data, headers=headers, method="POST")
    with _ur.urlopen(req, timeout=timeout) as resp:
        result = _json.loads(resp.read().decode("utf-8"))
    for block in result.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", ""))
    raise ValueError(f"No text in Claude response: {result}")


def _claude_vision_parse_blocks(raw_response, page, page_index, min_chars=2):
    import re as _re, json as _json
    text = raw_response.strip()
    text = _re.sub(r"^```(?:json)?\\s*", "", text, flags=_re.I)
    text = _re.sub(r"\\s*```$", "", text).strip()
    try:
        s, e = text.find("["), text.rfind("]")
        items = _json.loads(text[s:e+1]) if s >= 0 and e > s else _json.loads(text)
    except Exception as ex:
        print(f"      Claude Vision JSON parse error: {ex}")
        return []
    pw = float(page.rect.width); ph = float(page.rect.height); pcx = pw / 2.0
    blocks = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = _re.sub(r"\\s+", " ", normalize_special_chars(str(item.get("text","") or ""))).strip()
        if not should_keep_full_ocr_text_v16(raw, min_chars=min_chars):
            continue
        try:
            x0f,y0f,x1f,y1f = (max(0.,min(1.,float(item.get(k,d)))) for k,d in [("x0",0),("y0",0),("x1",1),("y1",1)])
        except Exception:
            continue
        if x1f<=x0f or y1f<=y0f:
            continue
        bbox = (x0f*pw, y0f*ph, x1f*pw, y1f*ph)
        ht = max(1., bbox[3]-bbox[1]); wt = max(1., bbox[2]-bbox[0])
        cr = str(item.get("role","")).lower()
        role = cr if cr in {"title","body","label","header","footer"} else ("title" if ht>=20 else ("body" if len(raw)>100 else "label"))
        fs = float(item.get("font_size_approx",0))
        if fs < 3.: fs = _estimate_font_size_v17(ht, raw, wt)
        bcx = (bbox[0]+bbox[2])/2.
        align = "center" if abs(bcx-pcx)<pw*0.10 and wt<pw*0.5 else ("right" if bbox[2]>pw*0.75 and bbox[0]>pw*0.45 else "left")
        blocks.append({
            "id": f"claudevision_p{page_index}_{len(blocks)}",
            "page_index": page_index, "text": raw, "bbox": bbox, "px_bbox": None,
            "confidence": 0.95, "role": role, "align": align, "font_size": fs,
            "bold": bool(item.get("bold",False)), "bg_color": (1.,1.,1.),
            "ocr_engine": "claude_vision",
        })
    return blocks


def ocr_page_to_all_line_blocks_claude_vision_v16(page, page_index, api_key, base_url,
        model, dpi=200, min_chars=2, anthropic_version="2023-06-01", timeout=120):
    from PIL import Image as _PI
    zoom = dpi / 72.
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = _PI.frombytes("RGB", [pix.width, pix.height], pix.samples)
    b64 = _claude_vision_page_to_b64(page, dpi=dpi)
    print(f"        Claude Vision API call (page {page_index+1}) ...")
    try:
        raw = _claude_vision_call(b64, api_key, base_url, model,
                                   anthropic_version=anthropic_version, timeout=timeout)
    except Exception as e:
        print(f"        Claude Vision error: {e}")
        return img, zoom, []
    blocks = _claude_vision_parse_blocks(raw, page, page_index, min_chars=min_chars)
    pw = float(page.rect.width); ph = float(page.rect.height)
    for b in blocks:
        px = (max(0,int(b["bbox"][0]*zoom)-2), max(0,int(b["bbox"][1]*zoom)-2),
              int(b["bbox"][2]*zoom)+2, int(b["bbox"][3]*zoom)+2)
        b["px_bbox"] = px
        b["bg_color"] = sample_bg_color_v17(img, px)
    print(f"        Claude Vision: {len(blocks)} blocks extracted")
    return img, zoom, blocks

'''

    insert_marker = "def full_ocr_rebuild_pdf_v16("
    if "ocr_page_to_all_line_blocks_claude_vision_v16" not in text and insert_marker in text:
        text = text.replace(insert_marker, vision_code + insert_marker)
        changed += 1
        print("✓ Patched: added Claude Vision OCR helper functions")
    else:
        print("⚠ Skip: Claude Vision helpers already present or marker not found")

    # ── 4. Add claude dispatch inside full_ocr_rebuild_pdf_v16 ────────────
    # Add claude params to function signature
    old_sig = "    tesseract_cmd: Optional[str] = None,\n):"
    new_sig = """    tesseract_cmd: Optional[str] = None,
    claude_vision_model: str = "claude-haiku-4-5",
    claude_vision_dpi: int = 200,
):"""
    if old_sig in text and "claude_vision_model" not in text:
        text = text.replace(old_sig, new_sig)
        changed += 1
        print("✓ Patched: added claude params to full_ocr_rebuild_pdf_v16 signature")

    # Add claude engine init block
    old_init = "    if ocr_engine == \"tesseract\":\n        _configure_tesseract_cmd(tesseract_cmd)"
    new_init = """    if ocr_engine == "tesseract":
        _configure_tesseract_cmd(tesseract_cmd)

    claude_api_key = ""; claude_base_url = "https://api.anthropic.com"; claude_version = "2023-06-01"
    if ocr_engine == "claude":
        claude_api_key  = os.getenv("LLM_API_KEY", "").strip()
        claude_base_url = os.getenv("LLM_BASE_URL", "https://api.anthropic.com").strip().rstrip("/")
        claude_version  = os.getenv("ANTHROPIC_VERSION", "2023-06-01").strip()
        if not claude_api_key:
            raise ValueError("Claude Vision requires LLM_API_KEY in .env")
        dpi = claude_vision_dpi
        print(f"      Claude Vision OCR: model={claude_vision_model} dpi={dpi}")"""
    if old_init in text and 'ocr_engine == "claude"' not in text:
        text = text.replace(old_init, new_init)
        changed += 1
        print("✓ Patched: added claude engine init in full_ocr_rebuild_pdf_v16")

    # Add claude dispatch in the per-page loop
    old_dispatch = '            if ocr_engine == "paddle":\n                img, zoom, blocks = ocr_page_to_all_line_blocks_paddle_v16('
    new_dispatch = '''            if ocr_engine == "claude":
                img, zoom, blocks = ocr_page_to_all_line_blocks_claude_vision_v16(
                    page=src_page, page_index=page_index,
                    api_key=claude_api_key, base_url=claude_base_url,
                    model=claude_vision_model, dpi=dpi, min_chars=min_chars,
                    anthropic_version=claude_version,
                )
            elif ocr_engine == "paddle":
                img, zoom, blocks = ocr_page_to_all_line_blocks_paddle_v16('''
    if old_dispatch in text and 'ocr_engine == "claude"' not in text:
        text = text.replace(old_dispatch, new_dispatch)
        changed += 1
        print("✓ Patched: added claude dispatch in page loop")

    # ── 5. Update main_v6 call to pass claude args ────────────────────────
    old_call = "            tesseract_cmd=args.tesseract_cmd,\n        )\n        if args.preview_dir:\n            print(f\"[Full OCR rebuild] Render preview PNGs:"
    new_call = """            tesseract_cmd=args.tesseract_cmd,
            claude_vision_model=getattr(args, "claude_vision_model", "claude-haiku-4-5"),
            claude_vision_dpi=getattr(args, "claude_vision_dpi", 200),
        )
        if args.preview_dir:
            print(f"[Full OCR rebuild] Render preview PNGs:"""
    if old_call in text and "claude_vision_model=getattr" not in text:
        text = text.replace(old_call, new_call)
        changed += 1
        print("✓ Patched: updated main_v6 call with claude args")

    if changed == 0:
        print("Nothing changed — file may already be fully patched.")
        return

    # Backup and write
    bak = str(src) + ".bak"
    shutil.copy2(str(src), bak)
    src.write_text(text, encoding="utf-8")
    print(f"\nDone: {changed} patches applied.")
    print(f"Backup saved to: {bak}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python apply_claude_vision_patch.py <path_to_pdf_generator.py>")
        sys.exit(1)
    patch(sys.argv[1])
