"""Add --cv-region-patch args and hook to existing file."""
from pathlib import Path
import re

f = Path("pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py")
t = f.read_text(encoding="utf-8")

# ── 1. Add args after --v263-debug-regions ────────────────────────────────
marker = '    p.add_argument("--v263-debug-regions", action="store_true",'
insert = '''    p.add_argument("--cv-region-patch", action="store_true",
                   help="Use Claude Vision to read/translate text inside image regions")
    p.add_argument("--cv-region-map", default=None,
                   help="Image region map JSON for Claude Vision region patch")
    p.add_argument("--cv-model", default="claude-haiku-4-5",
                   help="Claude model for --cv-region-patch")
    p.add_argument("--cv-dpi", type=int, default=220,
                   help="DPI for region rendering in --cv-region-patch")
    p.add_argument("--cv-debug", action="store_true",
                   help="Draw green outlines around processed image regions")
    '''
if "--cv-region-patch" not in t and marker in t:
    t = t.replace(marker, insert + marker)
    print("OK: --cv-region-patch args added")
else:
    print("SKIP: args already present or marker not found")

# ── 2. Add Claude Vision helper functions if not present ──────────────────
if "_cvr_render_region_b64" not in t:
    helpers = '''
# ── Claude Vision Region Patch helpers ────────────────────────────────────

def _cvr_render_region_b64(page, region, dpi=220):
    import base64
    zoom = dpi / 72.0
    clip = fitz.Rect(max(0,region.x0),max(0,region.y0),
                     min(page.rect.width,region.x1),min(page.rect.height,region.y1))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom,zoom), clip=clip, alpha=False)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8"), zoom

def _cvr_call_claude(b64_image, region, api_key, base_url, model,
                     anthropic_version="2023-06-01", timeout=60):
    prompt = ("Extract all visible text from this PDF region. "
              "Return ONLY a JSON array: [{\"text\":\"...\",\"x0\":0.0,\"y0\":0.0,\"x1\":1.0,\"y1\":1.0}] "
              "where coords are fractions 0.0-1.0 of this image. Include button/badge text. Skip QR codes.")
    body = {"model": model, "max_tokens": 2048,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_image}},
                {"type": "text", "text": prompt}]}]}
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": anthropic_version,
               "content-type": "application/json", "accept": "application/json"}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"        Claude Vision error: {e}"); return []
    raw = ""
    for block in result.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            raw = str(block.get("text", "")); break
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\\s*", "", raw, flags=re.I)
    raw = re.sub(r"\\s*```$", "", raw).strip()
    try:
        s, e = raw.find("["), raw.rfind("]")
        items = json.loads(raw[s:e+1]) if s>=0 and e>s else json.loads(raw)
    except Exception as ex:
        try:
            rep = raw.rstrip().rstrip(",")
            rep += "}" * max(0, rep.count("{")-rep.count("}")) + "]"
            s = rep.find("[")
            items = json.loads(rep[s:]) if s>=0 else []
        except Exception:
            print(f"        Claude Vision JSON error: {ex}"); return []
    rw, rh = region.width, region.height
    results = []
    for item in items:
        if not isinstance(item, dict): continue
        text = re.sub(r"\\s+", " ", str(item.get("text","") or "")).strip()
        if not text or len(text) < 2: continue
        try:
            x0f=max(0.,min(1.,float(item.get("x0",0.)))); y0f=max(0.,min(1.,float(item.get("y0",0.))))
            x1f=max(0.,min(1.,float(item.get("x1",1.)))); y1f=max(0.,min(1.,float(item.get("y1",1.))))
        except: continue
        if x1f<=x0f or y1f<=y0f: continue
        results.append({"text": text, "bbox": (
            region.x0+x0f*rw, region.y0+y0f*rh,
            region.x0+x1f*rw, region.y0+y1f*rh)})
    return results

def _cvr_translate_blocks(blocks, translator, source_lang, target_lang, glossary, batch_size=20):
    pending = []
    for i, b in enumerate(blocks):
        class _P:
            def __init__(self, blk):
                self.original_text=blk["text"]; self.role="label"
                self.font_size=max(4.,(blk["bbox"][3]-blk["bbox"][1])*.72)
                self.bbox=blk["bbox"]; self.lines=[type("L",(),{"text":blk["text"]})()]
        ov = compact_override_for_block(_P(b), target_lang)
        if ov: b["translated"] = sanitize_text_v9(ov, b["text"], "label")
        else: pending.append((i, b))
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start+batch_size]
        items = [{"id": str(idx), "role": "label", "text": b["text"],
                  "max_chars": 60, "max_lines": 3,
                  "instruction": "Translate to target language. Keep brand names."}
                 for idx, b in batch]
        res = translator.translate_batch(items, source_lang, target_lang, glossary)
        by_id = {x["id"]: x.get("translated","") for x in res}
        for idx, b in batch:
            b["translated"] = sanitize_text_v9(by_id.get(str(idx), b["text"]), b["text"], "label")
    return blocks

def apply_claude_vision_region_patch(
    source_pdf, translated_pdf, output_pdf, image_region_map, translator,
    source_lang="auto", target_lang="vi", glossary=None,
    api_key="", base_url="https://api.anthropic.com", model="claude-haiku-4-5",
    anthropic_version="2023-06-01", dpi=220,
    font_regular=None, font_bold=None, font_title=None, batch_size=20, debug=False,
):
    import shutil as _sh
    manual = _v263_load_region_map(image_region_map)
    auto   = _v263_auto_image_rects(source_pdf, min_area=800.0)
    all_r: Dict[int, List] = {}
    for m in (auto, manual):
        for pn, rects in m.items(): all_r.setdefault(pn,[]).extend(rects)
    if not all_r:
        print("      CV Region Patch: no regions"); return 0
    pdf = fitz.open(translated_pdf)
    reg  = _v263_first_existing(font_regular, "fonts/NotoSans-Regular.ttf")
    bold = _v263_first_existing(font_bold, "fonts/NotoSans-Bold.ttf", reg)
    total = 0
    for page_index, page in enumerate(pdf):
        pn = page_index + 1
        regions = all_r.get(pn, [])
        if not regions: continue
        print(f"      CV Region page {pn}: {len(regions)} regions")
        all_blocks = []
        for region in regions:
            if region.width < 20 or region.height < 10: continue
            if debug: page.draw_rect(region, color=(0,.6,0), width=1., overlay=True)
            b64, _ = _cvr_render_region_b64(page, region, dpi=dpi)
            ext = _cvr_call_claude(b64, region, api_key, base_url, model, anthropic_version)
            print(f"        [{region.x0:.0f},{region.y0:.0f}]: {len(ext)} blocks")
            all_blocks.extend(ext)
        if not all_blocks: continue
        all_blocks = _cvr_translate_blocks(all_blocks, translator, source_lang, target_lang, glossary, batch_size)
        for b in all_blocks:
            src_t = b.get("text",""); vi_t = b.get("translated","")
            if not vi_t or text_is_same(src_t, vi_t): continue
            rect = fitz.Rect(*b["bbox"])
            if rect.width<=2 or rect.height<=2: continue
            bg = _v263_sample_bg(page, rect)
            lum = _v263_luminance(bg); sat = max(bg)-min(bg)
            page.draw_rect(rect, color=None, fill=bg, overlay=True)
            color = (1.,1.,1.) if (lum<.50 or sat>.15) else (.06,.06,.10)
            fontfile = bold or reg; fontname = "FCVRBold"
            if not fontfile: fontname = "helv"
            ht = max(1., rect.height); wt = max(1., rect.width)
            fs = _estimate_font_size_v17(ht, vi_t, wt)
            size = fs; min_s = max(3.5, fs*.55)
            while size >= min_s:
                try:
                    rc = page.insert_textbox(rect, vi_t, fontsize=size, fontname=fontname,
                                             fontfile=fontfile, color=color,
                                             align=fitz.TEXT_ALIGN_CENTER, overlay=True)
                    if rc >= 0: break
                except: pass
                size -= .2
            total += 1
    out_path = Path(output_pdf).resolve(); tr_path = Path(translated_pdf).resolve()
    if tr_path == out_path:
        tmp = str(out_path.parent/(out_path.stem+"_cvrpatch_tmp.pdf"))
        pdf.save(tmp, garbage=4, deflate=True); pdf.close(); _sh.move(tmp, str(out_path))
    else:
        pdf.save(output_pdf, garbage=4, deflate=True); pdf.close()
    print(f"      CV Region Patch: {total} patches → {output_pdf}")
    return total

'''
    # Insert before build_arg_parser_v6
    marker2 = "def build_arg_parser_v6()"
    if marker2 in t:
        t = t.replace(marker2, helpers + marker2)
        print("OK: CV region helpers added")
    else:
        print("ERROR: build_arg_parser_v6 not found")

# ── 3. Add main_v6 hook before V26.3 step ────────────────────────────────
hook = '''    # Claude Vision Region Patch hook.
    if getattr(args, "cv_region_patch", False):
        _cv_out = args.output_pdf if not getattr(args,"ocr_output",None) else args.ocr_output
        _cv_key = os.getenv("LLM_API_KEY","").strip()
        _cv_url = os.getenv("LLM_BASE_URL","https://api.anthropic.com").strip().rstrip("/")
        _cv_ver = os.getenv("ANTHROPIC_VERSION","2023-06-01").strip()
        if not _cv_key:
            print("WARNING: --cv-region-patch requires LLM_API_KEY — skipping")
        else:
            print(f"[CV Region Patch] model={getattr(args,'cv_model','claude-haiku-4-5')}")
            apply_claude_vision_region_patch(
                source_pdf=args.input_pdf, translated_pdf=_cv_out, output_pdf=_cv_out,
                image_region_map=getattr(args,"cv_region_map",None), translator=translator,
                source_lang=args.source, target_lang=args.target, glossary=glossary,
                api_key=_cv_key, base_url=_cv_url,
                model=getattr(args,"cv_model","claude-haiku-4-5"),
                anthropic_version=_cv_ver,
                dpi=getattr(args,"cv_dpi",220),
                font_regular=args.font, font_bold=args.font_bold, font_title=args.font_title,
                batch_size=args.batch_size, debug=getattr(args,"cv_debug",False),
            )
    '''
v263_marker = "    # V26.3 inline region patch"
if "cv_region_patch" not in t and v263_marker in t:
    t = t.replace(v263_marker, hook + v263_marker)
    print("OK: CV region hook added in main_v6")
else:
    print("SKIP: hook already present or V26.3 marker not found")

f.write_text(t, encoding="utf-8")
print("Done.")
