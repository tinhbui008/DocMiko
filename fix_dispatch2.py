from pathlib import Path

f = Path("pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py")
t = f.read_text(encoding="utf-8")

old = '''            if ocr_engine == "paddle":
                img, zoom, blocks = ocr_page_to_all_line_blocks_paddle_v16(
                    page=src_page,
                    page_index=page_index,
                    engine=paddle_engine,
                    dpi=dpi,
                    min_conf=min_conf,
                    min_chars=min_chars,
                    temp_dir=Path(temp_dir_obj.name),
                )
            else:
                tess_lang = "eng" if ocr_lang == "en" else ocr_lang
                img, zoom, blocks = ocr_page_to_all_line_blocks_tesseract_v16(
                    page=src_page,
                    page_index=page_index,
                    dpi=dpi,
                    lang=tess_lang,
                    min_conf=min_conf,
                    min_chars=min_chars,
                )'''

new = '''            if ocr_engine == "claude":
                img, zoom, blocks = ocr_page_to_all_line_blocks_claude_vision_v16(
                    page=src_page,
                    page_index=page_index,
                    api_key=claude_api_key,
                    base_url=claude_base_url,
                    model=claude_vision_model,
                    dpi=dpi,
                    min_chars=min_chars,
                    anthropic_version=claude_version,
                )
            elif ocr_engine == "paddle":
                img, zoom, blocks = ocr_page_to_all_line_blocks_paddle_v16(
                    page=src_page,
                    page_index=page_index,
                    engine=paddle_engine,
                    dpi=dpi,
                    min_conf=min_conf,
                    min_chars=min_chars,
                    temp_dir=Path(temp_dir_obj.name),
                )
            else:
                tess_lang = "eng" if ocr_lang == "en" else ocr_lang
                img, zoom, blocks = ocr_page_to_all_line_blocks_tesseract_v16(
                    page=src_page,
                    page_index=page_index,
                    dpi=dpi,
                    lang=tess_lang,
                    min_conf=min_conf,
                    min_chars=min_chars,
                )'''

if old in t:
    t = t.replace(old, new)
    f.write_text(t, encoding="utf-8")
    print("OK: claude dispatch added in page loop")
else:
    print("ERROR: pattern not found")
    # Show what's there
    idx = t.find('ocr_page_to_all_line_blocks_paddle_v16(')
    while idx >= 0:
        print(f"  Found at pos {idx}:", repr(t[idx-100:idx+50]))
        idx = t.find('ocr_page_to_all_line_blocks_paddle_v16(', idx+1)
