"""Fix: add claude dispatch branch in full_ocr_rebuild_pdf_v16 page loop."""
from pathlib import Path

f = Path("pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py")
t = f.read_text(encoding="utf-8")

# The old dispatch only has paddle vs tesseract (else).
# We need to add claude as the first branch.
old = '''            if ocr_engine == "paddle":
                img, zoom, blocks = ocr_page_to_all_line_blocks_paddle_v16('''

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
                img, zoom, blocks = ocr_page_to_all_line_blocks_paddle_v16('''

if 'ocr_engine == "claude"' in t:
    print("SKIP: claude dispatch already present")
elif old in t:
    t = t.replace(old, new)
    f.write_text(t, encoding="utf-8")
    print("OK: claude dispatch added")
else:
    print("ERROR: dispatch marker not found — check manually")
    # Show context
    idx = t.find('ocr_page_to_all_line_blocks_paddle_v16(')
    if idx >= 0:
        print("Context:", repr(t[idx-200:idx+50]))
