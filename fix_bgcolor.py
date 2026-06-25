from pathlib import Path
f = Path("pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py")
t = f.read_text(encoding="utf-8")
t = t.replace("sample_bg_color_v17(img, px)", "sample_bg_color_v10(img, px)")
f.write_text(t, encoding="utf-8")
print("OK")
