# V29.1 Generic OCR Block Planner Refine

Purpose:
- Clean OCR noise inside block-level patches before translation.
- Preserve partner/logo grid pages, only patching their page title by default.
- Keep the renderer unchanged: use `pdf_image_region_only_patch_v27_style_presets.py`.

Recommended flow:

```powershell
python .\auto_patch_planner_v29_1_generic_block_refine.py `
  --ocr-report "longman_ocr_remaining_english.json" `
  --pdf "LONGMAN BFFB Renewable.pdf" `
  --output-patch-map "longman_patch_map_v29_1_generic_block_refine.json" `
  --output-image-region-map "longman_image_regions_v29_1_generic_block_refine.json" `
  --report-json "longman_planner_report_v29_1_generic_block_refine.json" `
  --translation-mode source `
  --min-confidence 0.35 `
  --protect-top-band 105 `
  --fill sample
```

Render:

```powershell
python .\pdf_image_region_only_patch_v27_style_presets.py `
  "LONGMAN BFFB Renewable.pdf" `
  "longman_output_v29_1_generic_block_refine_layout_test.pdf" `
  --patch-map "longman_patch_map_v29_1_generic_block_refine.json" `
  --image-region-map "longman_image_regions_v29_1_generic_block_refine.json" `
  --no-auto-image-rects `
  --font "fonts/NotoSans-Regular.ttf" `
  --font-bold "fonts/NotoSans-Bold.ttf" `
  --font-title "fonts/NotoSans-Bold.ttf" `
  --report-json "longman_render_report_v29_1_generic_block_refine.json"
```

Notes:
- This is still `translation-mode source` for layout/OCR cleanup testing.
- Do not attach real Vietnamese translation until this layout test is acceptable.
- Partner pages are detected generically via `Partners / ...` title plus logo/card density.
