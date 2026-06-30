# V29 Generic OCR Block Planner

This package contains `auto_patch_planner_v29_generic_block.py`.

Purpose:
- Generic OCR block planner for arbitrary image-heavy or hybrid PDFs.
- Groups OCR items into lines and blocks.
- Generates renderer-compatible patch map and full-page allowed regions.
- No Anthropic/API call.

Recommended local LONGMAN test:

```powershell
python .\auto_patch_planner_v29_generic_block.py `
  --ocr-report "longman_ocr_remaining_english.json" `
  --pdf "LONGMAN BFFB Renewable.pdf" `
  --output-patch-map "longman_patch_map_v29_generic_block.json" `
  --output-image-region-map "longman_image_regions_v29_generic_block.json" `
  --report-json "longman_planner_report_v29_generic_block.json" `
  --translation-mode source `
  --min-confidence 0.35 `
  --protect-top-band 105 `
  --fill sample
```

Render:

```powershell
python .\pdf_image_region_only_patch_v27_style_presets.py `
  "LONGMAN BFFB Renewable.pdf" `
  "longman_output_v29_generic_block_layout_test.pdf" `
  --patch-map "longman_patch_map_v29_generic_block.json" `
  --image-region-map "longman_image_regions_v29_generic_block.json" `
  --no-auto-image-rects `
  --font "fonts/NotoSans-Regular.ttf" `
  --font-bold "fonts/NotoSans-Bold.ttf" `
  --font-title "fonts/NotoSans-Bold.ttf" `
  --report-json "longman_render_report_v29_generic_block.json"
```

For actual translation, use:

```powershell
python .\auto_patch_planner_v29_generic_block.py `
  --ocr-report "longman_ocr_remaining_english.json" `
  --pdf "LONGMAN BFFB Renewable.pdf" `
  --output-patch-map "longman_patch_map_v29_generic_block_translated.json" `
  --output-image-region-map "longman_image_regions_v29_generic_block_translated.json" `
  --report-json "longman_planner_report_v29_generic_block_translated.json" `
  --translation-mode map_or_builtin `
  --translation-map "longman_translation_map.json" `
  --min-confidence 0.35 `
  --protect-top-band 105 `
  --fill sample
```

`translation_map` supports JSON `{ "source": "translation" }`, JSON `{"translations":[...]}`, or CSV with `source,translation` columns.
