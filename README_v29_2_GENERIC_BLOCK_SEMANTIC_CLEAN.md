# V29.2 Generic OCR Block Semantic Clean

Purpose:
- Continue from V29.1.
- Focus on cleaning OCR noise before real translation.
- Preserve partner/logo grid pages.
- Reduce duplicate OCR phrases and noisy tail runs.
- Still no API call. `translation-mode=source` is for layout test.

Run planner locally:

```powershell
python .\auto_patch_planner_v29_2_generic_block_semantic_clean.py `
  --ocr-report "longman_ocr_remaining_english.json" `
  --pdf "LONGMAN BFFB Renewable.pdf" `
  --output-patch-map "longman_patch_map_v29_2_generic_block_semantic_clean.json" `
  --output-image-region-map "longman_image_regions_v29_2_generic_block_semantic_clean.json" `
  --report-json "longman_planner_report_v29_2_generic_block_semantic_clean.json" `
  --translation-mode source `
  --min-confidence 0.35 `
  --protect-top-band 105 `
  --fill sample
```

Render:

```powershell
python .\pdf_image_region_only_patch_v27_style_presets.py `
  "LONGMAN BFFB Renewable.pdf" `
  "longman_output_v29_2_generic_block_semantic_clean_layout_test.pdf" `
  --patch-map "longman_patch_map_v29_2_generic_block_semantic_clean.json" `
  --image-region-map "longman_image_regions_v29_2_generic_block_semantic_clean.json" `
  --no-auto-image-rects `
  --font "fonts/NotoSans-Regular.ttf" `
  --font-bold "fonts/NotoSans-Bold.ttf" `
  --font-title "fonts/NotoSans-Bold.ttf" `
  --report-json "longman_render_report_v29_2_generic_block_semantic_clean.json"
```

Check reports:

```powershell
@'
import json
for fp in [
    "longman_planner_report_v29_2_generic_block_semantic_clean.json",
    "longman_render_report_v29_2_generic_block_semantic_clean.json"
]:
    print("\n===", fp, "===")
    d=json.load(open(fp,encoding="utf-8"))
    for k in ["ocr_items_total","ocr_items_kept","ocr_items_rejected","lines_total","blocks_total","patches_total","rejected_blocks_total","partner_logo_pages","applied","fit_fallbacks","skipped_outside_image_or_graphic_region","skipped_invalid"]:
        if k in d:
            print(k,"=",d[k])
'@ | python -
```

Notes:
- This version is still a layout/cleanup test. Do not use real translation until OCR blocks are visually acceptable.
- For production translation, run the next version with `translation_map` or LLM translation after V29.2 visual QA passes.
