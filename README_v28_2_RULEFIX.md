# v28.2 Auto Patch Planner Rule Fix

This package continues the v28 pipeline:

1. OCR only approved image/graphic regions.
2. Auto Patch Planner creates a v27-compatible patch map.
3. V27 Style Preset Renderer applies patches only inside approved regions.

## What v28.2 fixes

- Page 2 pricing badges no longer depend on OCR row order.
  OCR is used as evidence only; final badge positions use verified row slots.
- Generic OCR review candidates are noise-filtered and actionable-only.
- Page 5 diagram adds extra translations for:
  - GROUP A / GROUP B / GROUP C -> NHÓM A / NHÓM B / NHÓM C
  - From 40 cameras -> từ 40 cam

## Run OCR first

```powershell
python .\ocr_image_region_scanner_v28_1_4_rapidocr_numpyfix.py `
  "CXVIEW-SMART-AI-VIDEO-ANALYTICS-SOLUTION-PRICING.pdf" `
  --image-region-map "cxview_image_graphic_regions_v26.json" `
  --output-json "ocr_remaining_english_v28_1_4.json" `
  --debug-dir "ocr_debug_v28_1_4" `
  --dpi 420 `
  --lang en `
  --pages 2,4-5 `
  --min-confidence 0.05 `
  --engine rapidocr
```

## Run planner v28.2

```powershell
python .\auto_patch_planner_v28_2_rulefix.py `
  --ocr-report "ocr_remaining_english_v28_1_4.json" `
  --image-region-map "cxview_image_graphic_regions_v26.json" `
  --style-presets "style_presets_v27.json" `
  --output-patch-map "cxview_patch_map_v28_2_rulefix.json" `
  --report-json "v28_2_rulefix_report.json" `
  --base-recommended "ocr_report_dummy.pdf" `
  --include-needs-review
```

Expected planner summary on the provided sample:

```text
planned_patches_total=26
auto_approved_enabled=26
low_confidence_patch_count=0
needs_review_candidates_count=0
```

## Render with v27 Style Preset Engine

```powershell
python .\pdf_image_region_only_patch_v27_style_presets.py `
  "ocr_report_dummy.pdf" `
  "output_final_v28_2_rulefix.pdf" `
  --patch-map "cxview_patch_map_v28_2_rulefix.json" `
  --style-presets "style_presets_v27.json" `
  --source-pdf "CXVIEW-SMART-AI-VIDEO-ANALYTICS-SOLUTION-PRICING.pdf" `
  --image-region-map "cxview_image_graphic_regions_v26.json" `
  --font "fonts/NotoSans-Regular.ttf" `
  --font-bold "fonts/NotoSans-Bold.ttf" `
  --font-title "fonts/NotoSans-Bold.ttf" `
  --report-json "v28_2_render_report.json"
```
