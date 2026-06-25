# V28.1 OCR Improvement Layer

This package improves OCR before the V28 Auto Patch Planner.

## What changed

V28 used an existing OCR report. V28.1 adds a dedicated OCR scanner that:

- scans only approved image/graphic regions;
- renders crops at high DPI;
- runs multiple preprocessing variants;
- normalizes OCR boxes back to PDF coordinates;
- merges duplicate detections across variants;
- saves debug crop images for review;
- outputs a planner-compatible JSON report.

No LLM/API call is used.

## 1) Generate improved OCR report

```powershell
python .\ocr_image_region_scanner_v28_1.py `
  "CXVIEW-SMART-AI-VIDEO-ANALYTICS-SOLUTION-PRICING.pdf" `
  --image-region-map "cxview_image_graphic_regions_v26.json" `
  --output-json "ocr_remaining_english_v28_1.json" `
  --debug-dir "ocr_debug_v28_1" `
  --dpi 320 `
  --lang en `
  --min-confidence 0.35
```

If PaddleOCR is missing:

```powershell
pip install paddleocr paddlepaddle
```

## 2) Generate patch map from improved OCR

```powershell
python .\auto_patch_planner_v28.py `
  --ocr-report "ocr_remaining_english_v28_1.json" `
  --image-region-map "cxview_image_graphic_regions_v26.json" `
  --style-presets "style_presets_v27.json" `
  --output-patch-map "cxview_patch_map_v28_1_auto_planned.json" `
  --report-json "v28_1_auto_patch_planner_report.json" `
  --base-recommended "ocr_report_dummy.pdf" `
  --include-needs-review
```

## 3) Render final PDF

```powershell
python .\pdf_image_region_only_patch_v27_style_presets.py `
  "ocr_report_dummy.pdf" `
  "output_final_v28_1_ocr_improved.pdf" `
  --patch-map "cxview_patch_map_v28_1_auto_planned.json" `
  --style-presets "style_presets_v27.json" `
  --source-pdf "CXVIEW-SMART-AI-VIDEO-ANALYTICS-SOLUTION-PRICING.pdf" `
  --image-region-map "cxview_image_graphic_regions_v26.json" `
  --font "fonts/NotoSans-Regular.ttf" `
  --font-bold "fonts/NotoSans-Bold.ttf" `
  --font-title "fonts/NotoSans-Bold.ttf" `
  --report-json "v28_1_render_report.json"
```

## Practical tips

- Use `--dpi 320` for small diagram text.
- Try `--dpi 380` only when OCR still misses tiny labels.
- Use `--pages 2,4-5` when debugging CXVIEW to save time.
- Open `ocr_debug_v28_1` to check detected boxes visually.
- The OCR report is only a candidate list. The planner still filters by approved image/graphic regions and known rules.
