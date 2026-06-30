# V29.3.1 QA Fit Cleanup

Purpose: fix layout overflow and risky translated OCR blocks after V29.3 real translation.

Important:
- No OCR call.
- No Anthropic/LLM/API call.
- It reuses `longman_patch_map_v29_3_real_translate.json` and only edits risky translations/layout fields.
- Use `pdf_image_region_only_patch_v27_1_fit_safe.py` to avoid long fallback text overflowing outside boxes.

## Step 1 - Cleanup translated patch map

```powershell
python .\qa_patch_map_v29_3_1_fit_cleanup.py `
  --patch-map "longman_patch_map_v29_3_real_translate.json" `
  --render-report "longman_render_report_v29_3_real_translate.json" `
  --output-patch-map "longman_patch_map_v29_3_1_qa_fit_cleanup.json" `
  --report-json "longman_qa_report_v29_3_1_fit_cleanup.json"
```

Expected: no API/token cost.

## Step 2 - Render with fit-safe renderer

```powershell
python .\pdf_image_region_only_patch_v27_1_fit_safe.py `
  "LONGMAN BFFB Renewable.pdf" `
  "longman_output_v29_3_1_qa_fit_cleanup.pdf" `
  --patch-map "longman_patch_map_v29_3_1_qa_fit_cleanup.json" `
  --image-region-map "longman_image_regions_v29_3_real_translate.json" `
  --no-auto-image-rects `
  --font "fonts/NotoSans-Regular.ttf" `
  --font-bold "fonts/NotoSans-Bold.ttf" `
  --font-title "fonts/NotoSans-Bold.ttf" `
  --report-json "longman_render_report_v29_3_1_qa_fit_cleanup.json"
```

Expected:
- `applied = 49`
- `skipped_outside_image_or_graphic_region = 0`
- `skipped_invalid = 0`
- `fit_fallbacks` should be 0 or lower than V29.3.

## Step 3 - Check report

```powershell
@'
import json

for fp in [
    "longman_qa_report_v29_3_1_fit_cleanup.json",
    "longman_render_report_v29_3_1_qa_fit_cleanup.json"
]:
    print("\n===", fp, "===")
    d = json.load(open(fp, encoding="utf-8"))
    for k in [
        "patches_total", "edits_total", "risk_items_total",
        "applied", "fit_fallbacks", "skipped_outside_image_or_graphic_region", "skipped_invalid"
    ]:
        if k in d:
            print(k, "=", d[k])
'@ | python -
```

## What V29.3.1 fixes

- Page 6 map labels: line breaks + smaller font.
- Page 9 Indonesia card: reorder from OCR-like `Power Station Project Indonesia 3×330MW` to `Dự án Nhà máy Điện 3×330MW tại Indonesia`.
- Page 11 Algeria card: repairs incomplete OCR tail.
- Page 12 Jiangsu card: compact text to prevent horizontal overflow.
- Large paragraph pages 2/3/5: adds line breaks after headings for better readability.
- Fit-safe renderer: avoids drawing long fallback text outside the patch rectangle.

## Files to upload for QA

```text
longman_output_v29_3_1_qa_fit_cleanup.pdf
longman_qa_report_v29_3_1_fit_cleanup.json
longman_render_report_v29_3_1_qa_fit_cleanup.json
```
