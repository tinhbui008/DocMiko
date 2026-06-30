# V29.3 Generic Block Real Translate

This version keeps the V29.2.1 safe OCR cleanup and adds real LLM translation through Anthropic.

## Setup

Create `.env` in your project root:

```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
ANTHROPIC_MODEL=claude-3-5-haiku-latest
```

You can change the model with `--anthropic-model`.

## Build translated patch map

```powershell
python .\auto_patch_planner_v29_3_generic_block_real_translate.py `
  --ocr-report "longman_ocr_remaining_english.json" `
  --pdf "LONGMAN BFFB Renewable.pdf" `
  --output-patch-map "longman_patch_map_v29_3_real_translate.json" `
  --output-image-region-map "longman_image_regions_v29_3_real_translate.json" `
  --report-json "longman_planner_report_v29_3_real_translate.json" `
  --translation-mode llm `
  --source-lang en `
  --target-lang vi `
  --translation-cache ".translation_cache_v29_3.jsonl" `
  --min-confidence 0.35 `
  --protect-top-band 105 `
  --fill sample
```

## Render translated PDF

```powershell
python .\pdf_image_region_only_patch_v27_style_presets.py `
  "LONGMAN BFFB Renewable.pdf" `
  "longman_output_v29_3_real_translate.pdf" `
  --patch-map "longman_patch_map_v29_3_real_translate.json" `
  --image-region-map "longman_image_regions_v29_3_real_translate.json" `
  --no-auto-image-rects `
  --font "fonts/NotoSans-Regular.ttf" `
  --font-bold "fonts/NotoSans-Bold.ttf" `
  --font-title "fonts/NotoSans-Bold.ttf" `
  --report-json "longman_render_report_v29_3_real_translate.json"
```

## Notes

- `--translation-mode llm` calls Anthropic and consumes API credits.
- Translation cache prevents re-calling the same block text.
- Translation map is checked before LLM, so you can override specific blocks.
- Renderer does not call API; only the planner calls API in `llm` mode.
