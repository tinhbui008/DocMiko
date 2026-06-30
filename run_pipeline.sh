#!/usr/bin/env bash
# ============================================================
# Mikotech PDF Translation Pipeline - v28.2 RULEFIX (Linux / Docker)
# Mirrors README_v28_2_RULEFIX.md:
#   STEP 1: OCR approved image/graphic regions (RapidOCR)
#   STEP 2: Auto Patch Planner -> v27-compatible patch map
#   STEP 3: v27 Style Preset Renderer applies patches inside regions
#
# IMPORTANT - this v28.2 pipeline is tuned for the CXVIEW sample deck:
#   * region map (cxview_image_graphic_regions_v26.json) holds manually
#     approved coordinates for specific CXVIEW pages (2, 4-5);
#   * the planner only supports the "cxview" rule plugin;
#   * the renderer patches ONLY image regions on top of a pre-translated
#     base PDF (BASE_PDF). It does NOT re-run text-layer translation.
#   To translate a different deck you must regenerate region maps + base.
# ============================================================
# Usage (inside container):
#   run_pipeline.sh [SOURCE_PDF] [OUTPUT_PDF] [BASE_PDF]
set -euo pipefail

# --- Inputs (override via args; defaults match README) ---
SOURCE_PDF="${1:-CXVIEW-SMART-AI-VIDEO-ANALYTICS-SOLUTION-PRICING.pdf}"   # original deck
OUTPUT_PDF="${2:-/app/output/output_final_v28_2_rulefix.pdf}"             # final result
BASE_PDF="${3:-ocr_report_dummy.pdf}"                                     # pre-translated text-layer base

# --- Fixed CXVIEW assets (shipped in the repo) ---
REGION_MAP="cxview_image_graphic_regions_v26.json"
STYLE_PRESETS="style_presets_v27.json"

# --- Generated artifacts (written to mounted output dir) ---
OCR_REPORT="/app/output/ocr_remaining_english_v28_1_4.json"
PATCH_MAP="/app/output/cxview_patch_map_v28_2_rulefix.json"
OCR_DEBUG_DIR="/app/output/ocr_debug_v28_1_4"
PLANNER_REPORT="/app/output/v28_2_rulefix_report.json"
RENDER_REPORT="/app/output/v28_2_render_report.json"

echo ""
echo "=== STEP 1: OCR approved image/graphic regions (RapidOCR) ==="
python ocr_image_region_scanner_v28_1_4_rapidocr_numpyfix.py \
    "$SOURCE_PDF" \
    --image-region-map "$REGION_MAP" \
    --output-json      "$OCR_REPORT" \
    --debug-dir        "$OCR_DEBUG_DIR" \
    --dpi 420 \
    --lang en \
    --pages 2,4-5 \
    --min-confidence 0.05 \
    --engine rapidocr

echo ""
echo "=== STEP 2: Auto Patch Planner v28.2 ==="
python auto_patch_planner_v28_2_rulefix.py \
    --ocr-report       "$OCR_REPORT" \
    --image-region-map "$REGION_MAP" \
    --style-presets    "$STYLE_PRESETS" \
    --output-patch-map "$PATCH_MAP" \
    --report-json      "$PLANNER_REPORT" \
    --base-recommended "$BASE_PDF" \
    --include-needs-review

echo ""
echo "=== STEP 3: v27 Style Preset Renderer ==="
python pdf_image_region_only_patch_v27_style_presets.py \
    "$BASE_PDF" "$OUTPUT_PDF" \
    --patch-map        "$PATCH_MAP" \
    --style-presets    "$STYLE_PRESETS" \
    --source-pdf       "$SOURCE_PDF" \
    --image-region-map "$REGION_MAP" \
    --font             fonts/NotoSans-Regular.ttf \
    --font-bold        fonts/NotoSans-Bold.ttf \
    --font-title       fonts/NotoSans-Bold.ttf \
    --report-json      "$RENDER_REPORT"

echo ""
echo "=== DONE: $OUTPUT_PDF ==="
