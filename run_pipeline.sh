#!/usr/bin/env bash
# ============================================================
# Mikotech PDF Translation Pipeline (Linux / Docker port of run_pipeline.ps1)
# Step 1: Native text-layer translation
# Step 2: V26.3 patch for image/graphic regions
# ============================================================
# Usage (inside container):
#   run_pipeline.sh [INPUT_PDF] [OUTPUT_PDF] [CLEAN_PDF]
# Paths default to the mounted /app/input and /app/output folders.
set -euo pipefail

INPUT_PDF="${1:-/app/input/input.pdf}"
OUTPUT_PDF="${2:-/app/output/output_final.pdf}"
CLEAN_PDF="${3:-/app/output/output_clean.pdf}"

FONTS=(
  --font                    fonts/NotoSans-Regular.ttf
  --font-bold               fonts/NotoSans-Bold.ttf
  --font-title              fonts/NotoSansCondensed-Bold.ttf
  --font-medium             fonts/NotoSans-Medium.ttf
  --font-semibold           fonts/NotoSans-SemiBold.ttf
  --font-condensed          fonts/NotoSansCondensed-Regular.ttf
  --font-condensed-bold     fonts/NotoSansCondensed-Bold.ttf
  --font-condensed-semibold fonts/NotoSansCondensed-SemiBold.ttf
  --font-black              fonts/NotoSans-Black.ttf
)

echo ""
echo "=== STEP 1: Native text-layer translation ==="
python pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py \
    "$INPUT_PDF" "$CLEAN_PDF" "${FONTS[@]}"

echo ""
echo "=== STEP 2: V26.3 image/graphic region patch ==="
python v26_3_patch.py \
    "$CLEAN_PDF" "$OUTPUT_PDF" \
    --patch-map        patch_map.json \
    --source-pdf       "$INPUT_PDF" \
    --image-region-map image_region_map.json \
    --font             fonts/NotoSans-Regular.ttf \
    --font-bold        fonts/NotoSans-Bold.ttf \
    --font-title       fonts/NotoSansCondensed-Bold.ttf

echo ""
echo "=== DONE: $OUTPUT_PDF ==="
