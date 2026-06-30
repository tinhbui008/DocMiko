#!/usr/bin/env bash
# ============================================================
# Mikotech - Text-layer translation ONLY (v16 generator).
# This step DOES call the LLM API, but reuses a persistent disk
# cache so already-translated strings never hit the API again.
#
#   cache lives at /app/cache (mounted -> /root/pdf-translator/cache)
#   so it survives container exit and is reused across runs.
# ============================================================
# Usage (inside container):
#   run_text_translate.sh [INPUT_PDF] [OUTPUT_PDF] [CACHE_FILE]
set -euo pipefail

INPUT_PDF="${1:-/app/input/input.pdf}"
OUTPUT_PDF="${2:-/app/output/output_clean.pdf}"
CACHE_FILE="${3:-/app/cache/translation_cache.jsonl}"

# Make sure the cache dir exists (mounted volume)
mkdir -p "$(dirname "$CACHE_FILE")"

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
echo "=== Text-layer translation (cache: $CACHE_FILE) ==="
python pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py \
    "$INPUT_PDF" "$OUTPUT_PDF" \
    --translation-cache "$CACHE_FILE" \
    "${FONTS[@]}"

echo ""
echo "=== DONE: $OUTPUT_PDF (cache updated at $CACHE_FILE) ==="
