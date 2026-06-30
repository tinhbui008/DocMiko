# ============================================================
# Mikotech PDF Translation Pipeline
# Step 1: Native text-layer translation
# Step 2: V26.3 patch for image/graphic regions
# ============================================================

param(
    [string]$InputPdf  = "CXVIEW-SMART-AI-VIDEO-ANALYTICS-SOLUTION-PRICING.pdf",
    [string]$OutputPdf = "output_final.pdf",
    [string]$CleanPdf  = "output_clean.pdf"
)

$fonts = @(
    '--font',               'fonts/NotoSans-Regular.ttf',
    '--font-bold',          'fonts/NotoSans-Bold.ttf',
    '--font-title',         'fonts/NotoSansCondensed-Bold.ttf',
    '--font-medium',        'fonts/NotoSans-Medium.ttf',
    '--font-semibold',      'fonts/NotoSans-SemiBold.ttf',
    '--font-condensed',     'fonts/NotoSansCondensed-Regular.ttf',
    '--font-condensed-bold','fonts/NotoSansCondensed-Bold.ttf',
    '--font-condensed-semibold','fonts/NotoSansCondensed-SemiBold.ttf',
    '--font-black',         'fonts/NotoSans-Black.ttf'
)

Write-Host ""
Write-Host "=== STEP 1: Native text-layer translation ===" -ForegroundColor Cyan
python pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py `
    $InputPdf $CleanPdf @fonts

if ($LASTEXITCODE -ne 0) {
    Write-Host "Step 1 FAILED" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== STEP 2: V26.3 image/graphic region patch ===" -ForegroundColor Cyan
python v26_3_patch.py `
    $CleanPdf $OutputPdf `
    --patch-map        "patch_map.json" `
    --source-pdf       $InputPdf `
    --image-region-map "image_region_map.json" `
    --font             "fonts/NotoSans-Regular.ttf" `
    --font-bold        "fonts/NotoSans-Bold.ttf" `
    --font-title       "fonts/NotoSansCondensed-Bold.ttf"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Step 2 FAILED" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== DONE: $OutputPdf ===" -ForegroundColor Green
