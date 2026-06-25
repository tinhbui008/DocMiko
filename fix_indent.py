t = open('pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py', encoding='utf-8').read()
t = t.replace(
    '        p.add_argument("--full-ocr-min-conf"',
    '    p.add_argument("--full-ocr-min-conf"'
)
open('pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py', 'w', encoding='utf-8').write(t)
print('Fixed')
