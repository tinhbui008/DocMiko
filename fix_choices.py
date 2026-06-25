"""
Run: python fix_choices.py pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py
Adds 'claude' to --full-ocr-engine choices and adds --claude-vision-* args.
"""
import sys, shutil
from pathlib import Path

f = Path(sys.argv[1] if len(sys.argv) > 1 else "pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py")
shutil.copy2(f, str(f) + ".bak2")
t = f.read_text(encoding="utf-8")

# Fix 1: choices
old = 'choices=["tesseract", "paddle"], help="Engine for --full-ocr-rebuild"'
new = 'choices=["tesseract", "paddle", "claude"], help="Engine for --full-ocr-rebuild"'
if old in t:
    t = t.replace(old, new)
    print("OK: choices patched")
else:
    print("SKIP: choices already patched or not found")

# Fix 2: add --claude-vision-* args if not present
marker = 'p.add_argument("--full-ocr-min-conf"'
insert = ('    p.add_argument("--claude-vision-model", default="claude-haiku-4-5",'
          ' help="Claude model for --full-ocr-engine claude")\n'
          '    p.add_argument("--claude-vision-dpi", type=int, default=200,'
          ' help="DPI for Claude Vision page rendering")\n    ')
if "claude-vision-model" not in t and marker in t:
    t = t.replace("    " + marker, insert + "    " + marker)
    print("OK: --claude-vision-* args added")
else:
    print("SKIP: vision args already present")

f.write_text(t, encoding="utf-8")
print(f"Done. Backup: {f}.bak2")
