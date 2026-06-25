"""Fix: increase max_tokens and add JSON repair for truncated Claude Vision responses."""
from pathlib import Path

f = Path("pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py")
t = f.read_text(encoding="utf-8")

# Fix 1: increase max_tokens from 4096 to 8192 for complex pages
old1 = '"max_tokens": max_tokens=4096'
# Find actual pattern
import re
# Fix the call in ocr_page_to_all_line_blocks_claude_vision_v16
old1 = '        max_tokens=4096,\n        timeout=timeout,'
new1 = '        max_tokens=8192,\n        timeout=timeout,'
if old1 in t:
    t = t.replace(old1, new1)
    print("OK: max_tokens increased to 8192")
else:
    print("SKIP: max_tokens pattern not found")

# Fix 2: add JSON repair in _claude_vision_parse_blocks
old2 = '''    try:
        s, e = text.find("["), text.rfind("]")
        items = _json.loads(text[s:e+1]) if s >= 0 and e > s else _json.loads(text)
    except Exception as ex:
        print(f"      Claude Vision JSON parse error: {ex}")
        return []'''

new2 = '''    try:
        s, e = text.find("["), text.rfind("]")
        json_str = text[s:e+1] if s >= 0 and e > s else text
        try:
            items = _json.loads(json_str)
        except Exception:
            # Response was truncated — repair by closing open structures
            repaired = json_str.rstrip().rstrip(",")
            # Close any open string
            if repaired.count(\'"\') % 2 == 1:
                repaired += \'"\' 
            # Close open objects and array
            open_braces = repaired.count("{") - repaired.count("}")
            repaired += "}" * max(0, open_braces)
            repaired += "]"
            try:
                items = _json.loads(repaired)
                print(f"      Claude Vision: JSON repaired ({len(items)} items)")
            except Exception as ex2:
                print(f"      Claude Vision JSON parse error: {ex2}")
                return []
    except Exception as ex:
        print(f"      Claude Vision JSON parse error: {ex}")
        return []'''

if old2 in t:
    t = t.replace(old2, new2)
    print("OK: JSON repair added")
else:
    print("SKIP: JSON repair pattern not found")

f.write_text(t, encoding="utf-8")
print("Done")
