"""Download Noto Sans fonts for Vietnamese support."""
import urllib.request
from pathlib import Path

FONTS = {
    'NotoSans-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf',
    'NotoSans-Bold.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf',
}

Path('fonts').mkdir(exist_ok=True)

for filename, url in FONTS.items():
    target = Path('fonts') / filename
    if target.exists():
        print(f"  Exists: {filename}")
        continue
    print(f"  Downloading: {filename}")
    urllib.request.urlretrieve(url, target)
    print(f"  ✓ Saved: {target}")

print("Done!")