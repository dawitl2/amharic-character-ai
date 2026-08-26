import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

chars = ["ሀ", "ለ", "መ"]
for c in chars:
    count = len(list((Path("data") / c).glob("*.png")))
    print(f"{c}: {count} images")
