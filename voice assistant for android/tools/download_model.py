# -*- coding: utf-8 -*-
"""Downloads the Vosk streaming speech model into models/ (one time)."""
import sys
import urllib.request
import zipfile
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
URLS = [
    "https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip",
    "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
]

for url in URLS:
    name = url.rsplit("/", 1)[-1]
    target_dir = MODEL_DIR / name[:-4]
    if (target_dir / "conf" / "model.conf").exists():
        print(f"[ok] model already present: {name}")
        sys.exit(0)

for url in URLS:
    name = url.rsplit("/", 1)[-1]
    zp = MODEL_DIR / name
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print(f"downloading {name} …")
        urllib.request.urlretrieve(url, zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(MODEL_DIR)
        zp.unlink(missing_ok=True)
        print(f"[ok] extracted to {MODEL_DIR / name[:-4]}")
        sys.exit(0)
    except Exception as exc:
        print(f"[warn] {name} failed: {exc}")

print("[error] no model downloaded")
sys.exit(1)
