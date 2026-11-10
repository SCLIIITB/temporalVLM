"""
Extract Note overlay from plant images and save to CSV.

SETUP (run once):
    sudo apt-get install tesseract-ocr
    ./yp/bin/pip install pytesseract pillow
"""

import os
import re
import csv
import pytesseract
from PIL import Image, ImageOps, ImageEnhance

IMAGES_FOLDER = "testInput"
OUTPUT_CSV    = "97_DAP_2_plants_extracted.csv"

CROP_HEIGHT_PX = 300
CROP_WIDTH_PX  = 1300


def get_crop(img_path):
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    return img.crop((0, h - CROP_HEIGHT_PX, CROP_WIDTH_PX, h))


def ocr_text(crop):
    crop = ImageEnhance.Contrast(crop).enhance(1.5)
    crop = crop.convert("L")
    return pytesseract.image_to_string(crop, config="--psm 6").strip()


def extract_note(raw):
    """
    Match patterns like:
      Note: R5T2-a-97DAP
      Note:R5T2-b-97DAP
      NoteyR5T2-b-97DAP   (OCR reads : as y)
      Note: R5T1<a-97DAP  (OCR reads - as <)
      Note: R4AT3-c-97DAP (extra char)
      Note: R3T2-a        (split across lines, 97DAP on next)
    """
    # Join lines so split values are caught
    flat = " ".join(raw.splitlines())

    # Pattern: Not[e/3/...] + separator + R<digits>T<digits>-<letter>-97DAP
    # [A-Z][0-9]{1,2}[A-Z]?T[0-9]{1,2} correctly matches R5T2, R1T3, R4AT3 (extra OCR char)
    m = re.search(
        r'Not[e3E][^A-Z0-9]{0,4}'           # "Note" + separator (: ; y etc.)
        r'([A-Z][0-9]{1,2}[A-Z]?T[0-9]{1,2}'  # R5T2, R1T3, R4AT3 ...
        r'[^a-zA-Z0-9]{0,2}[a-z]'           # -a / -b / -c
        r'[^a-zA-Z0-9]{0,2}'                # separator
        r'9\/?7 ?DAP)',                       # 97DAP, 9/7DAP, or "97 DAP" (space OCR variant)
        flat   # no IGNORECASE: keeps [^A-Z0-9] strict so lowercase 'y' (OCR ":") is a valid separator
    )
    if not m:
        return ""

    note = m.group(1)
    # Normalise separators: replace any non-alphanumeric between segments with '-'
    note = re.sub(r'(?<=[0-9])([^a-zA-Z0-9]+)(?=[a-z])', '-', note)
    note = re.sub(r'(?<=[a-z])([^a-zA-Z0-9]+)(?=[0-9])', '-', note)
    note = re.sub(r'(?<=[0-9])([^a-zA-Z0-9]+)(?=DAP)', '-', note)
    note = re.sub(r'9[/. ]?7 ?DAP', '97DAP', note)
    # Remove spurious OCR letter between row digits and T (e.g. R4AT3 → R4T3)
    note = re.sub(r'([A-Z][0-9]{1,2})[A-Z](T[0-9])', r'\1\2', note)
    return note


def main():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, IMAGES_FOLDER)
    out_path   = os.path.join(base_dir, OUTPUT_CSV)

    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"))
    )

    if not image_files:
        print(f"No images found in: {images_dir}")
        return

    print(f"Processing {len(image_files)} images…")
    rows = []
    missed = []

    for fname in image_files:
        raw  = ocr_text(get_crop(os.path.join(images_dir, fname)))
        note = extract_note(raw)
        rows.append({"image_name": fname, "Note": note})
        if note:
            print(f"  {fname}: {note}")
        else:
            missed.append(fname)
            print(f"  {fname}: [NOT FOUND]  raw={repr(raw[:120])}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "Note"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {out_path}")
    if missed:
        print(f"Could not extract Note from {len(missed)} image(s): {missed}")


if __name__ == "__main__":
    main()