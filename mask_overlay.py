"""
Mask the burned-in NoteCam overlay in every image of a dataset and save the masked
copies to a separate folder.

The camera app writes a semi-transparent box in the bottom-left corner
("Time: <date>" / "Note: <plant code>-<view>-<DAP>") and a "NoteCam @ iOS" watermark
in the bottom-right. Both are the same size in pixels regardless of orientation, so the
masks are sized relative to the image's SHORT side. Each is covered with solid grey.

Every image is masked, including those without an overlay (Cycle A), so the mask itself
never tells two datasets or conditions apart.

The input folder is mirrored: each subfolder (one per DAP) becomes a subfolder of the
output with the same image filenames, and the `labels` folder (and any other non-image
files) is copied unchanged. The masked dataset can therefore be used directly, e.g.
    python Ollama.py ... --data-dir <output> --labels-dir <output>/labels

Images are saved upright (EXIF orientation applied) as JPEG.

Usage
-----
    python mask_overlay.py [input_folder] [output_folder] [--overwrite]

Defaults: input  C:\\Users\\Deeksha Aggarwal\\Documents\\IIRS\\Combined_TOMATO
          output <input>_masked
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

DEFAULT_INPUT = Path(r"C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Mask sizes as fractions of the image's short side (with a safety margin around
# the measured overlay: text box about 0.38 x 0.066, watermark about 0.25 x 0.04).
TEXT_BOX = (0.42, 0.08)    # bottom-left: width, height
WATERMARK = (0.30, 0.06)   # bottom-right: width, height
MASK_COLOR = (128, 128, 128)
JPEG_QUALITY = 95


def mask_image(image: Image.Image) -> Image.Image:
    """Return an upright RGB copy of `image` with the overlay regions filled grey."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    s = min(w, h)
    draw = ImageDraw.Draw(image)
    tw, th = round(TEXT_BOX[0] * s), round(TEXT_BOX[1] * s)
    draw.rectangle([0, h - th, tw, h], fill=MASK_COLOR)
    ww, wh = round(WATERMARK[0] * s), round(WATERMARK[1] * s)
    draw.rectangle([w - ww, h - wh, w, h], fill=MASK_COLOR)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Mask the burned-in overlay in every image of a dataset.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="dataset folder (one subfolder per DAP)")
    parser.add_argument("output", nargs="?", default=None, help="output folder (default: <input>_masked)")
    parser.add_argument("--overwrite", action="store_true", help="re-mask images that already exist in the output")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output) if args.output else src.with_name(src.name + "_masked")
    if not src.is_dir():
        raise SystemExit(f"Input folder not found: {src}")
    if dst.resolve() == src.resolve():
        raise SystemExit("Output folder must differ from the input folder")

    masked = skipped = copied = 0
    unreadable = []
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() not in IMAGE_EXTENSIONS or path.parent.name == "labels":
            shutil.copy2(path, out)
            copied += 1
            continue
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            with Image.open(path) as image:
                result = mask_image(image)
        except (UnidentifiedImageError, OSError) as exc:
            unreadable.append(f"{path.relative_to(src)} ({exc.__class__.__name__})")
            continue
        result.save(out, format="JPEG", quality=JPEG_QUALITY)
        masked += 1
        if masked % 50 == 0:
            print(f"  masked {masked} image(s)...")

    print(f"Masked {masked} image(s), skipped {skipped} already present, copied {copied} other file(s) -> {dst}")
    if unreadable:
        print(f"Could not read {len(unreadable)} file(s): {', '.join(unreadable)}")


if __name__ == "__main__":
    main()
