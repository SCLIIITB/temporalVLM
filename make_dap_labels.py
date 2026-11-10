"""
Build one labels CSV per DAP folder from the image filenames.

Each DAP folder (e.g. "ISRO 30 DAP 22.11.2023") holds images named like
"R1T1-2b.jpg": plot R1T1, plant 2 within the plot, camera angle b. For every
folder this writes <root>/labels/<DAP>.csv with columns filename, DAP, id, angle,
where id is the plot ID without the plant number (e.g. "R1T1"). The DAP value
comes from the folder name, since the filenames don't contain it.

Usage
-----
    python make_dap_labels.py [root_folder]

root_folder defaults to the RRSC-ADVRD_TOMATO dataset folder below.
"""

import csv
import re
import sys
from pathlib import Path

DEFAULT_ROOT = r"C:\Users\Deeksha Aggarwal\Documents\IIRS\RRSC-ADVRD_TOMATO"
LABELS_FOLDER = "labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# "ISRO 30 DAP 22.11.2023" -> 30
DAP_FOLDER_PATTERN = re.compile(r"(\d+)\s*DAP", re.IGNORECASE)
# "R1T1-2b.jpg" -> id "R1T1", angle "b" (the plant number "-2" is dropped)
FILENAME_PATTERN = re.compile(r"^(R\d+T\d+)-\d+([a-z])$", re.IGNORECASE)


def _natural_key(name: str) -> list:
    """Sort key that orders numbers numerically, so R2T1 comes before R10T1."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def build_labels(folder: Path, dap: int) -> tuple:
    """Parse every image filename in `folder` into a label row.

    Returns (rows, skipped), where skipped lists files that aren't images
    (e.g. partial downloads ending in .filepart) or don't match the
    R<row>T<plot>-<number><angle> naming pattern.
    """
    rows, skipped = [], []
    for path in sorted(folder.iterdir(), key=lambda p: _natural_key(p.name)):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            skipped.append(path.name)
            continue
        match = FILENAME_PATTERN.match(path.stem)
        if not match:
            skipped.append(path.name)
            continue
        rows.append({
            "filename": path.name,
            "DAP": dap,
            "id": match.group(1).upper(),
            "angle": match.group(2).lower(),
        })
    return rows, skipped


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_ROOT)
    if not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")

    labels_dir = root / LABELS_FOLDER
    labels_dir.mkdir(exist_ok=True)

    for folder in sorted(root.iterdir(), key=lambda p: _natural_key(p.name)):
        if not folder.is_dir() or folder == labels_dir:
            continue
        dap_match = DAP_FOLDER_PATTERN.search(folder.name)
        if not dap_match:
            print(f"Skipping folder without a DAP value in its name: {folder.name}")
            continue
        dap = int(dap_match.group(1))

        rows, skipped = build_labels(folder, dap)
        output_csv = labels_dir / f"{dap}.csv"
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "DAP", "id", "angle"])
            writer.writeheader()
            writer.writerows(rows)

        id_count = len({row["id"] for row in rows})
        print(f"{folder.name}: {len(rows)} image(s), {id_count} id(s) -> {output_csv}")
        if skipped:
            print(f"  skipped {len(skipped)} file(s): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
