"""
Merge the two tomato datasets into one combined dataset with five time points.

    Time point   Dataset 1 (Fruits, 2025)   Dataset 2 (RRSC-ADVRD_TOMATO, 2023-24)
    30 DAP       --                         30 DAP
    50 DAP       54 DAP                     50 DAP
    70 DAP       75 DAP                     72 DAP
    95 DAP       97 DAP                     --
    130 DAP      131 DAP                    --

Images are copied into <output>/<time point> DAP/ (e.g. "70 DAP"), and one labels CSV
per time point is written to <output>/labels/<time point>.csv. Folder and file
names follow what Ollama.py expects, so the combined dataset can be run with
--data-dir <output> --labels-dir <output>/labels --daps 30 50 70 95 130.

Both datasets use the same plot names (R1T1..R5T4) for different plants, so ids
are prefixed with the dataset (D1_R1T1, D2_R1T1) and never treated as the same
plant. Known typos in dataset 1's burned-in notes (e.g. "R3T2_", angle "C",
angle "6") are corrected, and every correction is printed.

Usage
-----
    python make_combined_dataset.py [output_folder]
"""

import csv
import re
import shutil
import sys
from pathlib import Path

DATASET1_IMAGES = Path(r"C:\Users\Deeksha Aggarwal\Documents\IIRS\Fruits")
DATASET1_LABELS = Path(__file__).resolve().parent / "output" / "labels_dap"
DATASET2_ROOT = Path(r"C:\Users\Deeksha Aggarwal\Documents\IIRS\RRSC-ADVRD_TOMATO")
DATASET2_LABELS = DATASET2_ROOT / "labels"
DEFAULT_OUTPUT = Path(r"C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO")

# time point -> list of (dataset tag, source DAP)
TIME_POINTS = {
    30: [("D2", 30)],
    50: [("D1", 54), ("D2", 50)],
    70: [("D1", 75), ("D2", 72)],
    95: [("D1", 97)],
    130: [("D1", 131)],
}

DATASETS = {
    "D1": {"name": "Fruits", "images": DATASET1_IMAGES, "labels": DATASET1_LABELS},
    "D2": {"name": "RRSC-ADVRD_TOMATO", "images": DATASET2_ROOT, "labels": DATASET2_LABELS},
}

FIELDNAMES = ["filename", "DAP", "id", "angle", "dataset", "source_dap", "source_folder"]
ANGLES = ("a", "b", "c")


def find_labels_csv(labels_dir: Path, dap: int) -> Path:
    """Return '<dap>.csv' if present, otherwise the first '<dap>_*.csv' in `labels_dir`."""
    exact = labels_dir / f"{dap}.csv"
    if exact.is_file():
        return exact
    matches = sorted(labels_dir.glob(f"{dap}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No labels CSV for DAP {dap} in {labels_dir}")
    return matches[0]


def find_image_dir(images_root: Path, dap: int) -> Path:
    """Return the session folder for `dap`: a name starting with the DAP number
    (e.g. '75DAP_ 2 plants') or containing '<dap> DAP' (e.g. 'ISRO 30 DAP 22.11.2023')."""
    patterns = [re.compile(rf"^{dap}(\D|$)"), re.compile(rf"(?<!\d){dap}\s*DAP\b", re.IGNORECASE)]
    for entry in sorted(images_root.iterdir()):
        if entry.is_dir() and any(p.search(entry.name) for p in patterns):
            return entry
    raise FileNotFoundError(f"No image folder for DAP {dap} in {images_root}")


def clean_rows(rows: list, dap: int, tag: str) -> list:
    """Normalise id/angle for one source DAP, fixing typos in the original labels.

    The id is reduced to its R<row>T<plot> part (dropping stray characters such as a
    trailing '_' or an angle letter glued on, as in 'R5T1a'). The angle is
    lower-cased; if it is missing or not a/b/c, it is taken from a letter glued to
    the id, or else inferred as the one angle that plant is missing in this DAP.
    """
    parsed = []
    for row in rows:
        raw_id, raw_angle = row["id"].strip(), row["angle"].strip()
        match = re.match(r"^(R\d+T\d+)([a-cA-C])?", raw_id)
        if not match:
            raise ValueError(f"{tag} DAP {dap}: cannot parse id {raw_id!r} ({row['filename']})")
        plant_id = match.group(1).upper()
        angle = raw_angle.lower() if raw_angle.lower() in ANGLES else (match.group(2) or "").lower()
        parsed.append([row, plant_id, angle, raw_id, raw_angle])

    for item in parsed:
        row, plant_id, angle, raw_id, raw_angle = item
        if angle not in ANGLES:
            used = {p[2] for p in parsed if p[1] == plant_id and p[2] in ANGLES}
            missing = [a for a in ANGLES if a not in used]
            if len(missing) != 1:
                raise ValueError(
                    f"{tag} DAP {dap}: cannot infer angle for {row['filename']} "
                    f"(id {raw_id!r}, angle {raw_angle!r})"
                )
            item[2] = angle = missing[0]
        if plant_id != raw_id or angle != raw_angle:
            print(f"  fixed {tag} {row['filename']} (DAP {dap}): id {raw_id!r} -> {plant_id!r}, "
                  f"angle {raw_angle!r} -> {angle!r}")

    return [(row, plant_id, angle) for row, plant_id, angle, _, _ in parsed]


def main() -> None:
    output_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    labels_out = output_root / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    for time_point, sources in TIME_POINTS.items():
        image_out = output_root / f"{time_point} DAP"
        image_out.mkdir(exist_ok=True)
        out_rows, copied, skipped = [], 0, 0

        for tag, source_dap in sources:
            dataset = DATASETS[tag]
            labels_csv = find_labels_csv(dataset["labels"], source_dap)
            image_dir = find_image_dir(dataset["images"], source_dap)
            with open(labels_csv, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))

            for row, plant_id, angle in clean_rows(rows, source_dap, tag):
                filename = row["filename"].strip()
                src = image_dir / filename
                if not src.is_file():
                    print(f"  missing image, skipped: {src}")
                    continue
                dst = image_out / filename
                if any(r["filename"] == filename for r in out_rows):
                    raise ValueError(f"Duplicate filename {filename} in time point {time_point}")
                if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                    skipped += 1  # already copied on an earlier run
                else:
                    shutil.copy2(src, dst)
                    copied += 1
                out_rows.append({
                    "filename": filename,
                    "DAP": time_point,
                    "id": f"{tag}_{plant_id}",
                    "angle": angle,
                    "dataset": dataset["name"],
                    "source_dap": source_dap,
                    "source_folder": image_dir.name,
                })

        out_csv = labels_out / f"{time_point}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(out_rows)

        per_source = ", ".join(
            f"{tag} {dap} DAP: {sum(1 for r in out_rows if r['id'].startswith(tag) and r['source_dap'] == dap)}"
            for tag, dap in sources
        )
        print(f"{time_point} DAP: {len(out_rows)} images ({per_source}) | copied {copied}, "
              f"already present {skipped} -> {out_csv}")
        grand_total += len(out_rows)

    print(f"Combined dataset: {grand_total} images in {output_root}")


if __name__ == "__main__":
    main()
