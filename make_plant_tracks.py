"""
Build ground_truth/plant_tracks.csv from ground_truth/labels.csv: one row per plant and
camera angle, listing that plant's image filename at every time point.

Tracks are matched per cycle:
- Cycle B: `id` + `angle` (e.g. D1_R5T2 viewed from angle a -> track D1_R5T2_a).
- Cycle A: the filename without its extension (e.g. R1T1-1a), which encodes plot,
  plant number and angle and is reused across sessions.

Columns: cycle, track_id, plant_id, angle, dap_30, dap_50, dap_70, dap_95, dap_130,
n_time_points. A cell is blank when the plant was not photographed at that time point
(Cycle B has no ~30 DAP session; Cycle A has none at ~95 or ~130 DAP).

Usage
-----
    python make_plant_tracks.py
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

GT = Path(__file__).resolve().parent / "ground_truth"
TIME_POINTS = ["30", "50", "70", "95", "130"]


def track_of(row: dict) -> tuple:
    """(track_id, plant_id, angle) for one image row of labels.csv."""
    if row["cycle"] == "Cycle B":
        return f"{row['id']}_{row['angle']}", row["id"], row["angle"]
    stem = os.path.splitext(row["filename"])[0]  # e.g. R1T1-1a
    return stem, f"D2_{stem[:-1]}", stem[-1]


def main() -> None:
    rows = list(csv.DictReader(open(GT / "labels.csv", encoding="utf-8")))
    tracks = {}
    files = defaultdict(dict)
    for r in rows:
        track_id, plant_id, angle = track_of(r)
        key = (r["cycle"], track_id)
        tracks[key] = {"cycle": r["cycle"], "track_id": track_id, "plant_id": plant_id, "angle": angle}
        if r["time_point"] in files[key]:
            raise SystemExit(f"Two images for {key} at {r['time_point']} DAP: "
                             f"{files[key][r['time_point']]}, {r['filename']}")
        files[key][r["time_point"]] = r["filename"]

    out_rows = []
    for key in sorted(tracks, key=lambda k: (k[0] != "Cycle B", k[1])):
        row = dict(tracks[key])
        for tp in TIME_POINTS:
            row[f"dap_{tp}"] = files[key].get(tp, "")
        row["n_time_points"] = len(files[key])
        out_rows.append(row)

    fields = ["cycle", "track_id", "plant_id", "angle", *(f"dap_{tp}" for tp in TIME_POINTS), "n_time_points"]
    with open(GT / "plant_tracks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} tracks to {GT / 'plant_tracks.csv'}")


if __name__ == "__main__":
    main()
