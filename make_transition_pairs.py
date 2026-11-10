"""
Build the Task 5 (growth stage transition) image pairs from ground_truth/labels.csv:
one CSV per camera angle, ground_truth/growth_transition_pairs_angle_<a|b|c>.csv.

Each row pairs two images of the same plant, seen from the same angle, at two different
time points (earlier, later); every such combination is included. Plants are tracked as
in make_plant_tracks.py: Cycle B by id + angle, Cycle A by filename stem (e.g. R1T1-1a).

Ground-truth columns come from `growth_stage` in labels.csv:
  earlier_growth_stage / later_growth_stage   stage numbers, e.g. "2, 3"
  gt_earlier_stages / gt_later_stages         stage names, e.g. "Flowering and Fruit development"
  gt_transition                               e.g. "Flowering to Flowering and Fruit development"
They are blank when either image has no growth_stage label.

Usage
-----
    python make_transition_pairs.py
"""

import csv
import itertools
from collections import defaultdict
from pathlib import Path

from make_plant_tracks import track_of
from prompts import growth_stage_set_name, growth_transition_name

GT = Path(__file__).resolve().parent / "ground_truth"
FIELDS = [
    "cycle", "track_id", "plant_id", "angle",
    "earlier_time_point", "earlier_source_dap", "earlier_filename", "earlier_growth_stage",
    "later_time_point", "later_source_dap", "later_filename", "later_growth_stage",
    "gt_earlier_stages", "gt_later_stages", "gt_transition",
]


def stages(text: str) -> list:
    return [s.strip() for s in text.split(",") if s.strip()]


def main() -> None:
    rows = list(csv.DictReader(open(GT / "labels.csv", encoding="utf-8")))
    tracks = defaultdict(list)
    for r in rows:
        track_id, plant_id, angle = track_of(r)
        tracks[(r["cycle"], track_id, plant_id, angle)].append(r)

    pairs_by_angle = defaultdict(list)
    for (cycle, track_id, plant_id, angle), images in tracks.items():
        images.sort(key=lambda r: int(r["time_point"]))
        for early, late in itertools.combinations(images, 2):
            labelled = bool(stages(early["growth_stage"]) and stages(late["growth_stage"]))
            pairs_by_angle[angle].append({
                "cycle": cycle, "track_id": track_id, "plant_id": plant_id, "angle": angle,
                "earlier_time_point": early["time_point"], "earlier_source_dap": early["source_dap"],
                "earlier_filename": early["filename"], "earlier_growth_stage": early["growth_stage"],
                "later_time_point": late["time_point"], "later_source_dap": late["source_dap"],
                "later_filename": late["filename"], "later_growth_stage": late["growth_stage"],
                "gt_earlier_stages": growth_stage_set_name(stages(early["growth_stage"])) if labelled else "",
                "gt_later_stages": growth_stage_set_name(stages(late["growth_stage"])) if labelled else "",
                "gt_transition": growth_transition_name(stages(early["growth_stage"]), stages(late["growth_stage"]))
                if labelled else "",
            })

    for angle in sorted(pairs_by_angle):
        out = sorted(pairs_by_angle[angle], key=lambda p: (p["cycle"] != "Cycle B", p["track_id"],
                                                           int(p["earlier_time_point"]), int(p["later_time_point"])))
        path = GT / f"growth_transition_pairs_angle_{angle}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(out)
        by_cycle = {c: sum(1 for p in out if p["cycle"] == c) for c in ("Cycle B", "Cycle A")}
        labelled = sum(1 for p in out if p["gt_transition"])
        classes = len({p["gt_transition"] for p in out if p["gt_transition"]})
        print(f"angle {angle}: {len(out)} pairs ({by_cycle['Cycle B']} Cycle B, {by_cycle['Cycle A']} Cycle A), "
              f"{labelled} with ground truth, {classes} transition classes -> {path.name}")


if __name__ == "__main__":
    main()
