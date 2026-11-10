"""
Build ground_truth/labels.csv: one row per image of the combined dataset (both cycles),
with every ground-truth label kept in one place, separate from model outputs.

Columns
-------
filename, cycle, dataset, time_point, source_dap, id, angle  -- image identity
growth_stage      T1 label, one or two stage numbers, e.g. "2, 3"
ripeness_stage    T2 per-image predominant ripeness stage: Raw / Semi-ripe / Ripe
raw_count, semi_ripe_count, ripe_count   T4 per-class fruit counts
notes             e.g. conflicting source labels that need resolving

A cell is blank when no label exists for that image. T3 needs no column: its ground
truth comes from the plant IDs. Filenames repeat across Cycle A sessions, so an image
is identified by (filename, time_point) or equivalently (filename, source_dap).

The labels were originally added by hand as `gt` / `*_count_gt` columns in the AGP
result files under output/AGP/gemma4cloud; this script collects them from there.
If ground_truth/labels.csv already exists it is not overwritten unless --overwrite is
given, so edits made directly to the file are kept.

Usage
-----
    python make_ground_truth.py [--overwrite]
"""

import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
MASTER_LABELS = Path(r"C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO\labels")
AGP = PROJECT / "output" / "AGP" / "gemma4cloud"
OUT = PROJECT / "ground_truth" / "labels.csv"
STAGES = ("Raw", "Semi-ripe", "Ripe")
FIELDS = ["filename", "cycle", "dataset", "time_point", "source_dap", "id", "angle",
          "growth_stage", "ripeness_stage", "raw_count", "semi_ripe_count", "ripe_count", "notes"]


def read(pattern):
    """Rows of every CSV matching `pattern`, each tagged with the DAP from its filename."""
    rows = []
    for f in sorted(glob.glob(str(pattern))):
        dap = re.search(r"_(\d+)\.csv$", f).group(1)
        for r in csv.DictReader(open(f, encoding="utf-8-sig", errors="replace")):
            r["_dap"] = dap
            rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect all ground-truth labels into ground_truth/labels.csv")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing labels.csv")
    args = parser.parse_args()
    if OUT.exists() and not args.overwrite:
        raise SystemExit(f"{OUT} already exists; use --overwrite to rebuild it (hand edits would be lost)")

    # Ground truth from the AGP result files, keyed by (filename, source DAP).
    growth, ripeness, counts = {}, defaultdict(set), {}
    for r in read(AGP / "growthStageClassification" / "*.csv"):
        if (r.get("gt") or "").strip():
            growth[(r["filename"], r["_dap"])] = ", ".join(s.strip() for s in r["gt"].split(",") if s.strip())
    for r in read(AGP / "ripeningTransitionResults" / "*.csv"):
        if r.get("gt") in STAGES and r.get("later_filename"):
            ripeness[(r["later_filename"], r["later_dap"])].add(r["gt"])
    for r in read(AGP / "ripenessCount" / "*.csv"):
        if (r.get("raw_count_gt") or "").strip():
            counts[(r["filename"], r["_dap"])] = (r["raw_count_gt"], r["semi_ripe_count_gt"], r["ripe_count_gt"])

    rows, used = [], set()
    for f in sorted(MASTER_LABELS.glob("*.csv"), key=lambda p: int(p.stem)):
        for m in csv.DictReader(open(f, encoding="utf-8-sig")):
            key = (m["filename"], m["source_dap"])
            notes = []
            stage = ""
            if key in ripeness:
                labels = ripeness[key]
                if len(labels) == 1:
                    stage = next(iter(labels))
                else:
                    notes.append("conflicting ripeness_stage labels: " + " vs ".join(sorted(labels)) + " (resolve)")
            raw, semi, ripe = counts.get(key, ("", "", ""))
            rows.append({
                "filename": m["filename"],
                "cycle": "Cycle B" if m["id"].startswith("D1_") else "Cycle A",
                "dataset": m["dataset"],
                "time_point": m["DAP"],
                "source_dap": m["source_dap"],
                "id": m["id"],
                "angle": m["angle"],
                "growth_stage": growth.get(key, ""),
                "ripeness_stage": stage,
                "raw_count": raw, "semi_ripe_count": semi, "ripe_count": ripe,
                "notes": "; ".join(notes),
            })
            used.add(key)

    # Labelled Cycle B first, then Cycle A; each by time point, keeping the source order.
    rows.sort(key=lambda r: (r["cycle"] != "Cycle B", int(r["time_point"])))
    unmatched = [k for k in set(growth) | set(ripeness) | set(counts) if k not in used]
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    filled = lambda col: sum(1 for r in rows if r[col])
    print(f"Wrote {len(rows)} images to {OUT}")
    print(f"  growth_stage: {filled('growth_stage')}, ripeness_stage: {filled('ripeness_stage')}, "
          f"counts: {filled('raw_count')}, notes: {filled('notes')}")
    if unmatched:
        print(f"  labels with no matching image (not written): {unmatched}")


if __name__ == "__main__":
    main()
