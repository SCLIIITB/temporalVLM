"""
CropEval: evaluate a VLM on TomatoTemporal along two axes.

1. Reasoning evaluation: mean of the per-answer `score` column (1 = the reasoning
   follows the task rubric in output/reasoning_score_rubric.md, 0 = it does not),
   over every answer in the folder, whether or not ground truth exists.
2. Quantitative evaluation: accuracy against ground truth on the old dataset
   (54/75/97/131 DAP), the only images with labels.

It also reports grounded accuracy: the share of answers that are both correct and
have a valid reasoning (score 1).

Every metric is computed per result file (one file per DAP) and then averaged over
those DAPs (macro average).
Growth stage and ripening transition are read from output/combined; re-identification
and ripeness counting from output/AGP and output/ZeroShot.

Ground truth is read from ground_truth/labels.csv (one row per image, blank where no
label exists; build it with make_ground_truth.py), matched by (filename, DAP):
- Growth stage: `growth_stage`.
- Ripeness counts: `raw_count`, `semi_ripe_count`, `ripe_count`.
- Per-image ripeness stage (for transitions): `ripeness_stage` where present, otherwise
  the predominant class of the ground-truth counts (no fruit counts as Raw).
- Re-identification: true_candidate_index, which comes from the plant IDs.

Usage:
    python cropeval.py
"""

import csv
import re
import io
import statistics
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"
MODEL = "gemma4cloud"
OLD_DAPS = {"54", "75", "97", "131"}
STAGE_RANK = {"Raw": 0, "Semi-ripe": 1, "Ripe": 2}


def load(path: Path) -> list:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return list(csv.DictReader(io.StringIO(raw.decode(enc), newline="")))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def task_rows(folder: Path) -> list:
    rows = []
    for f in sorted(folder.glob("*.csv")):
        file_dap = re.search(r"_(\d+)\.csv$", f.name).group(1)
        for r in load(f):
            r["_file_dap"] = file_dap
            rows.append(r)
    return rows


def macro(values_by_dap: dict):
    """Average of per-DAP means; None when there is nothing to average."""
    means = [statistics.mean(v) for v in values_by_dap.values() if v]
    return statistics.mean(means) if means else None


def stage_set(text: str) -> frozenset:
    return frozenset(s.strip() for s in text.split(",") if s.strip())


def transition(earlier: str, later: str) -> str:
    e, l = STAGE_RANK[earlier], STAGE_RANK[later]
    if l < e:
        return "Regression"
    if l == e:
        return "No transition"
    return {(0, 1): "Raw to Semi-ripe", (1, 2): "Semi-ripe to Ripe", (0, 2): "Raw to Ripe (skipped stage)"}[(e, l)]


# ---------------- ground truth ----------------
# All labels live in ground_truth/labels.csv (built by make_ground_truth.py), one row
# per image. Filenames repeat across Cycle A sessions, so labels are keyed by
# (filename, DAP); both the pooled time point (combined results, e.g. "50") and the
# session DAP (per-cycle results, e.g. "54") are registered as keys.
GT_FILE = Path(__file__).resolve().parent / "ground_truth" / "labels.csv"
gs_gt, count_gt, image_stage = {}, {}, {}
for g in load(GT_FILE):
    keys = {(g["filename"], g["time_point"]), (g["filename"], g["source_dap"])}
    if g["growth_stage"].strip():
        for k in keys:
            gs_gt[k] = stage_set(g["growth_stage"])
    counts = None
    if g["raw_count"].strip():
        counts = {k: int(g[f"{k}_count"]) for k in ("raw", "semi_ripe", "ripe")}
        for k in keys:
            count_gt[k] = counts
    # Per-image ripeness stage for T2: the annotated stage where one exists, otherwise
    # the predominant class of the ground-truth counts (no fruit counts as Raw).
    stage = g["ripeness_stage"].strip() or None
    if stage is None and counts is not None:
        if sum(counts.values()) == 0:
            stage = "Raw"
        else:
            best = max(counts, key=lambda k: (counts[k], -list(counts).index(k)))
            stage = {"raw": "Raw", "semi_ripe": "Semi-ripe", "ripe": "Ripe"}[best]
    if stage in STAGE_RANK:
        for k in keys:
            image_stage[k] = stage


def evaluate(task: str, rows: list, correct_fn, dap_key: str = "_file_dap") -> dict:
    """Return reasoning score, accuracy and grounded accuracy, each macro-averaged over DAPs."""
    reason, acc, grounded = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        if (r.get("error") or "").strip():
            continue
        dap = r[dap_key]
        score = r.get("score", "")
        if score in ("0", "1"):
            reason[dap].append(int(score))
        ok = correct_fn(r)
        if ok is not None:
            acc[dap].append(int(ok))
            if score in ("0", "1"):
                grounded[dap].append(int(ok and score == "1"))
    return {
        "reasoning": macro(reason), "reasoning_daps": sorted(reason, key=int),
        "accuracy": macro(acc), "accuracy_daps": sorted(acc, key=int), "n_gt": sum(len(v) for v in acc.values()),
        "grounded": macro(grounded) if any("score" in r for r in rows) else None,
    }


def gs_correct(r):
    g = gs_gt.get((r["filename"], r["dap"]))
    return None if g is None else stage_set(r["growth_stage_numbers"]) == g


def rt_correct(r):
    e = image_stage.get((r["earlier_filename"], r["earlier_dap"]))
    l = image_stage.get((r["later_filename"], r["later_dap"]))
    if e is None or l is None:
        return None
    return r["transition_type"] == transition(e, l)


def reid_correct(r):
    if r["query_dap"] not in OLD_DAPS:
        return None
    return r["matching_candidate_index"] == r["true_candidate_index"]


def count_correct(r):
    g = count_gt.get((r["filename"], r["dap"]))
    if g is None:
        return None
    pred = {k: int(r[f"{k}_count"] or 0) for k in ("raw", "semi_ripe", "ripe")}
    return pred == g


def count_mae(rows):
    by_dap = defaultdict(list)
    for r in rows:
        g = count_gt.get((r["filename"], r["dap"]))
        if g is None or (r.get("error") or "").strip():
            continue
        pred = sum(int(r[f"{k}_count"] or 0) for k in ("raw", "semi_ripe", "ripe"))
        by_dap[r["dap"]].append(abs(pred - sum(g.values())))
    return macro(by_dap)


T5_MIN_SUPPORT = 5


def macro_f1_min_support(y_true, y_pred, min_support=T5_MIN_SUPPORT):
    """Macro-F1 over classes with at least `min_support` ground-truth items (every
    prediction still counts as a false positive for the class it names)."""
    support = defaultdict(int)
    for t in y_true:
        support[t] += 1
    classes = [c for c, n in support.items() if n >= min_support]
    if not classes:
        return None, 0
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1s) / len(f1s), len(classes)


def growth_transition_file_metrics(rows):
    """Task 5 metrics for one result file (one camera angle): reasoning validity, exact
    accuracy, macro-F1 over classes with >= T5_MIN_SUPPORT pairs, grounded accuracy."""
    ok = [r for r in rows if not (r.get("error") or "").strip()]
    scores = [int(r["score"]) for r in ok if r.get("score") in ("0", "1")]
    gt = [r for r in ok if r.get("gt_transition")]
    y_true = [r["gt_transition"] for r in gt]
    y_pred = [r["pred_transition"] for r in gt]
    f1, n_classes = macro_f1_min_support(y_true, y_pred)
    grounded = [int(r["pred_transition"] == r["gt_transition"] and r["score"] == "1")
                for r in gt if r.get("score") in ("0", "1")]
    return {
        "reasoning": statistics.mean(scores) if scores else None,
        "accuracy": statistics.mean(t == p for t, p in zip(y_true, y_pred)) if gt else None,
        "macro_f1": f1, "f1_classes": n_classes, "n": len(gt),
        "grounded": statistics.mean(grounded) if grounded else None,
    }


def growth_transition_report():
    """Print Task 5 metrics for every growthTransition result folder under output/:
    per camera angle (file) and averaged over angles, overall and per cycle."""
    folders = sorted({p.parent for p in OUT.rglob("growthTransition/*.csv")})
    if not folders:
        return
    print(f"\nT5 Growth stage transition (macro-F1 over classes with >= {T5_MIN_SUPPORT} pairs)")
    print(f"{'Run':52} {'Subset':8} {'Reasoning':>9} {'Accuracy':>9} {'MacroF1':>8} {'Grounded':>9} {'n':>5}")
    for folder in folders:
        files = sorted(folder.glob("*.csv"))
        run = str(folder.parent.relative_to(OUT))
        for subset in ("All", "Cycle B", "Cycle A"):
            per_file = []
            for f in files:
                rows = [r for r in load(f) if subset == "All" or r.get("cycle") == subset]
                if rows:
                    per_file.append((f.stem.rsplit("_", 1)[-1], growth_transition_file_metrics(rows)))
            if not per_file:
                continue
            avg = lambda k: macro({a: [m[k]] for a, m in per_file if m[k] is not None})
            label = f"{run} (angles {','.join(a for a, _ in per_file)})" if subset == "All" else ""
            print(f"{label:52} {subset:8} {fmt(avg('reasoning')):>9} {fmt(avg('accuracy')):>9} "
                  f"{fmt(avg('macro_f1'), False):>8} {fmt(avg('grounded')):>9} {sum(m['n'] for _, m in per_file):>5}")


def fmt(x, pct=True):
    return "--" if x is None else (f"{100 * x:.1f}" if pct else f"{x:.2f}")


def main() -> None:
    results = []
    for mode in ("ZeroShot", "AGP"):
        comb = OUT / "combined" / mode / MODEL
        base = OUT / mode / MODEL
        if (comb / "growthStageClassification").exists():
            results.append(("T1 Growth stage", mode, "combined", evaluate("gs", task_rows(comb / "growthStageClassification"), gs_correct)))
        if (comb / "ripeningTransitionResults").exists():
            results.append(("T2 Ripening transition", mode, "combined", evaluate("rt", task_rows(comb / "ripeningTransitionResults"), rt_correct)))
        results.append(("T3 Re-identification", mode, mode, evaluate("reid", task_rows(base / "reid"), reid_correct)))
        rpc = task_rows(base / "ripenessCount")
        res = evaluate("rpc", rpc, count_correct)
        res["mae"] = count_mae(rpc)
        results.append(("T4 Ripeness counting", mode, mode, res))

    print(f"{'Task':26} {'Mode':9} {'Folder':9} {'Reasoning':>9} {'Accuracy':>9} {'Grounded':>9} {'n_GT':>5}  DAPs (reasoning | accuracy)")
    for task, mode, folder, r in results:
        extra = f"  MAE={fmt(r['mae'], False)}" if "mae" in r else ""
        print(f"{task:26} {mode:9} {folder:9} {fmt(r['reasoning']):>9} {fmt(r['accuracy']):>9} {fmt(r['grounded']):>9} {r['n_gt']:>5}  "
              f"{','.join(r['reasoning_daps']) or '-'} | {','.join(r['accuracy_daps'])}{extra}")

    growth_transition_report()


if __name__ == "__main__":
    main()
