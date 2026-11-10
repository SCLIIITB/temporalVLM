"""
Exploratory analysis of Task 1 (growth stage) results on the masked combined dataset
for three models (gemma4:cloud, gemma4e4b, minicpm-v:8b) in both prompting modes.

Correct = predicted stage set equals the ground-truth stage set (same rule as CropEval).
Reads   output/combined_masked/<ZeroShot|AGP>/<model>/growthStageClassification/gs_eval_*.csv
        ground_truth/labels.csv
Writes  output/eda/growth_stage/*.csv (tables) and *.png (figures)

Usage:  python eda_growth_stage.py
"""
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cropeval as c

ROOT = Path(__file__).resolve().parent
RES = ROOT / "output" / "combined_masked"
OUT = ROOT / "output" / "eda" / "growth_stage"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {"gemma4cloud": "gemma4:cloud", "gemma4e4b": "gemma4:e4b", "minicpmv8b": "minicpm-v:8b"}
MODES = {"ZeroShot": "Zero-shot", "AGP": "AGP"}
DAPS = ["30", "50", "70", "95", "130"]
STAGES = {"1": "Side shooting", "2": "Flowering", "3": "Fruit dev.", "4": "Ripening", "5": "Senescence"}

# Chart style: categorical slots 1-3 of the reference palette, one colour per model.
COLOR = {"gemma4cloud": "#2a78d6", "gemma4e4b": "#eb6834", "minicpmv8b": "#1baf7a"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def key(s):
    """Canonical text for a stage set, e.g. frozenset({'3','2'}) -> '2,3'."""
    return ",".join(sorted(s, key=int)) if s else "(none)"


# ---------------------------------------------------------------- load
labels = c.load(ROOT / "ground_truth" / "labels.csv")
cycle_of = {}
for g in labels:
    cycle_of[(g["filename"], g["time_point"])] = g["cycle"]

records = []
for m in MODELS:
    for mode in MODES:
        folder = RES / mode / m / "growthStageClassification"
        if not folder.exists():
            print("missing:", folder)
            continue
        for r in c.task_rows(folder):
            gt = c.gs_gt.get((r["filename"], r["dap"]))
            pred = c.stage_set(r.get("growth_stage_numbers", ""))
            records.append({
                "model": m, "mode": mode, "dap": r["_file_dap"], "filename": r["filename"],
                "angle": r.get("angle", ""), "cycle": cycle_of.get((r["filename"], r["_file_dap"]), ""),
                "pred": key(pred), "gt": key(gt) if gt is not None else None,
                "pred_set": pred, "gt_set": gt,
                "error": bool((r.get("error") or "").strip()),
                "score": int(r["score"]) if r.get("score") in ("0", "1") else np.nan,
                "reason_len": len((r.get("reasoning") or "").split()),
                "in_tok": pd.to_numeric(r.get("input_tokens"), errors="coerce"),
                "out_tok": pd.to_numeric(r.get("output_tokens"), errors="coerce"),
            })
df = pd.DataFrame(records)
lab = df[df["gt"].notna()].copy()
lab["correct"] = lab["pred_set"] == lab["gt_set"]
lab["grounded"] = lab["correct"] & (lab["score"] == 1)
lab["n_pred"] = lab["pred_set"].map(len)
lab["n_gt"] = lab["gt_set"].map(len)
lab["jaccard"] = [len(p & g) / len(p | g) if (p | g) else 1.0 for p, g in zip(lab["pred_set"], lab["gt_set"])]


def err_type(p, g):
    if p == g: return "Exact"
    if not p: return "No answer"
    if p < g: return "Missed a stage"
    if p > g: return "Extra stage"
    if p & g: return "Partly overlapping"
    return "Disjoint"
lab["error_type"] = [err_type(p, g) for p, g in zip(lab["pred_set"], lab["gt_set"])]

runs = [(m, mode) for m in MODELS for mode in MODES if ((lab["model"] == m) & (lab["mode"] == mode)).any()]
run_name = lambda m, mode: f"{MODELS[m]} {MODES[mode]}"
tables = {}

# ---------------------------------------------------------------- 1. headline metrics
rows = []
for m, mode in runs:
    s = lab[(lab.model == m) & (lab["mode"] == mode)]
    a = df[(df.model == m) & (df["mode"] == mode)]
    per_dap = s.groupby("dap")
    rows.append({
        "model": MODELS[m], "mode": MODES[mode], "n_images": len(a), "n_with_gt": len(s),
        "errors": int(a["error"].sum()),
        "R_macro": a.groupby("dap")["score"].mean().mean() * 100,
        "Acc_macro": per_dap["correct"].mean().mean() * 100,
        "GA_macro": per_dap["grounded"].mean().mean() * 100,
        "Acc_micro": s["correct"].mean() * 100,
        "Jaccard_macro": per_dap["jaccard"].mean().mean() * 100,
        "Acc_CycleB": s[s.cycle == "Cycle B"]["correct"].mean() * 100,
        "Acc_CycleA": s[s.cycle == "Cycle A"]["correct"].mean() * 100,
        "distinct_pred_sets": s["pred"].nunique(),
        "mean_out_tokens": a["out_tok"].mean(), "mean_reason_words": a["reason_len"].mean(),
    })
tables["01_headline"] = pd.DataFrame(rows).round(1)

# ---------------------------------------------------------------- 2. per-DAP accuracy
acc_dap = (lab.groupby(["model", "mode", "dap"])["correct"].mean().mul(100).unstack("dap")
           .reindex(columns=DAPS).round(1))
acc_dap.index = [run_name(m, mo) for m, mo in acc_dap.index]
tables["02_accuracy_by_dap"] = acc_dap
gt_mix = lab[(lab.model == runs[0][0]) & (lab["mode"] == runs[0][1])].groupby(["dap", "gt"]).size().unstack(fill_value=0)
tables["02b_gt_distribution_by_dap"] = gt_mix.reindex(DAPS)

# ---------------------------------------------------------------- 3. per-stage P/R/F1
rows = []
for m, mode in runs:
    s = lab[(lab.model == m) & (lab["mode"] == mode)]
    for st, name in STAGES.items():
        tp = sum(st in p and st in g for p, g in zip(s.pred_set, s.gt_set))
        fp = sum(st in p and st not in g for p, g in zip(s.pred_set, s.gt_set))
        fn = sum(st not in p and st in g for p, g in zip(s.pred_set, s.gt_set))
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec + rec) else 0.0
        rows.append({"run": run_name(m, mode), "model": m, "mode": mode, "stage": name, "gt_support": tp + fn,
                     "predicted": tp + fp, "precision": prec * 100, "recall": rec * 100, "f1": f1 * 100})
stage_tab = pd.DataFrame(rows)
tables["03_per_stage_prf"] = stage_tab.drop(columns=["model", "mode"]).round(1)

# ---------------------------------------------------------------- 4. error types
et_order = ["Exact", "Missed a stage", "Extra stage", "Partly overlapping", "Disjoint", "No answer"]
et = (lab.groupby(["model", "mode"])["error_type"].value_counts(normalize=True).mul(100)
      .unstack().reindex(columns=et_order, fill_value=0).round(1))
et.index = [run_name(m, mo) for m, mo in et.index]
tables["04_error_types"] = et

# ---------------------------------------------------------------- 5. answer diversity (collapse)
rows = []
for m, mode in runs:
    for d in DAPS:
        s = lab[(lab.model == m) & (lab["mode"] == mode) & (lab.dap == d)]
        if s.empty: continue
        cnt = Counter(s.pred)
        p = np.array(list(cnt.values())) / len(s)
        top, top_n = cnt.most_common(1)[0]
        gcnt = Counter(s["gt"])
        rows.append({"run": run_name(m, mode), "dap": d, "top_pred": top, "top_pred_share": 100 * top_n / len(s),
                     "pred_entropy_bits": float(-(p * np.log2(p)).sum()),
                     "gt_top": gcnt.most_common(1)[0][0], "gt_entropy_bits": float(-(np.array(list(gcnt.values())) / len(s) * np.log2(np.array(list(gcnt.values())) / len(s))).sum())})
tables["05_answer_diversity"] = pd.DataFrame(rows).round(2)

# ---------------------------------------------------------------- 6. confusion of stage sets
top_sets = [k for k, _ in Counter(lab["gt"]).most_common(8)]
for m, mode in runs:
    s = lab[(lab.model == m) & (lab["mode"] == mode)]
    conf = pd.crosstab(s["gt"], s.pred).reindex(index=top_sets, fill_value=0)
    keep = conf.sum().sort_values(ascending=False).index[:10]
    tables[f"06_confusion_{m}_{mode}"] = conf[keep]

# ---------------------------------------------------------------- 7. model agreement
rows = []
for mode in MODES:
    piv = lab[lab["mode"] == mode].pivot_table(index=["filename", "dap"], columns="model", values="pred", aggfunc="first")
    cor = lab[lab["mode"] == mode].pivot_table(index=["filename", "dap"], columns="model", values="correct", aggfunc="first")
    for a_, b_ in combinations([m for m in MODELS if m in piv.columns], 2):
        both = piv[[a_, b_]].dropna()
        rows.append({"mode": MODES[mode], "pair": f"{MODELS[a_]} vs {MODELS[b_]}", "same_answer_pct": 100 * (both[a_] == both[b_]).mean()})
    cc = cor.dropna().astype(bool)
    rows.append({"mode": MODES[mode], "pair": "all three wrong", "same_answer_pct": 100 * (~cc).all(axis=1).mean()})
    rows.append({"mode": MODES[mode], "pair": "only gemma4:cloud right",
                 "same_answer_pct": 100 * (cc["gemma4cloud"] & ~cc.drop(columns="gemma4cloud").any(axis=1)).mean()})
tables["07_agreement"] = pd.DataFrame(rows).round(1)

# ---------------------------------------------------------------- 8. reasoning vs correctness
rv = lab.groupby(["model", "mode", "correct"])["score"].mean().mul(100).unstack("correct").round(1)
rv.columns = ["R when wrong", "R when correct"]
rv.index = [run_name(m, mo) for m, mo in rv.index]
tables["08_reasoning_vs_correct"] = rv

for name, t in tables.items():
    t.to_csv(OUT / f"{name}.csv")

# ---------------------------------------------------------------- figures
def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# Fig 1: headline R / Acc / GA, small multiples by metric, colour = model, hatch = AGP
h = tables["01_headline"]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, (col, title) in zip(axes, [("R_macro", "Reasoning validity (R)"), ("Acc_macro", "Accuracy"), ("GA_macro", "Grounded accuracy")]):
    x = np.arange(len(MODELS))
    for j, mode in enumerate(MODES):
        vals = [h[(h.model == MODELS[m]) & (h["mode"] == MODES[mode])][col].squeeze() for m in MODELS]
        bars = ax.bar(x + (j - 0.5) * 0.38, vals, 0.36, color=[COLOR[m] for m in MODELS],
                      alpha=1.0 if mode == "ZeroShot" else 0.55, hatch="" if mode == "ZeroShot" else "////",
                      edgecolor=SURFACE, linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [n.replace(":", "\n") for n in MODELS.values()], fontsize=8)
    ax.set_title(title); ax.set_ylim(0, 110); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("% (macro over DAPs)")
from matplotlib.patches import Patch
fig.subplots_adjust(top=0.78)
fig.legend([Patch(facecolor=MUTED), Patch(facecolor=MUTED, alpha=0.55, hatch="////", edgecolor=SURFACE)],
           ["Zero-shot (solid)", "AGP (hatched)"], loc="upper right", bbox_to_anchor=(0.99, 0.97), ncol=2, fontsize=8)
fig.suptitle("Task 1 growth stage: CropEval metrics (masked dataset)", x=0.01, y=0.97, ha="left", fontweight="bold")
save(fig, "fig1_headline_metrics")

# Fig 2: accuracy by DAP, one panel per mode
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, mode in zip(axes, MODES):
    for m in MODELS:
        idx = run_name(m, mode)
        if idx not in acc_dap.index: continue
        y = acc_dap.loc[idx].values
        ax.plot(DAPS, y, color=COLOR[m], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=MODELS[m])
        ax.annotate(MODELS[m], (len(DAPS) - 1, y[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    ax.set_title(MODES[mode]); ax.set_xlabel("Days after planting (time point)"); ax.set_ylim(0, 100)
    ax.set_xlim(-0.3, len(DAPS) - 0.3); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Exact stage-set accuracy (%)")
fig.subplots_adjust(top=0.78, wspace=0.3)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.99, 0.97), ncol=3, fontsize=8)
fig.suptitle("Accuracy by time point", x=0.01, y=0.97, ha="left", fontweight="bold")
save(fig, "fig2_accuracy_by_dap")

# Fig 3: per-stage recall and precision, rows = metric, colour = model, panels = mode
fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharey=True, sharex=True)
for ci, mode in enumerate(MODES):
    for ri, metric in enumerate(["recall", "precision"]):
        ax = axes[ri, ci]
        x = np.arange(len(STAGES))
        for k, m in enumerate(MODELS):
            s = stage_tab[(stage_tab.model == m) & (stage_tab["mode"] == mode)]
            ax.bar(x + (k - 1) * 0.27, s[metric].fillna(0).values, 0.25, color=COLOR[m], edgecolor=SURFACE,
                   linewidth=1.5, label=MODELS[m])
        ax.set_title(f"{MODES[mode]}: {metric}"); ax.set_ylim(0, 105); ax.grid(axis="x", visible=False)
        ax.set_xticks(x, list(STAGES.values()), fontsize=8)
sup = stage_tab[stage_tab.run == stage_tab.run.iloc[0]].set_index("stage")["gt_support"]
axes[1, 0].set_xticks(np.arange(len(STAGES)), [f"{n}\n(n={int(sup[n])})" for n in STAGES.values()], fontsize=8)
axes[1, 1].set_xticks(np.arange(len(STAGES)), [f"{n}\n(n={int(sup[n])})" for n in STAGES.values()], fontsize=8)
for ax in axes[:, 0]: ax.set_ylabel("%")
fig.subplots_adjust(top=0.84, hspace=0.35)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.945), ncol=3, fontsize=8)
fig.suptitle("Per-stage recall and precision (a stage counts when it is in the predicted set)", x=0.01, y=0.975, ha="left", fontweight="bold")
save(fig, "fig3_per_stage_recall_precision")

# Fig 4: error-type composition, 100% horizontal stacked bars
fig, ax = plt.subplots(figsize=(9, 3.4))
seq = ["#2a78d6", "#86b6ef", "#eda100", "#e87ba4", "#e34948", "#898781"]
left = np.zeros(len(et))
for col, colr in zip(et_order, seq):
    v = et[col].values
    ax.barh(et.index, v, left=left, color=colr, edgecolor=SURFACE, linewidth=2, label=col)
    for y_, (l_, w_) in enumerate(zip(left, v)):
        if w_ >= 7:
            ax.text(l_ + w_ / 2, y_, f"{w_:.0f}", ha="center", va="center", fontsize=7,
                    color="#0b0b0b" if colr in ("#86b6ef", "#eda100", "#e87ba4") else "#ffffff")
    left += v
ax.set_xlim(0, 100); ax.invert_yaxis(); ax.grid(visible=False); ax.set_xlabel("% of labelled images")
ax.legend(ncol=6, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.16))
fig.suptitle("How predictions differ from the ground-truth stage set", x=0.01, y=1.06, ha="left", fontweight="bold")
save(fig, "fig4_error_types")

# Fig 5: answer collapse - share of the single most common predicted set, per DAP
div = tables["05_answer_diversity"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
for ax, mode in zip(axes, MODES):
    for m in MODELS:
        s = div[div.run == run_name(m, mode)]
        ax.plot(s.dap, s.top_pred_share, color=COLOR[m], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=MODELS[m])
    g = div[div.run == run_name("gemma4cloud", mode)]
    gshare = [100 * Counter(lab[(lab.model == "gemma4cloud") & (lab["mode"] == mode) & (lab.dap == d)]["gt"]).most_common(1)[0][1]
              / len(lab[(lab.model == "gemma4cloud") & (lab["mode"] == mode) & (lab.dap == d)]) for d in g.dap]
    ax.plot(g.dap, gshare, color=MUTED, lw=1.5, ls="--", label="ground truth")
    ax.set_title(MODES[mode]); ax.set_ylim(0, 100); ax.set_xlabel("Days after planting"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Share of the most common answer (%)")
fig.subplots_adjust(top=0.72)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.89), ncol=4, fontsize=8)
fig.suptitle("Answer collapse: how often a model gives the same stage set at a time point", x=0.01, y=0.98, ha="left", fontweight="bold")
save(fig, "fig5_answer_collapse")

# Fig 6: confusion heatmaps (rows = top GT sets, % of row), 2 x 3 grid
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
cols = ["1", "2", "2,3", "3", "3,4", "4,5", "1,3", "1,2", "3,5", "4"]
for ci, m in enumerate(MODELS):
    for ri, mode in enumerate(MODES):
        ax = axes[ri, ci]
        s = lab[(lab.model == m) & (lab["mode"] == mode)]
        conf = pd.crosstab(s["gt"], s.pred).reindex(index=top_sets, columns=cols, fill_value=0)
        pct = conf.div(pd.crosstab(s["gt"], s.pred).reindex(index=top_sets, fill_value=0).sum(axis=1), axis=0).fillna(0) * 100
        ax.imshow(pct.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#86b6ef", "#1c5cab", "#0d366b"]), vmin=0, vmax=100, aspect="auto")
        for (i, j), v in np.ndenumerate(pct.values):
            if v >= 5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="#ffffff" if v > 45 else INK)
        ax.set_xticks(range(len(cols)), cols, fontsize=7); ax.set_yticks(range(len(top_sets)), top_sets, fontsize=7)
        ax.grid(visible=False); ax.set_title(f"{MODELS[m]} {MODES[mode]}", fontsize=9)
        if ci == 0: ax.set_ylabel("Ground truth set")
        if ri == 1: ax.set_xlabel("Predicted set")
fig.suptitle("Stage-set confusion (% of each ground-truth row; 1 side shoot, 2 flower, 3 fruit dev., 4 ripening, 5 senescence)",
             x=0.01, ha="left", fontweight="bold")
save(fig, "fig6_confusion")

# ---------------------------------------------------------------- console summary
pd.set_option("display.width", 200, "display.max_columns", 30)
for name in ["01_headline", "02_accuracy_by_dap", "02b_gt_distribution_by_dap", "04_error_types", "07_agreement", "08_reasoning_vs_correct"]:
    print(f"\n=== {name}\n{tables[name]}")
print("\n=== 03 per-stage F1")
print(stage_tab.pivot(index="run", columns="stage", values="f1").reindex(columns=list(STAGES.values())).round(1))
print("\n=== 05 answer diversity (top predicted set / share)")
print(div.pivot(index="run", columns="dap", values="top_pred_share").reindex(columns=DAPS).round(0))
print(div.pivot(index="run", columns="dap", values="top_pred").reindex(columns=DAPS))
print(f"\nTables and figures written to {OUT}")
