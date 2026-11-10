"""
Trend of predicted growth stages across DAP (Task 1, masked combined dataset) for
gemma4:cloud, gemma4e4b and minicpm-v:8b, zero-shot and AGP, compared with ground truth.

1. Stage prevalence: % of images at each time point whose stage set contains stage k.
2. Developmental progression: mean earliest / latest reported stage per time point, and
   Spearman correlation between DAP and the latest reported stage of each image.
3. Per-plant temporal consistency (ground_truth/plant_tracks.csv): along each plant's
   consecutive time points, how often the latest reported stage goes backwards
   (regression), and how often a plant gets the same answer at every time point.
4. Number of stages reported per time point.

Writes output/eda/growth_stage/trend_*.csv and fig7-fig10 PNGs.
Usage:  python eda_stage_trend.py
"""
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
STAGES = {"1": "Side shooting", "2": "Flowering", "3": "Fruit development", "4": "Fruit ripening", "5": "Senescence"}
COLOR = {"gemma4cloud": "#2a78d6", "gemma4e4b": "#eb6834", "minicpmv8b": "#1baf7a", "gt": "#52514e"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})

# ---------------------------------------------------------------- load predictions + GT
rows = []
for m in MODELS:
    for mode in MODES:
        folder = RES / mode / m / "growthStageClassification"
        for r in c.task_rows(folder):
            gt = c.gs_gt.get((r["filename"], r["dap"]))
            rows.append({"run": f"{m}|{mode}", "model": m, "mode": mode, "dap": r["_file_dap"],
                         "filename": r["filename"], "pred": c.stage_set(r.get("growth_stage_numbers", "")),
                         "gt": gt})
df = pd.DataFrame(rows)
df = df[df["gt"].notna()].copy()           # same images for every run: those with a label
# Ground truth as a pseudo-run so it is analysed the same way.
gt_df = df[df.run == df.run.iloc[0]][["dap", "filename", "gt"]].copy()
gt_df["run"], gt_df["model"], gt_df["mode"], gt_df["pred"] = "gt|", "gt", "", gt_df["gt"]
data = pd.concat([df, gt_df], ignore_index=True)
data["n_stages"] = data["pred"].map(len)
data["latest"] = data["pred"].map(lambda s: max(map(int, s)) if s else np.nan)
data["earliest"] = data["pred"].map(lambda s: min(map(int, s)) if s else np.nan)
RUNS = [f"{m}|{mo}" for m in MODELS for mo in MODES] + ["gt|"]
label = lambda run: "Ground truth" if run == "gt|" else f"{MODELS[run.split('|')[0]]} {MODES[run.split('|')[1]]}"

# ---------------------------------------------------------------- 1. stage prevalence
prev = []
for run in RUNS:
    for d in DAPS:
        s = data[(data.run == run) & (data.dap == d)]
        for k, name in STAGES.items():
            prev.append({"run": label(run), "dap": d, "stage": name, "pct": 100 * s["pred"].map(lambda p: k in p).mean()})
prev = pd.DataFrame(prev)
prev_tab = prev.pivot_table(index=["stage", "run"], columns="dap", values="pct").reindex(columns=DAPS).round(1)
prev_tab = prev_tab.reindex(pd.MultiIndex.from_product([list(STAGES.values()), [label(r) for r in RUNS]]))
prev_tab.to_csv(OUT / "trend_01_stage_prevalence.csv")

# Mean absolute gap to ground-truth prevalence (percentage points), per run and stage.
gap = []
for run in RUNS[:-1]:
    for name in STAGES.values():
        p = prev_tab.loc[(name, label(run))].values
        g = prev_tab.loc[(name, "Ground truth")].values
        gap.append({"run": label(run), "stage": name, "mean_abs_gap_pp": np.nanmean(np.abs(p - g))})
gap = pd.DataFrame(gap).pivot(index="run", columns="stage", values="mean_abs_gap_pp")[list(STAGES.values())]
gap["mean"] = gap.mean(axis=1)
gap.round(1).to_csv(OUT / "trend_02_prevalence_gap_vs_gt.csv")

# ---------------------------------------------------------------- 2. progression
prog = data.groupby(["run", "dap"]).agg(earliest=("earliest", "mean"), latest=("latest", "mean"),
                                        n_stages=("n_stages", "mean")).reset_index()
prog["run"] = prog["run"].map(label)
prog_tab = prog.pivot(index="run", columns="dap", values="latest").reindex(columns=DAPS)
prog_tab = prog_tab.reindex([label(r) for r in RUNS]).round(2)
prog_tab.to_csv(OUT / "trend_03_mean_latest_stage.csv")
nst_tab = prog.pivot(index="run", columns="dap", values="n_stages").reindex(columns=DAPS).reindex([label(r) for r in RUNS]).round(2)
nst_tab.to_csv(OUT / "trend_04_mean_stages_reported.csv")

spear = []
for run in RUNS:
    s = data[data.run == run].dropna(subset=["latest"])
    rho_l = pd.Series(s.dap.astype(int).values).corr(pd.Series(s.latest.values), method="spearman")
    rho_e = pd.Series(s.dap.astype(int).values).corr(pd.Series(s.earliest.values), method="spearman")
    # Monotonic: mean latest stage never decreases from one time point to the next.
    seq = prog_tab.loc[label(run)].values
    spear.append({"run": label(run), "spearman_dap_vs_latest": rho_l, "spearman_dap_vs_earliest": rho_e,
                  "mean_latest_monotonic": bool(np.all(np.diff(seq) >= -1e-9)),
                  "latest_range": seq.max() - seq.min()})
spear = pd.DataFrame(spear).round(3)
spear.to_csv(OUT / "trend_05_progression_correlation.csv", index=False)

# ---------------------------------------------------------------- 3. per-plant consistency
tracks = c.load(ROOT / "ground_truth" / "plant_tracks.csv")
cons = []
for run in RUNS:
    lookup = {(r.filename, r.dap): r.pred for r in data[data.run == run].itertuples()}
    steps = regress = forward = same = 0
    stuck = n_tracks = 0
    for t in tracks:
        seq = [lookup.get((t[f"dap_{d}"], d)) for d in DAPS if t[f"dap_{d}"]]
        seq = [s for s in seq if s]
        if len(seq) < 2: continue
        n_tracks += 1
        stuck += all(s == seq[0] for s in seq)
        for a, b in zip(seq, seq[1:]):
            la, lb = max(map(int, a)), max(map(int, b))
            steps += 1
            regress += lb < la; forward += lb > la; same += lb == la
    cons.append({"run": label(run), "tracks": n_tracks, "steps": steps,
                 "forward_pct": 100 * forward / steps, "same_pct": 100 * same / steps,
                 "regression_pct": 100 * regress / steps, "same_answer_all_timepoints_pct": 100 * stuck / n_tracks})
cons = pd.DataFrame(cons).round(1)
cons.to_csv(OUT / "trend_06_track_consistency.csv", index=False)


# ---------------------------------------------------------------- figures
def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def gt_line(ax, y):
    ax.plot(DAPS, y, color=COLOR["gt"], lw=1.5, ls="--", marker="s", ms=4, label="Ground truth")


# Fig 7: stage prevalence, rows = mode, cols = stage
fig, axes = plt.subplots(2, 5, figsize=(14, 5.6), sharey=True, sharex=True)
for ri, mode in enumerate(MODES):
    for ci, name in enumerate(STAGES.values()):
        ax = axes[ri, ci]
        for m in MODELS:
            y = prev_tab.loc[(name, label(f"{m}|{mode}"))].values
            ax.plot(DAPS, y, color=COLOR[m], lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1.2, label=MODELS[m])
        gt_line(ax, prev_tab.loc[(name, "Ground truth")].values)
        ax.set_ylim(-3, 103); ax.grid(axis="x", visible=False)
        if ri == 0: ax.set_title(name)
        if ci == 0: ax.set_ylabel(f"{MODES[mode]}\n% of images with stage")
        if ri == 1: ax.set_xlabel("DAP")
fig.subplots_adjust(top=0.84, hspace=0.25, wspace=0.12)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.945), ncol=4, fontsize=8)
fig.suptitle("How often each growth stage is reported at each time point (dashed = ground truth)", x=0.01, y=0.99,
             ha="left", fontweight="bold")
save(fig, "fig7_stage_prevalence_by_dap")

# Fig 8: mean latest and earliest stage per DAP
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, mode in zip(axes, MODES):
    for m in MODELS:
        s = prog[prog.run == label(f"{m}|{mode}")].set_index("dap").reindex(DAPS)
        ax.plot(DAPS, s.latest, color=COLOR[m], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.2, label=MODELS[m])
        ax.fill_between(DAPS, s.earliest, s.latest, color=COLOR[m], alpha=0.08, lw=0)
    g = prog[prog.run == "Ground truth"].set_index("dap").reindex(DAPS)
    gt_line(ax, g.latest.values)
    ax.set_yticks(range(1, 6), ["1 Side shoot", "2 Flowering", "3 Fruit dev.", "4 Ripening", "5 Senescence"], fontsize=8)
    ax.set_ylim(0.8, 5.2); ax.set_title(MODES[mode]); ax.set_xlabel("Days after planting"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Mean latest reported stage")
fig.subplots_adjust(top=0.76, wspace=0.12)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.9), ncol=4, fontsize=8)
fig.suptitle("Developmental progression: mean latest stage per time point (band reaches the mean earliest stage)",
             x=0.01, y=0.99, ha="left", fontweight="bold")
save(fig, "fig8_stage_progression")

# Fig 9: per-plant transitions between consecutive time points
fig, ax = plt.subplots(figsize=(9, 3.6))
order = [label(r) for r in RUNS]
cc = cons.set_index("run").reindex(order)
segs = [("forward_pct", "Moves forward", "#2a78d6"), ("same_pct", "Same latest stage", "#86b6ef"),
        ("regression_pct", "Goes backwards", "#e34948")]
left = np.zeros(len(cc))
for col, name, colr in segs:
    v = cc[col].values
    ax.barh(order, v, left=left, color=colr, edgecolor=SURFACE, linewidth=2, label=name)
    for y_, (l_, w_) in enumerate(zip(left, v)):
        if w_ >= 6:
            ax.text(l_ + w_ / 2, y_, f"{w_:.0f}", ha="center", va="center", fontsize=7,
                    color="#0b0b0b" if colr == "#86b6ef" else "#ffffff")
    left += v
ax.set_xlim(0, 100); ax.invert_yaxis(); ax.grid(visible=False); ax.set_xlabel("% of consecutive time-point steps per plant")
ax.legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.15))
fig.suptitle("Per-plant consistency: latest reported stage between consecutive time points", x=0.01, y=1.06,
             ha="left", fontweight="bold")
save(fig, "fig9_track_consistency")

# Fig 10: number of stages reported per time point
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
for ax, mode in zip(axes, MODES):
    for m in MODELS:
        ax.plot(DAPS, nst_tab.loc[label(f"{m}|{mode}")].values, color=COLOR[m], lw=2, marker="o", ms=5,
                mec=SURFACE, mew=1.2, label=MODELS[m])
    gt_line(ax, nst_tab.loc["Ground truth"].values)
    ax.set_ylim(0.9, 2.1); ax.set_title(MODES[mode]); ax.set_xlabel("Days after planting"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Mean stages per image")
fig.subplots_adjust(top=0.74, wspace=0.12)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.89), ncol=4, fontsize=8)
fig.suptitle("Number of stages reported per image", x=0.01, y=0.99, ha="left", fontweight="bold")
save(fig, "fig10_stages_per_image")

# ---------------------------------------------------------------- console
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 100)
print("=== Stage prevalence (% of images reporting the stage)\n", prev_tab)
print("\n=== Mean |gap| to ground-truth prevalence (pp)\n", gap.round(1))
print("\n=== Mean latest stage\n", prog_tab)
print("\n=== Mean stages per image\n", nst_tab)
print("\n=== Progression correlation\n", spear)
print("\n=== Per-plant consistency\n", cons)
print(f"\nWritten to {OUT}")
