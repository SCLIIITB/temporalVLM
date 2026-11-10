"""Compute TomatoTemporal benchmark metrics (T1-T4) from the per-task result CSVs.

Usage:
    python compute_benchmark_metrics.py output/AGP/gemma4cloud
"""

import csv, glob, collections, itertools, os, statistics, sys

ROOT = sys.argv[1]
os.chdir(ROOT)


def load(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        out += list(csv.DictReader(open(f, encoding="utf-8", errors="replace")))
    return out


def macro_f1(yt, yp, labels=None):
    labels = labels or sorted(set(yt) | set(yp))
    fs, per = [], {}
    for l in labels:
        tp = sum(t == l and p == l for t, p in zip(yt, yp))
        fp = sum(t != l and p == l for t, p in zip(yt, yp))
        fn = sum(t == l and p != l for t, p in zip(yt, yp))
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0
        fs.append(f); per[l] = (round(f, 3), sum(t == l for t in yt))
    return sum(fs) / len(fs), per


def acc(yt, yp):
    return sum(t == p for t, p in zip(yt, yp)) / len(yt)


def stages(s):
    return frozenset(int(x) for x in s.replace(" ", "").split(",") if x)


# ---------------- T1 growth stage ----------------
print("=== T1 growth stage ===")
for dap in ["54", "75", "97", "131", "ALL"]:
    rows = load("growthStageClassification/gs_eval_*.csv")
    rows = [r for r in rows if not r["error"] and r["gt"].strip() and r["growth_stage_numbers"].strip()]
    if dap != "ALL":
        rows = [r for r in rows if r["dap"] == dap]
    P = [stages(r["growth_stage_numbers"]) for r in rows]
    G = [stages(r["gt"]) for r in rows]
    exact = acc(G, P)
    overlap = sum(bool(p & g) for p, g in zip(P, G)) / len(P)
    jacc = statistics.mean(len(p & g) / len(p | g) for p, g in zip(P, G))
    # multi-label per-stage macro F1 (over stages that occur in GT or pred)
    labs = sorted(set().union(*P, *G))
    f1s = []
    per = {}
    for l in labs:
        tp = sum(l in p and l in g for p, g in zip(P, G))
        fp = sum(l in p and l not in g for p, g in zip(P, G))
        fn = sum(l not in p and l in g for p, g in zip(P, G))
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0
        f1s.append(f); per[l] = round(f, 3)
    # set-as-class macro F1 (each stage combination is a class)
    sf1, _ = macro_f1([tuple(sorted(g)) for g in G], [tuple(sorted(p)) for p in P])
    omae = statistics.mean(abs(statistics.mean(p) - statistics.mean(g)) for p, g in zip(P, G))
    npred2 = sum(len(p) == 2 for p in P); ngt2 = sum(len(g) == 2 for g in G)
    print(f"DAP {dap}: n={len(P)} exact={exact:.3f} overlap={overlap:.3f} jacc={jacc:.3f} "
          f"mlF1={sum(f1s)/len(f1s):.3f} setF1={sf1:.3f} ordMAE={omae:.3f} pred2={npred2} gt2={ngt2} per={per}")

# ---------------- T2 ripeness transition ----------------
print("=== T2 ripeness transition ===")
MAP = {"Raw": "No transition", "Semi-ripe": "Raw to Semi-ripe", "Ripe": "Raw to Ripe (skipped stage)"}
allrows = load("ripeningTransitionResults/rt_eval_*.csv")
print("total rows", len(allrows), "errors", sum(1 for r in allrows if r["error"]))
rows = [r for r in allrows if not r["error"] and r["gt"] in MAP]
for name, sub in [("ALL", rows)] + [(f"gap {g}", [r for r in rows if int(r["later_dap"]) - int(r["earlier_dap"]) == g]) for g in sorted({int(r["later_dap"]) - int(r["earlier_dap"]) for r in rows})] \
        + [(f"pair {a}->{b}", [r for r in rows if (r['earlier_dap'], r['later_dap']) == (a, b)]) for a, b in sorted({(r['earlier_dap'], r['later_dap']) for r in rows}, key=lambda x: (int(x[0]), int(x[1])))]:
    if not sub: continue
    yt = [MAP[r["gt"]] for r in sub]; yp = [r["transition_type"] for r in sub]
    f, per = macro_f1(yt, yp)
    print(f"{name}: n={len(sub)} acc={acc(yt, yp):.3f} macroF1={f:.3f} gtdist={dict(collections.Counter(yt))} per={per}")
yt = [MAP[r["gt"]] for r in rows]; yp = [r["transition_type"] for r in rows]
print("confusion:", collections.Counter(zip(yt, yp)))
print("later-stage acc", acc([r["gt"] for r in rows], [r["later_stage"] for r in rows]))
print("consistency", sum(r["stage_order_consistent"] == "TRUE" for r in rows) / len(rows))
print("earlier stage dist", collections.Counter(r["earlier_stage"] for r in rows))

# ---------------- T3 re-id ----------------
print("=== T3 reid ===")
allrows = load("reid/reid_same_eval_*.csv")
print("total rows", len(allrows), "errors", sum(1 for r in allrows if r["error"]))
rows = [r for r in allrows if not r["error"]]
for name, sub in [("ALL", rows)] + [(f"qDAP {d}", [r for r in rows if r["query_dap"] == d]) for d in ["54", "75", "97", "131"]] \
        + [(f"gap {g}", [r for r in rows if abs(int(r["query_dap"]) - int(r["candidate_dap"])) == g]) for g in sorted({abs(int(r["query_dap"]) - int(r["candidate_dap"])) for r in rows})] \
        + [(f"dir {'fwd' if f else 'bwd'}", [r for r in rows if (int(r['candidate_dap']) > int(r['query_dap'])) == f]) for f in [True, False]]:
    yt = [r["true_candidate_index"] for r in sub]; yp = [r["matching_candidate_index"] for r in sub]
    f, _ = macro_f1(yt, yp)
    print(f"{name}: n={len(sub)} acc={acc(yt, yp):.3f} macroF1={f:.3f}")
print("pred idx", collections.Counter(r["matching_candidate_index"] for r in rows))
print("true idx", collections.Counter(r["true_candidate_index"] for r in rows))
cc = [float(r["confidence"]) for r in rows if r["correct"] == "True"]
ci = [float(r["confidence"]) for r in rows if r["correct"] == "False"]
print(f"conf correct {statistics.mean(cc):.3f} (n={len(cc)}) incorrect {statistics.mean(ci):.3f} (n={len(ci)})")
print("angles", collections.Counter(r["query_angle"] for r in rows))
txt = sum(1 for r in rows if any(k in r["reasoning"].lower() for k in ["label", "text", "tag", "r5t", "r1t", "r2t", "r3t", "r4t", "note"]))
print("reasoning mentioning label/text:", txt)
for r in rows:
    if r["correct"] == "False":
        print("  WRONG", r["query_id"], r["query_dap"], "->", r["candidate_dap"], r["confidence"])

# ---------------- T4 counting ----------------
print("=== T4 ripeness count ===")
allrows = load("ripenessCount/rpc_eval_*.csv")
rows = [r for r in allrows if not r["error"] and r.get("raw_count_gt") not in (None, "")]
K = ["raw", "semi_ripe", "ripe"]
for d in ["54", "75", "97", "ALL"]:
    sub = rows if d == "ALL" else [r for r in rows if r["dap"] == d]
    out = {}
    for k in K:
        out[k] = statistics.mean(abs(int(r[k + "_count"]) - int(r[k + "_count_gt"])) for r in sub)
    gt_tot = [sum(int(r[k + "_count_gt"]) for k in K) for r in sub]
    pr_tot = [sum(int(r[k + "_count"]) for k in K) for r in sub]
    out["total"] = statistics.mean(abs(a - b) for a, b in zip(gt_tot, pr_tot))
    bias = statistics.mean(b - a for a, b in zip(gt_tot, pr_tot))
    exact = acc(gt_tot, pr_tot)
    ripe_bias = {k: statistics.mean(int(r[k + "_count"]) - int(r[k + "_count_gt"]) for r in sub) for k in K}
    print(f"DAP {d}: n={len(sub)} MAE={ {k: round(v, 2) for k, v in out.items()} } bias_total={bias:.2f} exact_total={exact:.3f} "
          f"meanGT={statistics.mean(gt_tot):.2f} meanPred={statistics.mean(pr_tot):.2f} bias_per_class={ {k: round(v, 2) for k, v in ripe_bias.items()} }")
    # dominant ripeness class agreement (only images with fruit)
# derived pairwise comparison (same DAP, same angle, different plants)
pairs = []
for d in ["54", "75", "97"]:
    sub = [r for r in rows if r["dap"] == d]
    for a, b in itertools.combinations(sub, 2):
        if a["angle"] != b["angle"]: continue
        ga = sum(int(a[k + "_count_gt"]) for k in K); gb = sum(int(b[k + "_count_gt"]) for k in K)
        pa = int(a["raw_count"]) + int(a["semi_ripe_count"]) + int(a["ripe_count"])
        pb = int(b["raw_count"]) + int(b["semi_ripe_count"]) + int(b["ripe_count"])
        gw = "A" if ga > gb else "B" if gb > ga else "Equal"
        pw = "A" if pa > pb else "B" if pb > pa else "Equal"
        pairs.append((d, gw, pw, abs(ga - gb), abs(pa - pb)))
print("pairs", len(pairs))
for name, cond in [("all", lambda x: True), ("easy >8", lambda x: x[3] > 8), ("medium 4-8", lambda x: 4 <= x[3] <= 8), ("hard 1-3", lambda x: 1 <= x[3] <= 3), ("tie 0", lambda x: x[3] == 0)]:
    s = [p for p in pairs if cond(p)]
    if not s: continue
    f, _ = macro_f1([p[1] for p in s], [p[2] for p in s])
    print(f"{name}: n={len(s)} acc={acc([p[1] for p in s], [p[2] for p in s]):.3f} F1={f:.3f} diffMAE={statistics.mean(abs(p[3]-p[4]) for p in s):.2f}")
for d in ["54", "75", "97"]:
    s = [p for p in pairs if p[0] == d]
    f, _ = macro_f1([p[1] for p in s], [p[2] for p in s])
    print(f"DAP {d}: n={len(s)} acc={acc([p[1] for p in s], [p[2] for p in s]):.3f} F1={f:.3f} diffMAE={statistics.mean(abs(p[3]-p[4]) for p in s):.2f}")
