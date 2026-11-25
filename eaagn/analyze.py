"""Aggregate results across seeds, run paired patient-level bootstrap tests, and emit
LaTeX tables, figures and a numbers file for the paper.

    python -m eaagn.analyze --results results --data data_meta --out paper
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from . import metrics as M

SPLITS = ("test2024", "test2025")
BUDGETS = (0.125, 0.25, 0.5)
POL_ORDER = ["util", "util_hold", "util_kl", "change", "e2e", "periodic", "random", "acoustic", "energy", "oracle"]
POL_NAME = {"util": "Skip-harm utility (ours)", "util_hold": "Hold-only utility", "util_kl": "Label-free utility",
            "change": "Rep.\\ change (AGN-Net obj.)", "e2e": "End-to-end gate (task loss)", "periodic": "Periodic (stalest)", "random": "Random",
            "acoustic": "Acoustic change", "energy": "Energy", "oracle": "Hindsight oracle$^\\dagger$",
            "util_norefresh": "Ours w/o refresh"}
PLAIN = {k: v.replace("\\ ", " ").replace("$^\\dagger$", "") for k, v in POL_NAME.items()}
ARCH_NAME = {"pool": "Cheap encoder + pooling", "tcn": "Causal TCN", "gru": "GRU", "ggru": "Graph GRU (plain)",
             "gconv": "Freq.-conv cell (matched)", "agn": "Acoustic AGN-Net (full update)",
             "agnnf": "AGN-Net w/o fresh path", "agnh": "AGN-Net, harmonic graph"}
N_BOOT = 2000


def load_runs(root):
    runs = defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(root, "*_s[0-9]"))):
        name = os.path.basename(d)
        arch, seed = name.rsplit("_s", 1)
        f = os.path.join(d, "summary.json")
        if os.path.exists(f):
            summ = json.load(open(f))
            fe = os.path.join(d, "rev_e2e.json")
            if os.path.exists(fe) and "grid_C" in summ:
                for k, v in json.load(open(fe))["grid"].items():
                    if k != "full":
                        summ["grid_C"][k] = v
            runs[arch][int(seed)] = dict(dir=d, summary=summ)
    return runs


def patients(meta_dir, split):
    rows = [r for r in csv.DictReader(open(os.path.join(meta_dir, "manifest.csv"))) if r["split"] == split]
    return np.array([r["patient"] for r in rows])


def ms(vals, k=3):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return "--"
    if len(v) == 1:
        return f"{v[0]:.{k}f}"
    out = f"{v.mean():.{k}f}$\\pm${v.std(ddof=1):.{k}f}"
    return "$-$" + out[1:] if out.startswith("-") else out  # true minus sign in tables


def mean(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def rec(d, tag, key):
    p = os.path.join(d, f"rec_{tag}_{key.replace('@', '_')}.npz")
    if not os.path.exists(p) and key.startswith("e2e") and tag == "C":   # revision: end-to-end gate grid
        p = os.path.join(d, f"rec_Ce2e_{key.replace('@', '_')}.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ bootstrap
def boot_groups(pids, rng, n=N_BOOT):
    up, inv = np.unique(pids, return_inverse=True)
    idx_by_p = [np.nonzero(inv == i)[0] for i in range(len(up))]
    for _ in range(n):
        pick = rng.integers(0, len(up), len(up))
        yield np.concatenate([idx_by_p[i] for i in pick])


def ev_F(c):
    s = c.sum(0); return 2 * s[0] / max(s[4] + s[5], 1)


def ev_advF(c):
    s = c.sum(0); tp, nr, ne = s[6:13], s[13:20], s[20:27]
    return 2 * tp[1:].sum() / max(nr[1:].sum() + ne[1:].sum(), 1)


def ev_advR(c):
    s = c.sum(0); return s[7:13].sum() / max(s[14:20].sum(), 1)


def seg_F(c):
    s = c.sum(0); return 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1)


def seg_advF(c):
    s = c.sum(0)
    return 2 * s[7] / max(2 * s[7] + s[8] + s[9], 1) if len(s) >= 10 else float("nan")


def paired_boot(A, B, fn, pids, seed=0):
    """A, B: per-recording count arrays (summed over seeds) for two configs on the same
    recordings. Returns (delta, lo, hi, p_two_sided) with patient-level resampling."""
    rng = np.random.default_rng(seed)
    if len(pids) != len(A):  # only for truncated debug runs (--limit)
        pids = pids[: len(A)]
    d0 = fn(A) - fn(B)
    ds = np.array([fn(A[i]) - fn(B[i]) for i in boot_groups(pids, rng)])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return float(d0), float(lo), float(hi), float(min(p, 1.0))


def seed_sum(arrs):
    arrs = [a for a in arrs if a is not None]
    return sum(arrs[1:], arrs[0].astype(np.int64)) if arrs else None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data_meta")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    runs = load_runs(a.results)
    N = {}  # numbers for the text
    pid = {k: patients(a.data, k) for k in SPLITS}

    # ---------------- Table: architectures (full update), H1
    lines, split_lines = [], {}
    for arch in ["pool", "tcn", "gru", "ggru", "gconv", "agn", "agnh", "agnnf"]:
        if arch not in runs:
            continue
        S = [r["summary"].get("full") or r["summary"].get("grid_A", {}).get("full") for r in runs[arch].values()]
        S = [s for s in S if s]
        params = list(runs[arch].values())[0]["summary"].get("params", 0)
        row = [ARCH_NAME[arch], f"{params / 1e3:.0f}"]
        for sp in SPLITS:
            cells = [ms([s[sp]["F"] for s in S]), ms([s[sp].get("advF") for s in S]),
                     ms([s[sp].get("seg_advF") for s in S]), ms([s[sp]["frame_macroF1"] for s in S]),
                     ms([s[sp].get("seg_HS") for s in S])]
            row += cells
            split_lines.setdefault(sp, []).append(" & ".join([ARCH_NAME[arch], f"{params / 1e3:.0f}"] + cells) + " \\\\")
        lines.append(" & ".join(row) + " \\\\")
        for sp in SPLITS:
            for m in ("F", "advF", "seg_advF", "frame_macroF1", "seg_HS", "segF", "onsetF", "ER", "frame_acc"):
                N[f"{arch}_{sp}_{m}"] = mean([s[sp].get(m) for s in S])
        N[f"{arch}_nseeds"] = len(S)
    open(os.path.join(a.out, "tables", "arch.tex"), "w").write("\n".join(lines) + "\n")
    for sp, ls_ in split_lines.items():  # one block per test set (MDPI layout stacks them)
        if sp.startswith("test"):
            open(os.path.join(a.out, "tables", f"arch_{sp}.tex"), "w").write("\n".join(ls_) + "\n")

    # ---------------- policy grids (AGN): mean over seeds per (tag, policy, budget)
    for arch in ("agn", "agnnf", "ggrusel", "agnh"):
        if arch not in runs:
            continue
        for tag in ("A", "C"):
            G = [r["summary"].get(f"grid_{tag}") for r in runs[arch].values()]
            G = [g for g in G if g]
            if not G:
                continue
            for key in G[0]:
                for sp in SPLITS + ("val",):
                    for m in ("F", "advF", "seg_advF", "segF", "fidF", "fid_advR", "fid_agree", "fid_kl",
                              "frame_macroF1", "upd_frac", "forced_frac", "seg_HS"):
                        N[f"{arch}_{tag}_{key}_{sp}_{m}"] = mean([g[key][sp].get(m) for g in G if key in g])
            N[f"{arch}_{tag}_nseeds"] = len(G)

    # ---------------- Table: policies at matched budgets (C weights, test2024 + test2025)
    def policy_table(arch, tag, fname, metrics_):
        out = []
        G = [r["summary"].get(f"grid_{tag}") for r in runs.get(arch, {}).values()]
        G = [g for g in G if g]
        if not G:
            return
        for b in BUDGETS:
            out.append(f"\\multicolumn{{{1 + len(metrics_) * len(SPLITS)}}}{{l}}{{\\textit{{Budget {b * 100:g}\\% of nodes per step}}}} \\\\")
            best = {}
            for sp in SPLITS:
                for m in metrics_:
                    vals = {p: mean([g[f'{p}@{b}'][sp].get(m) for g in G if f'{p}@{b}' in g]) for p in POL_ORDER if p != "oracle"}
                    best[(sp, m)] = max(vals.values())
            for p in POL_ORDER + ["util_norefresh"]:
                key = f"{p}@{b}"
                if key not in G[0]:
                    continue
                row = [POL_NAME[p]]
                for sp in SPLITS:
                    for m in metrics_:
                        v = [g[key][sp].get(m) for g in G if key in g]
                        cell = ms(v)
                        if p not in ("oracle", "util_norefresh") and abs(mean(v) - best[(sp, m)]) < 1e-12:
                            cell = "\\textbf{" + cell + "}"
                        row.append(cell)
                out.append(" & ".join(row) + " \\\\")
            out.append("\\midrule")
        for key, nm in (("full", "Full update (reference)"), ("none@0.0", "No graph update (cheap path only)")):
            row = [nm]
            for sp in SPLITS:
                for m in metrics_:
                    row.append(ms([g[key][sp].get(m) for g in G]) if all(m in g[key][sp] for g in G) else "--")
            out.append(" & ".join(row) + " \\\\")
        open(os.path.join(a.out, "tables", fname), "w").write("\n".join(out) + "\n")

    policy_table("agn", "C", "policy_C.tex", ["fidF", "fid_advR", "F", "seg_advF"])
    policy_table("agn", "A", "policy_A.tex", ["fidF", "fid_advR", "F", "seg_advF"])
    policy_table("agnnf", "C", "policy_nf_C.tex", ["fidF", "fid_advR", "F", "seg_advF"])
    policy_table("ggrusel", "C", "policy_gg_C.tex", ["fidF", "fid_advR", "F", "seg_advF"])

    # ---------------- ranking diagnostics (combined A and C columns)
    DA = [r["summary"].get("diag_A") for r in runs.get("agn", {}).values()]
    DC = [r["summary"].get("diag_C") for r in runs.get("agn", {}).values()]
    DA, DC = [d for d in DA if d], [d for d in DC if d]
    if DA and DC:
        rows = []
        for nm in ["util", "util_hold", "util_kl", "change", "periodic", "random", "acoustic", "energy"]:
            rows.append(f"{POL_NAME[nm]} & {ms([d['spearman'][nm] for d in DA])} & {ms([d['topk_overlap'][nm] for d in DA])} & "
                        f"{ms([d['spearman'][nm] for d in DC])} & {ms([d['topk_overlap'][nm] for d in DC])} \\\\")
        open(os.path.join(a.out, "tables", "diag.tex"), "w").write("\n".join(rows) + "\n")
    for arch in ("agn", "ggrusel", "agnh"):
        for tag in ("A", "C"):
            D = [r["summary"].get(f"diag_{tag}") for r in runs.get(arch, {}).values()]
            D = [d for d in D if d]
            if not D:
                continue
            rows = []
            for nm in ["util", "util_hold", "util_kl", "change", "periodic", "random", "acoustic", "energy"]:
                rows.append(f"{POL_NAME[nm]} & {ms([d['spearman'][nm] for d in D])} & {ms([d['topk_overlap'][nm] for d in D])} \\\\")
                N[f"{arch}_diag{tag}_{nm}_rho"] = mean([d["spearman"][nm] for d in D])
                N[f"{arch}_diag{tag}_{nm}_topk"] = mean([d["topk_overlap"][nm] for d in D])
            N[f"{arch}_diag{tag}_conc_util"] = mean([d["conc_util_top25"] for d in D])
            N[f"{arch}_diag{tag}_conc_change"] = mean([d["conc_change_top25"] for d in D])
            N[f"{arch}_diag{tag}_conc_util_sqrt"] = mean([d.get("conc_util_sqrt_top25", float("nan")) for d in D])
            N[f"{arch}_diag{tag}_conc_change_sq"] = mean([d.get("conc_change_sq_top25", float("nan")) for d in D])
            open(os.path.join(a.out, "tables", f"diag_{tag}.tex"), "w").write("\n".join(rows) + "\n")

    # ---------------- paired patient-level bootstrap (seed-summed counts)
    tests = []
    for barch in ("agn", "ggrusel", "agnh"):
      if barch not in runs:
          continue
      dirs = [r["dir"] for r in runs[barch].values()]
      pref = {"agn": "", "ggrusel": "gg", "agnh": "h"}[barch]
      if True:
        for tag in ("C", "A"):
            for sp in SPLITS:
                for b in BUDGETS:
                    U = {p: seed_sum([(rec(d, tag, f"{p}@{b}") or {}).get(f"{sp}_fid") for d in dirs]) for p in POL_ORDER}
                    E = {p: seed_sum([(rec(d, tag, f"{p}@{b}") or {}).get(f"{sp}_ev") for d in dirs]) for p in POL_ORDER}
                    for other in ["random", "periodic", "change", "e2e", "acoustic", "energy", "util_kl", "util_hold"]:
                        if U["util"] is None or U[other] is None:
                            continue
                        for mname, fn, src in (("fidF", ev_F, U), ("fid_advR", ev_advR, U), ("F", ev_F, E)):
                            dlt, lo, hi, p = paired_boot(src["util"], src[other], fn, pid[sp])
                            tests.append(dict(tag=pref + tag, split=sp, budget=b, a="util", b=other, metric=mname,
                                              delta=dlt, lo=lo, hi=hi, p=p))
                # non-inferiority of selective vs full (event F, adv segment F) against margin
                full = seed_sum([(rec(d, tag, "full") or {}).get(f"{sp}_ev") for d in dirs])
                fsg = seed_sum([(rec(d, tag, "full") or {}).get(f"{sp}_sg") for d in dirs])
                for b in BUDGETS:
                    for p_ in ("util", "random", "periodic"):
                        e = seed_sum([(rec(d, tag, f"{p_}@{b}") or {}).get(f"{sp}_ev") for d in dirs])
                        g = seed_sum([(rec(d, tag, f"{p_}@{b}") or {}).get(f"{sp}_sg") for d in dirs])
                        if e is None or full is None:
                            continue
                        for mname, fn, X, Y in (("F", ev_F, e, full), ("advF", ev_advF, e, full), ("seg_advF", seg_advF, g, fsg)):
                            dlt, lo, hi, pp = paired_boot(X, Y, fn, pid[sp])
                            tests.append(dict(tag=pref + tag, split=sp, budget=b, a=p_, b="full", metric=mname,
                                              delta=dlt, lo=lo, hi=hi, p=pp))
    # architecture comparison (AGN full vs baselines; event metrics + segment adv F)
    for base in ["gru", "tcn", "ggru", "gconv", "pool"]:
        if base not in runs or "agn" not in runs:
            continue
        for sp in SPLITS:
            A_ = seed_sum([(rec(r["dir"], "A", "full") or {}).get(f"{sp}_ev") for r in runs["agn"].values()])
            B_ = seed_sum([(rec(r["dir"], "A", "full") or {}).get(f"{sp}_ev") for r in runs[base].values()])
            As = seed_sum([(rec(r["dir"], "A", "full") or {}).get(f"{sp}_sg") for r in runs["agn"].values()])
            Bs = seed_sum([(rec(r["dir"], "A", "full") or {}).get(f"{sp}_sg") for r in runs[base].values()])
            if A_ is None or B_ is None:
                continue
            for mname, fn, X, Y in (("F", ev_F, A_, B_), ("advF", ev_advF, A_, B_), ("seg_advF", seg_advF, As, Bs)):
                if X is None or Y is None:
                    continue
                dlt, lo, hi, p = paired_boot(X, Y, fn, pid[sp])
                tests.append(dict(tag="arch", split=sp, budget=1.0, a="agn", b=base, metric=mname, delta=dlt, lo=lo, hi=hi, p=p))
    json.dump(tests, open(os.path.join(a.out, "bootstrap.json"), "w"), indent=1)
    for t in tests:
        k = f"bt_{t['tag']}_{t['split']}_{t['budget']:g}_{t['a']}_{t['b']}_{t['metric']}"
        for f in ("delta", "lo", "hi", "p"):
            N[f"{k}_{f}"] = t[f]

    # ---------------- runtime
    for kind, pat in (("torch", "bench_C_t1.json"), ("numpy", "bench_np_C.json")):
        B = [json.load(open(os.path.join(r["dir"], pat))) for r in runs.get("agn", {}).values() if os.path.exists(os.path.join(r["dir"], pat))]
        if not B:
            continue
        rows = []
        for key in ["full@1.0", "none@0.0", "random@0.25", "periodic@0.25", "acoustic@0.25", "util@0.125", "util@0.25", "util@0.5"]:
            if key not in B[0]:
                continue
            g = lambda f: mean([f(b[key]) for b in B])
            st = {s: g(lambda x, s=s: x["stage_mean_ms"][s]) for s in ["features", "encoder", "selection", "graph", "readout"]}
            nm, bud = key.split("@")
            label = ("Full update" if nm == "full" else "Cheap path only" if nm == "none" else
                     f"{PLAIN.get(nm, nm)} @ {float(bud) * 100:g}\\%")
            sp_ = [b[key].get("speedup_median", float("nan")) for b in B]
            rows.append(f"{label} & {st['features']:.3f} & {st['encoder']:.3f} & {st['selection']:.3f} & {st['graph']:.3f} & "
                        f"{st['readout']:.3f} & {g(lambda x: x['mean_ms']):.3f} & {g(lambda x: x['p95_ms']):.3f} & "
                        f"{ms(sp_, 2)}$\\times$ \\\\")
            for s in st:
                N[f"rt_{kind}_{key}_{s}"] = st[s]
            N[f"rt_{kind}_{key}_speedup"] = mean(sp_)
            N[f"rt_{kind}_{key}_speedup_red"] = 1.0 - 1.0 / mean(sp_)
            for m in ("mean_ms", "median_ms", "p95_ms", "p99_ms", "max_ms", "rtf"):
                N[f"rt_{kind}_{key}_{m}"] = g(lambda x, m=m: x[m])
            if "macs_total" in B[0][key]:
                N[f"rt_{kind}_{key}_macs"] = g(lambda x: x["macs_total"])
        N[f"rt_{kind}_equiv"] = max(max(b["equivalence"].values()) for b in B)
        N[f"rt_{kind}_cpu"] = B[0].get("cpu", "").replace("Model name:", "").replace("(R)", "").replace("CPU", "").strip()
        open(os.path.join(a.out, "tables", f"runtime_{kind}.tex"), "w").write("\n".join(rows) + "\n")

    # ---------------- locality: does per-recording concentration predict selective fidelity?
    from scipy.stats import spearmanr
    val_names = [l.split(",")[0] for l in open(os.path.join(a.data, "val_pg_names.txt")).read().split("\n") if l]
    val_index = {nm: i for i, nm in enumerate(val_names)}
    loc = defaultdict(list)
    for r in runs.get("agn", {}).values():
        d = r["summary"].get("diag_C")
        if not d:
            continue
        # per_rec_conc is stored in evaluation-batch order (sorted by duration); the per-recording
        # count arrays are in split order, so pair the two by recording NAME, not by position
        cu = np.array([x[1] for x in d["per_rec_conc"]]); cc = np.array([x[2] for x in d["per_rec_conc"]])
        row = np.array([val_index[x[0]] for x in d["per_rec_conc"]])
        for pol_ in ("util", "random", "periodic", "change"):
            rr = rec(r["dir"], "C", f"{pol_}@0.25")
            if rr is None or "val_fid" not in rr:
                continue
            fc = rr["val_fid"][row]
            denom = fc[:, 4] + fc[:, 5]
            f = np.where(denom > 0, 2 * fc[:, 0] / np.maximum(denom, 1), np.nan)
            ok = np.isfinite(f) & np.isfinite(cu) & np.isfinite(cc)
            loc[f"{pol_}_rho_util_conc"].append(spearmanr(cu[ok], f[ok]).correlation)
            loc[f"{pol_}_rho_change_conc"].append(spearmanr(cc[ok], f[ok]).correlation)
            loc[f"{pol_}_fid"].append(f)
        if "util_fid" in loc and "random_fid" in loc:
            g = loc["util_fid"][-1] - loc["random_fid"][-1]
            ok = np.isfinite(g) & np.isfinite(cu)
            loc["gain_rho_util_conc"].append(spearmanr(cu[ok], g[ok]).correlation)
            loc["gain_rho_change_conc"].append(spearmanr(cc[ok], g[ok]).correlation)
    for k, v in loc.items():
        if not k.endswith("_fid"):
            N[f"loc_{k}"] = mean(v)
            N[f"loc_{k}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")

    # ---------------- per-seed contrasts (training-run level; complements the pooled bootstrap)
    for arch in ("agn", "agnnf", "ggrusel", "agnh"):
        G = [r["summary"].get("grid_C") for r in runs.get(arch, {}).values()]
        G = [g for g in G if g]
        if not G:
            continue
        for sp in SPLITS:
            for b in BUDGETS:
                for other in ["random", "periodic", "change", "e2e", "acoustic", "energy", "util_kl", "util_hold", "oracle"]:
                    for m in ("fidF", "fid_advR", "F"):
                        ds = [g[f"util@{b}"][sp][m] - g[f"{other}@{b}"][sp][m] for g in G if f"{other}@{b}" in g]
                        if ds:
                            k = f"ps_{arch}_{sp}_{b:g}_util_{other}_{m}"
                            N[k + "_min"], N[k + "_max"], N[k + "_npos"] = min(ds), max(ds), float(sum(d > 0 for d in ds))
                for p_ in ("util", "random", "periodic"):
                    ds = [g[f"{p_}@{b}"][sp]["F"] - g["full"][sp]["F"] for g in G]
                    k = f"ps_{arch}_{sp}_{b:g}_{p_}_full_F"
                    N[k + "_min"], N[k + "_max"] = min(ds), max(ds)
            # how many of the full model's adventitious detections match an annotated adventitious event
            tp = ne = 0
            for r in runs[arch].values():
                rr = rec(r["dir"], "C", "full")
                if rr is not None and f"{sp}_ev" in rr:
                    e = rr[f"{sp}_ev"].sum(0); tp += e[7:13].sum(); ne += e[21:27].sum()
            N[f"{arch}_C_full_{sp}_advP"] = float(tp / max(ne, 1))
            N[f"{arch}_C_full_{sp}_adv_tp"] = float(tp); N[f"{arch}_C_full_{sp}_adv_nest"] = float(ne)

    # ---------------- data-gated verdict strings for the graph-GRU transfer experiment
    def _g(k):
        return N.get(k, float("nan"))
    gg_have = "ggrusel_C_nseeds" in N
    gg_ok = gg_have and all(
        _g(f"bt_ggC_test2024_0.25_util_{o}_fidF_lo") > 0 and _g(f"ps_ggrusel_test2024_0.25_util_{o}_fidF_npos") == N["ggrusel_C_nseeds"]
        for o in ("random", "change"))
    N["gg_ok"] = float(gg_ok)
    if gg_ok:
        N["gg_intro"] = ("Retraining the same controller recipe on a plain graph gated recurrent unit (GRU) cell, with unchanged hyperparameters, "
                         "gives the same advantage in event preservation (Section~\\ref{sec:gg}).")
        N["gg_abs"] = ("Retrained with the same recipe, the controller shows the same preservation advantage on a plain graph-GRU cell.")
        N["gg_verdict"] = ("For this cell too, the skip-harm controller beats the change objective and random selection in event "
                           "preservation in every seed, so the advantage is not specific to the AGN-Net cell.")
        N["gg_disc"] = ("Retraining the same recipe on a plain graph-GRU cell gives the same ordering, so the advantage is not specific to "
                        "the AGN-Net cell. It is most plausibly due to the training target, although we tested only two graph cells "
                        "on one dataset.")
    elif gg_have:
        N["gg_intro"] = "We also apply the unchanged pipeline to a plain graph-GRU cell (Section~\\ref{sec:gg})."
        N["gg_abs"] = ""
        N["gg_verdict"] = "For this cell, the ordering observed with the AGN-Net cell is not reproduced in every seed with confidence."
        N["gg_disc"] = "Whether the advantage holds beyond the AGN-Net cell remains open (Section~\\ref{sec:gg})."
    else:  # transfer experiment not available: omit the section and every reference to it
        N["gg_intro"] = ""
        N["gg_abs"] = ""
        N["gg_verdict"] = ""
        N["gg_disc"] = ("Whether the advantage holds for other cells, such as the stronger graph GRU, remains to be tested.")
    N["gg_h1"] = ("Because the selection mechanism is independent of the cell, we also test it on the graph-GRU cell "
                  "(Section~\\ref{sec:gg})." if gg_have else "")
    open(os.path.join(a.out, "ggflag.tex"), "w").write("\\ggtrue\n" if gg_have else "\\ggfalse\n")

    json.dump(N, open(os.path.join(a.out, "numbers.json"), "w"), indent=1, default=float)
    figures(runs, N, a.out)
    print(f"wrote tables/figures; {len(tests)} bootstrap tests")


# ------------------------------------------------------------------ figures
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]


def figures(runs, N, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                         "font.family": "serif", "pdf.fonttype": 42})
    pols = ["util", "change", "e2e", "periodic", "random", "acoustic", "energy", "oracle"]
    for arch, tag in (("agn", "C"), ("agn", "A")):
        if f"{arch}_{tag}_nseeds" not in N:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.0))
        panels = [("test2024", "fidF", r"(a) $F_\mathrm{pres}$, 2024"), ("test2024", "fid_advR", r"(b) $R^\mathrm{adv}_\mathrm{pres}$, 2024"),
                  ("test2025", "fidF", r"(c) $F_\mathrm{pres}$, 2025"), ("test2025", "fid_advR", r"(d) $R^\mathrm{adv}_\mathrm{pres}$, 2025")]
        for ax, (sp, m, title) in zip(axes, panels):
            for i, p in enumerate(pols):
                ys = [N.get(f"{arch}_{tag}_{p}@{b}_{sp}_{m}", np.nan) for b in BUDGETS]
                ax.plot([b * 100 for b in BUDGETS], ys, color=COL[i], marker=MARK[i], ms=4, lw=1.6 if p == "util" else 1.0,
                        ls="--" if p == "oracle" else "-", label=PLAIN[p], zorder=3 if p == "util" else 2)
            none = N.get(f"{arch}_{tag}_none@0.0_{sp}_{m}", np.nan)
            ax.axhline(none, color="#8a8984", lw=0.8, ls=":")
            ax.set_xticks([12.5, 25, 50]); ax.set_xticklabels(["12.5", "25", "50"])
            ax.set_xlabel("budget (% of nodes)"); ax.set_title(title, fontsize=7.5)
            ax.set_ylim(-0.02, 1.0)
        axes[0].set_ylabel("preservation vs. full update")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=7, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(os.path.join(out, "figures", f"budget_{tag}.pdf"), bbox_inches="tight")
        plt.close(fig)

    # runtime stacked bars (numpy runtime)
    keys = ["full@1.0", "util@0.5", "util@0.25", "util@0.125", "random@0.25", "none@0.0"]
    for kind in ("numpy", "torch"):
        if f"rt_{kind}_full@1.0_mean_ms" not in N:
            continue
        fig, ax = plt.subplots(figsize=(3.5, 1.9))
        stages = ["features", "encoder", "selection", "graph", "readout"]
        labels = ["Full", "Ours 50%", "Ours 25%", "Ours 12.5%", "Random 25%", "Cheap only"]
        left = np.zeros(len(keys))
        for si, s in enumerate(stages):
            v = np.array([N.get(f"rt_{kind}_{k}_{s}", 0) for k in keys])
            ax.barh(range(len(keys)), v, left=left, color=COL[si], edgecolor="white", linewidth=0.8, height=0.62, label=s)
            left += v
        for i, t in enumerate(left):
            ax.text(t + 0.02, i, f"{t:.2f}", va="center", fontsize=6.5, color="#0b0b0b")
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels); ax.invert_yaxis()
        ax.set_xlabel(f"Mean latency per 40 ms chunk (ms), {'NumPy' if kind == 'numpy' else 'PyTorch'} runtime, one CPU core")
        ax.grid(axis="y", visible=False)
        ax.legend(ncol=5, fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0))
        fig.tight_layout()
        fig.savefig(os.path.join(out, "figures", f"runtime_{kind}.pdf"), bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
