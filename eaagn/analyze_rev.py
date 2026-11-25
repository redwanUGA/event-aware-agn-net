"""Tables and numbers for the revision experiments (run after eaagn.analyze; merges into numbers.json).

    python -m eaagn.analyze_rev --results results --results_hfl results_hfl --hfl_meta data_hflung_meta --out paper
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from .analyze import BUDGETS, POL_NAME, ev_F, ev_advR, load_runs, mean, ms, paired_boot, rec, seed_sum, seg_advF

TOL_ROWS = [("ev_c0.2", "Event, 0.2\\,s collar (main)"), ("ev_c0.5", "Event, 0.5\\,s collar"), ("ev_c1.0", "Event, 1\\,s collar"),
            ("onset_c0.5", "Onset only, 0.5\\,s"), ("onset_c1.0", "Onset only, 1\\,s"), ("overlap", "Any overlap"),
            ("seg1_F", "Segment 1\\,s, all classes"), ("seg2_F", "Segment 2\\,s, all classes"),
            ("seg1_advF", "Segment 1\\,s, adventitious"), ("seg2_advF", "Segment 2\\,s, adventitious")]


def jl(d, name):
    p = os.path.join(d, f"rev_{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def f3(v):
    return "--" if v is None or not np.isfinite(v) else f"{v:.3f}".replace("-", "$-$", 1) if v < 0 else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--results_hfl", default="results_hfl")
    ap.add_argument("--hfl_meta", default="data_hflung_meta")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    runs = load_runs(a.results)
    Np = os.path.join(a.out, "numbers.json")
    N = json.load(open(Np))
    T = os.path.join(a.out, "tables")

    # ---------------------------------------------------------------- R1.1 tolerant metrics (AGN, stage C)
    C = [jl(r["dir"], "collars") for r in runs.get("agn", {}).values()]
    C = [c for c in C if c]
    if C:
        cols = [("full@1.0", ""), ("util@0.25", ""), ("change@0.25", ""), ("random@0.25", ""),
                ("util@0.25", "pres_"), ("change@0.25", "pres_"), ("random@0.25", "pres_")]
        rows = []
        for key, lab in TOL_ROWS:
            cells = [lab]
            for cfg, pre in cols:
                v = mean([c[cfg]["test2024"][pre + key] for c in C if cfg in c])
                cells.append(f3(v))
                N[f"tol_{pre}{cfg}_{key}"] = v
                N[f"tol25_{pre}{cfg}_{key}"] = mean([c[cfg]["test2025"][pre + key] for c in C if cfg in c])
            rows.append(" & ".join(cells) + " \\\\")
        open(os.path.join(T, "tolerant.tex"), "w").write("\n".join(rows) + "\n")
        N["tol_nseeds"] = len(C)

    # ---------------------------------------------------------------- R1.3 / R2.1 end-to-end gate diagnostics
    for arch in ("agn", "ggrusel"):
        D = [m for m in (jl(r["dir"], "e2ediag") for r in runs.get(arch, {}).values()) if m]
        for nm in ("e2e", "util", "change", "random"):
            N[f"e2ediag_{arch}_{nm}_rho"] = mean([d["spearman"].get(nm) for d in D]) if D else float("nan")
            N[f"e2ediag_{arch}_{nm}_topk"] = mean([d["topk_overlap"].get(nm) for d in D]) if D else float("nan")
        E = [m for m in (jl(r["dir"], "e2e") for r in runs.get(arch, {}).values()) if m and m.get("hist")]
        if E:
            N[f"e2e_{arch}_loss_first"] = mean([e["hist"][0]["loss"] for e in E])
            N[f"e2e_{arch}_loss_last"] = mean([e["hist"][-1]["loss"] for e in E])
            N[f"e2e_{arch}_epoch_s"] = mean([np.mean([h["sec"] for h in e["hist"]]) for e in E])

    # ---------------------------------------------------------------- R1.6 multi-step harm
    combos = [(1, 1), (1, 5), (1, 25), (5, 5), (5, 25), (25, 25)]
    MS = {arch: [m for m in (jl(r["dir"], "multistep") for r in runs.get(arch, {}).values()) if m] for arch in ("agn", "ggrusel")}
    if MS["agn"]:
        rows = []
        for k, H in combos:
            key = f"k{k}_H{H}"
            cells = [f"{k} & {H}"]
            for arch, scs in (("agn", ("onestep", "util", "change")), ("ggrusel", ("onestep", "util", "change"))):
                for sc in scs:
                    v = mean([m["spearman"].get(f"{key}|{sc}") for m in MS[arch]]) if MS[arch] else float("nan")
                    N[f"ms_{arch}_{key}_{sc}"] = v
                    N[f"ms_{arch}_{key}_{sc}_top"] = mean([m["spearman"].get(f"{key}|{sc}_top") for m in MS[arch]]) if MS[arch] else float("nan")
                    cells.append(f3(v))
            N[f"ms_agn_{key}_ratio"] = mean([m["harm_ratio_to_onestep"].get(key) for m in MS["agn"]])
            cells.append(f"{N[f'ms_agn_{key}_ratio']:.2f}")
            rows.append(" & ".join(cells) + " \\\\")
        open(os.path.join(T, "multistep.tex"), "w").write("\n".join(rows) + "\n")

    # ---------------------------------------------------------------- R1.8 Lipschitz / bound check
    LP = {arch: [m for m in (jl(r["dir"], "lipschitz") for r in runs.get(arch, {}).values()) if m] for arch in ("agn", "ggrusel")}
    rows = []
    for arch, nm in (("agn", "AGN-Net"), ("ggrusel", "Graph GRU")):
        if not LP[arch]:
            continue
        for b in ("0.125", "0.25", "0.5"):
            g = lambda k: mean([m[b].get(k) for m in LP[arch]])
            for k in ("gamma_median", "gamma_p99", "frac_gamma_lt1", "one_step_bound_holds", "accumulated_bound_holds",
                      "readout_ratio_median", "spearman_state_err_vs_kl", "e_median", "accumulated_bound_median",
                      "jac_l2_median", "jac_l2_max", "gamma_max"):
                N[f"lip_{arch}_{b}_{k}"] = g(k)
            rows.append(f"{nm} & {float(b) * 100:g}\\% & {g('gamma_median'):.3f} & {g('gamma_p99'):.3f} & "
                        f"{100 * g('frac_gamma_lt1'):.1f} & {100 * g('one_step_bound_holds'):.1f} & "
                        f"{g('readout_ratio_median'):.3f} & {f3(g('spearman_state_err_vs_kl'))} \\\\")
    if rows:
        open(os.path.join(T, "lipschitz.tex"), "w").write("\n".join(rows) + "\n")

    # ---------------------------------------------------------------- R1.7 long streams + refresh rules
    L = [m for m in (jl(r["dir"], "longstream") for r in runs.get("agn", {}).values()) if m]
    if L:
        rules = ["none", "25", "50", "100", "200", "adaptive"]
        names = {"none": "No refresh", "25": "$A{=}25$", "50": "$A{=}50$ (main)", "100": "$A{=}100$", "200": "$A{=}200$",
                 "adaptive": "Adaptive $A{=}\\lceil 12.5/b\\rceil$"}
        rows = []
        for sp in ("test2024", "test2025"):
            N[f"long_{sp}_n"] = mean([m[sp]["n_streams"] for m in L if sp in m])
            N[f"long_{sp}_hours"] = mean([m[sp]["hours"] for m in L if sp in m])
            N[f"long_{sp}_mean_len"] = mean([m[sp]["mean_len_s"] for m in L if sp in m])
        for rule in rules:
            cells = [names[rule]]
            for b in BUDGETS:
                for sp in ("test2024", "test2025"):
                    v = mean([m[sp][f"util@{b}|{rule}"]["fidF"] for m in L if sp in m])
                    N[f"long_{sp}_{b:g}_{rule}_fidF"] = v
                    for k in ("fid_agree", "forced_frac", "agree_first15s", "agree_after30s", "fid_advR"):
                        N[f"long_{sp}_{b:g}_{rule}_{k}"] = mean([m[sp][f"util@{b}|{rule}"][k] for m in L if sp in m])
                    cells.append(f3(v))
            rows.append(" & ".join(cells) + " \\\\")
        open(os.path.join(T, "longstream.tex"), "w").write("\n".join(rows) + "\n")

    # ---------------------------------------------------------------- R1.2 overhead scaling
    sp_ = os.path.join(a.results, "scaling.json")
    if os.path.exists(sp_):
        S = json.load(open(sp_))
        rows = []
        for key in sorted(S, key=lambda k: (int(k.split("_")[0][1:]), int(k.split("_")[1][1:]))):
            r = S[key]
            d, Lr = key.split("_")
            m = r["med_ms"]
            rows.append(f"{d[1:]} & {Lr[1:]} & {r['cell_params'] / 1e3:.0f} & {m['graph@1.0']:.3f} & {m['sel@0.25']:.3f} & "
                        f"{m['graph@0.25']:.3f} & {r['graph_speedup@0.25']:.2f} & {100 * r['ctrl_share_of_saving@0.25']:.0f} \\\\")
            for k, v in r.items():
                if k != "med_ms":
                    N[f"scal_{key}_{k}"] = v
            for k, v in m.items():
                N[f"scal_{key}_{k}_ms"] = v
        open(os.path.join(T, "scaling.tex"), "w").write("\n".join(rows) + "\n")

    # ---------------------------------------------------------------- R1.4 HF_Lung
    H = load_runs(a.results_hfl)
    if H:
        pid = None
        mp = os.path.join(a.hfl_meta, "manifest.csv")
        if os.path.exists(mp):
            pid = np.array([r["patient"] for r in csv.DictReader(open(mp)) if r["split"] == "test2024"])
        pols = ["util", "util_hold", "util_kl", "change", "periodic", "random", "acoustic", "energy", "oracle"]
        rows = []
        for b in BUDGETS:
            rows.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{Budget {b * 100:g}\\% of nodes per step}}}} \\\\")
            best = {}
            for arch in ("agnsel", "ggrusel"):
                G = [r["summary"]["grid_C"] for r in H.get(arch, {}).values() if "grid_C" in r["summary"]]
                for m in ("fidF", "fid_advR", "F", "seg_advF"):
                    vals = [mean([g[f"{p}@{b}"]["test2024"][m] for g in G if f"{p}@{b}" in g]) for p in pols if p != "oracle"]
                    best[(arch, m)] = max(v for v in vals if np.isfinite(v)) if G else float("nan")
            for p in pols:
                cells = [POL_NAME[p]]
                for arch in ("agnsel", "ggrusel"):
                    G = [r["summary"]["grid_C"] for r in H.get(arch, {}).values() if "grid_C" in r["summary"]]
                    for m in ("fidF", "fid_advR", "F", "seg_advF"):
                        v = [g[f"{p}@{b}"]["test2024"][m] for g in G if f"{p}@{b}" in g]
                        N[f"hfl_{arch}_{p}@{b}_{m}"] = mean(v)
                        c = ms(v)
                        if p != "oracle" and v and abs(mean(v) - best[(arch, m)]) < 1e-12:
                            c = "\\textbf{" + c + "}"
                        cells.append(c)
                rows.append(" & ".join(cells) + " \\\\")
            rows.append("\\midrule")
        cells = ["Full update (reference)"]
        for arch in ("agnsel", "ggrusel"):
            G = [r["summary"]["grid_C"] for r in H.get(arch, {}).values() if "grid_C" in r["summary"]]
            for m in ("fidF", "fid_advR", "F", "seg_advF"):
                v = [g["full"]["test2024"].get(m) for g in G]
                N[f"hfl_{arch}_full_{m}"] = mean(v)
                cells.append("--" if m in ("fidF", "fid_advR") else ms(v))
            N[f"hfl_{arch}_nseeds"] = len(G)
            N[f"hfl_{arch}_none_fidF"] = mean([g["none@0.0"]["test2024"]["fidF"] for g in G if "none@0.0" in g])
            D = [r["summary"].get("diag_C") for r in H.get(arch, {}).values() if r["summary"].get("diag_C")]
            for nm in ("util", "change", "acoustic", "random"):
                N[f"hfl_{arch}_diag_{nm}_rho"] = mean([d["spearman"][nm] for d in D])
            N[f"hfl_{arch}_conc_util"] = mean([d["conc_util_top25"] for d in D])
            N[f"hfl_{arch}_conc_change"] = mean([d["conc_change_top25"] for d in D])
        rows.append(" & ".join(cells) + " \\\\")
        open(os.path.join(T, "hflung.tex"), "w").write("\n".join(rows) + "\n")
        # bootstrap + per-seed signs on HF_Lung
        for arch in ("agnsel", "ggrusel"):
            dirs = [r["dir"] for r in H.get(arch, {}).values()]
            G = [r["summary"]["grid_C"] for r in H.get(arch, {}).values() if "grid_C" in r["summary"]]
            for b in BUDGETS:
                U = seed_sum([(rec(d, "C", f"util@{b}") or {}).get("test2024_fid") for d in dirs])
                for other in ("change", "random", "periodic"):
                    O = seed_sum([(rec(d, "C", f"{other}@{b}") or {}).get("test2024_fid") for d in dirs])
                    if U is not None and O is not None and pid is not None:
                        dl, lo, hi, p = paired_boot(U, O, ev_F, pid)
                        for k_, v_ in (("delta", dl), ("lo", lo), ("hi", hi), ("p", p)):
                            N[f"hfl_bt_{arch}_{b:g}_util_{other}_fidF_{k_}"] = v_
                    ds = [g[f"util@{b}"]["test2024"]["fidF"] - g[f"{other}@{b}"]["test2024"]["fidF"] for g in G if f"{other}@{b}" in g]
                    if ds:
                        N[f"hfl_ps_{arch}_{b:g}_util_{other}_npos"] = float(sum(d > 0 for d in ds))
                        N[f"hfl_ps_{arch}_{b:g}_util_{other}_min"] = min(ds)
                E = seed_sum([(rec(d, "C", f"util@{b}") or {}).get("test2024_ev") for d in dirs])
                Fu = seed_sum([(rec(d, "C", "full") or {}).get("test2024_ev") for d in dirs])
                if E is not None and Fu is not None and pid is not None:
                    dl, lo, hi, p = paired_boot(E, Fu, ev_F, pid)
                    for k_, v_ in (("delta", dl), ("lo", lo), ("hi", hi), ("p", p)):
                        N[f"hfl_bt_{arch}_{b:g}_util_full_F_{k_}"] = v_
        N["hfl_nseeds"] = len(H.get("agnsel", {}))
    json.dump(N, open(Np, "w"), indent=1)
    print("analyze_rev: numbers", len(N))


if __name__ == "__main__":
    main()
