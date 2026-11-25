"""Print the facts that the paper's verbal claims depend on, so the text can be checked.

    python -m eaagn.check_claims --numbers paper/numbers.json
"""
from __future__ import annotations

import argparse
import json

BUDGETS = ("0.125", "0.25", "0.5")
OTHERS = ("random", "periodic", "energy", "acoustic", "change")
HEUR = ("random", "periodic", "energy", "acoustic", "change")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", required=True)
    a = ap.parse_args()
    N = json.load(open(a.numbers))
    for arch in ("agn", "agnnf", "ggrusel"):
        if f"{arch}_C_nseeds" not in N:
            continue
        n = N[f"{arch}_C_nseeds"]
        print(f"== {arch} (seeds={n})")
        fails = []
        for sp in ("test2024", "test2025"):
            for b in BUDGETS:
                bb = f"{float(b):g}"
                for o in OTHERS:
                    for m in ("fidF", "fid_advR", "F"):
                        k = f"ps_{arch}_{sp}_{bb}_util_{o}_{m}_npos"
                        if k in N and N[k] < n:
                            fails.append(f"{sp} b={b} vs {o} {m}: npos={N[k]:.0f} "
                                         f"[{N[k.replace('_npos', '_min')]:.3f},{N[k.replace('_npos', '_max')]:.3f}]")
                # ordering of non-utility policies by mean fidF
                vals = {p: N.get(f"{arch}_C_{p}@{b}_{sp}_fidF") for p in HEUR}
                order = sorted(vals, key=lambda p: -(vals[p] if vals[p] is not None else -1))
                print(f"  {sp} b={b}: util={N.get(f'{arch}_C_util@{b}_{sp}_fidF'):.3f} hold={N.get(f'{arch}_C_util_hold@{b}_{sp}_fidF'):.3f} "
                      f"kl={N.get(f'{arch}_C_util_kl@{b}_{sp}_fidF'):.3f} oracle={N.get(f'{arch}_C_oracle@{b}_{sp}_fidF'):.3f} | "
                      + " > ".join(f"{p} {vals[p]:.3f}" for p in order))
        print("  per-seed sign failures (util should beat each in every seed):")
        for f in fails:
            print("   ", f)
        if not fails:
            print("    none")
        for tag in ("A", "C"):
            keys = [f"{arch}_diag{tag}_{x}" for x in ("conc_util", "conc_change", "conc_util_sqrt", "conc_change_sq")]
            if keys[0] in N:
                print(f"  diag {tag}: " + ", ".join(f"{k.split('_diag')[1]}={N[k]:.3f}" for k in keys if k in N)
                      + f" | rho util={N.get(f'{arch}_diag{tag}_util_rho', float('nan')):.3f} change={N.get(f'{arch}_diag{tag}_change_rho', float('nan')):.3f} "
                      f"acoustic={N.get(f'{arch}_diag{tag}_acoustic_rho', float('nan')):.3f} periodic={N.get(f'{arch}_diag{tag}_periodic_rho', float('nan')):.3f}")
        # stage A ordering
        for b in BUDGETS:
            vals = {p: N.get(f"{arch}_A_{p}@{b}_test2024_fidF") for p in ("util", "util_hold", "util_kl") + HEUR}
            if all(v is not None for v in vals.values()):
                print(f"  stageA b={b}: " + " > ".join(f"{p} {vals[p]:.3f}" for p in sorted(vals, key=lambda p: -vals[p])))
    for k in ("gg_ok",):
        print(k, N.get(k))


if __name__ == "__main__":
    main()
