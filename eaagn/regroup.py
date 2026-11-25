"""Patient-disjoint train/validation re-split.

The official SPRSound detection 'valid' split shares 261/266 patients with 'train',
so it cannot measure generalisation to new patients. We pool train+valid and hold
out ~15% of patients (fixed seed) as validation; the test sets are untouched.

    python -m eaagn.regroup --data data
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--frac", type=float, default=0.15)
    a = ap.parse_args()
    pool = pickle.load(open(os.path.join(a.data, "train.pkl"), "rb")) + pickle.load(open(os.path.join(a.data, "val.pkl"), "rb"))
    seen, dedup = set(), []
    for it in pool:                       # drop exact duplicate recordings (keep first)
        if it["md5"] not in seen:
            seen.add(it["md5"]); dedup.append(it)
    n_dups = len(pool) - len(dedup)
    pool = dedup
    pids = sorted({it["pid"] for it in pool})
    cls_p = {c: {it["pid"] for it in pool for e in it["events"] if e[2] == c} for c in range(1, 8)}
    # smallest seed whose patient hold-out keeps every class (with >= 2 patients) on both sides
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        val_p = set(rng.choice(pids, size=int(round(a.frac * len(pids))), replace=False).tolist())
        if all((len(P) < 2) or (P & val_p and P - val_p) for P in cls_p.values()):
            break
    tr = [it for it in pool if it["pid"] not in val_p]
    va = [it for it in pool if it["pid"] in val_p]
    assert not ({it["md5"] for it in tr} & {it["md5"] for it in va})
    pickle.dump(tr, open(os.path.join(a.data, "train_pg.pkl"), "wb"))
    pickle.dump(va, open(os.path.join(a.data, "val_pg.pkl"), "wb"))
    cnt = lambda L: {c: int(sum(1 for it in L for e in it["events"] if e[2] == c)) for c in range(1, 8)}
    info = dict(seed=seed, dups_removed=n_dups, train_events_by_class=cnt(tr), val_events_by_class=cnt(va),
                pool_n=len(pool), pool_patients=len(pids), train_n=len(tr), val_n=len(va),
                train_patients=len(pids) - len(val_p), val_patients=len(val_p),
                val_events=int(sum(len(it["events"]) for it in va)),
                val_adv_events=int(sum(1 for it in va for e in it["events"] if e[2] >= 2)))
    json.dump(info, open(os.path.join(a.data, "regroup.json"), "w"), indent=1)
    print(info)


if __name__ == "__main__":
    main()
