"""Drift-robust latency protocol shared by the PyTorch and NumPy streaming benchmarks.

Configurations are interleaved per recording in a rotating order (so slow drifts of the
host's speed, e.g. frequency scaling or neighbouring jobs, affect every configuration
equally), the process is pinned to one core, and speedups are reported as paired
per-recording ratios against the full update on the same recording.
"""
from __future__ import annotations

import json
import os

import numpy as np

STAGES = ["features", "encoder", "selection", "graph", "readout"]
CONFIGS = [("full", 1.0), ("none", 0.0), ("random", 0.25), ("periodic", 0.25), ("acoustic", 0.25),
           ("util", 0.125), ("util", 0.25), ("util", 0.5)]


def pin_one_core():
    try:
        cores = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cores[0]})
        return cores[0]
    except (AttributeError, OSError):
        return None


def interleaved(run_one, items, extra=None, reps=1):
    """run_one(item, name, budget) -> (T (n_chunks, 5) seconds, nodes (n_chunks,), edges or None)."""
    Ts = {f"{n}@{b}": [] for n, b in CONFIGS}
    Ns = {k: [] for k in Ts}
    Es = {k: [] for k in Ts}
    per_rec = {k: [] for k in Ts}
    for r in range(reps):
        for ri, it in enumerate(items):
            off = (ri + r) % len(CONFIGS)
            for name, b in CONFIGS[off:] + CONFIGS[:off]:
                T, nn_, ee = run_one(it, name, b)
                k = f"{name}@{b}"
                Ts[k].append(T); Ns[k].append(nn_)
                if ee is not None:
                    Es[k].append(ee)
                per_rec[k].append(T.sum() / len(T))
    res = {}
    full = np.array(per_rec["full@1.0"])
    for k in Ts:
        T = np.concatenate(Ts[k]) * 1e3
        tot = T.sum(1)
        ratio = full / np.array(per_rec[k])           # paired per-recording speedup vs full
        d = dict(median_ms=float(np.median(tot)), p95_ms=float(np.percentile(tot, 95)),
                 p99_ms=float(np.percentile(tot, 99)), max_ms=float(tot.max()), mean_ms=float(tot.mean()),
                 stage_mean_ms=dict(zip(STAGES, T.mean(0).round(4).tolist())), rtf=float(tot.mean() / 40.0),
                 upd_nodes=float(np.concatenate(Ns[k]).mean()),
                 speedup_median=float(np.median(ratio)),
                 speedup_q25=float(np.percentile(ratio, 25)), speedup_q75=float(np.percentile(ratio, 75)))
        if Es[k]:
            d["upd_edges"] = float(np.concatenate(Es[k]).mean())
        if extra is not None:
            d.update(extra(k, d))
        res[k] = d
        print(k, json.dumps({x: (round(v, 4) if isinstance(v, float) else v) for x, v in d.items() if x != "stage_mean_ms"}), flush=True)
    return res
