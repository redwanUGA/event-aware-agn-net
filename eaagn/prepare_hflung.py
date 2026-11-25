"""Second dataset (revision): HF_Lung_V1 adult lung sounds (Hsu et al., PLOS ONE 2021; CC BY 4.0).

    python -m eaagn.prepare_hflung --repo HF_Lung_V1 --out data_hflung

15-s recordings at 4 kHz (resampled to 8 kHz so the front end is unchanged; nothing exists above
2 kHz). Labels are intervals of inhalation (I), exhalation (E), discontinuous adventitious sounds
(D, crackles) and continuous adventitious sounds (Wheeze, Stridor, Rhonchi); phase and adventitious
labels overlap. The detector is single-label per frame, so each frame takes the highest-priority
label present (Wheeze > Stridor > Rhonchi > D > E > I > background) and reference events are the
maximal runs of the resolved frame labels. Classes 1-2 (I, E) are non-adventitious, 3-6 adventitious
(set EAAGN_NONADV=2 so the adventitious metrics use classes 3-6).

Splits: the official train/test folders; 15% of the training subjects are held out for validation.
Subjects are approximated by recording device + date, as the dataset README recommends.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import pickle
import re

import numpy as np
import torch
from scipy.signal import resample_poly

from .audio import SR, CausalLogMel, read_wav

CLASSES = ["bg", "I", "E", "D", "Wheeze", "Stridor", "Rhonchi"]
PRIORITY = {"I": 1, "E": 2, "D": 3, "Rhonchi": 4, "Stridor": 5, "Wheeze": 6}   # higher wins
CIDX = {c: i for i, c in enumerate(CLASSES)}


def hms(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def subject(name):
    m = re.match(r"(steth|trunc)_(\d{8}|\d{4}-\d{2}-\d{2})", name)
    return f"{m.group(1)}_{m.group(2)}" if m else name


def load(folder):
    items = []
    for wp in sorted(glob.glob(os.path.join(folder, "*.wav"))):
        name = os.path.splitext(os.path.basename(wp))[0]
        lp = os.path.join(folder, name + "_label.txt")
        if not os.path.exists(lp):
            continue
        x, sr = read_wav(wp)
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
        T = len(x) * 100 // SR                                    # 10 ms frames
        lab = np.zeros(T, dtype=np.int64); pri = np.zeros(T, dtype=np.int64)
        for line in open(lp):
            p = line.split()
            if len(p) != 3 or p[0] not in PRIORITY:
                continue
            a, b = int(round(hms(p[1]) * 100)), int(round(hms(p[2]) * 100))
            seg = slice(max(a, 0), min(b, T))
            upd = PRIORITY[p[0]] > pri[seg]
            lab[seg] = np.where(upd, CIDX[p[0]], lab[seg]); pri[seg] = np.where(upd, PRIORITY[p[0]], pri[seg])
        evs, start = [], None
        for t in range(T + 1):
            cur = lab[t] if t < T else -1
            if start is not None and cur != lab[start]:
                if lab[start] != 0:
                    evs.append((start / 100.0, t / 100.0, int(lab[start])))
                start = None
            if start is None and t < T:
                start = t
        wav16 = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
        items.append(dict(name=name, pid=subject(name), orig_sr=int(sr), wav=wav16, events=evs, record="",
                          md5=hashlib.md5(wav16.tobytes()).hexdigest()))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="data_hflung")
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--manifest_only", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.manifest_only:   # split, name, patient in pickle order (for the patient-level bootstrap)
        import csv
        with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["split", "name", "patient", "duration_s", "n_events"])
            for k, fn in (("train", "train_pg"), ("val", "val_pg"), ("test2024", "test2024")):
                for it in pickle.load(open(os.path.join(a.out, fn + ".pkl"), "rb")):
                    w.writerow([k, it["name"], it["pid"], round(len(it["wav"]) / SR, 3), len(it["events"])])
        return
    tr = load(os.path.join(a.repo, "train"))
    te = load(os.path.join(a.repo, "test"))
    pids = sorted({it["pid"] for it in tr})
    rng = np.random.default_rng(0)
    val_p = set(rng.choice(pids, size=int(round(a.frac * len(pids))), replace=False).tolist())
    split = dict(train=[it for it in tr if it["pid"] not in val_p], val=[it for it in tr if it["pid"] in val_p], test=te)
    audit = dict(subject_overlap_train_test=len({it["pid"] for it in tr} & {it["pid"] for it in te}))
    for k, v in split.items():
        cnt = collections.Counter(CLASSES[e[2]] for it in v for e in it["events"])
        audit[k] = dict(n=len(v), subjects=len({it["pid"] for it in v}), hours=round(sum(len(it["wav"]) for it in v) / SR / 3600, 2),
                        events=dict(cnt))
    json.dump(audit, open(os.path.join(a.out, "audit.json"), "w"), indent=1)
    print(json.dumps(audit, indent=1))
    pickle.dump(split["train"], open(os.path.join(a.out, "train_pg.pkl"), "wb"))
    pickle.dump(split["val"], open(os.path.join(a.out, "val_pg.pkl"), "wb"))
    pickle.dump(split["test"], open(os.path.join(a.out, "test2024.pkl"), "wb"))   # 'test2024' = the HF_Lung test set
    fe = CausalLogMel()
    idx = rng.choice(len(split["train"]), size=min(600, len(split["train"])), replace=False)
    acc, acc2, n = 0, 0, 0
    with torch.no_grad():
        for i in idx:
            m = fe.raw(torch.from_numpy(split["train"][i]["wav"].astype(np.float32) / 32768.0)[None])[0]
            acc = acc + m.sum(-1); acc2 = acc2 + (m ** 2).sum(-1); n += m.shape[-1]
    mean = acc / n
    std = (acc2 / n - mean ** 2).clamp_min(1e-6).sqrt()
    np.savez(os.path.join(a.out, "stats.npz"), mean=mean.numpy(), std=std.numpy())


if __name__ == "__main__":
    main()
