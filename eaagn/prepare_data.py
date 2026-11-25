"""Build manifests + packed waveform/label files from an SPRSound checkout.

    python -m eaagn.prepare_data --repo /path/SPRSound --out data

Splits (official BioCAS-2024 detection protocol + later-release holdout):
  train    Detection/train_detection_{wav,json}
  val      Detection/valid_detection_{wav,json}
  test2024 Detection/test2024_detection_{wav,json}
  test2025 BioCAS2025/test2025_{wav,json}   (locked; evaluated once at the end)
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import pickle

import numpy as np
import torch
from scipy.signal import resample_poly

from .audio import SR, CausalLogMel, read_wav

CLASSES = ["bg", "Normal", "Rhonchi", "Wheeze", "Stridor", "Coarse Crackle", "Fine Crackle", "Wheeze+Crackle"]
CIDX = {c: i for i, c in enumerate(CLASSES)}

SPLITS = {
    "train": ("Detection/train_detection_wav", "Detection/train_detection_json"),
    "val": ("Detection/valid_detection_wav", "Detection/valid_detection_json"),
    "test2024": ("Detection/test2024_detection_wav", "Detection/test2024_detection_json"),
    "test2025": ("BioCAS2025/test2025_wav", "BioCAS2025/test2025_json"),
}


def load_split(repo, wav_dir, json_dir):
    items = []
    jsons = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(repo, json_dir, "**", "*.json"), recursive=True)}
    for wp in sorted(glob.glob(os.path.join(repo, wav_dir, "**", "*.wav"), recursive=True)):
        name = os.path.splitext(os.path.basename(wp))[0]
        if name not in jsons:
            continue
        x, sr = read_wav(wp)
        if sr != SR:
            g = np.gcd(int(sr), SR)
            x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
        ann = json.load(open(jsons[name], encoding="utf-8"))
        evs = []
        for e in ann.get("event_annotation", []):
            t = e["type"].strip()
            if t not in CIDX:
                continue
            on, off = float(e["start"]) / 1000.0, float(e["end"]) / 1000.0
            if off > on:
                evs.append((on, off, CIDX[t]))
        wav16 = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
        items.append(dict(name=name, pid=name.split("_")[0], orig_sr=int(sr), wav=wav16,
                          events=sorted(evs), record=ann.get("record_annotation", ""),
                          md5=hashlib.md5(wav16.tobytes()).hexdigest()))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {k: load_split(a.repo, *v) for k, v in SPLITS.items()}

    # duplicate + patient-overlap audit (reported, and exact duplicates removed from train/val)
    audit = {}
    test_md5 = {it["md5"] for k in ("test2024", "test2025") for it in data[k]}
    for k in ("train", "val"):
        before = len(data[k])
        data[k] = [it for it in data[k] if it["md5"] not in test_md5]
        audit[f"{k}_exact_dups_with_test_removed"] = before - len(data[k])
    val_md5 = {it["md5"] for it in data["val"]}
    before = len(data["train"])
    data["train"] = [it for it in data["train"] if it["md5"] not in val_md5]
    audit["train_exact_dups_with_val_removed"] = before - len(data["train"])
    pids = {k: {it["pid"] for it in v} for k, v in data.items()}
    for k in ("val", "test2024", "test2025"):
        audit[f"patients_{k}"] = len(pids[k])
        audit[f"patients_{k}_overlap_train"] = len(pids[k] & pids["train"])
    audit["patients_test2025_overlap_test2024"] = len(pids["test2025"] & pids["test2024"])
    audit["patients_train"] = len(pids["train"])
    for k, v in data.items():
        dur = [len(it["wav"]) / SR for it in v]
        cnt = collections.Counter(CLASSES[e[2]] for it in v for e in it["events"])
        audit[f"{k}_n"] = len(v)
        audit[f"{k}_hours"] = round(sum(dur) / 3600, 3)
        audit[f"{k}_dur_min_med_max"] = [round(min(dur), 2), round(float(np.median(dur)), 2), round(max(dur), 2)] if dur else []
        audit[f"{k}_events"] = dict(cnt)
        audit[f"{k}_orig_sr"] = dict(collections.Counter(it["orig_sr"] for it in v))
    json.dump(audit, open(os.path.join(a.out, "audit.json"), "w"), indent=1)
    print(json.dumps(audit, indent=1))

    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "name", "patient", "orig_sr", "duration_s", "n_events", "record_label", "md5"])
        for k, v in data.items():
            for it in v:
                w.writerow([k, it["name"], it["pid"], it["orig_sr"], round(len(it["wav"]) / SR, 3),
                            len(it["events"]), it["record"], it["md5"]])
    for k, v in data.items():
        pickle.dump(v, open(os.path.join(a.out, f"{k}.pkl"), "wb"))

    # fixed normalisation statistics from training audio only
    fe = CausalLogMel()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(data["train"]), size=min(600, len(data["train"])), replace=False)
    acc, acc2, n = 0, 0, 0
    with torch.no_grad():
        for i in idx:
            m = fe.raw(torch.from_numpy(data["train"][i]["wav"].astype(np.float32) / 32768.0)[None])[0]
            acc = acc + m.sum(-1); acc2 = acc2 + (m ** 2).sum(-1); n += m.shape[-1]
    mean = acc / n
    std = (acc2 / n - mean ** 2).clamp_min(1e-6).sqrt()
    np.savez(os.path.join(a.out, "stats.npz"), mean=mean.numpy(), std=std.numpy())
    print("saved", {k: len(v) for k, v in data.items()})


if __name__ == "__main__":
    main()
