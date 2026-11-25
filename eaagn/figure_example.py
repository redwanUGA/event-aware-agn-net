"""Qualitative figure: where each controller spends its update budget on one recording.

    python -m eaagn.figure_example --run results/agn_s0 --stats data_meta/stats.npz \
        --wav example.wav --json example.json --out paper/figures/example.pdf
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from .audio import STEP, CausalLogMel, read_wav
from .models import StreamModel
from .prepare_data import CLASSES
from .select import CONTROLLER_HEADS, Controller, make_policy, region_masks, utility_targets

COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


@torch.no_grad()
def rollout(model, fe, wav, name, budget, ctrls, R):
    masks, utils = [], []

    def collect(t, ctx):
        utils.append(utility_targets(model, ctx, R)["util"][0].numpy())

    base = make_policy(name, model, R, budget, ctrls)

    def pol(t, ctx):
        u = base(t, ctx)
        masks.append(u[0].numpy())
        return u

    logits, _ = model(fe(wav[None]), pol, budget, collect=collect)
    return np.array(masks).T, np.array(utils).T, torch.softmax(logits[0], -1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--weights", default="C")
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    st = np.load(a.stats)
    fe = CausalLogMel(st["mean"], st["std"])
    model = StreamModel("agn"); model.load_state_dict(torch.load(f"{a.run}/model_{a.weights}.pt", map_location="cpu")); model.eval()
    ctrls = nn.ModuleDict({h: Controller(model.dx, model.d) for h in CONTROLLER_HEADS})
    ctrls.load_state_dict(torch.load(f"{a.run}/ctrl_{a.weights}.pt", map_location="cpu")); ctrls.eval()
    R = region_masks(model.cell.adj)
    x, sr = read_wav(a.wav)
    n = len(x) // (80 * STEP) * (80 * STEP)
    wav = torch.from_numpy(x[:n].astype(np.float32))
    ann = json.load(open(a.json))["event_annotation"]
    mu, uu, pu = rollout(model, fe, wav, "util", a.budget, ctrls, R)
    mc, _, _ = rollout(model, fe, wav, "change", a.budget, ctrls, R)
    with torch.no_grad():
        spec = fe(wav[None])[0, 0].numpy()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    S = mu.shape[1]; T = S * STEP * 0.01
    fig, ax = plt.subplots(4, 1, figsize=(3.5, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1, 1, 1]))
    lo, hi = np.percentile(spec, [2, 99.5])
    ax[0].imshow(spec, origin="lower", aspect="auto", extent=[0, T, 0, 128], cmap="Greys", vmin=lo, vmax=hi)
    for e in ann:
        on, off, c = float(e["start"]) / 1000, float(e["end"]) / 1000, e["type"]
        k = CLASSES.index(c) if c in CLASSES else 1
        ax[0].plot([on, off], [124, 124], color=COL[(k - 1) % 8], lw=3, solid_capstyle="butt")
        ax[0].text((on + off) / 2, 113, c.replace("Fine Crackle", "FC").replace("Normal", "N").replace("Wheeze", "W"),
                   ha="center", va="top", fontsize=5.5, color="#0b0b0b")
    ax[0].set_ylabel("mel band"); ax[0].set_title("(a) Log-mel (25 ms) and annotated events", fontsize=7)
    ext = [0, T, 0, 64]
    share = uu / np.maximum(uu.sum(0, keepdims=True), 1e-12)      # each region's share of utility per step
    ax[1].imshow(share, origin="lower", aspect="auto", extent=ext, cmap="Blues", vmin=0, vmax=np.percentile(share, 99))
    ax[1].set_ylabel("region"); ax[1].set_title("(b) Share of measured skip-harm utility per step (hindsight)", fontsize=7)
    ax[2].imshow(mu, origin="lower", aspect="auto", extent=ext, cmap="Greys", vmin=0, vmax=1.6)
    ax[2].set_ylabel("node"); ax[2].set_title(f"(c) Updates chosen by skip-harm controller ({a.budget * 100:g}% budget)", fontsize=7)
    ax[3].imshow(mc, origin="lower", aspect="auto", extent=ext, cmap="Greys", vmin=0, vmax=1.6)
    ax[3].set_ylabel("node"); ax[3].set_title(f"(d) Updates chosen by change controller ({a.budget * 100:g}% budget)", fontsize=7)
    ax[3].set_xlabel("time (s)")
    fig.tight_layout(h_pad=0.4)
    fig.savefig(a.out, bbox_inches="tight", dpi=600)
    print("saved", a.out)


if __name__ == "__main__":
    main()
