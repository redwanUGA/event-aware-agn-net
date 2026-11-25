"""Batch-1 streaming CPU benchmark with true sparse (gather/scatter) graph updates.

Measures per-40-ms-chunk processing latency split into
  T_features + T_encoder + T_selection + T_graph + T_readout
for full updates, cheap-only, and selective policies, and checks that the
streaming implementation reproduces the batched model's outputs.

    python -m eaagn.bench --run results/agn_s0 --data data --n_rec 40 --threads 1
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio import HOP, N_MELS, STEP, WINS, CausalLogMel
from .models import StreamModel
from .select import CONTROLLER_HEADS, Controller, greedy_bitmask, make_policy, policy_scores, region_bits, region_masks

CHUNK = HOP * STEP


class Streamer:
    def __init__(self, model: StreamModel, fe: CausalLogMel, ctrls=None):
        self.m, self.fe, self.ctrls = model.eval(), fe, ctrls
        self.N, self.d, self.dx = model.N, model.d, model.dx
        self.R = region_masks(model.cell.adj)
        self.adj = model.cell.adj
        self.Rbits = region_bits(self.R)
        self.Rsize = self.R.sum(1).float().numpy()
        self.reset()

    def reset(self):
        self.buf = torch.zeros(max(WINS) - HOP + CHUNK)      # waveform history
        self.f_hist = torch.zeros(1, 2, N_MELS, 2)            # last 2 frames for conv1 (time k=3)
        self.p_hist = None                                    # last 2 pooled steps for conv2
        self.z = torch.zeros(self.N, self.d)
        self.age = torch.zeros(self.N)
        self.x_prev = torch.zeros(self.N, self.dx)
        self.spec_prev = torch.zeros(self.N, 4)

    def features(self, chunk):
        self.buf = torch.cat([self.buf[CHUNK:], chunk])
        outs = []
        for k, w in enumerate(WINS):
            seg = self.buf[-(w - HOP + CHUNK):]
            fr = seg.unfold(0, w, HOP)                        # (STEP, w)
            spec = torch.fft.rfft(fr * getattr(self.fe, f"win{k}"), n=w).abs() ** 2
            outs.append(torch.log(spec @ getattr(self.fe, f"fb{k}").T + 1e-8).T)
        f = (torch.stack(outs, 0) - self.fe.mean) / self.fe.std   # (2, M, STEP)
        return f[None]

    def encoder(self, f):
        enc = self.m.enc
        inp = torch.cat([self.f_hist, f], -1)                 # (1,2,M,STEP+2)
        self.f_hist = inp[..., -2:]
        h = F.gelu(enc.conv1(inp)).mean(-1, keepdim=True)     # (1,c1,M,1)
        if self.p_hist is None:
            self.p_hist = torch.zeros(1, h.shape[1], h.shape[2], 2)
        hin = torch.cat([self.p_hist, h], -1)
        self.p_hist = hin[..., -2:]
        x = F.gelu(enc.conv2(hin))[0, :, :, 0].T              # (N, dx)
        spec = f[0].reshape(2, self.N, 2, STEP).mean(-1).permute(1, 0, 2).reshape(self.N, 4)
        return enc.norm(x), spec

    def select(self, policy, budget, x, spec):
        if policy == "full":
            return None
        if policy == "none":
            return torch.zeros(0, dtype=torch.long)
        ctx = dict(x=x[None], x_prev=self.x_prev[None], z_prev=self.z[None], age=self.age[None],
                   spec=spec[None], spec_prev=self.spec_prev[None])
        s = policy_scores(policy, self.m, self.R, ctx, budget, self.ctrls)[0].numpy()
        idx = greedy_bitmask(s, self.Rbits, self.Rsize, self.age.numpy(), int(round(budget * self.N)), True)
        return torch.from_numpy(idx)

    def graph(self, x, idx):
        c = self.m.cell
        if idx is None:
            self.z = c(x[None], self.z[None])[0]
            self.age.zero_()
            return self.N, int(self.adj.sum())
        self.age += 1
        if idx.numel() == 0:
            return 0, 0
        xs, zs = x[idx], self.z[idx]
        mask = self.adj[idx][:, idx]
        h = F.gelu(c.inp(torch.cat([xs, zs], -1)) + c.node_emb[idx])
        hs = [h]
        for l in range(c.L):
            Wh = c.W[l](h)
            e = F.leaky_relu((Wh @ c.a_dst[l])[:, None] + (Wh @ c.a_src[l])[None, :], 0.2).clamp(-30.0, 30.0).masked_fill(~mask, -1e9)
            h = F.elu((torch.softmax(e, -1) * mask) @ Wh + c.beta[l] * h)
            hs.append(h)
        a = torch.softmax(c.fuse_w, 0)
        fused = c.ln(sum(a[i] * hs[i] for i in range(len(hs))))
        cand = torch.tanh(c.mlp(fused))
        lam = torch.sigmoid(c.gate(torch.cat([fused, zs], -1)))
        self.z = self.z.clone()
        self.z[idx] = (1 - lam) * zs + lam * cand
        self.age[idx] = 0
        return int(idx.numel()), int(mask.sum())

    @torch.no_grad()
    def run(self, wav, policy="full", budget=1.0, timing=True):
        self.reset()
        n = len(wav) // CHUNK
        T = np.zeros((n, 5)); probs = []; nodes = []; edges = []
        for t in range(n):
            t0 = time.perf_counter()
            f = self.features(wav[t * CHUNK:(t + 1) * CHUNK])
            t1 = time.perf_counter()
            x, spec = self.encoder(f)
            t2 = time.perf_counter()
            idx = self.select(policy, budget, x, spec)
            t3 = time.perf_counter()
            nu, ne = self.graph(x, idx)
            t4 = time.perf_counter()
            o = self.m.readout(self.z, x).view(STEP, self.m.C)
            probs.append(torch.softmax(o, -1))
            t5 = time.perf_counter()
            T[t] = [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4]
            nodes.append(nu); edges.append(ne)
            self.x_prev, self.spec_prev = x, spec
        return torch.cat(probs).numpy(), T, np.array(nodes), np.array(edges)


def macs_per_step(model, n_upd, n_edges, ctrl=False):
    """Analytic multiply-accumulates for one 40 ms step (sparse-equivalent graph cost)."""
    d, dx, N, L = model.d, model.dx, model.N, 2
    feat = sum(STEP * (w * np.log2(w) * 2 + (w // 2 + 1) * N_MELS) for w in WINS)
    enc = STEP * N_MELS * 16 * 2 * 15 + N * dx * 16 * 15
    graph = n_upd * ((dx + d) * d + L * d * d + 2 * d * d * 2 + 2 * d) + n_edges * d * L
    read = N * d + (d + 2 * dx) * 128 + 128 * model.C * STEP
    sel = (N * ((2 * dx + d + 2) * 24 + 24 * 24 * 5 + 24)) if ctrl else 0
    return dict(features=feat, encoder=enc, selection=sel, graph=graph, readout=read)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--n_rec", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--weights", default="C")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    st = np.load(os.path.join(a.data, "stats.npz"))
    fe = CausalLogMel(st["mean"], st["std"])
    model = StreamModel("agn")
    model.load_state_dict(torch.load(os.path.join(a.run, f"model_{a.weights}.pt"), map_location="cpu"))
    ctrls = nn.ModuleDict({h: Controller(model.dx, model.d) for h in CONTROLLER_HEADS})
    ctrls.load_state_dict(torch.load(os.path.join(a.run, f"ctrl_{a.weights}.pt"), map_location="cpu"))
    model.eval(); ctrls.eval()
    items = pickle.load(open(os.path.join(a.data, "test2024.pkl"), "rb"))[: a.n_rec]
    S = Streamer(model, fe, ctrls)

    # equivalence check vs batched model (full and a deterministic selective policy)
    wav = torch.from_numpy(items[0]["wav"].astype(np.float32) / 32768.0)
    wav = wav[: len(wav) // CHUNK * CHUNK]
    with torch.no_grad():
        ref_full = torch.softmax(model(fe(wav[None]))[0][0], -1).numpy()
        pol = make_policy("util", model, S.R, 0.25, ctrls)
        ref_sel = torch.softmax(model(fe(wav[None]), pol, 0.25)[0][0], -1).numpy()
    p_full = S.run(wav, "full")[0]; p_sel = S.run(wav, "util", 0.25)[0]
    equiv = dict(full_maxabs=float(np.abs(p_full - ref_full).max()), util25_maxabs=float(np.abs(p_sel - ref_sel).max()))
    print("equivalence", equiv, flush=True)

    from .benchutil import interleaved, pin_one_core
    core = pin_one_core()
    res = dict(equivalence=equiv, threads=a.threads, core=core, protocol="interleaved",
               cpu=os.popen("lscpu 2>/dev/null | grep 'Model name'").read().strip() or os.environ.get("PROCESSOR_IDENTIFIER", ""))
    S.run(torch.from_numpy(items[0]["wav"][: CHUNK * 50].astype(np.float32) / 32768.0), "full")  # warm-up

    def run_one(it, name, b):
        _, T, nn_, ee = S.run(torch.from_numpy(it["wav"].astype(np.float32) / 32768.0), name, b)
        return T, nn_, ee

    def extra(k, d):
        name = k.split("@")[0]
        mac = macs_per_step(model, d["upd_nodes"], d.get("upd_edges", 0.0), ctrl=name in CONTROLLER_HEADS)
        return dict(macs=mac, macs_total=float(sum(mac.values())))

    res.update(interleaved(run_one, items, extra))
    out = a.out or os.path.join(a.run, f"bench_{a.weights}_t{a.threads}.json")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
