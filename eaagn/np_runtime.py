"""Low-overhead NumPy streaming runtime (batch 1, float32) for deployment-style latency.

PyTorch eager mode spends ~10-30 us of dispatch overhead per operator, which dominates a
64-node graph and hides any saving from updating fewer nodes. This runtime executes the
same trained weights with plain NumPy/BLAS calls and a bitmask greedy selector, so that
latency tracks arithmetic. It is checked for numerical equivalence against the PyTorch model.

    python -m eaagn.np_runtime --run results/agn_s0 --data data --n_rec 40
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
from scipy.special import erf

from .audio import HOP, N_MELS, STEP, WINS
from .select import A_MAX, greedy_bitmask, region_bits

CHUNK = HOP * STEP
F32 = np.float32


def gelu(x):
    return (0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))).astype(F32)


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0))).astype(F32)


def layernorm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return ((x - m) / np.sqrt(v + eps) * w + b).astype(F32)


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20)))).astype(F32)


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(F32)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True))
    return (e / e.sum(axis, keepdims=True)).astype(F32)


class NPRuntime:
    def __init__(self, sd: dict, ctrl_sd: dict | None, stats, head="util", fresh=True):
        g = lambda k: sd[k].astype(F32)
        self.sd, self.fresh = sd, fresh
        self.N, self.d = sd["cell.node_emb"].shape
        self.dx = sd["enc.norm.weight"].shape[0]
        self.C = sd["readout.head.3.weight"].shape[0] // STEP
        self.L = sd["cell.beta"].shape[0]
        # features
        self.wins = [np.hanning(w + 1)[:-1].astype(F32) for w in WINS]   # periodic Hann (== torch)
        from .audio import mel_fbank
        self.fbs = [mel_fbank(w).numpy().astype(F32).T for w in WINS]   # (bins, M)
        self.mean = stats["mean"].reshape(2, N_MELS, 1).astype(F32)
        self.std = stats["std"].reshape(2, N_MELS, 1).astype(F32)
        # encoder as im2col matrices
        self.w1 = g("enc.conv1.weight").reshape(g("enc.conv1.weight").shape[0], -1).T   # (2*5*3, c1)
        self.b1 = g("enc.conv1.bias")
        self.w2 = g("enc.conv2.weight").reshape(g("enc.conv2.weight").shape[0], -1).T   # (c1*5*3, dx)
        self.b2 = g("enc.conv2.bias")
        self.adj = sd["cell.adj"].astype(bool)
        self.R = self.adj | np.eye(self.N, dtype=bool)
        self.Rsize = self.R.sum(1).astype(F32)
        self.Rbits = region_bits(self.R)
        self.ctrl = None
        if ctrl_sd is not None:
            c = {k.split(".", 1)[1]: v.astype(F32) for k, v in ctrl_sd.items() if k.startswith(head + ".")}
            c["inp.wT"] = np.ascontiguousarray(c["inp.weight"].T)
            O, H, K = c["conv.weight"].shape
            c["conv.wT"] = np.ascontiguousarray(c["conv.weight"].reshape(O, H * K).T)   # (H*K, O)
            c["out.wT"] = np.ascontiguousarray(c["out.weight"].T)
            self.ctrl = c
        self.reset()

    def reset(self):
        self.buf = np.zeros(max(WINS) - HOP + CHUNK, F32)
        self.f_hist = np.zeros((2, N_MELS, 2), F32)
        self.p_hist = None
        self.z = np.zeros((self.N, self.d), F32)
        self.age = np.zeros(self.N, F32)
        self.x_prev = np.zeros((self.N, self.dx), F32)
        self.spec_prev = np.zeros((self.N, 4), F32)

    # ------------------------------------------------------------------ stages
    def features(self, chunk):
        self.buf = np.concatenate([self.buf[CHUNK:], chunk])
        out = []
        for k, w in enumerate(WINS):
            seg = self.buf[-(w - HOP + CHUNK):]
            fr = np.lib.stride_tricks.sliding_window_view(seg, w)[::HOP]       # (STEP, w)
            spec = np.abs(np.fft.rfft(fr * self.wins[k], n=w)) ** 2
            out.append(np.log(spec.astype(F32) @ self.fbs[k] + 1e-8).T)
        return ((np.stack(out, 0) - self.mean) / self.std).astype(F32)       # (2, M, STEP)

    def encoder(self, f):
        sd = self.sd
        inp = np.concatenate([self.f_hist, f], -1)                           # (2, M, STEP+2)
        self.f_hist = inp[..., -2:]
        xp = np.pad(inp, ((0, 0), (2, 2), (0, 0)))
        cols = np.lib.stride_tricks.sliding_window_view(xp, (5, 3), axis=(1, 2))  # (2, M, STEP, 5, 3)
        cols = cols.transpose(1, 2, 0, 3, 4).reshape(N_MELS * STEP, -1)
        h = gelu(cols @ self.w1 + self.b1).reshape(N_MELS, STEP, -1).mean(1)   # (M, c1)
        h = h.T[:, :, None]                                                   # (c1, M, 1)
        if self.p_hist is None:
            self.p_hist = np.zeros((h.shape[0], N_MELS, 2), F32)
        hin = np.concatenate([self.p_hist, h], -1)                           # (c1, M, 3)
        self.p_hist = hin[..., -2:]
        hp = np.pad(hin, ((0, 0), (2, 2), (0, 0)))
        cols2 = np.lib.stride_tricks.sliding_window_view(hp, (5, 3), axis=(1, 2))[:, ::2, 0]  # (c1, N, 5, 3)
        cols2 = cols2.transpose(1, 0, 2, 3).reshape(self.N, -1)
        x = gelu(cols2 @ self.w2 + self.b2)                                   # (N, dx)
        x = layernorm(x, sd["enc.norm.weight"], sd["enc.norm.bias"])
        spec = f.reshape(2, self.N, 2, STEP).mean(-1).transpose(1, 0, 2).reshape(self.N, 4)
        return x, spec

    def controller(self, x, budget):
        c = self.ctrl
        a = (np.log1p(self.age) / np.log1p(A_MAX))[:, None].astype(F32)
        r = np.concatenate([x, x - self.x_prev, self.z, a, np.full_like(a, budget)], -1)
        h = gelu(r @ c["inp.wT"] + c["inp.bias"])                            # (N, H)
        hp = np.pad(h, ((2, 2), (0, 0)))
        cols = np.lib.stride_tricks.sliding_window_view(hp, 5, axis=0).reshape(self.N, -1)  # (N, H*5)
        h = h + gelu(cols @ c["conv.wT"] + c["conv.bias"])
        return softplus(h @ c["out.wT"] + c["out.bias"])[:, 0]

    def select(self, policy, budget, x, spec):
        bn = int(round(budget * self.N))
        if policy == "full":
            return None
        if policy == "none" or bn == 0:
            return np.zeros(0, dtype=np.int64)
        if policy == "util":
            s = self.controller(x, budget)
        elif policy == "random":   # region totals (see select.policy_scores)
            s = self.R @ np.random.rand(self.N).astype(F32)
        elif policy == "periodic":
            s = self.R @ (self.age + 1.0)
        elif policy == "acoustic":
            s = self.R @ np.linalg.norm(spec - self.spec_prev, axis=-1)
        else:
            raise ValueError(policy)
        return greedy_bitmask(s, self.Rbits, self.Rsize, self.age, bn, refresh=True)

    def graph(self, x, idx):
        sd = self.sd
        if idx is None:
            idx = np.arange(self.N)
        self.age += 1
        if len(idx) == 0:
            return 0, 0
        xs, zs = x[idx], self.z[idx]
        mask = self.adj[np.ix_(idx, idx)]
        h = gelu(np.concatenate([xs, zs], -1) @ sd["cell.inp.weight"].T + sd["cell.inp.bias"] + sd["cell.node_emb"][idx])
        hs = [h]
        for l in range(self.L):
            Wh = h @ sd[f"cell.W.{l}.weight"].T
            e = (Wh @ sd["cell.a_dst"][l])[:, None] + (Wh @ sd["cell.a_src"][l])[None, :]
            e = np.clip(np.where(e > 0, e, 0.2 * e), -30.0, 30.0)
            e = np.where(mask, e, -1e9)
            alpha = softmax(e, -1) * mask
            h = elu(alpha @ Wh + sd["cell.beta"][l] * h)
            hs.append(h)
        a = softmax(sd["cell.fuse_w"], 0)
        fused = layernorm(sum(a[i] * hs[i] for i in range(len(hs))), sd["cell.ln.weight"], sd["cell.ln.bias"])
        m = gelu(fused @ sd["cell.mlp.0.weight"].T + sd["cell.mlp.0.bias"])
        cand = np.tanh(m @ sd["cell.mlp.2.weight"].T + sd["cell.mlp.2.bias"]).astype(F32)
        lam = sigmoid(np.concatenate([fused, zs], -1) @ sd["cell.gate.weight"].T + sd["cell.gate.bias"])
        self.z[idx] = (1 - lam) * zs + lam * cand
        self.age[idx] = 0
        return len(idx), int(mask.sum())

    def readout(self, x):
        sd = self.sd
        w = softmax(self.z @ sd["readout.score.weight"].T + sd["readout.score.bias"], 0)
        g = (w * self.z).sum(0)
        if self.fresh:
            g = np.concatenate([g, x.mean(0), x.max(0)])
        h = gelu(g @ sd["readout.head.0.weight"].T + sd["readout.head.0.bias"])
        o = h @ sd["readout.head.3.weight"].T + sd["readout.head.3.bias"]
        return softmax(o.reshape(STEP, self.C), -1)

    def run(self, wav, policy="full", budget=1.0):
        self.reset()
        n = len(wav) // CHUNK
        T = np.zeros((n, 5)); probs = []; nodes = []
        pc = time.perf_counter
        for t in range(n):
            t0 = pc(); f = self.features(wav[t * CHUNK:(t + 1) * CHUNK])
            t1 = pc(); x, spec = self.encoder(f)
            t2 = pc(); idx = self.select(policy, budget, x, spec)
            t3 = pc(); nu, _ = self.graph(x, idx)
            t4 = pc(); p = self.readout(x)
            t5 = pc()
            T[t] = [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4]
            probs.append(p); nodes.append(nu)
            self.x_prev, self.spec_prev = x, spec
        return np.concatenate(probs), T, np.array(nodes)


def main():
    import torch
    from .audio import CausalLogMel
    from .models import StreamModel
    from .select import make_policy, region_masks, Controller, CONTROLLER_HEADS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--n_rec", type=int, default=40)
    ap.add_argument("--weights", default="C")
    ap.add_argument("--fresh", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    st = dict(np.load(os.path.join(a.data, "stats.npz")))
    sd_t = torch.load(os.path.join(a.run, f"model_{a.weights}.pt"), map_location="cpu")
    cs_t = torch.load(os.path.join(a.run, f"ctrl_{a.weights}.pt"), map_location="cpu")
    sd = {k: v.numpy() for k, v in sd_t.items()}
    cs = {k: v.numpy() for k, v in cs_t.items()}
    rt = NPRuntime(sd, cs, st, fresh=bool(a.fresh))
    items = pickle.load(open(os.path.join(a.data, "test2024.pkl"), "rb"))[: a.n_rec]

    # equivalence vs the PyTorch batched model
    model = StreamModel("agn", fresh=bool(a.fresh)); model.load_state_dict(sd_t); model.eval()
    ctrls = torch.nn.ModuleDict({h: Controller(model.dx, model.d) for h in CONTROLLER_HEADS}); ctrls.load_state_dict(cs_t)
    fe = CausalLogMel(st["mean"], st["std"])
    wav = items[0]["wav"].astype(np.float32) / 32768.0
    wav = wav[: len(wav) // CHUNK * CHUNK]
    with torch.no_grad():
        ref_full = torch.softmax(model(fe(torch.from_numpy(wav)[None]))[0][0], -1).numpy()
        pol = make_policy("util", model, region_masks(model.cell.adj), 0.25, ctrls)
        ref_sel = torch.softmax(model(fe(torch.from_numpy(wav)[None]), pol, 0.25)[0][0], -1).numpy()
    eq = dict(full_maxabs=float(np.abs(rt.run(wav, "full")[0] - ref_full).max()),
              util25_maxabs=float(np.abs(rt.run(wav, "util", 0.25)[0] - ref_sel).max()))
    print("equivalence", eq, flush=True)

    from .benchutil import interleaved, pin_one_core
    core = pin_one_core()
    res = dict(equivalence=eq, runtime="numpy", threads=1, core=core, protocol="interleaved",
               cpu=os.popen("lscpu 2>/dev/null | grep 'Model name'").read().strip() or os.environ.get("PROCESSOR_IDENTIFIER", ""))
    rt.run(items[0]["wav"][: CHUNK * 50].astype(np.float32) / 32768.0, "full")  # warm-up

    def run_one(it, name, b):
        _, T, nn_ = rt.run(it["wav"].astype(np.float32) / 32768.0, name, b)
        return T, nn_, None

    res.update(interleaved(run_one, items))
    json.dump(res, open(a.out or os.path.join(a.run, f"bench_np_{a.weights}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
