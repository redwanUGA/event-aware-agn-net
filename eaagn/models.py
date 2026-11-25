"""Streaming models: cheap always-on encoder, non-graph baselines, and the
stateful acoustic AGN-Net cell with selective (budgeted) updates.

Shapes: feats (B, 2, F=128, T) -> cheap node features x (B, S, N, dx), S = T // STEP.
Graph state z (B, N, d) persists across steps; inactive nodes carry their state.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio import N_MELS, STEP

N_CLASSES = 8


def chain_adj(N: int, hops=(1, 2)) -> torch.Tensor:
    A = torch.zeros(N, N, dtype=torch.bool)
    for h in hops:
        i = torch.arange(N - h)
        A[i, i + h] = True
        A[i + h, i] = True
    return A


def node_center_hz(N: int = 64, fmin: float = 50.0, fmax: float = 4000.0) -> torch.Tensor:
    """Centre frequency (Hz) of each graph node; node i owns mel channels 2i and 2i+1 of the
    N_MELS-band filterbank on [fmin, fmax] (HTK mel scale, as in audio.CausalLogMel)."""
    mel = lambda f: 2595.0 * math.log10(1.0 + f / 700.0)
    m = torch.linspace(mel(fmin), mel(fmax), N_MELS + 2)[1:-1]                 # band centres (mel)
    hz = 700.0 * (10 ** (m / 2595.0) - 1.0)
    return hz.view(N, N_MELS // N).mean(1)


def harmonic_adj(N: int = 64, harmonics=(2, 3), tol: float = 0.06) -> torch.Tensor:
    """Chain graph (hops 1, 2) plus harmonic edges: node i is linked to the node whose centre
    frequency is closest to k * f_i (k in harmonics) when the relative mismatch is below tol.
    Wheezes and stridor are (quasi-)harmonic, so these edges let one update propagate evidence
    between a fundamental and its overtones in a single message-passing step."""
    A = chain_adj(N)
    f = node_center_hz(N)
    for i in range(N):
        for k in harmonics:
            j = int(torch.argmin((f - k * f[i]).abs()))
            if j != i and abs(float(f[j]) - k * float(f[i])) / (k * float(f[i])) < tol:
                A[i, j] = A[j, i] = True
    return A


class CheapEncoder(nn.Module):
    """Always-on path: two small causal convs, receptive field ~150 ms.

    conv1 runs at the 10 ms frame rate; frames are averaged into 40 ms steps;
    conv2 runs at the step rate and halves frequency (128 mel -> 64 regions).
    """

    def __init__(self, dx: int = 32, c1: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(2, c1, (5, 3), padding=(2, 0))
        self.conv2 = nn.Conv2d(c1, dx, (5, 3), stride=(2, 1), padding=(2, 0))
        self.norm = nn.LayerNorm(dx)

    def forward(self, feats):
        h = F.gelu(self.conv1(F.pad(feats, (2, 0))))
        B, C, Fq, T = h.shape
        S = T // STEP
        h = h[..., : S * STEP].reshape(B, C, Fq, S, STEP).mean(-1)
        h = F.gelu(self.conv2(F.pad(h, (2, 0))))          # (B, dx, N, S)
        return self.norm(h.permute(0, 3, 2, 1))             # (B, S, N, dx)


def node_spectral(feats):
    """Per-node, per-step mean of the normalised log-mel bins it owns: (B, S, N, 4)."""
    B, R, Fq, T = feats.shape
    S = T // STEP
    m = feats[..., : S * STEP].reshape(B, R, Fq // 2, 2, S, STEP).mean(-1)   # B,R,N,2,S
    return m.permute(0, 4, 2, 1, 3).reshape(B, S, Fq // 2, R * 2)


class Readout(nn.Module):
    """Attention pooling over node states + fresh cheap evidence (residual path)."""

    def __init__(self, d: int, dx: int, C: int = N_CLASSES, fresh: bool = True, hidden: int = 128):
        super().__init__()
        self.fresh = fresh
        self.score = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.Linear(d + (2 * dx if fresh else 0), hidden), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(hidden, C))

    def forward(self, z, x):
        w = torch.softmax(self.score(z), dim=-2)
        g = (w * z).sum(-2)
        if self.fresh:
            f = torch.cat([x.mean(-2), x.amax(-2)], -1)
            g = torch.cat([g, f.expand(*g.shape[:-1], f.shape[-1])], -1)
        return self.head(g)


# ----------------------------------------------------------------------------- graph cells
class AGNCell(nn.Module):
    """Acoustic AGN-Net update: priority-free layered attention aggregation with
    learnable residual scalars beta_l, cross-layer fusion, and a gated temporal-momentum
    state update z <- (1-lam) z + lam * cand. With `upd` given, message passing runs on the
    subgraph induced by the update set U (AGN-Net's localized propagation) and nodes outside U
    carry their state unchanged."""

    def __init__(self, N, dx, d=96, L=2, adj=None, mp="gat"):
        super().__init__()
        self.N, self.d, self.L, self.mp = N, d, L, mp
        self.node_emb = nn.Parameter(torch.randn(N, d) * 0.02)
        self.inp = nn.Linear(dx + d, d)
        if mp == "gat":
            self.W = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
            self.a_src = nn.Parameter(torch.randn(L, d) * 0.05)
            self.a_dst = nn.Parameter(torch.randn(L, d) * 0.05)
        elif mp == "conv":  # matched frequency-convolution control (same 5-band neighbourhood)
            self.W = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2) for _ in range(L)])
        else:
            raise ValueError(mp)
        self.beta = nn.Parameter(torch.full((L,), 0.5))
        self.fuse_w = nn.Parameter(torch.zeros(L + 1))
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.gate = nn.Linear(2 * d, 1)
        self.register_buffer("adj", chain_adj(N) if adj is None else adj)

    def forward(self, x, z_prev, upd=None):
        h = F.gelu(self.inp(torch.cat([x, z_prev], -1)) + self.node_emb)
        if upd is not None:
            h = h * upd[..., None]
        mask = self.adj[None] if upd is None else (self.adj[None] & upd[:, :, None] & upd[:, None, :])
        hs = [h]
        for l in range(self.L):
            if self.mp == "gat":
                Wh = self.W[l](h)
                e = (Wh @ self.a_dst[l])[..., :, None] + (Wh @ self.a_src[l])[..., None, :]
                e = F.leaky_relu(e, 0.2).clamp(-30.0, 30.0).masked_fill(~mask, -1e9)
                alpha = torch.softmax(e, -1) * mask
                msg = alpha @ Wh
            else:
                msg = self.W[l](h.transpose(1, 2)).transpose(1, 2)
            h = F.elu(msg + self.beta[l] * h)
            if upd is not None:
                h = h * upd[..., None]
            hs.append(h)
        a = torch.softmax(self.fuse_w, 0)
        fused = self.ln(sum(a[i] * hs[i] for i in range(len(hs))))
        cand = torch.tanh(self.mlp(fused))
        lam = torch.sigmoid(self.gate(torch.cat([fused, z_prev], -1)))
        z = (1 - lam) * z_prev + lam * cand
        if upd is not None:
            z = torch.where(upd[..., None], z, z_prev)
        return z


class GraphGRUCell(nn.Module):
    """Plain temporal-graph baseline: neighbour-mean aggregation + shared node GRU."""

    def __init__(self, N, dx, d=96, adj=None):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(N, d) * 0.02)
        self.inp = nn.Linear(dx + d, d)
        self.gru = nn.GRUCell(d, d)
        A = chain_adj(N) if adj is None else adj.bool()
        self.register_buffer("adj", A)
        self.register_buffer("adjn", A.float() / A.float().sum(1, keepdim=True))

    def forward(self, x, z_prev, upd=None):
        B, N, d = z_prev.shape
        m = self.adjn @ z_prev
        inp = F.gelu(self.inp(torch.cat([x, m], -1)) + self.node_emb)
        z = self.gru(inp.reshape(B * N, d), z_prev.reshape(B * N, d)).view(B, N, d)
        if upd is not None:
            z = torch.where(upd[..., None], z, z_prev)
        return z


# ----------------------------------------------------------------------------- full models
class StreamModel(nn.Module):
    GRAPH = ("agn", "gconv", "ggru", "agnh", "ggruh")

    def __init__(self, arch="agn", N=64, dx=32, d=96, fresh=True, C=N_CLASSES):
        super().__init__()
        self.arch, self.N, self.dx, self.d = arch, N, dx, d
        self.enc = CheapEncoder(dx)
        if arch == "agn":
            self.cell = AGNCell(N, dx, d, mp="gat")
        elif arch == "gconv":
            self.cell = AGNCell(N, dx, d, mp="conv")
        elif arch == "ggru":
            self.cell = GraphGRUCell(N, dx, d)
        elif arch == "agnh":   # AGN-Net cell on the harmonic graph (chain + harmonic edges)
            self.cell = AGNCell(N, dx, d, mp="gat", adj=harmonic_adj(N))
        elif arch == "ggruh":  # graph GRU on the harmonic graph
            self.cell = GraphGRUCell(N, dx, d, adj=harmonic_adj(N))
        elif arch == "pool":
            self.node = nn.Sequential(nn.Linear(dx, d), nn.GELU(), nn.Linear(d, d))
        elif arch in ("tcn", "gru"):
            H = 128
            self.proj = nn.Linear(N * dx, H)
            if arch == "gru":
                self.rnn = nn.GRU(H, H, batch_first=True)
            else:
                self.tcn = nn.ModuleList([nn.Conv1d(H, H, 3, dilation=2 ** i) for i in range(6)])
                self.tln = nn.ModuleList([nn.LayerNorm(H) for _ in range(6)])
            d = H
        else:
            raise ValueError(arch)
        self.C = C
        # each 40 ms step emits STEP frame-level (10 ms) class logits
        self.readout = Readout(d, dx, C * STEP, fresh=fresh) if arch in self.GRAPH + ("pool",) else \
            nn.Sequential(nn.Linear(d + 2 * dx, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, C * STEP))

    @property
    def is_graph(self):
        return self.arch in self.GRAPH

    def init_state(self, B, device):
        return torch.zeros(B, self.N, self.d, device=device)

    def forward(self, feats, policy=None, budget=1.0, collect=None, lengths=None):
        """Returns frame logits (B, S*STEP, C) and a stats dict.

        policy: None = full update every step; else a callable
            policy(t, ctx) -> bool mask (B, N) of nodes to update (see select.py).
        collect: optional callback(t, ctx) invoked before each selective step (stage B).
        lengths: optional (B,) number of valid steps per row; update statistics count only
            valid (non-padded) steps: upd_nodes / valid_node_steps is the realised update fraction.
        """
        x = self.enc(feats)
        B, S, N, dx = x.shape
        fr = lambda o: o.reshape(B, S * STEP, self.C)
        fresh = torch.cat([x.mean(-2), x.amax(-2)], -1)
        if self.arch == "pool":
            return fr(self.readout(self.node(x), x)), {}
        if self.arch == "gru":
            h, _ = self.rnn(self.proj(x.reshape(B, S, N * dx)))
            return fr(self.readout(torch.cat([h, fresh], -1))), {}
        if self.arch == "tcn":
            h = self.proj(x.reshape(B, S, N * dx)).transpose(1, 2)
            for i, (c, ln) in enumerate(zip(self.tcn, self.tln)):
                y = F.gelu(c(F.pad(h, (2 * 2 ** i, 0))))
                h = ln((h + y).transpose(1, 2)).transpose(1, 2)
            return fr(self.readout(torch.cat([h.transpose(1, 2), fresh], -1))), {}

        dev = feats.device
        spec = node_spectral(feats) if policy is not None else None
        z = self.init_state(B, dev)
        age = torch.zeros(B, N, device=dev)
        lengths = torch.full((B,), S, device=dev) if lengths is None else lengths.to(dev)
        logits = []
        upd_nodes = torch.zeros((), device=dev); forced_nodes = torch.zeros((), device=dev)
        x_prev = torch.zeros_like(x[:, 0])
        for t in range(S):
            xt = x[:, t]
            valid = (t < lengths).float()
            if policy is None:
                z = self.cell(xt, z)
                upd_nodes = upd_nodes + valid.sum() * N
            else:
                ctx = dict(model=self, x=xt, x_prev=x_prev, z_prev=z, age=age, budget=budget,
                           spec=spec[:, t], spec_prev=spec[:, t - 1] if t > 0 else torch.zeros_like(spec[:, 0]), t=t)
                if collect is not None:
                    collect(t, ctx)
                upd = policy(t, ctx)
                z = self.cell(xt, z, upd)
                age = torch.where(upd, torch.zeros_like(age), age + 1)
                upd_nodes = upd_nodes + (upd.float().sum(-1) * valid).sum()
                if "forced" in ctx:
                    forced_nodes = forced_nodes + (ctx["forced"].float().sum(-1) * valid).sum()
            logits.append(self.readout(z, xt))
            x_prev = xt
        vns = float(lengths.clamp(max=S).sum()) * N
        return fr(torch.stack(logits, 1)), dict(upd_nodes=float(upd_nodes), forced_nodes=float(forced_nodes),
                                                valid_node_steps=vns)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
