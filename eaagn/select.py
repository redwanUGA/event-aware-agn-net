"""Budgeted update selection: region masks, cost-aware greedy, controllers,
policies, and the counterfactual task-utility targets used to train the controller.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio import STEP

A_MAX = 50  # maximum state age in steps (2 s) before a forced refresh (also the controller's age scale)
REFRESH_AGE = None  # override of the forced-refresh threshold (revision experiments); None -> A_MAX


def region_masks(adj: torch.Tensor) -> torch.Tensor:
    """R[i] = 1-hop expanded update region B_i of seed i (seed + graph neighbours)."""
    return adj.bool() | torch.eye(adj.shape[0], dtype=torch.bool, device=adj.device)


@torch.no_grad()
def forced_refresh(age, budget_nodes: int):
    """Safeguard: nodes whose cached state is >= A_MAX steps old are refreshed, charged against
    the budget and capped at budget_nodes per step (stalest first, ties by node index)."""
    over = age >= (A_MAX if REFRESH_AGE is None else REFRESH_AGE)
    if budget_nodes <= 0:
        return torch.zeros_like(over)
    N = age.shape[-1]
    key = torch.where(over, age - torch.arange(N, device=age.device) * 1e-3, torch.full_like(age, -1.0))
    top = key.topk(min(budget_nodes, N), -1).indices
    return torch.zeros_like(over).scatter_(-1, top, True) & over


@torch.no_grad()
def greedy_select(scores, R, budget_nodes: int, age=None, refresh=True):
    """Coverage-discounted cost-aware greedy over seed regions.

    Seed i carries a nonnegative utility s_i for its expanded region B_i. Adding seed i to the
    current update set U costs new_i = |B_i minus U| node updates and is credited the discounted gain
    s_i * new_i / |B_i|, so the cost-effectiveness ratio is s_i / |B_i| (independent of overlap,
    which avoids the degenerate contiguous growth of an undiscounted s_i / new_i rule). Seeds are
    visited in decreasing ratio (stable, ties by index) and added while |U| <= budget_nodes, where
    U (the actual computation region) is the union of expanded regions plus forced refreshes."""
    B, N = scores.shape
    dev = scores.device
    U = forced_refresh(age, budget_nodes) if (refresh and age is not None) else torch.zeros(B, N, dtype=torch.bool, device=dev)
    ratio = scores.clamp_min(0) / R.sum(1).float()
    order = torch.sort(ratio, dim=-1, descending=True, stable=True).indices
    for j in range(N):
        cand = R[order[:, j]]                                   # (B, N) region of j-th best seed
        new = (cand & ~U).sum(-1)
        ok = (new > 0) & (U.sum(-1) + new <= budget_nodes)
        U = U | (cand & ok[:, None])
        if j % 16 == 15 and bool((U.sum(-1) >= budget_nodes).all()):
            break
    return U


def region_bits(R):
    """Integer bitmasks of each seed region, for the batch-1 streaming selector."""
    Rn = R.cpu().numpy() if hasattr(R, "cpu") else R
    return [sum(1 << j for j in range(Rn.shape[1]) if Rn[i, j]) for i in range(Rn.shape[0])]


def greedy_bitmask(scores, Rbits, Rsize, age, budget_nodes: int, refresh=True):
    """Batch-1 equivalent of greedy_select on NumPy arrays with integer bitmasks (identical
    ordering: forced refresh stalest-first, then seeds by stable descending s_i/|B_i|)."""
    import numpy as np
    U, cnt = 0, 0
    if refresh:
        over = np.nonzero(age >= A_MAX)[0]
        for i in sorted(over.tolist(), key=lambda i: (-age[i], i))[:budget_nodes]:
            U |= 1 << i; cnt += 1
    ratio = np.maximum(scores, 0) / Rsize
    for i in np.argsort(-ratio, kind="stable"):
        if cnt >= budget_nodes:
            break
        new = (Rbits[i] & ~U).bit_count()
        if new and cnt + new <= budget_nodes:
            U |= Rbits[i]; cnt += new
    return np.array([i for i in range(len(Rbits)) if (U >> i) & 1], dtype=np.int64)


class Controller(nn.Module):
    """Lightweight per-seed scorer on current evidence + cached state + staleness + budget.
    A pointwise MLP followed by a 5-band conv over the node axis so a seed can see its region."""

    def __init__(self, dx, d, hidden=24):
        super().__init__()
        self.inp = nn.Linear(2 * dx + d + 2, hidden)
        self.conv = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, x_prev, z_prev, age, budget):
        a = (torch.log1p(age) / math.log1p(A_MAX))[..., None]
        b = torch.full_like(a, float(budget))
        h = F.gelu(self.inp(torch.cat([x, x - x_prev, z_prev, a, b], -1)))
        h = h + F.gelu(self.conv(h.transpose(1, 2)).transpose(1, 2))
        return F.softplus(self.out(h)).squeeze(-1)


CONTROLLER_HEADS = ("util", "util_hold", "util_kl", "change")


def frame_logp(readout, z, x, C):
    o = readout(z, x)
    return F.log_softmax(o.view(*o.shape[:-1], STEP, C), -1)


def _kl(lp, lq):  # KL(p || q) averaged over the STEP frames
    return (lp.exp() * (lp - lq)).sum(-1).mean(-1)


@torch.no_grad()
def utility_targets(model, ctx, R, y=None, alpha=1.0, C=8):
    """One-step counterfactual intervention targets for every seed region B_i.

    hold(i): all regions updated except B_i (B_i keeps its previous state)
    only(i): only B_i updated
    Skip-harm utility = 1/2 [KL(p_full||p_hold) + (KL(p_full||p_none) - KL(p_full||p_only))_+]
                        + alpha/2 [ (CE_hold - CE_full)_+ + (CE_none - CE_only)_+ ]   (labelled steps)
    Approximation (stated in the paper): region states under hold/only reuse the
    full-neighbourhood candidate, i.e. the boundary defect is ignored.
    """
    x, zp = ctx["x"], ctx["z_prev"]
    zf = model.cell(x, zp)
    Rm = R[None, :, :, None]
    z_hold = torch.where(Rm, zp[:, None], zf[:, None])
    z_only = torch.where(Rm, zf[:, None], zp[:, None])
    ro = model.readout
    lp_full = frame_logp(ro, zf, x, C)                  # (B, STEP, C)
    lp_none = frame_logp(ro, zp, x, C)
    lp_hold = frame_logp(ro, z_hold, x[:, None], C)     # (B, Ns, STEP, C)
    lp_only = frame_logp(ro, z_only, x[:, None], C)
    u_hold = _kl(lp_full[:, None], lp_hold).clamp_min(0)  # KL >= 0; clamp float round-off
    u_add = (_kl(lp_full, lp_none)[:, None] - _kl(lp_full[:, None], lp_only)).clamp_min(0)
    ch = (zf - zp).norm(dim=-1)                         # realised representation change
    Rf = R.float()
    change = (ch @ Rf.T) / Rf.sum(1)
    out = dict(util_kl=0.5 * (u_hold + u_add), util_hold=u_hold.clone(), change=change)
    if y is not None:
        valid = (y >= 0)                                # (B, STEP)
        yy = y.clamp_min(0)

        def ce(lp):  # mean frame CE over labelled frames; lp (B,STEP,C) or (B,Ns,STEP,C)
            single = lp.dim() == 3
            lp4 = lp[:, None] if single else lp
            idx = yy[:, None, :, None].expand(lp4.shape[0], lp4.shape[1], STEP, 1)
            g = -lp4.gather(-1, idx).squeeze(-1)
            vv = valid[:, None].float()
            r = (g * vv).sum(-1) / vv.sum(-1).clamp_min(1)
            return r[:, 0] if single else r

        l_hold = (ce(lp_hold) - ce(lp_full)[:, None]).clamp_min(0)
        l_add = (ce(lp_none)[:, None] - ce(lp_only)).clamp_min(0)
        out["util"] = out["util_kl"] + alpha * 0.5 * (l_hold + l_add)
        out["util_hold"] = out["util_hold"] + alpha * l_hold
    else:
        out["util"] = out["util_kl"]
    return out


def policy_scores(name, model, R, ctx, budget, controllers=None, gen=None):
    """Nonnegative per-seed REGION-TOTAL scores for every selection policy (shared by the batched
    model and the streaming benchmark so both select identically). The greedy ranks seeds by
    score / |B_i|, i.e. by the per-node mean of a region total, for every policy alike: node-level
    heuristics are summed over the region; the skip-harm utilities are whole-region quantities
    already; the change head was trained on a region MEAN, so it is rescaled by |B_i|."""
    Rf = R.float()
    size = Rf.sum(1)
    reg = lambda node: node @ Rf.T                              # region total of a node quantity
    if name == "random":       # region total of i.i.d. node draws -> per-node mean is random
        return reg(torch.rand(ctx["x"].shape[0], R.shape[0], device=ctx["x"].device, generator=gen))
    if name == "periodic":     # stalest region first == round-robin refresh
        return reg(ctx["age"] + 1.0)
    if name == "energy":       # loudest region (fixed monotone transform keeps scores > 0)
        return reg(F.softplus(ctx["spec"].mean(-1)))
    if name == "acoustic":     # observed spectral change of the region
        return reg((ctx["spec"] - ctx["spec_prev"]).norm(dim=-1))
    if name == "oracle":       # hindsight label-free utility (needs the full update): diagnostic only
        return utility_targets(model, ctx, R)["util"]
    if name == "change":       # target was the region-mean change -> rescale to a region total
        return controllers[name](ctx["x"], ctx["x_prev"], ctx["z_prev"], ctx["age"], budget) * size
    if name in CONTROLLER_HEADS or (controllers is not None and name in controllers):
        return controllers[name](ctx["x"], ctx["x_prev"], ctx["z_prev"], ctx["age"], budget)
    raise ValueError(name)


def make_policy(name, model, R, budget, controllers=None, refresh=True, gen=None):
    """Return policy(t, ctx) -> (B, N) bool update mask for a fixed budget fraction."""
    N = R.shape[0]
    budget_nodes = int(round(budget * N))

    @torch.no_grad()
    def pol(t, ctx):
        B = ctx["x"].shape[0]
        dev = ctx["x"].device
        if name == "none":
            return torch.zeros(B, N, dtype=torch.bool, device=dev)
        if name == "full" or budget >= 1.0:
            return torch.ones(B, N, dtype=torch.bool, device=dev)
        s = policy_scores(name, model, R, ctx, budget, controllers, gen)
        U = greedy_select(s, R, budget_nodes, ctx["age"], refresh)
        if refresh:
            ctx["forced"] = forced_refresh(ctx["age"], budget_nodes)
        return U

    return pol
