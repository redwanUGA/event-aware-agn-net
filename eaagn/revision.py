"""Experiments added in revision round 1 (MDPI Algorithms, algorithms-4716513).

Every subcommand loads the saved checkpoints of one finished run directory (stage-C weights and
controllers unless stated) and writes rev_<cmd>.json there. All are idempotent.

    python -m eaagn.revision e2e        --run results/agn_s0 --data data   # end-to-end gate baseline (R1.3, R2.1)
    python -m eaagn.revision multistep  --run results/agn_s0 --data data   # one-step vs multi-step skip harm (R1.6)
    python -m eaagn.revision longstream --run results/agn_s0 --data data   # long streams + adaptive refresh (R1.7)
    python -m eaagn.revision lipschitz  --run results/agn_s0 --data data   # measured Lipschitz ratios, bound check (R1.8)
    python -m eaagn.revision collars    --run results/agn_s0 --data data   # boundary-tolerant metrics (R1.1)
    python -m eaagn.revision scaling    --out results/scaling.json          # controller-overhead scaling (R1.2)
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import metrics as M
from . import select as S
from .audio import SR, STEP, CausalLogMel
from .models import StreamModel
from .select import CONTROLLER_HEADS, Controller, greedy_select, make_policy, region_masks, utility_targets
from .train import (CHUNK, class_weights, eval_batches, evaluate, log, prep_items, run_policy_grid,
                    train_batches, _spearman, _topk_mask)

E2E_VERSION = 1


# ----------------------------------------------------------------------------- loading
def load_run(a, weights="C", need_train=False):
    summ = json.load(open(os.path.join(a.run, "summary.json")))
    args = summ["args"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args["seed"]); np.random.seed(args["seed"]); random.seed(args["seed"])
    names = ["val", "test2024", "test2025"] + (["train"] if need_train else [])
    splits = {}
    for k in names:
        fn = f"{k}_pg.pkl" if k in ("train", "val") and args.get("split", "pg") == "pg" else f"{k}.pkl"
        p = os.path.join(a.data, fn)
        if os.path.exists(p):
            splits[k] = prep_items(pickle.load(open(p, "rb")))
    st = np.load(os.path.join(a.data, "stats.npz"))
    fe = CausalLogMel(st["mean"], st["std"]).to(dev)
    model = StreamModel(args["arch"], fresh=bool(args["fresh"])).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, f"model_{weights}.pt"), map_location=dev))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ctrls = nn.ModuleDict({h: Controller(model.dx, model.d) for h in CONTROLLER_HEADS}).to(dev)
    ctrls.load_state_dict(torch.load(os.path.join(a.run, f"ctrl_{weights}.pt"), map_location=dev))
    ctrls.eval()
    R = region_masks(model.cell.adj)
    return summ, args, dev, splits, fe, model, ctrls, R


def out_path(a, name):
    return os.path.join(a.run, f"rev_{name}.json")


def done(a, name):
    p = out_path(a, name)
    if os.path.exists(p):
        log("exists, skipping", p)
        return True
    return False


# ----------------------------------------------------------------------------- R1.3 / R2.1: end-to-end gate
def e2e_rollout(model, gate, feats, R, budget, tau_g, temp, gen):
    """Differentiable selective rollout for the end-to-end gate (model frozen, gate trained).

    Forward pass is exactly the deployed one: hard update set U from the same budget-exact greedy
    on (Gumbel-perturbed) gate scores, and the cell on the subgraph induced by U. The backward pass
    uses a straight-through relaxation: node j's soft inclusion q_j = sigmoid((log r_j - theta)/temp),
    r_j = s_j / |B_j|, theta = the (1-b) quantile of log r (detached), and the state update is
    z = m * z_cand + (1 - m) * z_prev with m = U + q - q.detach(). z_cand equals the selective update
    on U and the full update elsewhere, so unselected nodes also receive a task gradient."""
    x = model.enc(feats)
    B, Sn, N, dx = x.shape
    z = model.init_state(B, feats.device)
    age = torch.zeros(B, N, device=feats.device)
    x_prev = torch.zeros_like(x[:, 0])
    k = int(round(budget * N))
    size = R.float().sum(1)
    logits = []
    for t in range(Sn):
        xt = x[:, t]
        s = gate(xt, x_prev, z, age, budget)                       # (B, N) >= 0 seed scores
        lr = torch.log(s / size + 1e-8)
        with torch.no_grad():
            g = -torch.log(-torch.log(torch.rand(lr.shape, device=lr.device, generator=gen).clamp(1e-9, 1 - 1e-9)))
            U = greedy_select(torch.exp(lr + tau_g * g), R, k, age, True)
            theta = torch.quantile(lr.detach(), 1.0 - budget, dim=-1, keepdim=True)
        q = torch.sigmoid((lr - theta) / temp)
        m = (U.float() + q - q.detach())[..., None]
        z_sel = model.cell(xt, z, U)
        z_full = model.cell(xt, z)
        z_cand = torch.where(U[..., None], z_sel, z_full)
        z = m * z_cand + (1 - m) * z
        age = torch.where(U, torch.zeros_like(age), age + 1)
        logits.append(model.readout(z, xt))
        x_prev = xt
    return torch.stack(logits, 1).reshape(B, Sn * STEP, model.C)


def cmd_e2e(a):
    if done(a, "e2e"):
        return
    summ, args, dev, splits, fe, model, ctrls, R = load_run(a, "C", need_train=True)
    tr = splits.pop("train")
    cw = class_weights(tr).to(dev)
    rng = np.random.default_rng(args["seed"] + 101)
    gen = torch.Generator(device=dev); gen.manual_seed(args["seed"] + 101)
    gate = Controller(model.dx, model.d).to(dev)
    ck = os.path.join(a.run, "ctrl_e2e.pt")
    hist = []
    if os.path.exists(ck):
        gate.load_state_dict(torch.load(ck, map_location=dev))
    else:
        opt = torch.optim.Adam(gate.parameters(), lr=2e-3)
        n_ep = a.e2e_epochs
        for ep in range(n_ep):
            frac = ep / max(n_ep - 1, 1)
            tau_g = 0.5 * (1 - frac) + 0.05 * frac
            tl, n, t0 = 0.0, 0, time.time()
            for W, Y in train_batches(tr, a.bs, a.crop_s, rng):
                budget = float(rng.choice([0.125, 0.25, 0.5]))
                feats = fe(W.to(dev))
                with torch.no_grad():
                    tlog, _ = model(feats)                        # full-update teacher (same weights)
                lo = e2e_rollout(model, gate, feats, R, budget, tau_g, a.temp, gen)
                Yd = Y.to(dev).reshape(-1)
                ce = F.cross_entropy(lo.reshape(-1, model.C), Yd, weight=cw, ignore_index=-1)
                mk = (Yd >= 0).float()
                kd = (F.kl_div(F.log_softmax(lo.reshape(-1, model.C), -1), F.log_softmax(tlog.reshape(-1, model.C), -1),
                               log_target=True, reduction="none").sum(-1) * mk).sum() / mk.sum().clamp_min(1)
                loss = ce + kd                                     # the stage-C task objective
                opt.zero_grad(); loss.backward()
                if torch.isfinite(loss):
                    nn.utils.clip_grad_norm_(gate.parameters(), 5.0); opt.step()
                    tl += loss.item(); n += 1
            hist.append(dict(epoch=ep, loss=tl / max(n, 1), tau_g=tau_g, sec=time.time() - t0))
            log(f"[e2e] ep{ep} loss {tl / max(n, 1):.4f} tau {tau_g:.2f} ({time.time() - t0:.0f}s)")
        torch.save(gate.state_dict(), ck)
    gate.eval()
    for p in gate.parameters():
        p.requires_grad_(False)
    ext = nn.ModuleDict(dict(ctrls.items()))
    ext["e2e"] = gate
    a2 = argparse.Namespace(seed=args["seed"], quick=0)
    grid = run_policy_grid(model, fe, splits, dev, ext, R, "Ce2e", a.run, a2, only=["e2e"], res={})
    json.dump(dict(version=E2E_VERSION, hist=hist, grid=grid), open(out_path(a, "e2e"), "w"), indent=1)
    log("e2e done")


def cmd_e2ediag(a):
    """Ranking quality of the end-to-end gate vs. the skip-harm controller against measured utility
    (same protocol as train.ranking_diagnostics, stage-C weights)."""
    if done(a, "e2ediag"):
        return
    from . import train as TR
    summ, args, dev, splits, fe, model, ctrls, R = load_run(a, "C")
    gate = Controller(model.dx, model.d).to(dev)
    gate.load_state_dict(torch.load(os.path.join(a.run, "ctrl_e2e.pt"), map_location=dev)); gate.eval()
    ext = nn.ModuleDict(dict(ctrls.items())); ext["e2e"] = gate
    TR.CONTROLLER_HEADS = tuple(CONTROLLER_HEADS) + ("e2e",)
    d = TR.ranking_diagnostics(model, fe, splits["val"], dev, ext, R)
    d.pop("per_rec_conc", None)
    json.dump(d, open(out_path(a, "e2ediag"), "w"), indent=1)
    log("e2ediag", json.dumps(d["spearman"]))


# ----------------------------------------------------------------------------- R1.6: multi-step harm
@torch.no_grad()
def cmd_multistep(a):
    if done(a, "multistep2"):
        return
    summ, args, dev, splits, fe, model, ctrls, R = load_run(a, "C")
    items = splits["val"][: a.n_rec]
    N = R.shape[0]
    C = model.C
    combos = [(1, 1), (1, 5), (1, 25), (5, 5), (5, 25), (25, 25)]   # (hold steps k, horizon H)
    Hmax = max(h for _, h in combos)
    k25 = int(round(0.25 * N))
    acc = collections.defaultdict(list)
    ratio = collections.defaultdict(list)
    for idx, W in eval_batches(items, 8):
        feats = fe(W.to(dev))
        x = model.enc(feats)
        B, Sn = x.shape[:2]
        Tst = torch.tensor([items[i]["T"] // STEP for i in idx], device=dev)
        # full-update trajectory: state before each step
        Z = [model.init_state(B, dev)]
        for t in range(Sn):
            Z.append(model.cell(x[:, t], Z[-1]))
        for t in range(10, Sn - Hmax, a.stride):
            valid = (t + Hmax) < Tst
            if not bool(valid.any()):
                continue
            zp = Z[t]
            ctx = dict(x=x[:, t], x_prev=x[:, t - 1], z_prev=zp, age=torch.zeros(B, N, device=dev), budget=0.25,
                       spec=None, spec_prev=None)
            u1 = utility_targets(model, ctx, R)
            sc = {h: ctrls[h](x[:, t], x[:, t - 1], zp, ctx["age"], 0.25) for h in ("util", "util_kl", "change")}
            sc["change"] = sc["change"] * R.float().sum(1)
            sc["change_meas"] = u1["change"] * R.float().sum(1)
            # branches: B*N trajectories, region i held for k steps from t, then full updates
            Rm = R[None, :, :, None]
            for kh in sorted({k for k, _ in combos}):
                zb = zp[:, None].expand(B, N, N, model.d).reshape(B * N, N, model.d).clone()
                zf = zp.clone()
                harm = torch.zeros(B, N, device=dev)
                hs = {}
                for h in range(Hmax):
                    xt = x[:, t + h]
                    zf = model.cell(xt, zf)
                    zn = model.cell(xt[:, None].expand(B, N, N, -1).reshape(B * N, N, -1), zb)
                    if h < kh:
                        zn = torch.where(Rm.expand(B, N, N, 1).reshape(B * N, N, 1), zb, zn)
                    zb = zn
                    lpf = F.log_softmax(model.readout(zf, xt).view(B, STEP, C), -1)
                    lpb = F.log_softmax(model.readout(zb, xt[:, None].expand(B, N, N, -1).reshape(B * N, N, -1))
                                        .view(B, N, STEP, C), -1)
                    harm = harm + (lpf.exp()[:, None] * (lpf[:, None] - lpb)).sum(-1).mean(-1)
                    hs[h + 1] = harm.clone()
                for (k_, H) in combos:
                    if k_ != kh:
                        continue
                    tgt = hs[H]
                    v = valid
                    key = f"k{k_}_H{H}"
                    acc[f"{key}|onestep"].append(_spearman(u1["util_kl"], tgt)[v].cpu())
                    acc[f"{key}|onestep_top"].append(((_topk_mask(u1["util_kl"], k25) & _topk_mask(tgt, k25)).sum(-1).float() / k25)[v].cpu())
                    for nm, s in sc.items():
                        acc[f"{key}|{nm}"].append(_spearman(s, tgt)[v].cpu())
                        acc[f"{key}|{nm}_top"].append(((_topk_mask(s, k25) & _topk_mask(tgt, k25)).sum(-1).float() / k25)[v].cpu())
                    ratio[key].append((tgt.sum(-1) / hs[1].sum(-1).clamp_min(1e-12))[v].cpu())  # vs. the one-step hold harm
    res = dict(n_rec=len(items), stride=a.stride,
               spearman={k: float(torch.cat(v).nanmean()) for k, v in acc.items()},
               harm_ratio_to_onestep={k: float(torch.cat(v).median()) for k, v in ratio.items()},
               n_points=int(sum(len(x) for x in acc[f"k1_H1|onestep"])))
    json.dump(res, open(out_path(a, "multistep2"), "w"), indent=1)
    log("multistep", json.dumps(res["spearman"])[:600])


# ----------------------------------------------------------------------------- R1.7: long streams + adaptive refresh
def long_streams(items, min_s=60.0, max_s=120.0):
    """Concatenate the recordings of each patient (sorted by name) into one continuous stream;
    events are shifted by the running offset. Streams shorter than min_s are dropped."""
    by = collections.defaultdict(list)
    for it in items:
        by[it["name"].split("_")[0]].append(it)
    out = []
    for pid, L in sorted(by.items()):
        L = sorted(L, key=lambda it: it["name"])
        wav, evs, off = [], [], 0.0
        for it in L:
            n = len(it["wav"]) // CHUNK * CHUNK
            if off + n / SR > max_s:
                break
            wav.append(it["wav"][:n])
            evs += [(o + off, f + off, c) for o, f, c in it["events"] if f + off <= off + n / SR]
            off += n / SR
        if off >= min_s:
            out.append(dict(name=f"{pid}_long", pid=pid, wav=np.concatenate(wav), events=sorted(evs)))
    return prep_items(out)


@torch.no_grad()
def cmd_longstream(a):
    if done(a, "longstream"):
        return
    summ, args, dev, splits, fe, model, ctrls, R = load_run(a, "C")
    res = {}
    for split in ("test2024", "test2025"):
        items = long_streams(splits[split], a.min_s, a.max_s)
        if not items:
            continue
        dec = tuple(summ["grid_C"]["full"]["decoding"])
        sp = {"long": items}
        full, _, fprobs = evaluate(model, fe, sp, dev, decoding=dec)
        ref = dict(probs=fprobs, decoding=dec)
        r = dict(n_streams=len(items), hours=sum(len(it["wav"]) for it in items) / SR / 3600,
                 mean_len_s=float(np.mean([len(it["wav"]) / SR for it in items])), full_F=full["long"]["F"])
        for b in (0.125, 0.25, 0.5):
            for rule in ("none", "25", "50", "100", "200", "adaptive"):
                if rule == "none":
                    S.REFRESH_AGE, refresh = None, False
                elif rule == "adaptive":                      # refresh share <= 1/(A b) of the budget, A = 12.5 / b
                    S.REFRESH_AGE, refresh = int(math.ceil(12.5 / b)), True
                else:
                    S.REFRESH_AGE, refresh = int(rule), True
                mk = lambda b=b, refresh=refresh: make_policy("util", model, R, b, ctrls, refresh=refresh)
                s, _, probs = evaluate(model, fe, sp, dev, mk, b, decoding=dec, ref=ref)
                # agreement with the full model by position in the stream (first 15 s vs after 30 s)
                early, late = [0, 0], [0, 0]
                for p, q in zip(probs["long"], fprobs["long"]):
                    ag = p.argmax(1) == q.argmax(1)
                    early[0] += int(ag[:1500].sum()); early[1] += min(len(ag), 1500)
                    late[0] += int(ag[3000:].sum()); late[1] += max(len(ag) - 3000, 0)
                st = s["long"]
                r[f"util@{b}|{rule}"] = dict(fidF=st["fidF"], fid_advR=st["fid_advR"], fid_agree=st["fid_agree"],
                                             fid_kl=st["fid_kl"], F=st["F"], forced_frac=st["forced_frac"],
                                             upd_frac=st["upd_frac"], A=S.REFRESH_AGE if refresh else None,
                                             agree_first15s=early[0] / max(early[1], 1), agree_after30s=late[0] / max(late[1], 1))
                log(f"[long {split}] util@{b} {rule}: fidF {st['fidF']:.4f} agree {st['fid_agree']:.4f} forced {st['forced_frac']:.3f}")
        S.REFRESH_AGE = None
        res[split] = r
    json.dump(res, open(out_path(a, "longstream"), "w"), indent=1)


# ----------------------------------------------------------------------------- R1.8: Lipschitz ratios + bound check
def rownorm_max(D):
    return D.norm(dim=-1).amax(-1)


@torch.no_grad()
def cmd_lipschitz(a):
    if done(a, "lipschitz"):
        return
    summ, args, dev, splits, fe, model, ctrls, R = load_run(a, "C")
    items = splits["val"][: a.n_rec]
    out = {}
    for b in (0.125, 0.25, 0.5):
        gam, e_list, bound_ok, kl_e, Lr, local_jac = [], [], [], [], [], []
        acc_bound, acc_e = [], []
        for idx, W in eval_batches(items, 16):
            feats = fe(W.to(dev))
            x = model.enc(feats)
            B, Sn, N, _ = x.shape
            Tst = torch.tensor([items[i]["T"] // STEP for i in idx], device=dev)
            pol = make_policy("util", model, R, b, ctrls)
            zf = model.init_state(B, dev); zs = zf.clone()
            age = torch.zeros(B, N, device=dev); x_prev = torch.zeros_like(x[:, 0])
            e_prev = torch.zeros(B, device=dev); bnd = torch.zeros(B, device=dev)
            gmax = 0.0
            for t in range(Sn):
                xt = x[:, t]
                valid = t < Tst
                ctx = dict(model=model, x=xt, x_prev=x_prev, z_prev=zs, age=age, budget=b, spec=None, spec_prev=None, t=t)
                U = pol(t, ctx)
                phi_f = model.cell(xt, zf)                 # Phi(Z_full_{t-1})
                phi_s = model.cell(xt, zs)                 # Phi(Z_sel_{t-1})  (full update from the selective state)
                sel = model.cell(xt, zs, U)                # selective update
                d_prev = rownorm_max(zf - zs)
                g_t = rownorm_max(phi_f - phi_s) / d_prev.clamp_min(1e-9)
                skip_def = ((phi_s - zs).norm(dim=-1) * (~U)).amax(-1)
                bnd_def = ((phi_s - sel).norm(dim=-1) * U).amax(-1)
                e_t = rownorm_max(phi_f - sel)
                ok = d_prev > 1e-6
                m = valid & ok
                if bool(m.any()):
                    gam.append(g_t[m].cpu())
                    bound_ok.append((e_t <= g_t * d_prev + torch.maximum(skip_def, bnd_def) + 1e-5)[m].cpu())
                # accumulated bound with a single constant gamma (the running max is a conservative stand-in)
                if bool(m.any()):
                    gmax = max(gmax, float(g_t[m].max()))
                bnd = g_t.clamp_max(10.0) * bnd + torch.maximum(skip_def, bnd_def)
                acc_bound.append(bnd[valid].cpu()); acc_e.append(e_t[valid].cpu())
                lf = model.readout(phi_f, xt); ls_ = model.readout(sel, xt)
                lpf = F.log_softmax(lf.view(B, STEP, -1), -1); lps = F.log_softmax(ls_.view(B, STEP, -1), -1)
                kl = (lpf.exp() * (lpf - lps)).sum(-1).mean(-1)
                m2 = valid & (e_t > 1e-6)
                if bool(m2.any()):
                    kl_e.append(torch.stack([e_t[m2], kl[m2]], -1).cpu())
                    Lr.append(((lf - ls_).abs().amax(-1) / e_t.clamp_min(1e-9))[m2].cpu())
                zf, zs = phi_f, sel
                age = torch.where(U, torch.zeros_like(age), age + 1)
                x_prev = xt
            # local Jacobian spectral norm (l2) at a few full-update states, by power iteration
            if b == 0.25 and len(local_jac) < 40:
                for t in range(20, min(Sn, 200), 60):
                    z0 = model.init_state(B, dev)
                    for tt in range(t):
                        z0 = model.cell(x[:, tt], z0)
                    v = torch.randn_like(z0); v = v / v.flatten(1).norm(dim=-1)[:, None, None]
                    for _ in range(8):
                        with torch.enable_grad():
                            zz = z0.clone().requires_grad_(True)
                            zout = model.cell(x[:, t], zz)
                            jv = torch.autograd.functional.jvp(lambda z: model.cell(x[:, t], z), (zz,), (v,))[1]
                            jtjv = torch.autograd.grad(zout, zz, grad_outputs=jv)[0]
                        nrm = jtjv.flatten(1).norm(dim=-1)
                        v = jtjv / nrm[:, None, None].clamp_min(1e-12)
                    local_jac.append(nrm.sqrt().cpu())
        g = torch.cat(gam)
        ke = torch.cat(kl_e)
        r = dict(gamma_median=float(g.median()), gamma_p90=float(g.quantile(0.9)), gamma_p99=float(g.quantile(0.99)),
                 gamma_max=float(g.max()), frac_gamma_lt1=float((g < 1).float().mean()),
                 one_step_bound_holds=float(torch.cat(bound_ok).float().mean()),
                 e_median=float(torch.cat(acc_e).median()), accumulated_bound_median=float(torch.cat(acc_bound).median()),
                 accumulated_bound_holds=float((torch.cat(acc_e) <= torch.cat(acc_bound) + 1e-5).float().mean()),
                 readout_ratio_median=float(torch.cat(Lr).median()), readout_ratio_p99=float(torch.cat(Lr).quantile(0.99)),
                 spearman_state_err_vs_kl=float(_spearman(ke[:, 0][None], ke[:, 1][None])[0]))
        if local_jac:
            lj = torch.cat(local_jac)
            r.update(jac_l2_median=float(lj.median()), jac_l2_max=float(lj.max()))
        out[str(b)] = r
        log(f"[lip] b={b}: " + json.dumps(r))
    json.dump(out, open(out_path(a, "lipschitz"), "w"), indent=1)


# ----------------------------------------------------------------------------- R1.1: boundary-tolerant metrics
def tolerant_counts(ref, est, collar, offset=True, overlap=False):
    """Class-aware hit count under a given onset collar (offset collar max(collar, 20%) if offset),
    or under 'any temporal overlap' matching. Returns [TP, Nref, Nest]."""
    Mx = np.zeros((len(ref), len(est)), dtype=bool)
    for i, (ro, rf, rc) in enumerate(ref):
        oc = max(collar, 0.2 * (rf - ro))
        for j, (eo, ef, ec) in enumerate(est):
            if rc != ec:
                continue
            if overlap:
                Mx[i, j] = min(rf, ef) > max(ro, eo)
            else:
                Mx[i, j] = abs(ro - eo) <= collar and (not offset or abs(rf - ef) <= oc)
    r, _ = M._max_match(Mx)
    return np.array([len(r), len(ref), len(est)], dtype=np.int64)


TOL = [("ev_c0.2", 0.2, True, False), ("ev_c0.5", 0.5, True, False), ("ev_c1.0", 1.0, True, False),
       ("onset_c0.5", 0.5, False, False), ("onset_c1.0", 1.0, False, False), ("overlap", 0.0, True, True)]


def tolerant_metrics(refs, ests, durs):
    out = {}
    for nm, c, off, ov in TOL:
        cnt = np.stack([tolerant_counts(r, e, c, off, ov) for r, e in zip(refs, ests)]).sum(0)
        out[nm] = float(2 * cnt[0] / max(cnt[1] + cnt[2], 1))
    for seg in (1.0, 2.0):
        sg = np.stack([M.segment_counts(r, e, d, seg) for r, e, d in zip(refs, ests, durs)])
        s = M.summarize_segments(sg)
        out[f"seg{seg:g}_F"] = s["segF"]; out[f"seg{seg:g}_advF"] = s["seg_advF"]
    return out


@torch.no_grad()
def cmd_collars(a):
    if done(a, "collars2"):
        return
    summ, args, dev, splits, fe, model, ctrls, R = load_run(a, "C")
    ck = os.path.join(a.run, "ctrl_e2e.pt")
    if os.path.exists(ck):
        gate = Controller(model.dx, model.d).to(dev)
        gate.load_state_dict(torch.load(ck, map_location=dev)); gate.eval()
        ctrls = nn.ModuleDict(dict(ctrls.items())); ctrls["e2e"] = gate
    full, _, fprobs = evaluate(model, fe, splits, dev)
    w, mf = full["decoding"]
    res = {}
    configs = [("full", 1.0)] + [(p, 0.25) for p in ("util", "change", "periodic", "random")] + [("util", 0.125), ("util", 0.5)]
    if "e2e" in ctrls:
        configs.append(("e2e", 0.25))
    for name, b in configs:
        if name == "full":
            probs, dec = fprobs, (w, mf)
        else:
            mk = lambda name=name, b=b: make_policy(name, model, R, b, ctrls, gen=torch.Generator(device=dev).manual_seed(args["seed"]))
            s, _, probs = evaluate(model, fe, splits, dev, mk, b)
            dec = tuple(s["decoding"])
        r = {}
        for k in ("test2024", "test2025"):
            items = splits[k]
            est = [M.decode(p, *dec) for p in probs[k]]
            refs = [it["events"] for it in items]
            durs = [len(it["wav"]) / SR for it in items]
            r[k] = tolerant_metrics(refs, est, durs)
            # tolerant preservation: the full model's own decoded events as reference
            fref = [M.decode(p, w, mf) for p in fprobs[k]]
            est_pres = [M.decode(p, w, mf) for p in probs[k]]   # same decoding as F_pres (train.evaluate)
            r[k].update({f"pres_{kk}": v for kk, v in tolerant_metrics(fref, est_pres, durs).items()})
        res[f"{name}@{b}"] = r
        log(f"[collars] {name}@{b}: " + json.dumps({k: round(v, 3) for k, v in r['test2024'].items()}))
    json.dump(res, open(out_path(a, "collars2"), "w"), indent=1)


# ----------------------------------------------------------------------------- R1.2: overhead scaling
def cmd_scaling(a):
    """Batch-1 NumPy timing of the controller vs. the graph update as the cell grows (random weights;
    arithmetic cost does not depend on the weight values). One pinned core."""
    from .benchutil import pin_one_core
    from .np_runtime import NPRuntime
    from .select import greedy_bitmask
    pin_one_core()
    res = {}
    rng = np.random.default_rng(0)
    for d in (96, 192, 384, 768):
        for L in (2, 4):
            torch.manual_seed(0)
            m = StreamModel("agn", d=d)
            if L != 2:
                from .models import AGNCell
                m.cell = AGNCell(m.N, m.dx, d, L=L)
            sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
            ctrl = Controller(m.dx, d)
            csd = {f"util.{k}": v.detach().numpy() for k, v in ctrl.state_dict().items()}
            stats = dict(mean=np.zeros(256, np.float32), std=np.ones(256, np.float32))
            rt = NPRuntime(sd, csd, stats)
            rt.z = rng.standard_normal((rt.N, d)).astype(np.float32) * 0.3
            x = rng.standard_normal((rt.N, rt.dx)).astype(np.float32)
            timings = collections.defaultdict(list)
            for rep in range(a.reps):
                for b in (1.0, 0.25, 0.125):
                    t0 = time.perf_counter()
                    if b < 1.0:
                        s = rt.controller(x, b)
                        idx = greedy_bitmask(s, rt.Rbits, rt.Rsize, rt.age * 0, int(round(b * rt.N)), refresh=True)
                    else:
                        idx = None
                    t1 = time.perf_counter()
                    rt.graph(x, idx)
                    t2 = time.perf_counter()
                    timings[f"sel@{b}"].append(t1 - t0); timings[f"graph@{b}"].append(t2 - t1)
            med = {k: float(np.median(v) * 1e3) for k, v in timings.items()}
            full = med["graph@1.0"]
            r = dict(med_ms=med, cell_params=int(sum(v.size for k, v in sd.items() if k.startswith("cell."))),
                     ctrl_params=int(sum(v.size for v in csd.values())))
            for b in (0.25, 0.125):
                r[f"graph_speedup@{b}"] = full / (med[f"sel@{b}"] + med[f"graph@{b}"])
                r[f"ctrl_share_of_saving@{b}"] = med[f"sel@{b}"] / max(full - med[f"graph@{b}"], 1e-9)
            res[f"d{d}_L{L}"] = r
            log(f"[scaling] d={d} L={L}: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != 'med_ms'}))
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--run", default="")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/scaling.json")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--crop_s", type=float, default=8.0)
    ap.add_argument("--e2e_epochs", type=int, default=6)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--n_rec", type=int, default=150)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--min_s", type=float, default=60.0)
    ap.add_argument("--max_s", type=float, default=120.0)
    a = ap.parse_args()
    dict(e2e=cmd_e2e, e2ediag=cmd_e2ediag, multistep=cmd_multistep, longstream=cmd_longstream, lipschitz=cmd_lipschitz,
         collars=cmd_collars, scaling=cmd_scaling)[a.cmd](a)


if __name__ == "__main__":
    main()
