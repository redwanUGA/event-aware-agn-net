"""Train + evaluate one configuration (idempotent: skips if results exist).

    python -m eaagn.train --arch agn --seed 0 --data data --out results/agn_s0

Graph arch 'agn' with --selective 1 runs the full pipeline:
  A  full-update training
  B  controller learning on frozen weights with selective rollouts (4 heads:
     skip-harm utility, hold-only utility, label-free utility, representation change)
  eval_A  every policy x budget on the same weights
  C  policy-agnostic selective fine-tuning with distillation from A
  B' controller refresh on C weights
  eval_C
"""
from __future__ import annotations

import argparse
import copy
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
from .audio import HOP, SR, STEP, CausalLogMel
from .models import StreamModel, count_params
from .select import CONTROLLER_HEADS, Controller, make_policy, policy_scores, region_masks, utility_targets

CHUNK = HOP * STEP  # samples per graph step (320 = 40 ms)
REEVAL_VERSION = 3
POLICIES = ("random", "periodic", "energy", "acoustic", "change", "util_kl", "util_hold", "util", "oracle")
BUDGETS = (0.125, 0.25, 0.5)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- data
def frame_labels(it):
    T = len(it["wav"]) // HOP
    y = np.zeros(T, dtype=np.int64)
    for on, off, c in it["events"]:
        a, b = int(round(on / 0.01)), int(round(off / 0.01))
        y[max(a, 0):min(b, T)] = c
    return y


def prep_items(items):
    for it in items:
        it["y"] = frame_labels(it)
        it["T"] = len(it["wav"]) // HOP
    return items


def train_batches(items, bs, crop_s, rng):
    crop = int(crop_s * SR) // CHUNK * CHUNK
    order = rng.permutation(len(items))
    for k in range(0, len(order) - bs + 1, bs):
        W = np.zeros((bs, crop), dtype=np.float32)
        Y = np.full((bs, crop // HOP), -1, dtype=np.int64)
        for b, i in enumerate(order[k:k + bs]):
            it = items[i]
            n = len(it["wav"])
            s = rng.integers(0, max(n - crop, 0) + 1)
            s = s // HOP * HOP
            seg = it["wav"][s:s + crop].astype(np.float32) / 32768.0
            W[b, :len(seg)] = seg
            ys = it["y"][s // HOP: s // HOP + crop // HOP]
            Y[b, :len(ys)] = ys
        yield torch.from_numpy(W), torch.from_numpy(Y)


def eval_batches(items, bs):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["wav"]))
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        L = max(len(items[i]["wav"]) for i in idx)
        L = int(math.ceil(L / CHUNK) * CHUNK)
        W = np.zeros((len(idx), L), dtype=np.float32)
        for b, i in enumerate(idx):
            W[b, :len(items[i]["wav"])] = items[i]["wav"].astype(np.float32) / 32768.0
        yield idx, torch.from_numpy(W)


def augment(feats, wav_gain_db, rng):
    B = feats.shape[0]
    # gain of g dB shifts log-power by g*ln(10)/10; divided by ~2 to approximate the per-bin std normalisation
    g = torch.from_numpy(rng.uniform(-wav_gain_db, wav_gain_db, size=(B, 1, 1, 1)).astype(np.float32)).to(feats.device)
    feats = feats + g * (math.log(10) / 10) / 2.0
    for b in range(B):
        if rng.random() < 0.5:
            w = int(rng.integers(4, 17)); f0 = int(rng.integers(0, feats.shape[2] - w))
            feats[b, :, f0:f0 + w] = 0.0
    return feats


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def infer(model, fe, items, dev, policy_fn=None, budget=1.0, bs=64):
    model.eval()
    probs = [None] * len(items)
    upd, forced, vns = 0.0, 0.0, 0.0
    for idx, W in eval_batches(items, bs):
        feats = fe(W.to(dev))
        pol = policy_fn() if policy_fn is not None else None
        lengths = torch.tensor([int(math.ceil(items[i]["T"] / STEP)) for i in idx])
        logits, st = model(feats, pol, budget, lengths=lengths)
        p = torch.softmax(logits.float(), -1).cpu().numpy()
        for b, i in enumerate(idx):
            probs[i] = p[b, : items[i]["T"]].astype(np.float32)
        if "valid_node_steps" in st:
            upd += st["upd_nodes"]; forced += st["forced_nodes"]; vns += st["valid_node_steps"]
    # realised update fraction over valid (non-padded) node-steps; forced = refresh share
    return probs, dict(upd_frac=upd / vns if vns else 1.0, forced_frac=forced / vns if vns else 0.0)


def evaluate(model, fe, splits, dev, policy_fn=None, budget=1.0, decoding=None, ref=None):
    """Infer on val + tests; tune decoding on val (unless given); return summary + per-rec counts.

    ref: optional {"probs": {split: [probs]}, "decoding": (w, m)} from the full-update model on the
    same weights. Adds fidelity metrics that use the full model's own outputs as reference:
    event preservation (event-F of selective vs full-model decoded events), adventitious-event
    preservation recall, frame agreement and mean frame KL(full || selective)."""
    out, per_rec = {}, {}
    probs = {}
    stats = {}
    for k, items in splits.items():
        probs[k], stats[k] = infer(model, fe, items, dev, policy_fn, budget)
    if decoding is None:
        decoding = M.tune_decoding(probs["val"], [it["events"] for it in splits["val"]])
    w, mf = decoding
    for k, items in splits.items():
        dec = [M.decode(p, w, mf) for p in probs[k]]
        ev = np.stack([M.event_counts(it["events"], d) for d, it in zip(dec, items)])
        on = np.stack([M.onset_counts(it["events"], d) for d, it in zip(dec, items)])
        sg = np.stack([M.segment_counts(it["events"], d, len(it["wav"]) / SR) for d, it in zip(dec, items)])
        segs = [M.segment_preds(p, it["events"]) for p, it in zip(probs[k], items)]
        pairs = [pp for sq in segs for pp in sq]
        mf1, acc = M.frame_macro_f1(probs[k], [it["y"] for it in items])
        s = M.summarize_events(ev)
        s.update(M.summarize_segments(sg))
        s.update(onsetF=M.summarize_onsets(on), frame_macroF1=mf1, frame_acc=acc, **stats[k])
        s.update({f"seg_{a}": b for a, b in M.sprsound_scores(pairs).items()})
        rec = dict(ev=ev.astype(np.int32), on=on.astype(np.int32), sg=sg.astype(np.int32),
                   seg=np.array(pairs, dtype=np.int8).reshape(-1, 2),
                   seg_rec=np.repeat(np.arange(len(items)), [len(sq) for sq in segs]).astype(np.int32))
        if ref is not None:
            rw, rm = ref["decoding"]
            rdec = [M.decode(p, rw, rm) for p in ref["probs"][k]]
            fev = np.stack([M.event_counts(r, M.decode(p, rw, rm)) for r, p in zip(rdec, probs[k])])
            fs = M.summarize_events(fev)
            agree = np.array([[(p.argmax(1) == q.argmax(1)).sum(), len(p)] for p, q in zip(probs[k], ref["probs"][k])])
            kl = np.array([float((q * (np.log(q + 1e-8) - np.log(p + 1e-8))).sum(1).sum()) for p, q in zip(probs[k], ref["probs"][k])])
            s.update(fidF=fs["F"], fid_advR=fs.get("advR", 0.0), fid_advF=fs.get("advF", 0.0),
                     fid_agree=float(agree[:, 0].sum() / agree[:, 1].sum()), fid_kl=float(kl.sum() / agree[:, 1].sum()))
            rec.update(fid=fev.astype(np.int32), agree=agree.astype(np.int64), kl=kl.astype(np.float32))
        out[k] = s
        per_rec[k] = rec
    out["decoding"] = [int(w), int(mf)]
    return out, per_rec, probs


# ----------------------------------------------------------------------------- stages
def class_weights(items, C=8):
    cnt = np.bincount(np.concatenate([it["y"] for it in items]), minlength=C).astype(float)
    w = np.where(cnt > 0, (np.maximum(cnt, 1) / cnt.sum()) ** -0.5, 0.0)
    w = w / w[cnt > 0].mean()
    return torch.tensor(np.minimum(w, 10.0), dtype=torch.float32)


def guarded_step(model, opt, loss, state):
    """Skip updates with a non-finite loss or gradient norm; roll back to the last good
    weights if parameters ever become non-finite. Returns True if the step was applied."""
    if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True); state["skipped"] += 1; return False
    gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gn):
        opt.zero_grad(set_to_none=True); state["skipped"] += 1; return False
    opt.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        model.load_state_dict(state["good"]); state["rollbacks"] += 1
        opt.state.clear()  # moments may hold non-finite values
        return False
    return True


def stage_a(model, fe, tr, va, dev, a, rng, cw):
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    steps = a.epochs_a * (len(tr) // a.bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.1)
    best, best_sd, hist = -1, None, []
    gs = dict(skipped=0, rollbacks=0, good=copy.deepcopy(model.state_dict()))
    lr_scale, restarts = 1.0, 0
    for ep in range(a.epochs_a):
        model.train(); tl, n = 0.0, 0
        gs["good"] = copy.deepcopy(model.state_dict())
        sk0, nb = gs["skipped"], 0
        for W, Y in train_batches(tr, a.bs, a.crop_s, rng):
            feats = augment(fe(W.to(dev)), 6.0, rng)
            logits, _ = model(feats)
            loss = F.cross_entropy(logits.reshape(-1, model.C), Y.to(dev).reshape(-1), weight=cw, ignore_index=-1)
            opt.zero_grad(); loss.backward()
            if guarded_step(model, opt, loss, gs):
                tl += loss.item(); n += 1
            sch.step(); nb += 1
            if lr_scale != 1.0:
                for g in opt.param_groups:
                    g["lr"] *= lr_scale
        # divergence rule (revision): if >10% of an epoch's batches had a non-finite loss, restore the
        # epoch's starting weights and continue with half the learning rate. Never triggers on stable runs.
        if gs["skipped"] - sk0 > 0.1 * max(nb, 1) and restarts < 3:
            model.load_state_dict(gs["good"]); opt.state.clear()
            lr_scale *= 0.5; restarts += 1
            log(f"[A] ep{ep}: {gs['skipped'] - sk0}/{nb} non-finite batches -> restored epoch start, lr scale {lr_scale}")
            hist.append(dict(epoch=ep, restart=restarts, lr_scale=lr_scale))
        if ep % 2 == 1 or ep == a.epochs_a - 1:
            probs, _ = infer(model, fe, va, dev)
            ev = np.stack([M.event_counts(it["events"], M.decode(p, 5, 10)) for p, it in zip(probs, va)])
            sm = M.summarize_events(ev)
            f = M.selection_score(sm)
            mf1, _ = M.frame_macro_f1(probs, [it["y"] for it in va])
            hist.append(dict(epoch=ep, loss=tl / max(n, 1), val_F=sm["F"], val_advF=sm["advF"], val_sel=f, val_frameF1=mf1,
                             skipped=gs["skipped"], rollbacks=gs["rollbacks"]))
            log(f"[A] ep{ep} loss {tl / max(n, 1):.4f} valF {sm['F']:.4f} advF {sm['advF']:.4f} frameF1 {mf1:.4f} "
                f"skipped {gs['skipped']} rollbacks {gs['rollbacks']}")
            if f > best:
                best, best_sd = f, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_sd)
    return hist


def stage_b(model, fe, tr, dev, a, rng, ctrls, epochs, R, warm=False):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(ctrls.parameters(), lr=2e-3)
    scale = dict(util=0.01, util_hold=0.01, util_kl=0.01, change=0.1)
    hist = []
    for ep in range(epochs):
        tl, n = 0.0, 0
        for W, Y in train_batches(tr, a.bs, a.crop_s, rng):
            Y = Y.to(dev)
            budget = float(rng.choice([0.125, 0.25, 0.5, 0.75]))
            use_ctrl = (ep > 0 or warm) and rng.random() < 0.5
            roll = str(rng.choice(CONTROLLER_HEADS)) if use_ctrl else "random"
            pol = make_policy(roll, model, R, budget, ctrls)
            acc = []

            def collect(t, ctx):
                y = Y[:, t * STEP:(t + 1) * STEP]
                tgt = utility_targets(model, ctx, R, y=y)
                with torch.enable_grad():
                    l = 0.0
                    for h in CONTROLLER_HEADS:
                        pred = ctrls[h](ctx["x"], ctx["x_prev"], ctx["z_prev"], ctx["age"], ctx["budget"])
                        l = l + F.huber_loss(torch.log1p(pred), torch.log1p(tgt[h] / scale[h]))
                    acc.append(l)

            with torch.no_grad():
                model(fe(W.to(dev)), pol, budget, collect=collect)
            loss = torch.stack(acc).mean()
            opt.zero_grad(); loss.backward()
            if torch.isfinite(loss):
                nn.utils.clip_grad_norm_(ctrls.parameters(), 5.0); opt.step()
                tl += loss.item(); n += 1
        hist.append(dict(epoch=ep, loss=tl / max(n, 1)))
        log(f"[B] ep{ep} ctrl loss {tl / max(n, 1):.4f}")
    for p in model.parameters():
        p.requires_grad_(True)
    for p in ctrls.parameters():
        p.requires_grad_(False)
    return hist


def stage_c(model, teacher, fe, tr, va, dev, a, rng, cw, ctrls, R):
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr_c, weight_decay=1e-2)
    teacher.eval()
    hist = []
    gs = dict(skipped=0, rollbacks=0, good=copy.deepcopy(model.state_dict()))
    for ep in range(a.epochs_c):
        model.train(); tl, n = 0.0, 0
        gs["good"] = copy.deepcopy(model.state_dict())
        for W, Y in train_batches(tr, a.bs, a.crop_s, rng):
            feats = augment(fe(W.to(dev)), 6.0, rng)
            name = str(rng.choice([p for p in POLICIES if p != "oracle"]))
            budget = float(rng.choice([0.125, 0.25, 0.5, 0.75, 1.0]))
            pol = make_policy(name, model, R, budget, ctrls)
            logits, _ = model(feats, pol, budget)
            with torch.no_grad():
                tl_, _ = teacher(feats)
            Yd = Y.to(dev).reshape(-1)
            ce = F.cross_entropy(logits.reshape(-1, model.C), Yd, weight=cw, ignore_index=-1)
            m = (Yd >= 0).float()
            kd = (F.kl_div(F.log_softmax(logits.reshape(-1, model.C), -1), F.log_softmax(tl_.reshape(-1, model.C), -1),
                           log_target=True, reduction="none").sum(-1) * m).sum() / m.sum().clamp_min(1)
            loss = ce + a.distill * kd
            opt.zero_grad(); loss.backward()
            if guarded_step(model, opt, loss, gs):
                tl += loss.item(); n += 1
        hist.append(dict(epoch=ep, loss=tl / max(n, 1), skipped=gs["skipped"], rollbacks=gs["rollbacks"]))
        log(f"[C] ep{ep} loss {tl / max(n, 1):.4f}")
    return hist


def _rank(v):
    """Average (fractional) ranks along the last axis, so ties are handled correctly."""
    less = (v[..., None, :] < v[..., :, None]).sum(-1).float()
    eq = (v[..., None, :] == v[..., :, None]).sum(-1).float()
    return less + (eq - 1) / 2


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(-1, keepdim=True); rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-9)


def _topk_mask(v, k):
    m = torch.zeros_like(v, dtype=torch.bool)
    return m.scatter_(-1, v.topk(k, -1).indices, True)


@torch.no_grad()
def ranking_diagnostics(model, fe, items, dev, ctrls, R, n_rec=200, budget=0.25):
    """Rank quality of each scorer against independently measured (label-free) one-step
    skip-harm utility, under random-policy rollouts; plus locality statistics."""
    model.eval()
    sub = items[:n_rec]
    names = ["random", "periodic", "energy", "acoustic"] + list(CONTROLLER_HEADS)
    rho = {k: [] for k in names}; topk = {k: [] for k in names}
    conc_change, conc_util, per_rec_conc = [], [], []
    conc_util_sqrt, conc_change_sq = [], []
    Rf = R.float()
    k = int(round(budget * R.shape[0]))
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    for idx, W in eval_batches(sub, 32):
        T_steps = torch.tensor([sub[i]["T"] // STEP for i in idx], device=dev)
        rec_cu = torch.zeros(len(idx), device=dev); rec_cc = torch.zeros(len(idx), device=dev)
        rec_n = torch.zeros(len(idx), device=dev)

        def collect(t, ctx):
            if t < 5:
                return
            tg = utility_targets(model, ctx, R)
            u, ch = tg["util"], tg["change"]
            # each policy's selection key is its score divided by region size (greedy ratio)
            sc = {nm: policy_scores(nm, model, R, ctx, budget, ctrls, gen) / Rf.sum(1) for nm in names}
            u = u / Rf.sum(1)  # rank by per-node utility, the quantity the greedy ratio targets
            valid = t < T_steps
            if not bool(valid.any()):
                return
            tm = _topk_mask(u, k)
            for nm in names:
                rho[nm].append(_spearman(sc[nm], u)[valid].cpu())
                topk[nm].append(((tm & _topk_mask(sc[nm], k)).sum(-1).float() / k)[valid].cpu())
            share = lambda v: v.topk(k, -1).values.sum(-1) / v.sum(-1).clamp_min(1e-12)
            cu = share(u)
            cc = share(ch)
            # scale-matched variants: KL utility is ~quadratic in the output perturbation, the
            # change norm is linear, so also compare sqrt(utility) with change and utility with change^2
            conc_util_sqrt.append(share(u.clamp_min(0).sqrt())[valid].cpu())
            conc_change_sq.append(share(ch ** 2)[valid].cpu())
            conc_util.append(cu[valid].cpu()); conc_change.append(cc[valid].cpu())
            rec_cu.add_(cu * valid); rec_cc.add_(cc * valid); rec_n.add_(valid.float())

        pol = make_policy("random", model, R, budget, ctrls)
        model(fe(W.to(dev)), pol, budget, collect=collect)
        for b, i in enumerate(idx):
            n = max(float(rec_n[b]), 1.0)
            per_rec_conc.append((sub[i]["name"], float(rec_cu[b]) / n, float(rec_cc[b]) / n))
    cat = lambda v: torch.cat(v).float()
    return dict(spearman={k: float(cat(v).nanmean()) for k, v in rho.items()},
                topk_overlap={k: float(cat(v).mean()) for k, v in topk.items()},
                conc_util_top25=float(cat(conc_util).mean()), conc_change_top25=float(cat(conc_change).mean()),
                conc_util_sqrt_top25=float(cat(conc_util_sqrt).mean()), conc_change_sq_top25=float(cat(conc_change_sq).mean()),
                per_rec_conc=per_rec_conc)


def run_policy_grid(model, fe, splits, dev, ctrls, R, tag, outdir, a, only=None, res=None):
    """Evaluate every policy x budget on fixed weights. With `only`, re-evaluate just those
    policies and merge them into an existing grid `res` (the full-update reference is recomputed)."""
    res = {} if res is None else res
    full, pr, full_probs = evaluate(model, fe, splits, dev)
    ref = dict(probs=full_probs, decoding=full["decoding"])
    res["full"] = full
    np.savez_compressed(os.path.join(outdir, f"rec_{tag}_full.npz"), **{f"{k}_{m}": v for k, d in pr.items() for m, v in d.items()})
    log(f"[{tag}] full: " + json.dumps({k: [round(v['F'], 4), round(v['segF'], 4)] for k, v in full.items() if isinstance(v, dict)}))
    budgets, policies = ((0.25,), ("random", "util", "oracle")) if a.quick else (BUDGETS, POLICIES)
    configs = [("none", 0.0)] + [(p, b) for b in budgets for p in policies] + [("util_norefresh", b) for b in budgets]
    if only is not None:
        configs = [(p, b) for b in budgets for p in only]
    for name, b in configs:
        pname = name.replace("_norefresh", "")
        refresh = "norefresh" not in name
        mk = lambda pname=pname, b=b, refresh=refresh: make_policy(pname, model, R, b, ctrls, refresh=refresh,
                                                                  gen=torch.Generator(device=dev).manual_seed(a.seed))
        t0 = time.time()
        r, pr, _ = evaluate(model, fe, splits, dev, mk, b, ref=ref)
        key = f"{name}@{b}"
        res[key] = r
        np.savez_compressed(os.path.join(outdir, f"rec_{tag}_{key.replace('@', '_')}.npz"),
                            **{f"{k}_{m}": v for k, d in pr.items() for m, v in d.items()})
        log(f"[{tag}] {key}: t24 F {r['test2024']['F']:.4f} segF {r['test2024']['segF']:.4f} fidF {r['test2024']['fidF']:.4f} "
            f"seg_advF {r['test2024']['seg_advF']:.4f} fid_advR {r['test2024']['fid_advR']:.4f} agree {r['test2024']['fid_agree']:.4f} | t25 segF {r.get('test2025', r['test2024'])['segF']:.4f} "
            f"fidF {r.get('test2025', r['test2024'])['fidF']:.4f} "
            f"upd {r['test2024']['upd_frac']:.3f} ({time.time() - t0:.0f}s)")
        json.dump(res, open(os.path.join(outdir, f"grid_{tag}.json"), "w"), indent=1)
    return res


REGRID_VERSION = 1


def regrid(a, final):
    """Evaluation-only: re-run selected policies (and the ranking diagnostics) on the saved
    stage-A and stage-C weights/controllers, merging into summary.json. No training."""
    summary = json.load(open(final))
    if summary.get("regrid_version") == REGRID_VERSION:
        log("already regridded", final); return
    only = [p for p in a.regrid.split(",") if p]
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("regrid", only, "device", dev)
    splits = {}
    for k in ("val", "test2024", "test2025"):
        fn = f"{k}_pg.pkl" if (a.split == "pg" and k == "val") else f"{k}.pkl"
        splits[k] = prep_items(pickle.load(open(os.path.join(a.data, fn), "rb")))
    st = np.load(os.path.join(a.data, "stats.npz"))
    fe = CausalLogMel(st["mean"], st["std"]).to(dev)
    model = StreamModel(a.arch, fresh=bool(a.fresh)).to(dev)
    ctrls = nn.ModuleDict({h: Controller(model.dx, model.d) for h in CONTROLLER_HEADS}).to(dev)
    R = region_masks(model.cell.adj)
    for tag, ck_m, ck_c in (("A", "model_A.pt", "ctrl_A.pt"), ("C", "model_C.pt", "ctrl_C.pt")):
        model.load_state_dict(torch.load(os.path.join(a.out, ck_m), map_location=dev)); model.eval()
        ctrls.load_state_dict(torch.load(os.path.join(a.out, ck_c), map_location=dev)); ctrls.eval()
        summary[f"diag_{tag}"] = ranking_diagnostics(model, fe, splits["val"], dev, ctrls, R)
        log(f"[diag {tag}] " + json.dumps({k: v for k, v in summary[f'diag_{tag}'].items() if k != 'per_rec_conc'}))
        summary[f"grid_{tag}"] = run_policy_grid(model, fe, splits, dev, ctrls, R, tag, a.out, a,
                                                 only=only, res=summary[f"grid_{tag}"])
    summary["regrid_version"] = REGRID_VERSION
    summary["regrid_policies"] = only
    json.dump(summary, open(final, "w"), indent=1)
    log("regrid done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="agn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fresh", type=int, default=1)
    ap.add_argument("--selective", type=int, default=0)
    ap.add_argument("--epochs_a", type=int, default=40)
    ap.add_argument("--epochs_b", type=int, default=6)
    ap.add_argument("--epochs_c", type=int, default=8)
    ap.add_argument("--epochs_b2", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--crop_s", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr_c", type=float, default=3e-4)
    ap.add_argument("--distill", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap items per split")
    ap.add_argument("--quick", type=int, default=0, help="smoke test: tiny policy grid")
    ap.add_argument("--reeval", type=int, default=0, help="non-selective runs: recompute metrics from model_A.pt")
    ap.add_argument("--split", default="pg", help="pg = patient-disjoint train/val (regroup.py); official = Detection valid")
    ap.add_argument("--regrid", default="", help="comma list of policies to re-evaluate on saved A/C weights")
    a = ap.parse_args()
    final = os.path.join(a.out, "summary.json")
    if a.regrid:
        return regrid(a, final)
    if os.path.exists(final) and not a.reeval:
        log("exists, skipping", final); return
    if a.reeval and os.path.exists(final) and json.load(open(final)).get("reeval_version") == REEVAL_VERSION:
        log("already re-evaluated", final); return
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")

    splits = {}
    for k in ("train", "val", "test2024", "test2025"):
        fn = f"{k}_pg.pkl" if (a.split == "pg" and k in ("train", "val")) else f"{k}.pkl"
        if k == "test2025" and not os.path.exists(os.path.join(a.data, fn)):
            continue  # second dataset (HF_Lung) has a single test set
        items = pickle.load(open(os.path.join(a.data, fn), "rb"))
        if a.limit:
            items = items[: a.limit]
        splits[k] = prep_items(items)
    st = np.load(os.path.join(a.data, "stats.npz"))
    fe = CausalLogMel(st["mean"], st["std"]).to(dev)
    tr = splits.pop("train")
    va = splits["val"]
    cw = class_weights(tr).to(dev)

    model = StreamModel(a.arch, fresh=bool(a.fresh)).to(dev)
    summary = dict(args=vars(a), params=count_params(model))
    if a.reeval and os.path.exists(final):
        summary = json.load(open(final)); summary["reeval_version"] = REEVAL_VERSION
    log(f"arch {a.arch} params {summary['params']}")
    ck_a = os.path.join(a.out, "model_A.pt")
    if os.path.exists(ck_a):
        model.load_state_dict(torch.load(ck_a, map_location=dev))
    else:
        t0 = time.time()
        summary["hist_A"] = stage_a(model, fe, tr, va, dev, a, rng, cw)
        summary["time_A_s"] = time.time() - t0
        torch.save(model.state_dict(), ck_a)

    if not (a.selective and model.is_graph):
        r, pr, _ = evaluate(model, fe, splits, dev)
        summary["full"] = r
        np.savez_compressed(os.path.join(a.out, "rec_A_full.npz"), **{f"{k}_{m}": v for k, d in pr.items() for m, v in d.items()})
        log("full: " + json.dumps({k: round(v['F'], 4) for k, v in r.items() if isinstance(v, dict)}))
        json.dump(summary, open(final, "w"), indent=1)
        return

    R = region_masks(model.cell.adj)
    ctrls = nn.ModuleDict({h: Controller(model.dx, model.d) for h in CONTROLLER_HEADS}).to(dev)
    summary["ctrl_params"] = count_params(ctrls[CONTROLLER_HEADS[0]])
    ck_b = os.path.join(a.out, "ctrl_A.pt")
    if os.path.exists(ck_b):
        ctrls.load_state_dict(torch.load(ck_b, map_location=dev))
        for p in ctrls.parameters():
            p.requires_grad_(False)
    else:
        t0 = time.time()
        summary["hist_B"] = stage_b(model, fe, tr, dev, a, rng, ctrls, a.epochs_b, R)
        summary["time_B_s"] = time.time() - t0
        torch.save(ctrls.state_dict(), ck_b)
    summary["diag_A"] = ranking_diagnostics(model, fe, va, dev, ctrls, R)
    log("[diag A] " + json.dumps({k: v for k, v in summary["diag_A"].items() if k != "per_rec_conc"}))
    summary["grid_A"] = run_policy_grid(model, fe, splits, dev, ctrls, R, "A", a.out, a)
    json.dump(summary, open(os.path.join(a.out, "summary_partial.json"), "w"), indent=1)

    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    ck_c = os.path.join(a.out, "model_C.pt")
    if os.path.exists(ck_c):
        model.load_state_dict(torch.load(ck_c, map_location=dev))
    else:
        t0 = time.time()
        summary["hist_C"] = stage_c(model, teacher, fe, tr, va, dev, a, rng, cw, ctrls, R)
        summary["time_C_s"] = time.time() - t0
        torch.save(model.state_dict(), ck_c)
    for p in ctrls.parameters():
        p.requires_grad_(True)
    summary["hist_B2"] = stage_b(model, fe, tr, dev, a, rng, ctrls, a.epochs_b2, R, warm=True)
    torch.save(ctrls.state_dict(), os.path.join(a.out, "ctrl_C.pt"))
    summary["diag_C"] = ranking_diagnostics(model, fe, va, dev, ctrls, R)
    log("[diag C] " + json.dumps({k: v for k, v in summary["diag_C"].items() if k != "per_rec_conc"}))
    summary["grid_C"] = run_policy_grid(model, fe, splits, dev, ctrls, R, "C", a.out, a)
    json.dump(summary, open(final, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
