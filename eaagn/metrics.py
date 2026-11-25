"""Event decoding and evaluation.

Event-based F-score / error rate follow the sed_eval definitions with DCASE-style
collars (onset 200 ms; offset max(200 ms, 20% of reference length)); the SPRSound
organisers have not released their evaluator, so the collars are stated explicitly.
Matching is maximum-cardinality bipartite (Hungarian), class-aware for hits, then
class-agnostic on the residue to count substitutions.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment

FRAME_S = 0.01
T_COLLAR, PCT = 0.2, 0.2
# classes 1..N_NONADV are non-adventitious (SPRSound: 1 = Normal; HF_Lung: 1-2 = inhalation, exhalation)
N_NONADV = int(__import__("os").environ.get("EAAGN_NONADV", "1"))
ADV = tuple(range(N_NONADV + 1, 8))  # adventitious event classes


def decode(prob: np.ndarray, w: int = 1, min_frames: int = 0):
    """prob (T, C) frame posteriors -> list of (on_s, off_s, cls).

    Causal: the smoother is a TRAILING moving average (frame t averages t-w+1..t), so no
    decision uses future posteriors. Dropping runs shorter than min_frames only delays the
    emission of an event by at most min_frames x 10 ms (reported as decoding delay)."""
    p = uniform_filter1d(prob, size=w, axis=0, mode="nearest", origin=(w - 1) // 2) if w > 1 else prob
    lab = p.argmax(1)
    evs, start = [], None
    for t in range(len(lab) + 1):
        cur = lab[t] if t < len(lab) else -1
        if start is not None and cur != lab[start]:
            if lab[start] != 0 and t - start >= max(min_frames, 1):
                evs.append((start * FRAME_S, t * FRAME_S, int(lab[start])))
            start = None
        if start is None and t < len(lab):
            start = t
    return evs


def _eligible(ref, est, class_aware):
    M = np.zeros((len(ref), len(est)), dtype=bool)
    for i, (ro, rf, rc) in enumerate(ref):
        off_col = max(T_COLLAR, PCT * (rf - ro))
        for j, (eo, ef, ec) in enumerate(est):
            if class_aware and rc != ec:
                continue
            M[i, j] = abs(ro - eo) <= T_COLLAR and abs(rf - ef) <= off_col
    return M


def _max_match(M):
    if M.size == 0 or not M.any():
        return [], []
    r, c = linear_sum_assignment(-M.astype(float))
    keep = M[r, c]
    return list(r[keep]), list(c[keep])


def event_counts(ref, est):
    """Per-recording counts: [TP, S, D, I, Nref, Nest] + per-class (1..7) TP, Nref, Nest."""
    r, c = _max_match(_eligible(ref, est, True))
    tp = len(r)
    rset, cset = set(r), set(c)
    ref_u = [e for i, e in enumerate(ref) if i not in rset]
    est_u = [e for j, e in enumerate(est) if j not in cset]
    rs, _ = _max_match(_eligible(ref_u, est_u, False))
    s = len(rs)
    tpc = np.zeros(7, np.int64); nrc = np.zeros(7, np.int64); nec = np.zeros(7, np.int64)
    for i in r:
        tpc[ref[i][2] - 1] += 1
    for e in ref:
        nrc[e[2] - 1] += 1
    for e in est:
        nec[e[2] - 1] += 1
    return np.concatenate([[tp, s, len(ref_u) - s, len(est_u) - s, len(ref), len(est)], tpc, nrc, nec]).astype(np.int64)


def onset_counts(ref, est):
    """Onset-only class-aware hits (collar 200 ms): TP, Nref, Nest."""
    M = np.zeros((len(ref), len(est)), dtype=bool)
    for i, (ro, _, rc) in enumerate(ref):
        for j, (eo, _, ec) in enumerate(est):
            M[i, j] = rc == ec and abs(ro - eo) <= T_COLLAR
    r, _ = _max_match(M)
    return np.array([len(r), len(ref), len(est)], dtype=np.int64)


def summarize_events(counts: np.ndarray):
    c = counts.sum(0)
    tp, s, d, i, nref, nest = c[:6]
    f = 2 * tp / max(nref + nest, 1)
    er = (s + d + i) / max(nref, 1)
    out = dict(F=float(f), ER=float(er), P=float(tp / max(nest, 1)), R=float(tp / max(nref, 1)))
    if len(c) >= 27:
        tpc, nrc, nec = c[6:13], c[13:20], c[20:27]
        a = slice(N_NONADV, 7)  # adventitious classes
        out["advF"] = float(2 * tpc[a].sum() / max(nrc[a].sum() + nec[a].sum(), 1))
        out["advR"] = float(tpc[a].sum() / max(nrc[a].sum(), 1))
        out["normF"] = float(2 * tpc[0] / max(nrc[0] + nec[0], 1))
        fs = [2 * tpc[k] / max(nrc[k] + nec[k], 1) for k in range(7) if nrc[k] > 0]
        out["macroF"] = float(np.mean(fs)) if fs else 0.0
        out["classR"] = {str(k + 1): float(tpc[k] / nrc[k]) for k in range(7) if nrc[k] > 0}
    return out


def segment_counts(ref, est, dur_s, seg_s=1.0):
    """Segment-based (1 s) class-wise activity counts, micro over classes 1..7 (sed_eval
    segment-based metric): [TP, FP, FN, S, D, I, N_ref_active]."""
    n = max(int(np.ceil(dur_s / seg_s)), 1)
    def act(evs):
        A = np.zeros((n, 8), dtype=bool)
        for on, off, c in evs:
            a, b = int(np.floor(on / seg_s)), int(np.ceil(off / seg_s))
            A[max(a, 0):min(max(b, a + 1), n), c] = True
        return A[:, 1:]
    R, E = act(ref), act(est)
    tp = (R & E).sum(); fp = (~R & E).sum(); fn = (R & ~E).sum()
    fn_s = (R & ~E).sum(1); fp_s = (~R & E).sum(1)
    S = np.minimum(fn_s, fp_s).sum(); D = np.maximum(0, fn_s - fp_s).sum(); I = np.maximum(0, fp_s - fn_s).sum()
    Ra, Ea = R[:, N_NONADV:], E[:, N_NONADV:]  # adventitious classes only
    return np.array([tp, fp, fn, S, D, I, R.sum(), (Ra & Ea).sum(), (~Ra & Ea).sum(), (Ra & ~Ea).sum()], dtype=np.int64)


def summarize_segments(counts: np.ndarray):
    tp, fp, fn, S, D, I, nr, atp, afp, afn = counts.sum(0)
    return dict(segF=float(2 * tp / max(2 * tp + fp + fn, 1)), segER=float((S + D + I) / max(nr, 1)),
                seg_advF=float(2 * atp / max(2 * atp + afp + afn, 1)))


def summarize_onsets(counts: np.ndarray):
    tp, nref, nest = counts.sum(0)
    return float(2 * tp / max(nref + nest, 1))


def segment_preds(prob: np.ndarray, ref):
    """Classify each reference segment by mean frame posterior over classes 1..7."""
    out = []
    for on, off, c in ref:
        a, b = int(round(on / FRAME_S)), max(int(round(off / FRAME_S)), int(round(on / FRAME_S)) + 1)
        seg = prob[a:b, 1:]
        if len(seg) == 0:
            continue
        out.append((c, int(seg.mean(0).argmax()) + 1))
    return out


def sprsound_scores(pairs):
    """SPRSound Task-1 style: SE over adventitious events (exact subtype must match),
    SP over normal events, AS=(SE+SP)/2, HS=2SE*SP/(SE+SP); plus macro-F1 over 7 classes
    and binary (normal vs adventitious) scores."""
    if len(pairs) == 0:
        return {}
    y = np.array([p[0] for p in pairs]); yh = np.array([p[1] for p in pairs])
    adv = y != 1
    se = float((yh[adv] == y[adv]).mean()) if adv.any() else 0.0
    sp = float((yh[~adv] == 1).mean()) if (~adv).any() else 0.0
    bse = float((yh[adv] != 1).mean()) if adv.any() else 0.0
    f1s = []
    for c in range(1, 8):
        tp = ((y == c) & (yh == c)).sum(); fp = ((y != c) & (yh == c)).sum(); fn = ((y == c) & (yh != c)).sum()
        if (y == c).any():
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    hs = lambda a, b: 2 * a * b / (a + b) if a + b > 0 else 0.0
    return dict(SE=se, SP=sp, AS=(se + sp) / 2, HS=hs(se, sp), macroF1=float(np.mean(f1s)),
                bin_SE=bse, bin_AS=(bse + sp) / 2, bin_HS=hs(bse, sp))


def frame_macro_f1(probs, labels):
    """Frame-level macro-F1 over classes present in the labels (bg included)."""
    yh = np.concatenate([p.argmax(1) for p in probs]); y = np.concatenate(labels)
    m = y >= 0
    yh, y = yh[m], y[m]
    f1 = []
    for c in np.unique(y):
        tp = ((y == c) & (yh == c)).sum(); fp = ((y != c) & (yh == c)).sum(); fn = ((y == c) & (yh != c)).sum()
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1)), float((yh == y).mean())


DECODE_GRID = [(w, m) for w in (1, 5, 11, 21) for m in (0, 10, 20, 30)]


def selection_score(summary):
    """Validation criterion: mean of overall event-F (Normal-dominated) and adventitious event-F."""
    return 0.5 * (summary["F"] + summary.get("advF", 0.0))


def tune_decoding(probs, refs):
    best = None
    for w, m in DECODE_GRID:
        c = np.stack([event_counts(r, decode(p, w, m)) for p, r in zip(probs, refs)])
        f = selection_score(summarize_events(c))
        if best is None or f > best[0]:
            best = (f, w, m)
    return best[1], best[2]
