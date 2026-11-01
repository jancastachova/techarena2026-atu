#!/usr/bin/env python3
"""
TechArena CTR -- experiment 26: the AUDIENCE view, written as two extra-group caches for
24_ensemble/ensemble.py (--extra aud=... --extra audc=...).

What: a candidate is represented by its audience -- the mean taste of the users who clicked it,
each clicker's taste being the mean MF vector of their OTHER known clicks -- instead of its own
learned item vector. Scores per row (user vector hv = mean of the user's known clicks):
    aud    hv . A_i                 A_i = mean over i's clicks of the clicker's other-click taste
    audc   hv . (A_i - B_i)         B_i = mean taste of the users i was SHOWN to (impressions)
Everything as of each day d: clicks and impressions before d, MF vectors of an MF trained on
days < d (experiment 22's exported vectors, 1 seed); the test events use the vectors of the MF
trained on all fit days and every click/impression before the cutoff.

Why: rare items are underfit for lack of information (RESULTS.md, 23); their own vector is
learned from a handful of clicks, while their audience pools the clickers' whole histories.
It is a different information path (user-based CF), like the IDF and clicks-only views.
Day-6 screening, dot product alone (lab/audience.py): MF 0.81278, aud 0.81363, audc 0.81451
(rank corr with the MF 0.945), MF + audc 0.81786. As LightGBM features on the current best
feature set (lab/audience_feature.py, half B): 0.83854 -> 0.83973 (+0.00119; half A +0.00111).

Cache columns (the ensemble adds the event-relative versions of column 0):
    [score (scaled to std 3 on the fit rows), log(1 + audience clicks), same, score]
Days 0-1 are NaN, as in the MF caches.

--variants also writes screening variants (same format, <prefix>_<name>_c7.npz):
    audc05  hv . (A_i - 0.5 B_i)    (day-6 alone 0.81608 vs audc 0.81451, 1 seed)
    wb      hv . (W_i - B_i)        the item's own vector, exposure-corrected
    hop2    u2 . W_i                u2 = mean of (A_j - B_j) over the user's known clicks j
    hop2c   u2 . (A_i - B_i)
--hops adds one more hop: the audience operator applied to C = A - B instead of W, giving
A2 (mean over i's clickers of their other clicks' C) and B2 (same over the users i was shown to):
    hop3    hv . (A2_i - B2_i)
    hop4    u2 . (A2_i - B2_i)

    .venv/bin/python experiments/26_audience/audience.py --vectors data/cache/22_vectors_c7.npz \
        --out-prefix data/cache/26
"""
import argparse, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)


def tastes(fit, W, day):
    """Per user: sum and count of the vectors of their clicks before `day`; plus the click arrays."""
    k = fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY
    cu, ci = fit["clk_user"][k].astype(np.int64), fit["clk_item"][k].astype(np.int64)
    n_users = int(max(cu.max(), fit["user"].max())) + 1
    S = np.zeros((n_users, W.shape[1])); np.add.at(S, cu, W[ci])
    n = np.bincount(cu, minlength=n_users).astype(np.float64)
    return S, n, cu, ci


def audience(fit, W, day):
    """Per item: A (mean other-click taste of its clickers), its click count, B (mean taste of the
    users it was shown to in fit events before `day`)."""
    S, n, cu, ci = tastes(fit, W, day)
    other = (S[cu] - W[ci]) / np.maximum(n[cu] - 1, 1)[:, None]
    if UNIT_CLICKERS:
        other = other / np.maximum(np.linalg.norm(other, axis=1, keepdims=True), 1e-12)
    ok = n[cu] > 1
    A = np.zeros_like(W); np.add.at(A, ci[ok], other[ok])
    na = np.bincount(ci[ok], minlength=W.shape[0]).astype(np.float64)
    A /= np.maximum(na, 1)[:, None]
    ev = np.flatnonzero(fit["day"] < day)
    su = np.repeat(fit["user"][ev].astype(np.int64), e13.EV_SIZE)
    si = fit["item"].reshape(-1, e13.EV_SIZE)[ev].ravel()
    taste = S / np.maximum(n, 1)[:, None]
    if UNIT_CLICKERS:
        taste = taste / np.maximum(np.linalg.norm(taste, axis=1, keepdims=True), 1e-12)
    B = np.zeros_like(W); np.add.at(B, si, taste[su])
    nb = np.bincount(si, minlength=W.shape[0]).astype(np.float64)
    B /= np.maximum(nb, 1)[:, None]
    return A, na, B


COSINE = False                    # --cosine: unit item vectors in, every score a cosine (experiment 34)
UNIT_CLICKERS = False             # --unit-clickers: every clicker's / viewer's taste as a unit vector before averaging
SOFT_PROJ = 0.0                   # --soft-proj K: item vectors = top-8 principal part + clicks/(clicks+K) x the rest


def soft_project(fit, W, day):
    if not SOFT_PROJ:
        return W
    _, U = np.linalg.eigh(W[1:].T @ W[1:]); U = U[:, ::-1][:, :8]
    top = (W @ U) @ U.T
    c = np.bincount(fit["clk_item"][fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY], minlength=len(W))[:len(W)]
    return top + (c / (c + SOFT_PROJ))[:, None] * (W - top)


def _unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12) if COSINE else X


def scores(items, hv, A, na, B, W=None, hist=None, C2=None):
    hv = _unit(hv)
    s_a = np.einsum("ed,ekd->ek", hv, _unit(A)[items]).ravel()
    s_c = np.einsum("ed,ekd->ek", hv, _unit(A - B)[items]).ravel()
    out = [s_a, s_c, np.log1p(na[items]).ravel()]
    if W is not None:
        H, n = hist
        C = A - B
        msk = (np.arange(H.shape[1])[None, :] < n[:, None])[:, :, None]
        u2 = (C[H] * msk).sum(1) / np.maximum(n, 1)[:, None]
        dot = lambda u, V: np.einsum("ed,ekd->ek", _unit(u), _unit(V)[items]).ravel()
        out += [dot(hv, A - 0.5 * B), dot(hv, W - B), dot(u2, W), dot(u2, C)]
        if C2 is not None:
            out += [dot(hv, C2), dot(u2, C2)]
    return out


def second_hop(fit, A, B, day):
    A2, _, B2 = audience(fit, A - B, day)
    return A2 - B2


VARIANTS = ["audc05", "wb", "hop2", "hop2c"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
    ap.add_argument("--vectors", required=True,
                    help="comma-separated --save-vectors outputs (cutoff 7; 22's file or vectors.py --vec-seed N);"
                         " the scores are averaged over them (dimensions differ per seed, scores do not)")
    ap.add_argument("--out-prefix", required=True, help="writes <prefix>_aud_c7.npz and <prefix>_audc_c7.npz")
    ap.add_argument("--variants", action="store_true", help="also write the screening variants (docstring)")
    ap.add_argument("--hops", action="store_true", help="with --variants: also hop3 and hop4")
    ap.add_argument("--cosine", action="store_true", help="unit item vectors, every score a cosine (experiment 34)")
    ap.add_argument("--unit-clickers", action="store_true", help="unit clicker / viewer tastes (lab/aud_weighting.py)")
    ap.add_argument("--soft-proj", type=float, default=0.0, help="soft projection of the item vectors (lab/soft_proj.py)")
    a = ap.parse_args()
    global COSINE, UNIT_CLICKERS, SOFT_PROJ
    COSINE, UNIT_CLICKERS, SOFT_PROJ = a.cosine, a.unit_clickers, a.soft_proj
    fit, test, _, _, _ = e13.load_base(a.base)
    files = a.vectors.split(",")
    n_fit = len(fit["item"])
    names = ["aud", "audc"] + (VARIANTS if a.variants else []) + (["hop3", "hop4"] if a.hops else [])
    hf = e13.long_history(fit, fit, e13.HIST_LONG) if a.variants else None
    ht = e13.long_history(test, fit, e13.HIST_LONG) if a.variants else None
    out_f = {k: np.full((n_fit, 3), np.nan, np.float32) for k in names}
    for day in range(e13.FIRST_MODEL_DAY, e13.OFFICIAL_CUTOFF):
        acc = []
        for f in files:
            z = np.load(f)
            W = z[f"W_{day}"].astype(np.float64); evd = z[f"ev_{day}"]; hv = z[f"hv_{day}"].astype(np.float64)
            if SOFT_PROJ:
                W = soft_project(fit, W, day)
                hf_ = e13.long_history({"user": fit["user"][evd], "ts": fit["ts"][evd]}, fit, e13.HIST_LONG)
                msk_ = (np.arange(hf_[0].shape[1])[None] < hf_[1][:, None])[:, :, None]
                hv = (W[hf_[0]] * msk_).sum(1) / np.maximum(hf_[1], 1)[:, None]
            if COSINE:
                W = _unit(W); W[0] = 0.0
            if not np.all(fit["day"][evd] == day):
                raise AssertionError(f"{f}: events of day {day} do not match the base cache")
            A, na, B = audience(fit, W, day)
            extra = (W, (hf[0][evd], hf[1][evd])) if a.variants else ()
            if a.hops:
                extra += (second_hop(fit, A, B, day),)
            acc.append(scores(fit["item"].reshape(-1, e13.EV_SIZE)[evd], hv, A, na, B, *extra))
        sc = [np.mean([x[j] for x in acc], 0) for j in range(len(acc[0]))]
        ln = sc.pop(2)
        r = (evd[:, None] * e13.EV_SIZE + np.arange(e13.EV_SIZE)).ravel()
        y = fit["label"][r]
        for k, s_ in zip(names, sc):
            out_f[k][r] = np.stack([s_, ln, ln], 1)
        e13.log(f"day {day} ({len(files)} seed(s)): " + ", ".join(
            f"{k} AUC {e13.roc_auc(s_, y):.5f}" for k, s_ in zip(names, sc)))
    acc = []
    ht_ = e13.long_history(test, fit, e13.HIST_LONG) if SOFT_PROJ else None
    for f in files:
        z = np.load(f)
        W = z["W_test"].astype(np.float64); hv = z["hv_test"].astype(np.float64)
        if SOFT_PROJ:
            W = soft_project(fit, W, e13.OFFICIAL_CUTOFF)
            msk_ = (np.arange(ht_[0].shape[1])[None] < ht_[1][:, None])[:, :, None]
            hv = (W[ht_[0]] * msk_).sum(1) / np.maximum(ht_[1], 1)[:, None]
        if COSINE:
            W = _unit(W); W[0] = 0.0
        if len(hv) != len(test["user"]):
            raise AssertionError(f"{f}: test events do not match the base cache")
        A, na, B = audience(fit, W, e13.OFFICIAL_CUTOFF)
        extra = (W, ht) if a.variants else ()
        if a.hops:
            extra += (second_hop(fit, A, B, e13.OFFICIAL_CUTOFF),)
        acc.append(scores(test["item"].reshape(-1, e13.EV_SIZE), hv, A, na, B, *extra))
    sc = [np.mean([x[j] for x in acc], 0) for j in range(len(acc[0]))]
    ln = sc.pop(2)
    out_t = {k: np.stack([s_, ln, ln], 1) for k, s_ in zip(names, sc)}
    for k in names:
        f, t = out_f[k], out_t[k].astype(np.float32)
        scale = 3.0 / np.nanstd(f[:, 0])
        f[:, 0] *= scale; t[:, 0] *= scale
        mf_fit = np.concatenate([f, f[:, :1]], 1)
        mf_test = np.concatenate([t, t[:, :1]], 1)
        path = f"{a.out_prefix}_{k}_c7.npz"
        np.savez(path, mf_fit=mf_fit, mf_test=mf_test)
        e13.log(f"{k}: wrote {path} (test score mean {mf_test[:, 0].mean():.3f} sd {mf_test[:, 0].std():.3f}, "
                f"fit days 2-6 mean {np.nanmean(mf_fit[:, 0]):.3f} sd {np.nanstd(mf_fit[:, 0]):.3f})")


if __name__ == "__main__":
    main()
