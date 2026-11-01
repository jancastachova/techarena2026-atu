#!/usr/bin/env python3
"""
Experiment 34 view: the taste match as a COSINE, plus the vector norms the dot product hides, from the saved
MF vectors of experiments 22 / 26 (MF trained on days < d, 10 seeds, averaged), as an extra-group cache for
24_ensemble/ensemble.py (--extra cos=...). lab/score_form.py: on day 6, cosine(user, item) of the dot-trained
vectors scores 0.81777 vs 0.81278 for their dot product (item-normalised alone 0.81347, user-normalised
0.81677; within-event 0.8201 vs 0.8191); LightGBM only ever saw the dot. Columns [cosine scaled to std 3,
item vector norm, user vector norm, item-normalised dot]; days 0-1 NaN.

    .venv/bin/python experiments/34_cosine/cos_view.py --out data/cache/34_cos_c7.npz
    # the same for the IDF-pooled MF (user = idf-weighted mean, gamma 1) and the clicks-only MF (5 seeds each):
    .venv/bin/python experiments/34_cosine/cos_view.py --vectors 26_vectors_idf_c7_s{0..4} --idf 1.0 --out data/cache/34_cos_idf_c7.npz
    .venv/bin/python experiments/34_cosine/cos_view.py --vectors 26_vectors_clk_c7_s{0..4} --out data/cache/34_cos_clk_c7.npz
"""
import argparse, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
EV = e13.EV_SIZE


def block(H, n, items, W, pw=None):
    m = (np.arange(H.shape[1])[None] < n[:, None]).astype(np.float32)
    if pw is not None:
        m = m * pw[H]
    hv = (W[H] * m[:, :, None]).sum(1) / np.maximum(m.sum(1), 1e-9)[:, None]
    V = W[items]
    nv = np.linalg.norm(V, axis=2); nu = np.linalg.norm(hv, axis=1)
    dot = np.einsum("ed,ekd->ek", hv, V)
    cos = dot / np.maximum(nv * nu[:, None], 1e-9)
    return np.stack([cos, nv, np.repeat(nu[:, None], EV, 1), dot / np.maximum(nv, 1e-9)], 2)


PROJ = 0            # --proj k: vectors projected onto their top-k principal directions first (lab/geometry.py)
RESID = False       # --resid: keep only what the top-k projection drops (lab/replica_gap.py: real-only signal)
SOFT = 0.0          # --soft K: keep the remaining directions scaled by c / (c + K), c = item clicks (lab/soft_proj.py)
CLICKS = None       # item clicks known before the scored day (set per day when --soft is used)


def project(W):
    if not PROJ:
        return W
    C = W[1:].T.astype(np.float64) @ W[1:].astype(np.float64)
    _, U = np.linalg.eigh(C)
    U = U[:, ::-1][:, :PROJ].astype(np.float32)
    top = (W @ U) @ U.T
    if RESID:
        return W - top
    if not SOFT:
        return top
    lam = (CLICKS[:len(W)] / (CLICKS[:len(W)] + SOFT)).astype(np.float32)[:, None]
    return top + lam * (W - top)


def scores(H, n, items, files, key, pw=None):
    out = np.zeros(items.shape + (4,))
    for f in files:
        W = project(np.load(f)[key].astype(np.float32))
        for s in range(0, len(items), 5000):
            sl = slice(s, s + 5000)
            out[sl] += block(H[sl], n[sl].astype(int), items[sl], W, pw) / len(files)
    return out.reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors", nargs="*", help="vector caches in data/cache (default: 22 + 26 seeds 1-9)")
    ap.add_argument("--proj", type=int, default=0, help="project the vectors onto their top-k principal directions")
    ap.add_argument("--soft", type=float, default=0.0, help="with --proj: keep the rest scaled by clicks / (clicks + K)")
    ap.add_argument("--resid", action="store_true", help="with --proj: the residual outside the top-k directions")
    ap.add_argument("--idf", type=float, help="pool the user with weights (1 + item clicks before the day)^-gamma")
    a = ap.parse_args()
    global PROJ, SOFT, CLICKS, RESID
    PROJ, SOFT, RESID = a.proj, a.soft, a.resid
    fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
    files = ([ROOT / "data/cache" / (v if v.endswith(".npz") else v + ".npz") for v in a.vectors] if a.vectors else
             [ROOT / "data/cache/22_vectors_c7.npz"] + [ROOT / f"data/cache/26_vectors_c7_s{s}.npz" for s in range(1, 10)])
    n_items = int(max(fit["item"].max(), test["item"].max())) + 1
    def pool_w(day):
        if a.idf is None:
            return None
        k = fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY
        return (1.0 + np.bincount(fit["clk_item"][k], minlength=n_items)) ** -a.idf
    hf, lf = e13.long_history(fit, fit, e13.HIST_LONG); ht, lt = e13.long_history(test, fit, e13.HIST_LONG)
    out_f = np.full((len(fit["item"]), 4), np.nan, np.float32)
    for day in range(e13.FIRST_MODEL_DAY, e13.OFFICIAL_CUTOFF):
        evd = np.flatnonzero(fit["day"] == day)
        r = (evd[:, None] * EV + np.arange(EV)).ravel()
        CLICKS = np.bincount(fit["clk_item"][fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY], minlength=n_items).astype(np.float64)
        out_f[r] = scores(hf[evd], lf[evd], fit["item"].reshape(-1, EV)[evd], files, f"W_{day}", pool_w(day))
        e13.log(f"day {day}: cosine AUC {e13.roc_auc(out_f[r, 0], fit['label'][r]):.5f}  "
                f"item-normalised dot {e13.roc_auc(out_f[r, 3], fit['label'][r]):.5f}")
    CLICKS = np.bincount(fit["clk_item"][fit["clk_ts"] < e13.TRAIN_START + e13.OFFICIAL_CUTOFF * e13.DAY], minlength=n_items).astype(np.float64)
    t = scores(ht, lt, test["item"].reshape(-1, EV), files, "W_test", pool_w(e13.OFFICIAL_CUTOFF)).astype(np.float32)
    scale = 3.0 / np.nanstd(out_f[:, 0]); out_f[:, 0] *= scale; t[:, 0] *= scale
    np.savez(a.out, mf_fit=out_f, mf_test=t)
    e13.log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
