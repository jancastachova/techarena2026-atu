#!/usr/bin/env python3
"""
Experiment 33 view: the taste match with SET pooling (each unique clicked item weighted by
sqrt(times clicked) instead of once per click), from the saved MF vectors of experiments 22 / 26
(MF trained on days < d, 10 seeds, scores averaged), as an extra-group cache for 24_ensemble/ensemble.py
(--extra sqrtpool=...). Columns [score scaled to std 3, sqrt-pooled minus count-pooled dot, same, score];
days 0-1 NaN.

    .venv/bin/python experiments/33_set_pool/sqrt_view.py --out data/cache/33_sqrtpool_c7.npz
"""
import argparse, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
EV = e13.EV_SIZE


def pooled(H, n, W):
    """count-pooled and sqrt-set-pooled user vectors for histories H (newest first, 0 = padding)."""
    m = np.arange(H.shape[1])[None] < n[:, None]
    same = (H[:, :, None] == H[:, None, :]) & m[:, None, :]
    cnt = np.maximum(same.sum(2), 1)
    w_sq = m / np.sqrt(cnt); w_ct = m.astype(float)
    V = W[H]
    pool = lambda w: (V * w[:, :, None]).sum(1) / np.maximum(w.sum(1), 1e-9)[:, None]
    return pool(w_ct), pool(w_sq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
    files = [ROOT / "data/cache/22_vectors_c7.npz"] + [ROOT / f"data/cache/26_vectors_c7_s{s}.npz" for s in range(1, 10)]
    hf, lf = e13.long_history(fit, fit, e13.HIST_LONG); ht, lt = e13.long_history(test, fit, e13.HIST_LONG)
    out_f = np.full((len(fit["item"]), 3), np.nan, np.float32)
    for day in range(e13.FIRST_MODEL_DAY, e13.OFFICIAL_CUTOFF):
        evd = np.flatnonzero(fit["day"] == day); items = fit["item"].reshape(-1, EV)[evd]
        sq = np.zeros(items.shape); df = np.zeros(items.shape)
        for f in files:
            W = np.load(f)[f"W_{day}"].astype(np.float32)
            for s in range(0, len(evd), 5000):
                sl = slice(s, s + 5000)
                hc, hs = pooled(hf[evd[sl]], lf[evd[sl]].astype(int), W)
                a_ = np.einsum("ed,ekd->ek", hs, W[items[sl]]); b_ = np.einsum("ed,ekd->ek", hc, W[items[sl]])
                sq[sl] += a_ / len(files); df[sl] += (a_ - b_) / len(files)
        r = (evd[:, None] * EV + np.arange(EV)).ravel()
        out_f[r] = np.stack([sq.ravel(), df.ravel(), df.ravel()], 1)
        e13.log(f"day {day}: sqrt-pooled dot AUC {e13.roc_auc(sq.ravel(), fit['label'][r]):.5f}")
    items = test["item"].reshape(-1, EV); sq = np.zeros(items.shape); df = np.zeros(items.shape)
    for f in files:
        W = np.load(f)["W_test"].astype(np.float32)
        for s in range(0, len(items), 5000):
            sl = slice(s, s + 5000)
            hc, hs = pooled(ht[sl], lt[sl].astype(int), W)
            a_ = np.einsum("ed,ekd->ek", hs, W[items[sl]]); b_ = np.einsum("ed,ekd->ek", hc, W[items[sl]])
            sq[sl] += a_ / len(files); df[sl] += (a_ - b_) / len(files)
    t = np.stack([sq.ravel(), df.ravel(), df.ravel()], 1).astype(np.float32)
    scale = 3.0 / np.nanstd(out_f[:, 0]); out_f[:, 0] *= scale; t[:, 0] *= scale
    np.savez(a.out, mf_fit=np.concatenate([out_f, out_f[:, :1]], 1), mf_test=np.concatenate([t, t[:, :1]], 1))
    e13.log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
