# Run from the repository root: .venv/bin/python experiments/lab/segment_blend.py
# 2026-11-03 (review: "is the final scoring function calibrated ACROSS events?"). lab/combine_check.py tested global
# blends and event x share decompositions on day 6 only. Here the question is per SEGMENT: should the trees / choice
# weight, or an additive score offset, differ by known-clicks group or by hours since the last click? Members: SET40
# LightGBM (lab_preds_boxbase_dT) and the 10-seed choice model (lab_preds_chC3x10_dT), both on half B of day T, fit days
# 2..T-1. Weights and offsets are chosen on day 6 (coordinate ascent, gap-weighted AUC) and only SCORED on days 5 and 4.
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
y_all = fit["label"][rows]
_, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
_, hl_eval = e13.long_history(test, fit, e13.HIST_LONG)
hl_row = hlen[rows // 10].astype(int)
BUCKETS = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100)]
def bucket(h):
    b = np.zeros(len(h), int)
    for k, (lo, hi) in enumerate(BUCKETS):
        b[(h >= lo) & (h <= hi)] = k
    return b
cu, ct = fit["clk_user"].astype(np.int64), fit["clk_ts"].astype(np.int64); o_ = np.lexsort((ct, cu)); cu, ct = cu[o_], ct[o_]
def gap_bucket(src):                                       # as lab/paired_ci.py
    key = src["user"].astype(np.int64) * 10**10 + src["ts"].astype(np.int64)
    j = np.searchsorted(cu * 10**10 + ct, key, side="left") - 1; ok = (j >= 0) & (cu[np.maximum(j, 0)] == src["user"])
    g_ = np.full(len(key), np.inf); g_[ok] = (src["ts"][ok] - ct[j[ok]]) / 3600
    return np.searchsorted([0, 6, 12, 24, 48, 96, 192], g_, side="right") - 1
gb_fit, gb_ev = gap_bucket(fit), gap_bucket(test)
L = ROOT / "data/cache/box/lab"


def wauc(s, y_, w):
    o = np.argsort(s, kind="mergesort"); s, y_, w = s[o], y_[o], w[o]
    new = np.r_[True, s[1:] != s[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, w * y_); wn = np.bincount(g, w * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


def day(T):
    t = np.load(L / f"lab_preds_boxbase_d{T}.npz"); c = np.load(L / f"lab_preds_chC3x10_d{T}.npz")
    rB = t["rows_B"]; assert np.array_equal(rB, c["rows_B"])
    ev_ra = rows[rB][::10] // 10
    hb = bucket(hl_row[rB]); gb = np.repeat(gb_fit[ev_ra], 10)
    d_ = np.repeat(fit["day"], 10)[rows] == T
    ratio = (np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)) / (np.bincount(bucket(hl_row[d_][::10]), minlength=5) / d_.sum() * 10)
    gshare = np.bincount(gb_fit[ev_ra], minlength=7) / len(ev_ra)
    w = {"eval": ratio[hb], "gap": (np.bincount(gb_ev, minlength=7) / len(gb_ev) / np.maximum(gshare, 1e-12))[gb]}
    r = lambda p: rankdata(p) / len(p)
    return dict(y=y_all[rB].astype(float), t=r(t["base (= the best member)|B"]), c=r(c["base (= the best member)|B"]),
                seg={"hist": hb, "gap": gb}, w=w, n_ev=len(rB) // 10)


def score(D, seg, wt, off):
    s = D["seg"][seg]
    return (1 - wt[s]) * D["t"] + wt[s] * D["c"] + off[s]


D6 = day(6)
def fit_seg(seg, what, metric="gap"):
    k = D6["seg"][seg].max() + 1
    wt, off = np.full(k, 0.5), np.zeros(k)
    grid = {"wt": np.linspace(0, 1, 9), "off": np.linspace(-0.03, 0.03, 13)}[what]
    for _ in range(2):
        for b in range(k):
            best = None
            for v in grid:
                (wt if what == "wt" else off)[b] = v
                a = wauc(score(D6, seg, wt, off), D6["y"], D6["w"][metric])
                if best is None or a > best[0]:
                    best = (a, v)
            (wt if what == "wt" else off)[b] = best[1]
    return wt, off


FITS = {f"{seg} {what}": (seg, *fit_seg(seg, what)) for seg in ("hist", "gap") for what in ("wt", "off")}
for name, (seg, wt, off) in FITS.items():
    print(f"{name:9s} chosen on day 6: choice weight {np.round(wt, 3)}  offset {np.round(off, 3)}", flush=True)
for T in (6, 5, 4):
    D = D6 if T == 6 else day(T)
    rng = np.random.default_rng(0)
    cnt = [np.repeat(rng.multinomial(D["n_ev"], np.full(D["n_ev"], 1 / D["n_ev"])), 10) for _ in range(150)]
    base = 0.5 * D["t"] + 0.5 * D["c"]
    print(f"### day {T}{' (fitted here)' if T == 6 else ''}: base = rank avg .5/.5, eval / gap-weighted "
          f"{wauc(base, D['y'], D['w']['eval']):.5f} / {wauc(base, D['y'], D['w']['gap']):.5f}", flush=True)
    for name, (seg, wt, off) in FITS.items():
        s = score(D, seg, wt, off)
        out = []
        for m in ("eval", "gap"):
            w = D["w"][m]
            d = np.array([wauc(s, D["y"], w * c) - wauc(base, D["y"], w * c) for c in cnt])
            out.append(f"{m} {wauc(s, D['y'], w) - wauc(base, D['y'], w):+.5f} [{np.percentile(d, 2.5):+.5f}, {np.percentile(d, 97.5):+.5f}]")
        print(f"  {name:9s} " + "   ".join(out), flush=True)
