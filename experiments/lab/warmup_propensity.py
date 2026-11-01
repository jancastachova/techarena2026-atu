# Run from the repository root: .venv/bin/python experiments/lab/warmup_propensity.py
# Does the warm-up click INTENSITY estimate a user's click propensity? If events arrive at a stable
# per-user rate lam_u and each is clicked with probability p_u, the visible warm-up clicks (up to 20
# before the user's first train event, with timestamps) give the click intensity lam_u * p_u, and the
# train events give lam_u, so (warm-up intensity / train event rate) estimates p_u from data the
# per-user click rate never sees. Checked on users with many train events (their click rate is
# well measured): correlation of the warm-up estimate with the train click rate, compared with the
# correlation between two halves of their own train events (the reliability ceiling).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
T0, DAY = e13.TRAIN_START, e13.DAY

eu = fit["user"].astype(np.int64); ets = fit["ts"].astype(np.int64)
ylab = fit["label"].reshape(-1, 10).max(1)
nu = int(max(eu.max(), fit["clk_user"].max())) + 1
n_ev = np.bincount(eu, minlength=nu)
n_clk = np.bincount(eu, weights=ylab, minlength=nu)
first_ev = np.full(nu, np.iinfo(np.int64).max); np.minimum.at(first_ev, eu, ets)
last_ev = np.zeros(nu, np.int64); np.maximum.at(last_ev, eu, ets)

cu, cts = fit["clk_user"].astype(np.int64), fit["clk_ts"].astype(np.int64)
w = cts < T0                                    # warm-up clicks (visible ones)
wu, wts = cu[w], cts[w]
n_w = np.bincount(wu, minlength=nu)
w_first = np.full(nu, np.iinfo(np.int64).max); np.minimum.at(w_first, wu, wts)

ok = (n_ev >= 20) & (n_w >= 5)
span_days = (np.minimum(first_ev, T0) - w_first) / DAY          # from the oldest visible warm-up click to the start
intensity = (n_w - 1) / np.maximum(span_days, 1e-3)             # warm-up clicks per day
lam = n_ev / 7.0                                                # train events per day
p_train = n_clk / np.maximum(n_ev, 1)
p_warm = intensity / lam

# reliability ceiling: click rate on odd vs even events of the same user
order = np.lexsort((ets, eu)); su, sy = eu[order], ylab[order]
rank_in_user = np.arange(len(su)) - np.searchsorted(su, su, side="left")
odd = rank_in_user % 2 == 1
p_odd = np.bincount(su[odd], weights=sy[odd], minlength=nu) / np.maximum(np.bincount(su[odd], minlength=nu), 1)
p_even = np.bincount(su[~odd], weights=sy[~odd], minlength=nu) / np.maximum(np.bincount(su[~odd], minlength=nu), 1)

def corr(a, b, m): return np.corrcoef(a[m], b[m])[0, 1]
print(f"users with >= 20 train events and >= 5 visible warm-up clicks: {ok.sum()}")
print(f"  corr(train click rate odd events, even events)           {corr(p_odd, p_even, ok):.3f}   (reliability of a half)")
print(f"  corr(warm-up intensity / train event rate, train rate)   {corr(np.log(p_warm + 1e-3), p_train, ok):.3f}")
print(f"  corr(warm-up intensity, train rate)                      {corr(np.log(intensity + 1e-3), p_train, ok):.3f}")
print(f"  corr(train event rate, train rate)                       {corr(np.log(lam), p_train, ok):.3f}")
print(f"  corr(train click intensity, warm-up intensity) (log)     {corr(np.log(n_clk / 7 + 1e-3), np.log(intensity + 1e-3), ok):.3f}")
print(f"  corr(train event rate, warm-up intensity) (log)          {corr(np.log(lam), np.log(intensity + 1e-3), ok):.3f}")
print(f"  median warm-up span {np.median(span_days[ok]):.2f} days, median warm-up clicks/day {np.median(intensity[ok]):.2f}, "
      f"train clicks/day {np.median(n_clk[ok] / 7):.2f}")
