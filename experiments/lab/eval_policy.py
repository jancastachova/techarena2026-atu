# Run from the repository root: .venv/bin/python experiments/lab/eval_policy.py
# Is the candidate policy of evaluation.csv the same as in training? Earlier checks compared item
# exposure shares only. Here, per event, against the user's known clicks (MF vectors of days 0-5,
# 22_vectors_c7.npz): share of candidates already in the user's clicks (repeats), share re-shown to
# the user, and mean cosine between candidates and the user's taste. Day 6 vs evaluation, both split
# by known clicks so the different user mix does not explain a difference.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
W = np.load(ROOT / "data/cache/22_vectors_c7.npz")["W_6"].astype(np.float64)
Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
rng = np.random.default_rng(0)

# earlier impressions per user (train events only), as sorted keys
fu = fit["user"].astype(np.int64); fts = fit["ts"].astype(np.int64)
imp_u = np.repeat(fu, 10); imp_t = np.repeat(fts, 10); imp_i = fit["item"].astype(np.int64)
o = np.lexsort((imp_t, imp_u)); imp_u, imp_t, imp_i = imp_u[o], imp_t[o], imp_i[o]
ukey = imp_u * 10_000_000_000 + imp_t

def stats(d, evs, name):
    H, n = e13.long_history(d, fit, 100)
    H, n = H[evs], n[evs].astype(int)
    items = d["item"].reshape(-1, 10)[evs].astype(np.int64)
    us, ts = d["user"][evs].astype(np.int64), d["ts"][evs].astype(np.int64)
    e_ = np.searchsorted(ukey, us * 10_000_000_000 + ts, "left"); s_ = np.searchsorted(ukey, us * 10_000_000_000, "left")
    out = {}
    for lo, hi in [(1, 5), (6, 20), (21, 100)]:
        m = np.flatnonzero((n >= lo) & (n <= hi))
        if len(m) > 20000:
            m = rng.choice(m, 20000, replace=False)
        rep, resh, cos, cos_rand = [], [], [], []
        for i in m:
            hs = H[i, :n[i]]
            hset = set(hs.tolist())
            rep.append(np.mean([c in hset for c in items[i]]))
            seen = set(imp_i[s_[i]:e_[i]].tolist())
            resh.append(np.mean([(c in seen) and (c not in hset) for c in items[i]]) if seen else 0.0)
            tv = Wn[hs].mean(0); tv /= max(np.linalg.norm(tv), 1e-9)
            cos.append(float((Wn[items[i]] @ tv).mean()))
            cos_rand.append(float((Wn[rng.integers(1, n_items, 10)] @ tv).mean()))
        out[(lo, hi)] = (len(m), np.mean(rep), np.mean(resh), np.mean(cos), np.mean(cos_rand))
        print(f"{name:10s} known clicks {lo:3d}-{hi:3d}: {len(m):6d} events  repeats {np.mean(rep):.4f}  "
              f"re-shown (not clicked) {np.mean(resh):.4f}  cos(cand, taste) {np.mean(cos):+.4f}  "
              f"random items {np.mean(cos_rand):+.4f}", flush=True)

stats(fit, np.flatnonzero(fit["day"] == 6), "day 6")
# day-6-like reference for "first event after a gap": each user's first day-6 event
d6 = np.flatnonzero(fit["day"] == 6); _, first = np.unique(fit["user"][d6], return_index=True)
stats(fit, d6[first], "day 6 1st")
stats(test, np.arange(len(test["user"])), "eval")
