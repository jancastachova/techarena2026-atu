# Run from the repository root: .venv/bin/python experiments/lab/context_taste.py
# Needs data/cache/22_vectors_c7.npz. Is taste CONTEXT-dependent? Earlier context checks were global
# (click share per time bucket, item click rate per bucket); none asked whether a user's clicks made in
# the same time bucket / day type as the event say more about this event than their other clicks.
# Day 6, MF vectors of days 0-5. For events whose user has >= m known clicks both in the event's
# context and outside it, m clicks are drawn from each side and the user vector is their mean: if
# taste depends on context, the same-context half scores clearly higher than the other half.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
z = np.load(ROOT / "data/cache/22_vectors_c7.npz")
W = z["W_6"].astype(np.float64); evd = z["ev_6"]
items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd]

u, it, ts = fit["clk_user"].astype(np.int64), fit["clk_item"], fit["clk_ts"].astype(np.int64)
order = np.lexsort((it, ts, u)); u, it, ts = u[order], it[order], ts[order]
key = u * 10_000_000_000 + ts
eu, ets = fit["user"][evd].astype(np.int64), fit["ts"][evd].astype(np.int64)
start = np.searchsorted(key, eu * 10_000_000_000, side="left")
end = np.searchsorted(key, eu * 10_000_000_000 + ets, side="left")

def bucket(t): return (t % 86400) // 21600
def daytype(t): return (((t // 86400) + 3) % 7 >= 5).astype(int)   # 1970-01-01 was a Thursday
def hour(t): return (t % 86400) / 3600.0
assert daytype(np.array([e13.TRAIN_START]))[0] == 0 and daytype(np.array([e13.TRAIN_START + 5 * 86400]))[0] == 1

rng = np.random.default_rng(0)
for ctx_name, ctx in [("time bucket", bucket), ("day type", daytype)]:
    for m in (3, 6):
        same_s, oth_s, all_s, ys = [], [], [], []
        for i in range(len(evd)):
            a, b = start[i], end[i]
            if b - a < 2 * m:
                continue
            c_ev = ctx(ets[i:i + 1])[0]
            cc = ctx(ts[a:b]); its = it[a:b]
            s_it, o_it = its[cc == c_ev], its[cc != c_ev]
            if len(s_it) < m or len(o_it) < m:
                continue
            V = W[items[i]]
            same_s.append(V @ W[rng.choice(s_it, m, replace=False)].mean(0))
            oth_s.append(V @ W[rng.choice(o_it, m, replace=False)].mean(0))
            all_s.append(V @ W[its].mean(0))
            ys.append(y[i])
        S, O, A, Y = map(np.concatenate, (same_s, oth_s, all_s, ys))
        print(f"{ctx_name:12s} m={m}: {len(Y) // 10:6d} events  same-context {auc(S, Y):.5f}  "
              f"other-context {auc(O, Y):.5f}  all known clicks {auc(A, Y):.5f}", flush=True)

# Hour-of-day proximity as a soft weight over ALL known clicks (circular distance, kernel width h).
for width in (None, 6.0, 3.0, 1.5):
    sc = np.zeros((len(evd), 10)); ok = np.zeros(len(evd), bool)
    for i in range(len(evd)):
        a, b = start[i], end[i]
        if b == a:
            continue
        dh = np.abs(hour(ts[a:b]) - hour(ets[i:i + 1])[0]); dh = np.minimum(dh, 24 - dh)
        w = np.ones(b - a) if width is None else np.exp(-0.5 * (dh / width) ** 2) + 1e-3
        sc[i] = W[items[i]] @ ((W[it[a:b]] * w[:, None]).sum(0) / w.sum()); ok[i] = True
    print(f"hour kernel {str(width):5s}: AUC {auc(sc[ok].ravel(), y[ok].ravel()):.5f}", flush=True)
