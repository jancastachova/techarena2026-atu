# Run from the repository root: .venv/bin/python experiments/lab/boundary_shift.py
# Does taste shift at a PERIOD boundary? Within the train week taste does not drift (lab/mild_decay.py),
# but the leaderboard rewarded sequence (recency-aware) features ~2x their half-B gain, and a simulator
# about "changing user behaviour" could shift preferences between periods -- the evaluation starts at
# one. The warm-up -> train boundary is visible: for day-6 events whose user has >= m warm-up clicks
# AND >= m train-period clicks known, m of each are drawn and the user vector is their mean (MF vectors
# of days 0-5). A boundary shift makes the warm-up half clearly worse. Control for plain age: the m
# OLDEST vs the m NEWEST train-period clicks of users with >= 2m of them.
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
o = np.lexsort((it, ts, u)); u, it, ts = u[o], it[o], ts[o]
key = u * 10_000_000_000 + ts
eu, ets = fit["user"][evd].astype(np.int64), fit["ts"][evd].astype(np.int64)
s = np.searchsorted(key, eu * 10_000_000_000, "left"); e = np.searchsorted(key, eu * 10_000_000_000 + ets, "left")
wend = np.clip(np.searchsorted(key, eu * 10_000_000_000 + e13.TRAIN_START, "left"), s, e)
rng = np.random.default_rng(0)
for m in (3, 6, 10):
    W_s, T_s, A_s, O_s, N_s, Y1, Y2, age_w, age_t = [], [], [], [], [], [], [], [], []
    for i in range(len(evd)):
        nw, nt = wend[i] - s[i], e[i] - wend[i]
        V = W[items[i]]
        if nw >= m and nt >= m:
            wi = rng.choice(np.arange(s[i], wend[i]), m, replace=False)
            ti = rng.choice(np.arange(wend[i], e[i]), m, replace=False)
            W_s.append(V @ W[it[wi]].mean(0)); T_s.append(V @ W[it[ti]].mean(0)); A_s.append(V @ W[it[s[i]:e[i]]].mean(0))
            Y1.append(y[i]); age_w.append((ets[i] - ts[wi]).mean() / 86400); age_t.append((ets[i] - ts[ti]).mean() / 86400)
        if nt >= 2 * m:
            O_s.append(V @ W[it[wend[i]:wend[i] + m]].mean(0)); N_s.append(V @ W[it[e[i] - m:e[i]]].mean(0)); Y2.append(y[i])
    Y1, Y2 = np.concatenate(Y1), np.concatenate(Y2)
    print(f"m={m:2d}: {len(Y1) // 10:6d} events  warm-up clicks {auc(np.concatenate(W_s), Y1):.5f} (mean age {np.mean(age_w):.1f} d)  "
          f"train-period clicks {auc(np.concatenate(T_s), Y1):.5f} (age {np.mean(age_t):.1f} d)  all {auc(np.concatenate(A_s), Y1):.5f}", flush=True)
    print(f"      control {len(Y2) // 10:6d} events  oldest {m} train-period clicks {auc(np.concatenate(O_s), Y2):.5f}  "
          f"newest {m} {auc(np.concatenate(N_s), Y2):.5f}", flush=True)
