# Run from the repository root: .venv/bin/python experiments/lab/first_event_check.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cutoff-5 caches in data/cache.
#
# The cutoff-5 validation (first event per user at/after day 5, statistics frozen at the cutoff,
# as in evaluation.csv) for LightGBM-level ideas that should matter MORE for first events than for
# day 6 as a whole (user activity: the gap between the user's last activity and the first event
# after the cutoff exists only there). 13's procedure: trees by early stopping on day 4 (fit days
# 2-3), refit on days 2-4; AUC on the test events, overall and by known clicks.
# Fit-period features from lab/user_activity.py's activity().
import importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
sys.argv = sys.argv[:1] + ["--seq", "x"]
src = (ROOT / "experiments/lab/user_activity.py").read_text()
ns = {"e13": e13, "np": np}
exec(src[src.index("def activity"):src.index("ACT = activity")], ns)
activity = ns["activity"]

fit, test, _, _, _ = e13.load_base(str(CACHE / "13_base_c5_a2ecdd59dd03.npz"))
z = np.load(str(CACHE / "13_mf_c5_2889ef80775f.npz")); mf_fit, mf_test = z["mf_fit"], z["mf_test"]
auc = e13.roc_auc
row_day = np.repeat(fit["day"], 10)
rows, X, names = e13.feature_matrix(fit, mf_fit, np.flatnonzero(row_day >= 2))
y, d = fit["label"][rows], row_day[rows]
_, Xt, _ = e13.feature_matrix(test, mf_test, np.arange(len(test["item"])))
yt = test["label"]
Af, At = activity(fit, fit), activity(test, fit)
cols = list(Af)
Xa = np.concatenate([X, np.stack([Af[c][rows // 10] for c in cols], 1)], 1)
Xta = np.concatenate([Xt, np.stack([np.repeat(At[c], 10) for c in cols], 1)], 1)
known = np.repeat(e13.long_history(test, fit, e13.HIST_LONG)[1], 10)
segs = {"all": slice(None), "0 known": known == 0, "1-5": (known >= 1) & (known <= 5),
        "6-20": (known >= 6) & (known <= 20), "21+": known > 20}
for c in cols:
    print(f"  {c:22s} test mean {np.nanmean(At[c]):7.3f}  fit days 2-4 mean {np.nanmean(Af[c][rows[d <= 4] // 10]):7.3f}")

import lightgbm as lgb
def run(label, Xf, Xtt, nm, seeds=(0, 1, 2)):
    es, al = d < 4, d <= 4
    ps = []
    for s in seeds:
        p = {**e13.LGB_PARAMS, "seed": s}
        m = lgb.train(p, lgb.Dataset(Xf[es], y[es], feature_name=nm), 3000,
                      valid_sets=[lgb.Dataset(Xf[d == 4], y[d == 4])], callbacks=[lgb.early_stopping(100, verbose=False)])
        n = max(1, round(m.best_iteration * 3 / 2))
        m = lgb.train(p, lgb.Dataset(Xf[al], y[al], feature_name=nm), n)
        ps.append(m.predict(Xtt))
    pt = np.mean(ps, 0)
    print(f"  {label:28s} " + "  ".join(f"{k} {auc(pt[v], yt[v]):.5f}" for k, v in segs.items()), flush=True)
    return m
print("\ncutoff-5 validation (first events), LightGBM averaged over 3 seeds:")
run("13 features", X, Xt, names)
m = run("+ activity", Xa, Xta, names + cols)
g = sorted(zip(m.feature_importance("gain"), names + cols), reverse=True)
print("    by gain: " + ", ".join(f"{n_} #{i + 1}" for i, (_, n_) in enumerate(g) if n_ in cols))
