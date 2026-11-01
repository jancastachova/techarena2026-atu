# Run from the repository root: .venv/bin/python experiments/lab/stale_fix.py
# TRAIN/TEST INCONSISTENCY found by lab/adversarial.py: item_log_pop_decay is a 24 h-decayed click count, refreshed
# from every incoming event's history during the fit period (near real time), but after the cutoff no new clicks
# arrive, so for test events it only decays (median 0.39 on evaluation rows vs 0.67 on day 6; a day-13 event sees
# e^-6 of it). rep_recency (24 h decay of the time since the candidate was last clicked) has the same clock, while
# real repeats follow recency in CLICKS (lab/replica_ppc.py) and first-event repeats are under-predicted (0.91).
# Since nothing updates after the freeze, the value AT THE CUTOFF is exact: f' = log1p(expm1(f) e^{(ts - T)/tau})
# for pop, r' = r e^{(ts - T)/tau} for rep_recency. Cutoff-5 validation (first event per user at/after day 5,
# stats frozen at day 5, test events 0-2 days stale; real labels): experiment 13's LightGBM trained once, test
# scored with the original and the frozen features.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
CUT = 5
fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c5_a2ecdd59dd03.npz"))
z = np.load(ROOT / "data/cache/13_mf_c5_2889ef80775f.npz")
mf_fit, mf_test = z["mf_fit"], z["mf_test"]
T = e13.TRAIN_START + CUT * e13.DAY
i_pop, i_rep = e13.CAND_FEATURES.index("item_log_pop_decay"), e13.CAND_FEATURES.index("rep_recency")
age = np.repeat((test["ts"].astype(np.float64) - T), 10)          # seconds since the freeze, per row
grow = np.exp(age / e13.TAU_POP)
model = e13.lgbm_procedure(fit, mf_fit, CUT)
rows = np.arange(len(test["item"]))
y = test["label"]; tday = np.repeat(test["day"], 10)
def score(cand):
    t2 = dict(test); t2["cand"] = cand
    _, X, _ = e13.feature_matrix(t2, mf_test, rows)
    return model.predict(X)
c0 = test["cand"]
c_pop = c0.copy(); c_pop[:, i_pop] = np.log1p(np.expm1(c0[:, i_pop]) * grow)
c_rep = c0.copy(); c_rep[:, i_rep] = np.minimum(c0[:, i_rep] * np.exp(age / e13.TAU_REPEAT), 1.0)
c_both = c_pop.copy(); c_both[:, i_rep] = c_rep[:, i_rep]
rep = c0[:, e13.CAND_FEATURES.index("rep_any")] > 0
print(f"test events {len(test['user'])}; rows by test day: " + ", ".join(f"day {d}: {np.mean(tday == d):.3f}" for d in np.unique(tday)))
print(f"pop_decay median: fit day 4 {np.median(fit['cand'][np.repeat(fit['day'] == 4, 10), i_pop]):.3f}, "
      f"test {np.median(c0[:, i_pop]):.3f}, test frozen {np.median(c_pop[:, i_pop]):.3f}")
base = None
for name, c in [("original", c0), ("pop frozen at cutoff", c_pop), ("rep_recency frozen", c_rep), ("both", c_both)]:
    p = score(c)
    if base is None:
        base = p
    parts = "  ".join(f"day {d} {e13.roc_auc(p[tday == d], y[tday == d]):.5f}" for d in np.unique(tday))
    print(f"{name:22s} AUC {e13.roc_auc(p, y):.5f}  ({parts})  repeats pred/actual {p[rep].mean() / y[rep].mean():.3f}", flush=True)
    if name != "original":
        evs = np.arange(len(test["user"])); rs = np.random.default_rng(0); diffs = []
        for _ in range(200):
            wt = np.repeat(np.bincount(rs.integers(0, len(evs), len(evs)), minlength=len(evs)), 10).astype(float)
            m_ = wt > 0
            diffs.append(e13.roc_auc(np.repeat(p[m_], wt[m_].astype(int)), np.repeat(y[m_], wt[m_].astype(int)))
                         - e13.roc_auc(np.repeat(base[m_], wt[m_].astype(int)), np.repeat(y[m_], wt[m_].astype(int))))
        diffs = np.array(diffs)
        print(f"{'':22s} vs original {diffs.mean():+.5f}  95% [{np.percentile(diffs, 2.5):+.5f}, {np.percentile(diffs, 97.5):+.5f}]", flush=True)
