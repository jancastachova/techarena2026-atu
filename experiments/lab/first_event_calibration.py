# Run from the repository root: .venv/bin/python experiments/lab/first_event_calibration.py
# Loads experiment 13 and its cutoff-5 caches (first event per user at/after day 5, as in
# evaluation.csv). DIAGNOSTIC only (validation test events are never used to choose settings).
#
# Global AUC compares events of different users, so a segment whose events are systematically
# over- or under-predicted costs AUC. RESULTS.md found the model calibrated across segments on
# ALL day-6 events; here the same check on FIRST events after a cutoff: mean predicted event click
# probability vs the actual share of events with a click, by known clicks, hours since the user's
# last click, whether the user had train events, and event day.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, _ = e13.load_base(str(CACHE / "13_base_c5_a2ecdd59dd03.npz"))
z = np.load(str(CACHE / "13_mf_c5_2889ef80775f.npz")); mf_fit, mf_test = z["mf_fit"], z["mf_test"]
auc = e13.roc_auc
model = e13.lgbm_procedure(fit, mf_fit, 5)
_, Xt, names = e13.feature_matrix(test, mf_test, np.arange(len(test["item"])))
p = model.predict(Xt).reshape(-1, 10); y = test["label"].reshape(-1, 10)
pe, ye = 1 - np.prod(1 - p, 1), y.max(1)
print(f"first events: {len(pe):,}, AUC {auc(p.ravel(), y.ravel()):.5f}, event-level AUC {auc(pe, ye):.5f}, "
      f"click share {ye.mean():.3f}, mean predicted {pe.mean():.3f}")
_, hl = e13.long_history(test, fit, e13.HIST_LONG)
gap = Xt[::10, names.index("log_hours_since_last")]
newu = ~np.isin(test["user"], np.unique(fit["user"]))
def table(label, keys, groups):
    print(label)
    for g in groups:
        k = keys == g
        if k.sum() > 50:
            print(f"  {str(g):>12s}: {k.mean():6.1%} of events  actual {ye[k].mean():.3f}  predicted {pe[k].mean():.3f}  "
                  f"ratio {pe[k].mean() / ye[k].mean():.3f}")
hb = np.digitize(hl, [1, 6, 21, 51]); table("known clicks (0, 1-5, 6-20, 21-50, 51+)", hb, range(5))
gb = np.digitize(np.expm1(gap), [6, 24, 72, 168]); table("hours since last click (<6, 6-24, 24-72, 72-168, 168+)", gb, range(5))
table("user without train events before", newu.astype(int), [0, 1])
table("event day", test["day"], np.unique(test["day"]))
# fit-period reference: the same on fit day 4 events (all events, in-sample-ish)
rows, Xf, _ = e13.feature_matrix(fit, mf_fit, np.flatnonzero(np.repeat(fit["day"], 10) == 4))
pf = model.predict(Xf).reshape(-1, 10); yf = fit["label"][rows].reshape(-1, 10)
print(f"fit day 4, all events: click share {yf.max(1).mean():.3f}, mean predicted {(1 - np.prod(1 - pf, 1)).mean():.3f}")
