# Run from the repository root: .venv/bin/python experiments/lab/repeat_first_events.py
# Loads experiment 13 and its cutoff-5 caches (first event per user at/after day 5 = the
# evaluation setup) in data/cache.
#
# The other team's file ranks repeat-click candidates (already in the user's history) higher than
# ours. Is our model miscalibrated on repeats in FIRST events after a cutoff (the evaluation
# distribution), while calibrated on the fit events it is trained on? 13's LightGBM (trees by early
# stopping on day 4, refit on days 2-4), predicted on the cutoff-5 test events; mean prediction vs
# actual click rate for repeat / non-repeat candidates, and the log-odds shift for repeats that
# maximises AUC (scanned; reported on the test events, so a diagnostic, not a setting).
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
_, Xt, _ = e13.feature_matrix(test, mf_test, np.arange(len(test["item"])))
pt, yt = model.predict(Xt), test["label"]
rows, Xf, _ = e13.feature_matrix(fit, mf_fit, np.flatnonzero(np.repeat(fit["day"], 10) == 4))
pf, yf = model.predict(Xf), fit["label"][rows]
ri = e13.CAND_FEATURES.index("rep_any")
for label, p, y, rep in [("fit day 4 (all events, in-sample)", pf, yf, fit["cand"][rows, ri] > 0),
                         ("cutoff-5 test (first events)", pt, yt, test["cand"][:, ri] > 0)]:
    print(f"{label}: AUC {auc(p, y):.5f}")
    for name, k in [("repeat", rep), ("not repeat", ~rep)]:
        print(f"  {name:10s} {k.mean():6.2%} of rows  mean pred {p[k].mean():.4f}  actual {y[k].mean():.4f}  "
              f"ratio {p[k].mean() / y[k].mean():.3f}")
lo = np.log(np.clip(pt, 1e-6, 1 - 1e-6) / (1 - np.clip(pt, 1e-6, 1 - 1e-6)))
rep = test["cand"][:, ri] > 0
print("log-odds shift on repeats (test, diagnostic): " + ", ".join(
    f"{s:+.1f} {auc(lo + s * rep, yt):.5f}" for s in [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6]))
