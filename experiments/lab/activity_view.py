# Run from the repository root: .venv/bin/python experiments/lab/activity_view.py
# REOPENED (2026-10-29): user-activity features (lab/user_activity.py; +0.0002 eval-weighted on the 26_ensemble_hops set,
# lab/activity_eval.py) and the scoring day (lab/mf_strength.py; +0.00014 [-0.00009, +0.00034]) were closed as noise on
# older feature sets and are not in SET36. Written as extra-group caches (lab/extra_member.py / 24_ensemble --extra format:
# 4 columns, no NaN; missing values -1) so they can be screened as a bundle:
#   38_act_a  [log known clicks, log clicks 3 d, log clicks 7 d, log span of the known history (h)]
#   38_act_b  [log clicks/day over the last 20, log mean gap (h), log warm-up clicks, log events so far]
#   38_act_c  [train clicks per event so far (shrunk), scoring day (test = 7: unseen days act like the latest), 0, 0]
# All as of the event (clicks and events strictly before it; no labels of the event).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
C = ROOT / "data/cache"
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, _ = e13.load_base(str(C / "13_base_c7_9c47a117c762.npz"))


def activity(dd):
    H = 3600.0
    u, ts = fit["clk_user"].astype(np.int64), fit["clk_ts"].astype(np.int64)
    o = np.lexsort((ts, u)); u, ts = u[o], ts[o]
    key = u * 10_000_000_000 + ts
    eu, et = dd["user"].astype(np.int64), dd["ts"].astype(np.int64)
    base = eu * 10_000_000_000
    s = np.searchsorted(key, base, "left"); e = np.searchsorted(key, base + et, "left")
    n = e - s
    def count_since(sec):
        return e - np.maximum(s, np.searchsorted(key, base + np.maximum(et - sec, 0), "left"))
    first = np.where(n > 0, ts[np.minimum(s, len(ts) - 1)], et)
    k20 = np.minimum(n, 20)
    t20 = np.where(n > 0, ts[np.clip(e - k20, 0, len(ts) - 1)], et)
    span20 = np.maximum((et - t20) / H, 1.0)
    warm = np.maximum(np.searchsorted(key, base + e13.TRAIN_START, "left") - s, 0)
    fu, fts = fit["user"].astype(np.int64), fit["ts"].astype(np.int64)
    fo = np.lexsort((fts, fu)); fkey = fu[fo] * 10_000_000_000 + fts[fo]
    n_ev = np.searchsorted(fkey, base + et, "left") - np.searchsorted(fkey, base, "left")
    cols = [np.log1p(n), np.log1p(count_since(3 * 86400)), np.log1p(count_since(7 * 86400)), np.log1p((et - first) / H),
            np.log(k20 / span20 * 24 + 1e-3),
            np.where(k20 > 1, np.log1p((et - t20) / H / np.maximum(k20 - 1, 1)), -1.0),
            np.log1p(warm), np.log1p(n_ev),
            np.where(n_ev > 0, (n - warm + 0.63 * 3) / (n_ev + 3), -1.0)]
    return np.stack(cols, 1).astype(np.float32)


Af, At = activity(fit), activity(test)
day_f = fit["day"].astype(np.float32)[:, None]; day_t = np.full((len(test["user"]), 1), 7.0, np.float32)
Af = np.concatenate([Af, day_f, np.zeros((len(Af), 2), np.float32)], 1)
At = np.concatenate([At, day_t, np.zeros((len(At), 2), np.float32)], 1)
for k, sl in (("a", slice(0, 4)), ("b", slice(4, 8)), ("c", slice(8, 12))):
    mf_fit, mf_test = np.repeat(Af[:, sl], 10, 0), np.repeat(At[:, sl], 10, 0)
    np.savez(C / f"38_act_{k}_c7.npz", mf_fit=mf_fit, mf_test=mf_test)
    print(f"wrote data/cache/38_act_{k}_c7.npz", mf_fit.shape, mf_test.shape,
          "fit means", np.round(mf_fit[np.repeat(fit['day'], 10) == 6].mean(0), 3), "test means", np.round(mf_test.mean(0), 3))
