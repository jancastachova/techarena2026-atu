# Run from the repository root: .venv/bin/python experiments/lab/last_event_view.py
# LAST-EVENT ACTIVITY (2026-10-31). DATASET.md: the gap since the user's last EVENT does not change a user's own click
# behaviour but "changes who is in the set" -- inactive users click less per event (event click rate 0.65 -> 0.54 over the
# gap buckets, lab/gap_profile.py), which is the cross-event part of the global AUC. The act_a/b/c views count CLICKS and
# events so far; nothing measures time since the last event (clicked or not) or recent visit counts. Evaluation knows it
# exactly: its event is the user's first after the cutoff, so the last event is the last training event.
#   43_act_d  [log hours since the user's last event (-1 none), log events in 24 h, log events in 7 d,
#              no-click streak: previous consecutive events without a click (capped 20; -1 none)]
# As of the event: earlier events only (their clicks are in later histories, so their labels are inputs); not the event's own.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); C = ROOT / "data/cache"
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, _ = e13.load_base(str(C / "13_base_c7_9c47a117c762.npz"))
fu, fts = fit["user"].astype(np.int64), fit["ts"].astype(np.int64)
clk = fit["label"].reshape(-1, 10).sum(1) > 0
o = np.lexsort((fts, fu)); su, sts, sc = fu[o], fts[o], clk[o]
key = su * 10_000_000_000 + sts
pos = np.arange(len(o)); lcp = np.maximum.accumulate(np.where(sc, pos, -1))
def view(dd):
    eu, et = dd["user"].astype(np.int64), dd["ts"].astype(np.int64); base = eu * 10_000_000_000
    s = np.searchsorted(key, base, "left"); e = np.searchsorted(key, base + et, "left"); n = e - s
    prev = np.clip(e - 1, 0, len(o) - 1)
    gap = np.where(n > 0, np.log1p((et - sts[prev]) / 3600.0), -1.0)
    cnt = lambda sec: e - np.maximum(s, np.searchsorted(key, base + np.maximum(et - sec, 0), "left"))
    streak = np.where(n > 0, np.minimum(e - 1 - np.maximum(lcp[prev], s - 1), 20), -1)
    return np.stack([gap, np.log1p(cnt(86400)), np.log1p(cnt(7 * 86400)), streak], 1).astype(np.float32)
Vf, Vt = view(fit), view(test)
mf_fit, mf_test = np.repeat(Vf, 10, 0), np.repeat(Vt, 10, 0)
np.savez(C / "43_act_d_c7.npz", mf_fit=mf_fit, mf_test=mf_test)
d6 = fit["day"] == 6
print("wrote data/cache/43_act_d_c7.npz", mf_fit.shape, mf_test.shape)
print("day-6 means", np.round(Vf[d6].mean(0), 3), " evaluation means", np.round(Vt.mean(0), 3))
y6 = clk[d6]
for j, nm in enumerate(["log gap", "events 24h", "events 7d", "no-click streak"]):
    print(f"  {nm:16s} day-6 event-click AUC {e13.roc_auc(Vf[d6, j], y6.astype(float)):.4f}")
