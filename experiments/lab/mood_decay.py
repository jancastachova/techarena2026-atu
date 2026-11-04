# Run from the repository root: .venv/bin/python experiments/lab/mood_decay.py
# 2026-11-03 forensic follow-up. lab/residuals.py found a per-user residual shared by events of the SAME day (+0.052) but not
# across days (+0.015), and called it a same-day "mood" the first-event test cannot use. That holds only if the mood resets
# at UTC midnight. If it decays with TIME instead, an evaluation event a few hours after the user's last training event
# (late day 6) would share it, and the residual of those recent labelled events would be a legal feature.
# Test on training events (labels of the fit period only): for pairs of consecutive events of a user, the correlation of
# their "event has a click" residuals, where the user's level is the click share on OTHER calendar days (so a per-day mood
# is not subtracted away), by the gap between the two events and by whether the pair crosses UTC midnight.
# Confounds that are continuous in the gap (a click lowers the next event's click share, repeats) cannot make a step at
# midnight, so same-day vs cross-midnight at the same gap isolates a calendar-day effect.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
y = fit["label"].reshape(-1, 10).max(1).astype(float)
u, ts = fit["user"].astype(np.int64), fit["ts"]
cday = (ts - e13.TRAIN_START) // e13.DAY                         # UTC calendar day 0..6
o = np.lexsort((ts, u)); u, ts, y, cday = u[o], ts[o], y[o], cday[o]
nu = u.max() + 1
S = np.zeros((nu, 7)); N = np.zeros((nu, 7))
np.add.at(S, (u, cday), y); np.add.at(N, (u, cday), 1)
St, Nt = S.sum(1), N.sum(1)

same_user = u[1:] == u[:-1]
i0 = np.flatnonzero(same_user); i1 = i0 + 1
d0, d1 = cday[i0], cday[i1]
uu = u[i0]
excl_S = S[uu, d0] + np.where(d1 != d0, S[uu, d1], 0)
excl_N = N[uu, d0] + np.where(d1 != d0, N[uu, d1], 0)
nn = Nt[uu] - excl_N
ok = nn >= 5
r = np.where(ok, (St[uu] - excl_S) / np.maximum(nn, 1), np.nan)
r = np.clip(r, 0.02, 0.98)
a, b = y[i0] - r, y[i1] - r
gap_h = (ts[i1] - ts[i0]) / 3600
cross = d1 != d0
edges = [0, 0.5, 1, 2, 4, 8, 16, 24, 48, 96, 1e9]
lab = ["<0.5h", "0.5-1h", "1-2h", "2-4h", "4-8h", "8-16h", "16-24h", "1-2d", "2-4d", "4d+"]
gb = np.searchsorted(edges, gap_h, side="right") - 1


vr = r * (1 - r)                                                 # binomial variance at the user's level
er = vr / np.maximum(nn, 1)                                      # variance of the level's own estimate, shared by a and b


def corr(m):
    """Correlation of the two residuals after removing the part both share through the error of the estimated level."""
    if m.sum() < 300:
        return "      -      "
    c = (np.mean(a[m] * b[m]) - np.mean(er[m])) / np.mean(vr[m])
    se = np.std(a[m] * b[m]) / np.sqrt(m.sum()) / np.mean(vr[m])
    return f"{c:+.4f}±{se:.4f} n{m.sum():>6d}"


print(f"{ok.sum():,} consecutive event pairs with >= 5 events on other days; event click share {y.mean():.3f}")
print("residual correlation of consecutive events (level = click share on other days, its estimation error removed), "
      "same UTC day vs across midnight:")
print(f"  {'gap':8s} {'same day':>26s} {'across midnight':>26s}")
for k, nm in enumerate(lab):
    m = ok & (gb == k)
    print(f"  {nm:8s} {corr(m & ~cross):>26s} {corr(m & cross):>26s}")
# split by whether the earlier event had a click (satiation / repeat confounds act through it)
print("same, split by the earlier event's outcome (mean residual of the LATER event, same day / across midnight):")
for k, nm in enumerate(lab):
    m = ok & (gb == k)
    cells = []
    for prev in (1, 0):
        for cr in (False, True):
            mm = m & (y[i0] == prev) & (cross == cr)
            cells.append(f"{b[mm].mean():+.4f} n{mm.sum():>6d}" if mm.sum() >= 300 else f"{'-':>13s}")
    print(f"  {nm:8s} prev click: {cells[0]} / {cells[1]}   prev no click: {cells[2]} / {cells[3]}")
