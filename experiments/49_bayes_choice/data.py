# Aligns family outputs and existing members per snapshot (2026-11-04). Existing members exist on half B of lab days 3-6 only.
#   ch  = neural choice member, raw probabilities
#   ens = rank mean of tree + choice members, mapped onto the choice member's probability scale (raw tree probabilities were
#         not saved for evaluation, so both sides use this same mapping)
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import stack as st
T0 = 1737331200
FAMS = ("F1", "F2", "F3", "F4")
Zr = np.load("data/cache/dataset_raw.npz")
DAY = (Zr["tr_ts"] - T0) // 86400
OFF = int((DAY < 2).sum()) * 10
LAB = "data/cache/box/lab/"


def _lab(f):
    Z = np.load(f); k = [x for x in Z.files if x != "rows_B"][0]
    a = np.full(len(DAY) * 10, np.nan); a[Z["rows_B"] + OFF] = Z[k]
    return a.reshape(-1, 10)


def qmap(score, ref):
    """Monotone map of score onto the empirical distribution of ref."""
    r = rankdata(score.ravel(), method="ordinal").astype(np.int64) - 1
    return np.sort(ref.ravel())[r].reshape(score.shape)


def members():
    ch = np.full((len(DAY), 10), np.nan); en = ch.copy()
    for d in (3, 4, 5, 6):
        c = _lab(LAB + (f"lab_preds_d3_choice.npz" if d == 3 else f"lab_preds_chC3x10_d{d}.npz"))
        t = _lab(LAB + (f"lab_preds_d3_lgb.npz" if d == 3 else f"lab_preds_boxbase_d{d}.npz"))
        m = (DAY == d) & ~np.isnan(c[:, 0]) & ~np.isnan(t[:, 0])
        ch[m] = c[m]
        n = m.sum() * 10
        en[m] = qmap((rankdata(c[m].ravel()) / n + rankdata(t[m].ravel()) / n).reshape(-1, 10), c[m])
    return ch, en


def family(C, fams=FAMS, seeds=(0, 1), path="data/cache/49"):
    out, base = {}, None
    for F in fams:
        Zs = [np.load(f"{path}/{F}_s{s}_c{C}.npz") for s in seeds if Path(f"{path}/{F}_s{s}_c{C}.npz").exists() or F != "F3"]   # F3: only its prof field is used
        out[F] = (np.mean([z["P"] for z in Zs], 0), np.mean([z["P0"] for z in Zs], 0))
        if F == "F3": base = Zs[0]
    return out, base


def features(fam, base, ch=None, en=None, fams=FAMS, prof=True):
    """Columns: ch(2) ens(2) then 2 per family, then prof."""
    c = []
    f2 = lambda p: st.family_feats(p, np.clip(1 - p.sum(1), 1e-3, 1))
    if ch is not None: c += [f2(ch), f2(en)]
    for F in fams: c.append(st.family_feats(*fam[F]))
    if prof and base is not None: c.append(base["prof"][:, :, None].astype(np.float64))
    return np.concatenate(c, 2)


def snapshot(C, ch, en, maxday=99, **kw):
    fam, base = family(C, **kw)
    t = base["tgt_idx"]
    m = ~np.isnan(ch[t, 0]) & (DAY[t] < maxday)
    famm = {k: (v[0][m], v[1][m]) for k, v in fam.items()}
    bm = dict(prof=base["prof"][m])
    F = features(famm, bm, ch[t][m], en[t][m], fams=tuple(fam))
    return F, base["y"][m].astype(float), base["user"][m].astype(int), t[m]
