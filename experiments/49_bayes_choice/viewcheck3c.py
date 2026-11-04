# Members trained with the point-estimate view, against the 49d inputs. Jan24 rows before Jan25, the rule: +0.0001, 4 of 5 folds, valid fits.
import sys; sys.path.insert(0, "experiments/49_bayes_choice")
from pathlib import Path
import numpy as np, stack as st, data as D
seeds = tuple(range(16)); P = Path("data/cache/49")
ch, en = D.members()
def lab(pre):
    c = np.full((len(D.DAY), 10), np.nan)
    for d in (3, 4):
        a = D._lab(f"data/cache/49/{pre}_d{d}.npz"); m = D.DAY == d; c[m] = a[m]
    return c
cm, cc, tv, cv, tb, cb = lab("catm"), lab("chcat"), lab("tv"), lab("chv"), lab("tb"), lab("chb")
cw = lab("chw"); tw = cw
f2 = lambda p: st.family_feats(p, np.clip(1 - p.sum(1), 1e-3, 1))
def snap(c, **k):
    fam, _ = D.family(c, fams=("F1", "F2"), seeds=seeds); _, b = D.family(c, fams=("F3",), seeds=(0,))
    t = b["tgt_idx"]; m = ~np.isnan(ch[t, 0]) & ~np.isnan(cm[t, 0]) & ~np.isnan(cc[t, 0]) & ~np.isnan(tv[t, 0]) & ~np.isnan(cv[t, 0]) & ~np.isnan(tb[t, 0]) & ~np.isnan(cb[t, 0]) & ~np.isnan(tw[t, 0]) & ~np.isnan(cw[t, 0]) & (D.DAY[t] < k.get("maxday", 99))
    F = D.features({f: (v[0][m], v[1][m]) for f, v in fam.items()}, None, ch[t][m], en[t][m], fams=("F1", "F2"), prof=False)
    Z = [np.load(P / f"G{n}_c{c}.npz") for n in ("c16", "gau")]
    return np.concatenate([F, f2(cm[t][m]), f2(cc[t][m])] + [st.family_feats(z["P"][m], z["P0"][m]) for z in Z] + [f2(tv[t][m]), f2(cv[t][m]), f2(tb[t][m]), f2(cb[t][m]), f2(tw[t][m]), f2(cw[t][m])], 2), b["y"][m].astype(float), b["user"][m].astype(int)
a, b = snap(3), snap(4, maxday=5)
Ft, yt, ut = [np.concatenate([a[i], b[i]]) for i in range(3)]; F4, y4, u4 = b
A = lambda c: st.auc(y4.ravel(), F4[:, :, c].ravel())
A = lambda c: st.auc(y4.ravel(), F4[:, :, c].ravel())
print(f"alone Jan24: tree+bayes {A(20):.5f} choice+bayes {A(22):.5f} | tree+all views {A(24):.5f} choice+all views {A(26):.5f}", flush=True)
base = list(range(24))
def show(n, ca, cb_):
    r = st.cv_compare(Ft, yt, ut, F4, y4, u4, ca, cb_)
    print(f"{n}: {r['auc_a']:.5f} -> {r['auc_b']:.5f} {r['auc_b'] - r['auc_a']:+.5f} folds {np.round(r['gains'], 5)} pos {int((r['gains'] > 0).sum())}/5 valid {r['valid']}", flush=True)
show("49f + choice+all views", base, base + [26, 27])
