# Greedy selection of point-estimate variants on top of the 49d inputs, JAN23 ONLY; the selected set is tested once on Jan24.
import sys; sys.path.insert(0, "experiments/49_bayes_choice")
from pathlib import Path
import numpy as np, stack as st, data as D
seeds = tuple(range(16)); P = Path("data/cache/49")
names = sorted({f.name[1:].rsplit("_c", 1)[0] for f in P.glob("G*_c3.npz")} & {f.name[1:].rsplit("_c", 1)[0] for f in P.glob("G*_c4.npz")} - {"c16", "gau"})
ch, en = D.members()
def lab(pre):
    c = np.full((len(D.DAY), 10), np.nan)
    for d in (3, 4):
        a = D._lab(f"data/cache/49/{pre}_d{d}.npz"); m = D.DAY == d; c[m] = a[m]
    return c
cm, cc = lab("catm"), lab("chcat")
f2 = lambda p: st.family_feats(p, np.clip(1 - p.sum(1), 1e-3, 1))
def snap(c, **k):
    fam, _ = D.family(c, fams=("F1", "F2"), seeds=seeds); _, b = D.family(c, fams=("F3",), seeds=(0,))
    t = b["tgt_idx"]; m = ~np.isnan(ch[t, 0]) & ~np.isnan(cm[t, 0]) & ~np.isnan(cc[t, 0]) & (D.DAY[t] < k.get("maxday", 99))
    F = D.features({f: (v[0][m], v[1][m]) for f, v in fam.items()}, None, ch[t][m], en[t][m], fams=("F1", "F2"), prof=False)
    Z = [np.load(P / f"G{n}_c{c}.npz") for n in ["c16", "gau"] + names]
    f5 = [np.load(P / f"F5_s{s}_c{c}.npz") for s in range(4)]
    F5 = st.family_feats(np.mean([z["P"] for z in f5], 0)[m], np.mean([z["P0"] for z in f5], 0)[m])
    return np.concatenate([F, f2(cm[t][m]), f2(cc[t][m])] + [st.family_feats(z["P"][m], z["P0"][m]) for z in Z] + [F5], 2), b["y"][m].astype(float), b["user"][m].astype(int)
a, b = snap(3), snap(4, maxday=5)
Ft, yt, ut = [np.concatenate([a[i], b[i]]) for i in range(3)]; F4, y4, u4 = b; F3, y3, u3 = a
allv = names + ["F5"]; c0 = 16
print("Jan23 alone:", {n: round(float(st.auc(y3.ravel(), F3[:, :, c0 + 2 * i].ravel())), 4) for i, n in enumerate(allv)}, flush=True)
cur = list(range(16)); chosen = []
for step in range(5):
    res = {}
    for i, n in enumerate(allv):
        if n in chosen: continue
        q = st.cv_compare(F3, y3, u3, F3, y3, u3, cur, cur + [c0 + 2 * i, c0 + 1 + 2 * i])
        res[n] = (q["auc_b"] - q["auc_a"], int((q["gains"] > 0).sum()))
    top = sorted(res.items(), key=lambda kv: -kv[1][0])[:6]
    print(f"step {step}:", [(k, round(float(v[0]), 5), v[1]) for k, v in top], flush=True)
    best = top[0][0]
    if res[best][0] < 0.0001 or res[best][1] < 4: break
    chosen.append(best); i = allv.index(best); cur += [c0 + 2 * i, c0 + 1 + 2 * i]
print("chosen on Jan23:", chosen, flush=True)
if chosen:
    r = st.cv_compare(Ft, yt, ut, F4, y4, u4, list(range(16)), cur)
    print(f"Jan24 once: {r['auc_a']:.5f} -> {r['auc_b']:.5f} {r['auc_b'] - r['auc_a']:+.5f} folds {np.round(r['gains'], 5)} pos {int((r['gains'] > 0).sum())}/5 valid {r['valid']}")
