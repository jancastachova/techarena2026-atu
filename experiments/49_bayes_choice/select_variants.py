# Step 1: choice member with category columns, rule on Jan24. Step 2: point-estimate variants, greedy forward selection on JAN23 ONLY
# (whole-user folds inside Jan23). The selected set is then tested once on Jan24.
import sys; sys.path.insert(0, "experiments/49_bayes_choice")
import numpy as np, stack as st, data as D
seeds = tuple(range(16)); NAMES = "K16 K32 K12 pr3 k12p pw05 pw2 c16 c0 gau y1".split()
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
    Ms = [np.load(f"data/cache/49/{g}_s0_c{c}.npz") for g in ("M1", "M2")] + [np.load(f"data/cache/49/G{n}_c{c}.npz") for n in NAMES]
    cols = [F, f2(cm[t][m]), f2(cc[t][m])] + [st.family_feats(z["P"][m], z["P0"][m]) for z in Ms]
    return np.concatenate(cols, 2), b["y"][m].astype(float), b["user"][m].astype(int)
a, b = snap(3), snap(4, maxday=5)
Ft, yt, ut = [np.concatenate([a[i], b[i]]) for i in range(3)]; F4, y4, u4 = b; F3, y3, u3 = a
print("alone Jan24: chcat", round(st.auc(y4.ravel(), F4[:, :, 10].ravel()), 5), "catm", round(st.auc(y4.ravel(), F4[:, :, 8].ravel()), 5), flush=True)
base = list(range(10))
r = st.cv_compare(Ft, yt, ut, F4, y4, u4, base, base + [10, 11])
print(f"STEP1 Jan24 + chcat: {r['auc_a']:.5f} -> {r['auc_b']:.5f} {r['auc_b'] - r['auc_a']:+.5f} folds {np.round(r['gains'], 5)} pos {int((r['gains'] > 0).sum())}/5 valid {r['valid']}", flush=True)
names = ["M1", "M2"] + NAMES; cur = list(range(12)); chosen = []
print("Jan23 alone:", {n: round(st.auc(y3.ravel(), F3[:, :, 12 + 2 * i].ravel()), 4) for i, n in enumerate(names)}, flush=True)
for step in range(4):
    res = {}
    for i, n in enumerate(names):
        if n in chosen: continue
        c = cur + [12 + 2 * i, 13 + 2 * i]
        q = st.cv_compare(F3, y3, u3, F3, y3, u3, cur, c)
        res[n] = (q["auc_b"] - q["auc_a"], int((q["gains"] > 0).sum()))
    best = max(res, key=lambda k: res[k][0])
    print(f"step {step}:", {k: (round(v[0], 5), v[1]) for k, v in sorted(res.items(), key=lambda kv: -kv[1][0])[:5]}, flush=True)
    if res[best][0] < 0.0001 or res[best][1] < 4: break
    chosen.append(best); i = names.index(best); cur += [12 + 2 * i, 13 + 2 * i]
print("chosen on Jan23:", chosen, flush=True)
if chosen:
    r = st.cv_compare(Ft, yt, ut, F4, y4, u4, list(range(12)), cur)
    print(f"Jan24 once: {r['auc_a']:.5f} -> {r['auc_b']:.5f} {r['auc_b'] - r['auc_a']:+.5f} folds {np.round(r['gains'], 5)} pos {int((r['gains'] > 0).sum())}/5 valid {r['valid']}")
