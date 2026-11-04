# Does the category tree member add to the candidate? Jan24 rows before Jan25, whole-user folds, the rule: >= +0.0001, 4 of 5 folds, valid fits.
#   python experiments/49_bayes_choice/catcheck.py "0 1 2 3 4 5 6 7"
import sys; sys.path.insert(0, "experiments/49_bayes_choice")
import numpy as np, stack as st, data as D
seeds = tuple(int(s) for s in sys.argv[1].split())
ch, en = D.members()
cm = np.full((len(D.DAY), 10), np.nan)
for d in (3, 4):
    a = D._lab(f"data/cache/49/catm_d{d}.npz"); m = D.DAY == d; cm[m] = a[m]


def snap(c, **k):
    fam, _ = D.family(c, fams=("F1", "F2"), seeds=seeds); _, b = D.family(c, fams=("F3",), seeds=(0,))
    t = b["tgt_idx"]; m = ~np.isnan(ch[t, 0]) & ~np.isnan(cm[t, 0]) & (D.DAY[t] < k.get("maxday", 99))
    F = D.features({f: (v[0][m], v[1][m]) for f, v in fam.items()}, dict(prof=b["prof"][m]), ch[t][m], en[t][m], fams=("F1", "F2"), prof=False)
    p = cm[t][m]
    return np.concatenate([F, st.family_feats(p, np.clip(1 - p.sum(1), 1e-3, 1))], 2), b["y"][m].astype(float), b["user"][m].astype(int)


a, b = snap(3), snap(4, maxday=5)
Ft, yt, ut = [np.concatenate([a[i], b[i]]) for i in range(3)]; F4, y4, u4 = b
print(f"Jan24 events {len(y4):,}; alone: ch {st.auc(y4.ravel(), F4[:, :, 0].ravel()):.5f} old ens {st.auc(y4.ravel(), F4[:, :, 2].ravel()):.5f} cat tree {st.auc(y4.ravel(), F4[:, :, 8].ravel()):.5f}")
cand = list(range(8)); full = list(range(10))
r = st.cv_compare(Ft, yt, ut, F4, y4, u4, cand, full)
print(f"candidate {r['auc_a']:.5f} + cat tree {r['auc_b']:.5f} gain {r['auc_b'] - r['auc_a']:+.5f} folds {np.round(r['gains'], 5)} positive {int((r['gains'] > 0).sum())}/5 valid {r['valid']}")
