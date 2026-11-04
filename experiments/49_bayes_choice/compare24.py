# Selected full-stack comparison on Jan24 (rows before Jan25 only; Jan25 labels stay out). Python from the repo root.
import sys; sys.path.insert(0, "experiments/49_bayes_choice")
import numpy as np, stack as st, data as D
ch, en = D.members()
F3, y3, u3, _ = D.snapshot(3, ch, en)
F4, y4, u4, t4 = D.snapshot(4, ch, en, maxday=5)
print(f"Jan23 half-B events {len(y3):,}; Jan24 half-B events before Jan25 {len(y4):,}")
print("alone on Jan24: ch", round(st.auc(y4.ravel(), F4[:, :, 0].ravel()), 5), "ens", round(st.auc(y4.ravel(), F4[:, :, 2].ravel()), 5),
      *[f"{f} {st.auc(y4.ravel(), F4[:, :, 4 + 2 * i].ravel()):.5f}" for i, f in enumerate(D.FAMS)])
Ft, yt, ut = np.concatenate([F3, F4]), np.concatenate([y3, y4]), np.concatenate([u3, u4])
base = [0, 1, 2, 3]; allc = list(range(F4.shape[2]))
r = st.cv_compare(Ft, yt, ut, F4, y4, u4, base, allc)
print(f"existing stack {r['auc_a']:.5f}  full stack {r['auc_b']:.5f}  gain {r['auc_b'] - r['auc_a']:+.5f}  folds {np.round(r['gains'], 5)} positive {int((r['gains'] > 0).sum())}/5  valid {r['valid']}")
for i, f in enumerate(D.FAMS):
    drop = [c for c in allc if c not in (4 + 2 * i, 5 + 2 * i)]
    q = st.cv_compare(Ft, yt, ut, F4, y4, u4, drop, allc)
    print(f"  {f} on top of the others: {q['auc_b'] - q['auc_a']:+.5f}  folds {np.round(q['gains'], 5)} positive {int((q['gains'] > 0).sum())}/5  valid {q['valid']}")
q = st.cv_compare(Ft, yt, ut, F4, y4, u4, allc[:-1], allc)
print(f"  prof on top of the rest: {q['auc_b'] - q['auc_a']:+.5f}  folds {np.round(q['gains'], 5)} positive {int((q['gains'] > 0).sum())}/5  valid {q['valid']}")
