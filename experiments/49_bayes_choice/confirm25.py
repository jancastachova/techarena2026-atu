# Frozen candidate (existing choice + ensemble + F1 + F2), one purged Jan25 confirmation. All meta-training events precede Jan25.
#   python experiments/49_bayes_choice/confirm25.py "0 1 2 3"
import sys; sys.path.insert(0, "experiments/49_bayes_choice")
import numpy as np, stack as st, data as D
seeds = tuple(int(s) for s in sys.argv[1].split()); fams = ("F1", "F2")
ch, en = D.members()
S = lambda c, **k: D.snapshot(c, ch, en, seeds=seeds, fams=("F1", "F2", "F3"), **k)
keep = [0, 1, 2, 3, 4, 5, 6, 7]                                 # ch, ens, F1, F2 (F3 is loaded only for the shared profile fields)
F3_, y3, u3, t3 = S(3); F4_, y4, u4, t4 = S(4, maxday=5); F5_, y5, u5, t5 = S(5)
assert D.DAY[np.r_[t3, t4]].max() < 5 and D.DAY[t5].min() >= 5 and not set(t5) & set(np.r_[t3, t4])
Ft, yt, ut = np.concatenate([F3_, F4_]), np.concatenate([y3, y4]), np.concatenate([u3, u4])
print(f"seeds {seeds}; meta-train events {len(yt):,} (all before Jan25); Jan25 half-B test events {len(y5):,}")
r = st.cv_compare(Ft, yt, ut, F5_, y5, u5, [0, 1, 2, 3], keep)
print(f"Jan25 all rows: existing {r['auc_a']:.5f} candidate {r['auc_b']:.5f} gain {r['auc_b'] - r['auc_a']:+.5f} folds {np.round(r['gains'], 5)} positive {int((r['gains'] > 0).sum())}/5 valid {r['valid']}")
m = D.DAY[t5] == 5
print(f"Jan25 day-5 rows only ({m.sum():,}): existing {st.auc(y5[m].ravel(), r['pa'][m].ravel()):.5f} candidate {st.auc(y5[m].ravel(), r['pb'][m].ravel()):.5f}")
