# Final stacker on the legal Jan23, Jan24 (before Jan25) and Jan25 snapshots, applied to the Jan27 evaluation fits.
#   python experiments/49_bayes_choice/final.py "0 1 2 3" submissions/49_bayes_stack.csv
import sys; sys.path.insert(0, "experiments/49_bayes_choice")
import numpy as np, pandas as pd, stack as st, data as D
seeds = tuple(int(s) for s in sys.argv[1].split()); out = sys.argv[2]
ch, en = D.members()
S = lambda c, **k: D.snapshot(c, ch, en, seeds=seeds, fams=("F1", "F2", "F3"), **k)
keep = list(range(8))
A = [S(3), S(4, maxday=5), S(5, maxday=6)]   # day-6 rows of Jan25 come from members fit after the cutoff: left out
ids = np.concatenate([a[3] for a in A]); assert len(ids) == len(set(ids)), "overlapping target events"
F, y = np.concatenate([a[0] for a in A])[:, :, keep], np.concatenate([a[1] for a in A])
m = st.fit(F, y); print("stacker:", m["ok"], m["nit"], "gmax", round(m["gmax"], 3), m["msg"]); assert m["ok"]
print("beta", np.round(m["b"], 3), "intercept", round(m["c"], 3))
# evaluation inputs, in file order
base = pd.read_csv("submissions/47s6_spec_choice.csv"); rid = pd.read_csv("data/evaluation.csv", usecols=["rowid", "event_id"])
assert (base.rowid.values == rid.rowid.values).all(); ev = rid.event_id.values
assert (ev[::10] == ev[9::10]).all() and len(set(ev[::10])) == len(ev) // 10
Z = np.load("data/cache/dataset_raw.npz"); assert (Z["te_ev"] == np.repeat(np.arange(len(Z["te_user"])), 10)).all()
B = "data/cache/box/24_choice_full_"
chE = np.mean([np.load(f"{B}{t}_s{s}_test.npy") for t in ("sp", "d45") for s in range(6)], 0).reshape(-1, 10)
enE = D.qmap(base.predicted_probability.values.reshape(-1, 10), chE)
fam, b0 = D.family(7, fams=("F1", "F2", "F3"), seeds=seeds)
assert (b0["tgt_idx"] == np.arange(len(chE))).all() and (b0["user"] == Z["te_user"]).all()
Fe = D.features(fam, b0, chE, enE, fams=("F1", "F2", "F3"))[:, :, keep]
p = st.predict(m, Fe)
pd.DataFrame(dict(rowid=rid.rowid.values, predicted_probability=p.ravel())).to_csv(out, index=False)
from scipy.stats import spearmanr
print("wrote", out, "mean event sum", round(p.sum(1).mean(), 4), "max", round(p.sum(1).max(), 4), "Spearman vs 47s6", round(spearmanr(p.ravel(), base.predicted_probability.values)[0], 4))
