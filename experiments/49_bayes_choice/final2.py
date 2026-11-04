# 49 + extra stacker members that passed the Jan24 rule. Final stacker on Jan23, Jan24 (before Jan25) and Jan25 day-5 rows, applied to evaluation.
#   python experiments/49_bayes_choice/final2.py "<seeds>" out.csv name=lab_prefix:eval.npz:key [...]
#   lab files: <lab_prefix>_d{3,4,5}.npz with rows_B and predictions; eval file: raw evaluation probabilities under key
import sys; sys.path.insert(0, "experiments/49_bayes_choice")
import numpy as np, pandas as pd, stack as st, data as D
from scipy.stats import spearmanr
seeds = tuple(int(s) for s in sys.argv[1].split()); out = sys.argv[2]; specs = [s.split("=", 1)[1].split(":") for s in sys.argv[3:]]
ch, en = D.members()
ex = []
for pre, _, _ in specs:
    c = np.full((len(D.DAY), 10), np.nan)
    for d in (3, 4, 5):
        a = D._lab(f"{pre}_d{d}.npz"); m = D.DAY == d; c[m] = a[m]
    ex.append(c)
f2 = lambda p: st.family_feats(p, np.clip(1 - p.sum(1), 1e-3, 1))


def snap(c, maxday):
    fam, _ = D.family(c, fams=("F1", "F2"), seeds=seeds); _, b = D.family(c, fams=("F3",), seeds=(0,))
    t = b["tgt_idx"]; m = ~np.isnan(ch[t, 0]) & (D.DAY[t] < maxday)
    for e in ex: m &= ~np.isnan(e[t, 0])
    F = D.features({f: (v[0][m], v[1][m]) for f, v in fam.items()}, None, ch[t][m], en[t][m], fams=("F1", "F2"), prof=False)
    return np.concatenate([F] + [f2(e[t][m]) for e in ex], 2), b["y"][m].astype(float), t[m]


A = [snap(3, 99), snap(4, 5), snap(5, 6)]
ids = np.concatenate([a[2] for a in A]); assert len(ids) == len(set(ids))
F, y = np.concatenate([a[0] for a in A]), np.concatenate([a[1] for a in A])
m = st.fit(F, y); print(f"stacker on {len(y):,} events: ok {m['ok']} iters {m['nit']} gmax {m['gmax']:.3f} {m['msg']}"); assert m["ok"]
print("beta", np.round(m["b"], 3), "intercept", round(m["c"], 3))
base = pd.read_csv("submissions/47s6_spec_choice.csv"); rid = pd.read_csv("data/evaluation.csv", usecols=["rowid", "event_id"])
assert (base.rowid.values == rid.rowid.values).all()
Z = np.load("data/cache/dataset_raw.npz")
B = "data/cache/box/24_choice_full_"
chE = np.mean([np.load(f"{B}{t}_s{s}_test.npy") for t in ("sp", "d45") for s in range(6)], 0).reshape(-1, 10)
enE = D.qmap(base.predicted_probability.values.reshape(-1, 10), chE)
fam, _ = D.family(7, fams=("F1", "F2"), seeds=seeds); _, b0 = D.family(7, fams=("F3",), seeds=(0,))
assert (b0["tgt_idx"] == np.arange(len(chE))).all() and (b0["user"] == Z["te_user"]).all()
Fe = D.features(fam, None, chE, enE, fams=("F1", "F2"), prof=False)
exE = [np.load(f)[k].reshape(-1, 10) for _, f, k in specs]
for (_, f, k), e in zip(specs, exE): print(f, k, "event mean sum", round(e.sum(1).mean(), 4))
Fe = np.concatenate([Fe] + [f2(e) for e in exE], 2)
p = st.predict(m, Fe)
pd.DataFrame(dict(rowid=rid.rowid.values, predicted_probability=p.ravel())).to_csv(out, index=False)
print("wrote", out, "mean event sum", round(p.sum(1).mean(), 4), "max", round(p.sum(1).max(), 4), "Spearman vs 47s6", round(spearmanr(p.ravel(), base.predicted_probability.values)[0], 4))
