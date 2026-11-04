# 49e: the 49d inputs + tree and choice members trained with the point-estimate view (run on Jan24: +0.00104, 5/5). Jan23, Jan24 (before Jan25) and
# Jan25 day-5 rows for the fit; applied to evaluation.   python experiments/49_bayes_choice/final4.py out.csv <chcat eval .npy> <choice+view eval .npy> <tree+view eval .npz> [lab prefix of tree] [lab prefix of choice]
import sys; sys.path.insert(0, "experiments/49_bayes_choice")
import numpy as np, pandas as pd, stack as st, data as D
from scipy.stats import spearmanr
out, chcat_eval, chv_eval, tv_eval = sys.argv[1:5]; TVP = sys.argv[5] if len(sys.argv) > 5 else "tv"; CVP = sys.argv[6] if len(sys.argv) > 6 else "chv"; seeds = tuple(range(16)); GN = ("c16", "gau")
ch, en = D.members()
def lab(pre):
    c = np.full((len(D.DAY), 10), np.nan)
    for d in (3, 4, 5):
        a = D._lab(f"data/cache/49/{pre}_d{d}.npz"); m = D.DAY == d; c[m] = a[m]
    return c
cm, cc, tv, cv = lab("catm"), lab("chcat"), lab(TVP), lab(CVP)
f2 = lambda p: st.family_feats(p, np.clip(1 - p.sum(1), 1e-3, 1))
def snap(c, maxday):
    fam, _ = D.family(c, fams=("F1", "F2"), seeds=seeds); _, b = D.family(c, fams=("F3",), seeds=(0,))
    t = b["tgt_idx"]; m = ~np.isnan(ch[t, 0]) & ~np.isnan(cm[t, 0]) & ~np.isnan(cc[t, 0]) & ~np.isnan(tv[t, 0]) & ~np.isnan(cv[t, 0]) & (D.DAY[t] < maxday)
    F = D.features({f: (v[0][m], v[1][m]) for f, v in fam.items()}, None, ch[t][m], en[t][m], fams=("F1", "F2"), prof=False)
    Gs = [np.load(f"data/cache/49/G{n}_c{c}.npz") for n in GN]
    return np.concatenate([F, f2(cm[t][m]), f2(cc[t][m])] + [st.family_feats(z["P"][m], z["P0"][m]) for z in Gs] + [f2(tv[t][m]), f2(cv[t][m])], 2), b["y"][m].astype(float), t[m]
A = [snap(3, 99), snap(4, 5), snap(5, 6)]
ids = np.concatenate([a[2] for a in A]); assert len(ids) == len(set(ids))
F, y = np.concatenate([a[0] for a in A]), np.concatenate([a[1] for a in A])
m = st.fit(F, y); print(f"stacker on {len(y):,} events: ok {m['ok']} iters {m['nit']} gmax {m['gmax']:.3f} {m['msg']}"); assert m["ok"]
print("beta", np.round(m["b"], 3), "intercept", round(m["c"], 3))
base = pd.read_csv("submissions/47s6_spec_choice.csv"); rid = pd.read_csv("data/evaluation.csv", usecols=["rowid", "event_id"])
assert (base.rowid.values == rid.rowid.values).all()
Z = np.load("data/cache/dataset_raw.npz"); B = "data/cache/box/24_choice_full_"
chE = np.mean([np.load(f"{B}{t}_s{s}_test.npy") for t in ("sp", "d45") for s in range(6)], 0).reshape(-1, 10)
enE = D.qmap(base.predicted_probability.values.reshape(-1, 10), chE)
fam, _ = D.family(7, fams=("F1", "F2"), seeds=seeds); _, b0 = D.family(7, fams=("F3",), seeds=(0,))
assert (b0["tgt_idx"] == np.arange(len(chE))).all() and (b0["user"] == Z["te_user"]).all()
Fe = D.features(fam, None, chE, enE, fams=("F1", "F2"), prof=False)
Gs = [np.load(f"data/cache/49/G{n}_c7.npz") for n in GN]
assert all((g["user"] == Z["te_user"]).all() for g in Gs)
Fe = np.concatenate([Fe, f2(np.load("data/cache/49/catm_eval.npz")["lgb_full"].reshape(-1, 10)), f2(np.load(chcat_eval).reshape(-1, 10))] + [st.family_feats(z["P"], z["P0"]) for z in Gs] + [f2(np.load(tv_eval)["lgb_full"].reshape(-1, 10)), f2(np.load(chv_eval).reshape(-1, 10))], 2)
p = st.predict(m, Fe)
pd.DataFrame(dict(rowid=rid.rowid.values, predicted_probability=p.ravel())).to_csv(out, index=False)
print("wrote", out, "mean event sum", round(p.sum(1).mean(), 4), "max", round(p.sum(1).max(), 4), "Spearman vs 47s6", round(spearmanr(p.ravel(), base.predicted_probability.values)[0], 4))
