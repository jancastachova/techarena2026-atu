# Bayes view: as build_view.py from the F1 and F2 chains, 4 seeds averaged. Fit day d from chains on days < d; evaluation from the cutoff-7 chains.
import numpy as np
Z = np.load("data/cache/dataset_raw.npz"); T0 = 1737331200
day = (Z["tr_ts"] - T0) // 86400; NE = len(day)
rows = np.flatnonzero(np.repeat(day, 10) >= 2)
def feats(P, P0):
    P = np.clip(P, 1e-7, 1); P0 = np.clip(P0, 1e-7, 1); l = np.log(P)
    return np.stack([P, l, l - np.log(P0)[:, None], l - l.mean(1, keepdims=True), np.repeat(np.log(P0)[:, None], 10, 1)], 2).astype(np.float32)
cols = ("p", "logp", "logit0", "rel", "logp0"); names = []; fit = []; test = []
for v in ("F1", "F2"):
    A = np.full((NE, 10, 5), np.nan, np.float32)
    for c in (2, 3, 4, 5, 6):
        zs = [np.load(f"data/cache/49/V{v}_s{s}_c{c}.npz") for s in range(4)]; t = zs[0]["tgt_idx"]
        assert (day[t] == c).all() and len(t) == (day == c).sum()
        A[t] = feats(np.mean([z["P"] for z in zs], 0), np.mean([z["P0"] for z in zs], 0))
    gs = [np.load(f"data/cache/49/{v}_s{s}_c7.npz") for s in range(4)]
    fit.append(A.reshape(-1, 5)); test.append(feats(np.mean([z["P"] for z in gs], 0), np.mean([z["P0"] for z in gs], 0)).reshape(-1, 5)); names += [f"b_{v}_{c}" for c in cols]
mf_fit, mf_test = np.concatenate(fit, 1), np.concatenate(test, 1)
np.savez("data/cache/49/view2.npz", X=mf_fit[rows], X_test=mf_test, names=np.array(names), rows=rows, mf_fit=mf_fit, mf_test=mf_test)
print("view2", mf_fit.shape, mf_test.shape, "NaN in lab rows", int(np.isnan(mf_fit[rows]).sum()))
