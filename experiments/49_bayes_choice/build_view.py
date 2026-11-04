# Point-estimate view for member training: row of fit day d from the model fit on days < d (d = 2..6), evaluation rows from the cutoff-7 fit.
import numpy as np
Z = np.load("data/cache/dataset_raw.npz"); T0 = 1737331200
day = (Z["tr_ts"] - T0) // 86400; NE = len(day)
exp = np.load("data/cache/lab_export_lean_d6.npz"); rows = np.flatnonzero(np.repeat(day, 10) >= 2)
assert np.array_equal(exp["y"].astype(np.int8), Z["tr_label"][rows]), "export row order differs"
def feats(P, P0):
    P = np.clip(P, 1e-7, 1); P0 = np.clip(P0, 1e-7, 1); l = np.log(P)
    return np.stack([P, l, l - np.log(P0)[:, None], l - l.mean(1, keepdims=True), np.repeat(np.log(P0)[:, None], 10, 1)], 2).astype(np.float32)
cols = ("p", "logp", "logit0", "rel", "logp0"); names = []; fit = []; test = []
for v in ("gau", "c16"):
    A = np.full((NE, 10, 5), np.nan, np.float32)
    for c in (2, 3, 4, 5, 6):
        z = np.load(f"data/cache/49/V{v}_c{c}.npz"); t = z["tgt_idx"]
        assert (day[t] == c).all() and len(t) == (day == c).sum()
        A[t] = feats(z["P"], z["P0"])
    g = np.load(f"data/cache/49/G{v}_c7.npz")
    fit.append(A.reshape(-1, 5)); test.append(feats(g["P"], g["P0"]).reshape(-1, 5)); names += [f"v_{v}_{c}" for c in cols]
mf_fit, mf_test = np.concatenate(fit, 1), np.concatenate(test, 1)
np.savez("data/cache/49/view.npz", X=mf_fit[rows], X_test=mf_test, names=np.array(names), rows=rows, mf_fit=mf_fit, mf_test=mf_test)
print("view", mf_fit.shape, mf_test.shape, "NaN in lab rows", int(np.isnan(mf_fit[rows]).sum()))
