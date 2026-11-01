# Run from the repository root: .venv/bin/python experiments/lab/predictability_view.py
# PER-USER PREDICTABILITY AS A VIEW (lab/user_predictability.py: users whose past clicks stood out from their events' other 9
# candidates under their own taste stay more predictable on day 6, a little beyond coherence). For fit day d the statistic
# uses only events before day d and the item vectors W_d of an MF trained on days < d (22's vectors, 1 seed, projected on
# the 8 strong directions, unit); the evaluation rows use W_test and all fit events. Columns (lab/extra_member.py format):
#   0  trust-weighted cosine: projected cosine(user taste, candidate) x the user's shrunk margin / global mean margin
#   1  the user's shrunk margin (clicked item's cosine minus the mean of the event's other 9, leave-one-out taste)
#   2  the user's shrunk rank percentile of the clicked item among the 10
#   3  log1p(number of the user's scored clicked events)
# Cache: data/cache/37_predict_c7.npz (mf_fit, mf_test).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
C = ROOT / "data/cache"
fit, test, _, _, _ = e13.load_base(str(C / "13_base_c7_9c47a117c762.npz"))
V = np.load(C / "22_vectors_c7.npz")
def unit(A): return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
def proj(W):
    W = W.astype(np.float64); _, U = np.linalg.eigh(W[1:].T @ W[1:]); Z = unit(W @ U[:, ::-1][:, :8]); Z[0] = 0
    return Z
items = fit["item"].reshape(-1, 10).astype(np.int64); lab = fit["label"].reshape(-1, 10); eu = fit["user"].astype(np.int64)
NU = int(max(eu.max(), fit["clk_user"].max(), test["user"].max())) + 1
H, n = e13.long_history(fit, fit, 100)
Ht, nt = e13.long_history(test, fit, 100)
K = 5.0
def user_stats(Z, upto_ts, ev_mask):
    k = fit["clk_ts"] < upto_ts
    cu, ci = fit["clk_user"][k].astype(np.int64), fit["clk_item"][k].astype(np.int64)
    P = np.zeros((NU, 8)); np.add.at(P, cu, Z[ci]); npos = np.bincount(cu, minlength=NU)
    tr = np.flatnonzero(ev_mask & (lab.max(1) > 0))
    c = items[tr][lab[tr] > 0]; u = eu[tr]
    ok = npos[u] >= 2; tr, c, u = tr[ok], c[ok], u[ok]
    th = unit(P[u] - Z[c])
    cs = np.einsum("ed,ekd->ek", th, Z[items[tr]]); cc = np.einsum("ed,ed->e", th, Z[c])
    margin = cc - (cs.sum(1) - cc) / 9; pct = (cs < cc[:, None]).sum(1) / 9.0
    cnt = np.bincount(u, minlength=NU)
    m0, p0 = margin.mean(), pct.mean()
    mg = (np.bincount(u, margin, NU) + K * m0) / (cnt + K)
    pc = (np.bincount(u, pct, NU) + K * p0) / (cnt + K)
    return mg / m0, pc, np.log1p(cnt)
def rows_for(Z, Hh, nn, it, users, stats):
    mg, pc, lc = stats
    msk = np.arange(100)[None] < nn[:, None]
    S = (Z[Hh.astype(np.int64)] * msk[:, :, None]).sum(1)
    cos = np.einsum("ed,ekd->ek", unit(S), Z[it])
    out = np.zeros((len(users), 10, 4), np.float32)
    out[..., 0] = cos * mg[users][:, None]
    out[..., 1] = mg[users][:, None]; out[..., 2] = pc[users][:, None]; out[..., 3] = lc[users][:, None]
    return out.reshape(-1, 4)
mf_fit = np.full((len(fit["label"]), 4), np.nan, np.float32)
for d in range(2, 7):
    Z = proj(V[f"W_{d}"])
    st = user_stats(Z, e13.TRAIN_START + d * e13.DAY, fit["day"] < d)
    e = np.flatnonzero(fit["day"] == d)
    mf_fit[(e[:, None] * 10 + np.arange(10)).ravel()] = rows_for(Z, H[e], n[e].astype(int), items[e], eu[e], st)
    print(f"day {d}: mean shrunk margin ratio {st[0][eu[e]].mean():.3f}", flush=True)
mf_fit[np.isnan(mf_fit)] = 0                         # days 0-1 are never used (LightGBM fits days >= 2)
Z = proj(V["W_test"])
st = user_stats(Z, e13.TRAIN_START + 7 * e13.DAY, np.ones(len(eu), bool))
mf_test = rows_for(Z, Ht, nt.astype(int), test["item"].reshape(-1, 10).astype(np.int64), test["user"].astype(np.int64), st)
np.savez(C / "37_predict_c7.npz", mf_fit=mf_fit, mf_test=mf_test)
print("wrote data/cache/37_predict_c7.npz", mf_fit.shape, mf_test.shape)
