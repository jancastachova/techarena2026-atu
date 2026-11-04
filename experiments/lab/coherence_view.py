# Run from the repository root: .venv/bin/python experiments/lab/coherence_view.py
# COHERENCE / TYPICALITY VIEW (2026-11-02; flagged as untested in LightGBM by the tercile check, and asked for by the external
# review as "preference breadth"). From the 10 seeds of 40's vectors projected on their top-8 directions (as 40_cosproj8),
# with u = mean unit vector of the user's known clicks (up to 100) and c_k = cos(history item k, u):
#   col 0  coherence |u| (1 = all clicks in one direction, ~0 = spread out)            [event-level]
#   col 1  std over k of c_k (how unevenly the user's own clicks fit their taste)       [event-level]
#   col 2  typicality of the candidate: (cos(candidate, u) - mean_k c_k) / (std_k c_k + 0.05)
#   col 3  cos(candidate, u) x coherence
# Days 0-1 NaN, empty histories -1 (extra-group cache format of lab/extra_member.py / 24_ensemble --extra).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
EV = e13.EV_SIZE
fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
files = [ROOT / f"data/cache/40_vectors_c7_s{s}.npz" for s in range(10)]


def project(W, k=8):
    C = W[1:].T.astype(np.float64) @ W[1:].astype(np.float64)
    _, U = np.linalg.eigh(C)
    U = U[:, ::-1][:, :k].astype(np.float32)
    return (W @ U) @ U.T


def block(H, n, items, W):
    Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
    m = (np.arange(H.shape[1])[None] < n[:, None]).astype(np.float32)
    Uh = Wn[H] * m[:, :, None]
    u = Uh.sum(1) / np.maximum(m.sum(1), 1)[:, None]
    coh = np.linalg.norm(u, axis=1)
    un = u / np.maximum(coh, 1e-9)[:, None]
    ck = np.einsum("ekd,ed->ek", Uh, un)
    cnt = np.maximum(m.sum(1), 1)
    mu = (ck * m).sum(1) / cnt
    sd = np.sqrt(np.maximum((((ck - mu[:, None]) ** 2) * m).sum(1) / cnt, 0))
    cc = np.einsum("ekd,ed->ek", Wn[items], un)
    out = np.stack([np.repeat(coh[:, None], EV, 1), np.repeat(sd[:, None], EV, 1),
                    (cc - mu[:, None]) / (sd[:, None] + 0.05), cc * coh[:, None]], 2)
    out[n == 0] = -1
    return out


def scores(H, n, items, key):
    out = np.zeros(items.shape + (4,))
    for f in files:
        W = project(np.load(f)[key].astype(np.float32))
        for s in range(0, len(items), 5000):
            sl = slice(s, s + 5000)
            out[sl] += block(H[sl], n[sl].astype(int), items[sl], W) / len(files)
    return out.reshape(-1, 4)


hf, lf = e13.long_history(fit, fit, e13.HIST_LONG); ht, lt = e13.long_history(test, fit, e13.HIST_LONG)
out_f = np.full((len(fit["item"]), 4), np.nan, np.float32)
for day in range(e13.FIRST_MODEL_DAY, e13.OFFICIAL_CUTOFF):
    evd = np.flatnonzero(fit["day"] == day)
    r = (evd[:, None] * EV + np.arange(EV)).ravel()
    out_f[r] = scores(hf[evd], lf[evd], fit["item"].reshape(-1, EV)[evd], f"W_{day}")
    e13.log(f"day {day}: typicality AUC {e13.roc_auc(out_f[r, 2], fit['label'][r]):.5f}  "
            f"cos x coherence {e13.roc_auc(out_f[r, 3], fit['label'][r]):.5f}")
t = scores(ht, lt, test["item"].reshape(-1, EV), "W_test").astype(np.float32)
np.savez(ROOT / "data/cache/46_coh_c7.npz", mf_fit=out_f, mf_test=t)
e13.log("wrote data/cache/46_coh_c7.npz")
