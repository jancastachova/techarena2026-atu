# Run from the repository root: .venv/bin/python experiments/lab/replica_cells.py <replica base> <replica vectors s0,s1,s2>
# WHERE DO REAL DATA AND REPLICA DISAGREE, CELL BY CELL? (the route that found the cosine / projection gains: real clicks were
# more coherent than the replica's while the MF scored the same). Each world with its own MF vectors of days 0-5 (3 seeds),
# day-6 events with >= 1 known click, statistics per cell of (user known clicks) x (candidate clicks before day 6):
#   auc      AUC of the projected cosine within the cell (rows of the cell vs rows of the cell)
#   lift     click rate of the top cosine-quintile rows / bottom quintile, within the cell
#   coh      mean cosine between the clicked item and the user's taste (clicked rows only)
#   ctr      click rate of the cell
# and per user-activity group: mean pairwise cosine of the user's known clicks (taste coherence), event click share,
# share of clicks that are repeats. A cell where real minus replica stands out is a mechanism the replica (= our model
# family) lacks.
import importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
auc = e13.roc_auc
C = ROOT / "data/cache"
WORLDS = [("real", C / "13_base_c7_9c47a117c762.npz", [C / "22_vectors_c7.npz", C / "26_vectors_c7_s1.npz", C / "26_vectors_c7_s2.npz"]),
          ("replica", Path(sys.argv[1]), [Path(p) for p in sys.argv[2].split(",")])]
UB = [("1-5", 1, 5), ("6-20", 6, 20), ("21-50", 21, 50), ("51+", 51, 10**9)]
IB = [("0-10", 0, 10), ("11-30", 11, 30), ("31-100", 31, 100), ("101+", 101, 10**9)]
def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)

def world(base, vecs):
    fit, _, _, _, _ = e13.load_base(str(base))
    H, n = e13.long_history(fit, fit, 100)
    d6 = np.flatnonzero((fit["day"] == 6) & (n > 0))
    h, nn = H[d6].astype(np.int64), n[d6].astype(int)
    m = (np.arange(100)[None] < nn[:, None])
    it = fit["item"].reshape(-1, 10)[d6].astype(np.int64)
    y = fit["label"].reshape(-1, 10)[d6]
    NI = int(max(fit["item"].max(), fit["clk_item"].max())) + 1
    k = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
    clk = np.bincount(fit["clk_item"][k], minlength=NI)
    cosv = 0; pair = 0
    for v in vecs:
        W = np.load(v)["W_6"].astype(np.float64)
        _, U = np.linalg.eigh(W[1:].T @ W[1:]); Z = unit(W @ U[:, ::-1][:, :8]); Z[0] = 0
        S = (Z[h] * m[:, :, None]).sum(1)
        cosv = cosv + np.einsum("ed,ekd->ek", unit(S), Z[it]) / len(vecs)
        # mean pairwise cosine of the known clicks: (|S|^2 - n) / (n (n - 1))
        pair = pair + (np.einsum("ed,ed->e", S, S) - nn) / np.maximum(nn * (nn - 1), 1) / len(vecs)
    rep = (it[:, :, None] == np.where(m, h, -1)[:, None, :]).any(2)
    return dict(cos=cosv, y=y, nn=nn, cpop=clk[it], pair=pair, rep=rep)

R = {name: world(b, v) for name, b, v in WORLDS}
def cellstats(w, um, im):
    mm = um[:, None] & im
    s, yy = w["cos"][mm], w["y"][mm]
    if yy.sum() < 30 or len(yy) < 2000:
        return None
    q = np.quantile(s, [0.2, 0.8])
    lift = yy[s >= q[1]].mean() / max(yy[s <= q[0]].mean(), 1e-9)
    return auc(s, yy), lift, s[yy > 0].mean(), yy.mean(), len(yy)
print("overall projected-cosine AUC (3 seeds): " + "  ".join(f"{k} {auc(R[k]['cos'].ravel(), R[k]['y'].ravel()):.5f}" for k in R))
DIFF = []                                   # (weight, real - replica) for the mismatch summary
print("cells: user known clicks x candidate clicks; real | replica | real - replica")
for un, ulo, uhi in UB:
    for iname, ilo, ihi in IB:
        out = []
        for name in ("real", "replica"):
            w = R[name]
            um = (w["nn"] >= ulo) & (w["nn"] <= uhi); im = (w["cpop"] >= ilo) & (w["cpop"] <= ihi)
            out.append(cellstats(w, um, im))
        if None in out:
            continue
        (a1, l1, c1, r1, n1), (a2, l2, c2, r2, n2) = out
        DIFF.append((n1, a1 - a2))
        print(f"  user {un:6s} cand {iname:7s} n {n1:7d}/{n2:7d} | auc {a1:.4f} {a2:.4f} {a1 - a2:+.4f} | lift {l1:5.2f} {l2:5.2f} "
              f"{l1 - l2:+5.2f} | coh {c1:.3f} {c2:.3f} {c1 - c2:+.3f} | ctr {r1:.4f} {r2:.4f}", flush=True)
print("per user group: pairwise cosine of known clicks, event click share, repeat-candidate click rate, repeat share of clicks")
for un, ulo, uhi in UB:
    row = []
    for name in ("real", "replica"):
        w = R[name]; um = (w["nn"] >= ulo) & (w["nn"] <= uhi) & (w["nn"] >= 2)
        ev = w["y"][um]; rp = w["rep"][um]
        row.append(f"{name} pair {w['pair'][um].mean():.4f} evclick {ev.max(1).mean():.3f} "
                   f"rep-ctr {ev[rp].mean():.3f} rep-share {ev[rp].sum() / max(ev.sum(), 1):.3f}")
    print(f"  {un:6s} " + " | ".join(row), flush=True)
    for name_ in ("real", "replica"):
        w = R[name_]; um = (w["nn"] >= ulo) & (w["nn"] <= uhi) & (w["nn"] >= 2)
        ev = w["y"][um]; rp = w["rep"][um]
        R[name_].setdefault("grp", []).append((ev.max(1).mean(), ev[rp].mean()))
w_, d_ = np.array(DIFF).T
rep_d = np.array([g1[1] - g2[1] for g1, g2 in zip(R["real"]["grp"], R["replica"]["grp"])])
ev_d = np.array([g1[0] - g2[0] for g1, g2 in zip(R["real"]["grp"], R["replica"]["grp"])])
print(f"MISMATCH: cell AUC rms {np.sqrt(np.average(d_ ** 2, weights=w_)):.4f} (mean {np.average(d_, weights=w_):+.4f})  "
      f"repeat-ctr rms {np.sqrt(np.mean(rep_d ** 2)):.4f}  event-click rms {np.sqrt(np.mean(ev_d ** 2)):.4f}", flush=True)
