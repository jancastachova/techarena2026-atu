# Run from the repository root: .venv/bin/python experiments/lab/mlp_blend.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
# A neural network (MLP) on exactly LightGBM's features, blended with LightGBM. Fit days 2-5;
# day 6 split by event: half A = early stopping and blend weight, half B = report (LightGBM
# baseline 0.83571). torch and LightGBM run in separate processes (they crash together on macOS):
# stage "nn" trains the MLPs and saves their day-6 predictions, stage "lgb" (default) does the rest.
import importlib.util, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
ROOT = Path.cwd()
OUT = ROOT / "data" / "cache" / "lab_mlp_day6.npz"
SEEDS = 3
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.load(str(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
row_day = np.repeat(fit["day"], 10)
rows, X, names = e13.feature_matrix(fit, mf, np.flatnonzero(row_day >= 2))
y, d, ev = fit["label"][rows], row_day[rows], rows // 10
rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]
auc = e13.roc_auc


def gauss_rank(X, fit_rows):
    """Per feature: map to normal scores by the rank among the training rows (NaN -> 0 + flag)."""
    out, extra = np.zeros(X.shape, np.float32), []
    for j in range(X.shape[1]):
        col = X[:, j].astype(np.float64)
        nan = np.isnan(col)
        ref = np.sort(col[fit_rows & ~nan])
        q = (np.searchsorted(ref, col, side="left") + np.searchsorted(ref, col, side="right")) / 2
        out[:, j] = np.where(nan, 0.0, ndtri(np.clip((q + 0.5) / (len(ref) + 1), 1e-4, 1 - 1e-4)))
        if nan.any():
            extra.append(nan.astype(np.float32))
    return np.concatenate([out] + [e[:, None] for e in extra], 1) if extra else out


if len(sys.argv) > 1 and sys.argv[1] == "nn":
    import torch
    torch.set_num_threads(8)
    Z = gauss_rank(X, tr)
    Zt, yt = torch.from_numpy(Z), torch.from_numpy(y.astype(np.float32))
    idx_tr, idx_A, idx_B = np.flatnonzero(tr), np.flatnonzero(A), np.flatnonzero(B)

    def predict(m, idx):
        m.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(m(Zt[idx[s:s + 65536]])).squeeze(1).numpy()
                                   for s in range(0, len(idx), 65536)])

    preds_A, preds_B = [], []
    for seed in range(SEEDS):
        torch.manual_seed(seed); r = np.random.default_rng(seed)
        m = torch.nn.Sequential(
            torch.nn.Linear(Z.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-6)
        best, best_state, bad, t0 = -1.0, None, 0, time.time()
        for epoch in range(20):
            m.train()
            perm = r.permutation(idx_tr)
            for s in range(0, len(perm), 4096):
                b = torch.from_numpy(perm[s:s + 4096])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Zt[b]).squeeze(1), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            a = auc(predict(m, idx_A), y[A])
            print(f"  seed {seed} epoch {epoch}: half-A AUC {a:.5f} ({time.time()-t0:.0f}s)", flush=True)
            if a > best:
                best, best_state, bad = a, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 2:
                    break
        m.load_state_dict(best_state)
        preds_A.append(predict(m, idx_A)); preds_B.append(predict(m, idx_B))
    np.savez(OUT, A=np.mean(preds_A, 0), B=np.mean(preds_B, 0))
    sys.exit(0)

subprocess.run([sys.executable, __file__, "nn"], check=True)
z = np.load(OUT)
nn_A, nn_B = z["A"], z["B"]
import lightgbm as lgb
m = lgb.train(e13.LGB_PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=names), 3000,
              valid_sets=[lgb.Dataset(X[A], y[A])], callbacks=[lgb.early_stopping(100, verbose=False)])
lg_A, lg_B = m.predict(X[A], num_iteration=m.best_iteration), m.predict(X[B], num_iteration=m.best_iteration)
r = lambda p: rankdata(p) / len(p)
print(f"\nLightGBM (13)            A {auc(lg_A, y[A]):.5f}  B {auc(lg_B, y[B]):.5f}")
print(f"MLP ({SEEDS} seeds)            A {auc(nn_A, y[A]):.5f}  B {auc(nn_B, y[B]):.5f}")
print(f"correlation of ranks on B: {np.corrcoef(r(lg_B), r(nn_B))[0, 1]:.4f}")
ws = np.linspace(0, 1, 11)
for w in ws:
    print(f"  w_mlp {w:.1f}  A {auc((1-w)*r(lg_A) + w*r(nn_A), y[A]):.5f}  B {auc((1-w)*r(lg_B) + w*r(nn_B), y[B]):.5f}")
w = ws[int(np.argmax([auc((1-w)*r(lg_A) + w*r(nn_A), y[A]) for w in ws]))]
b = auc((1-w)*r(lg_B) + w*r(nn_B), y[B])
print(f"chosen on A: w_mlp {w:.1f} -> B {b:.5f}  gain {b - auc(lg_B, y[B]):+.5f}")
