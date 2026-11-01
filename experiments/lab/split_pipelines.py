# Run from the repository root (same arguments as lab/extra_member.py, --base-extra = the full set):
#   .venv/bin/python experiments/lab/split_pipelines.py --seq <cache,...> --base-extra name=<cache,...> [...]
#
# Our members (LightGBM / XGBoost / MLP) all see the same ~200 features and agree to rank corr 0.997,
# so blending them gains little; blends help when SEPARATELY built pipelines with different errors
# are combined (lab/output_blend.py found nothing to add to the full set itself). Here our own
# feature set is split into separate pipelines, each a LightGBM on the base features + MF30 + one
# family of views:
#   P1 MF family       the alternative MF views (idf*, clicks*, impronly, uoff*, dim*, choice, uvec*, deep)
#   P2 audience family aud, audc, hop*, idf_hop2
#   P3 sequence        the sequence model seeds
#   full               everything (the current best member)
# and blended at the output (rank average; logistic weights on half A). AUC on half B, plain and
# evaluation-weighted, with paired intervals against the full pipeline alone.
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtri
from scipy.stats import rankdata
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
import lightgbm as lgb
FAMILY = {
    "P1 MF family": ("idf", "idf15", "idf2", "clicks", "impronly", "uoff10", "dim64", "choice",
                     "clicks_idf", "clicks_unif", "uvec09", "deep"),
    "P2 audience family": ("aud", "audc", "hop2c", "hop3", "hop2", "idf_hop2"),
    "P3 sequence": ("seq",),
}
def prefix(n):
    best = None
    for fam in FAMILY.values():
        for p in fam:
            if n.startswith(p + "_") and (best is None or len(p) > len(best)):
                best = p
    return best
pref = [prefix(n) for n in base_names]
assert all(p is not None for p in pref[len(names) + 2:]), "every extra column should belong to a family"
core = np.array([p is None for p in pref])                       # base features + MF30 + popularity taste
print(f"{base_X.shape[1]} features: {core.sum()} core, " + ", ".join(
    f"{k} {sum(p in v for p in pref)}" for k, v in FAMILY.items()), flush=True)

P = {**e13.LGB_PARAMS, "learning_rate": 0.02}
preds = {}
def run(label, cols):
    Xm = base_X[:, cols]; nm = [n for n, c in zip(base_names, cols) if c]
    m = lgb.train(P, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 8000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pa, pb = m.predict(Xm[A], num_iteration=m.best_iteration), m.predict(Xm[B], num_iteration=m.best_iteration)
    preds[label] = (pa, pb)
    print(f"  {label:22s} half-B {auc(pb, y[B]):.5f}  eval-weighted {wauc(pb, y[B], W_EVAL[B]):.5f}  "
          f"({m.best_iteration} trees)", flush=True)
run("full", np.ones(base_X.shape[1], bool))
for k, fam in FAMILY.items():
    run(k, core | np.array([p in fam for p in pref]))

yA_, yB_, wB = y[A].astype(float), y[B].astype(float), W_EVAL[B]
nr = lambda p: ndtri(rankdata(p) / (len(p) + 1))
nB = len(yB_) // 10
rng = np.random.default_rng(0)
COUNTS = [np.repeat(rng.multinomial(nB, np.full(nB, 1 / nB)), 10) for _ in range(150)]
def ci(base, new, w):
    d = np.array([wauc(new, yB_, w * c) - wauc(base, yB_, w * c) for c in COUNTS])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f"{wauc(new, yB_, w) - wauc(base, yB_, w):+.5f} [{lo:+.5f}, {hi:+.5f}]"
def lr_w(XA):
    X1 = np.c_[XA, np.ones(len(XA))]
    def f(t):
        q = X1 @ t
        return (np.logaddexp(0, q) - yA_ * q).sum() + 0.5 * (t[:-1] @ t[:-1]), \
               X1.T @ (1 / (1 + np.exp(-q)) - yA_) + np.r_[t[:-1], 0]
    return minimize(f, np.zeros(X1.shape[1]), jac=True, method="L-BFGS-B").x[:-1]
full_B = preds["full"][1]
print("\nrank correlations on half B: " + ", ".join(
    f"full-{k.split()[0]} {np.corrcoef(rankdata(full_B), rankdata(preds[k][1]))[0, 1]:.4f}" for k in FAMILY))
print("blends (half B; vs full alone, eval-weighted, paired interval):")
combos = [["full", k] for k in FAMILY] + [["full"] + list(FAMILY), list(FAMILY)]
for c in combos:
    eq = sum(nr(preds[k][1]) for k in c)
    t = lr_w(np.column_stack([nr(preds[k][0]) for k in c]))
    fw = np.column_stack([nr(preds[k][1]) for k in c]) @ t
    lab = " + ".join(k.split()[0] for k in c)
    print(f"  {lab:22s} equal: half-B {auc(eq, yB_):.5f} eval-w {wauc(eq, yB_, wB):.5f} {ci(full_B, eq, wB)} | "
          f"fitted ({', '.join(f'{v:.2f}' for v in t)}): eval-w {wauc(fw, yB_, wB):.5f} {ci(full_B, fw, wB)}", flush=True)
