# Run from the repository root (same arguments as lab/extra_member.py, --base-extra only):
#   source experiments/lab/bundle_args.sh
#   .venv/bin/python experiments/lab/repeat_feature.py --seq "$SEQ" "${BUNDLE[@]}" --base-extra sqrtview=data/cache/33_sqrtpool_c7.npz
# lab/repeat_position.py: on half B the bundle LightGBM under-predicts a re-click of the user's LAST clicked item
# (click rate / prediction 1.068) and over-predicts repeats 21+ clicks back (0.956); rep_recency counts time, not
# clicks. Features: position of the candidate's most recent occurrence among the known clicks (in clicks), and
# its occurrences among the known clicks (up to 100; rep_count only sees the 20 visible).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/warmup_feature.py").read_text()
exec(src[:src.index("if __name__")])
H, n = e13.long_history(fit, fit, 100)
cand = fit["item"][rows].astype(np.int64)
Hr, nr = H[rows // 10], n[rows // 10].astype(int)
pos = np.zeros(len(rows)); cnt = np.zeros(len(rows))
for j in range(99, -1, -1):
    hit = (Hr[:, j] == cand) & (nr > j)
    pos[hit] = j + 1; cnt += hit
F = np.stack([np.where(pos > 0, np.log(pos), -1.0), (pos == 1).astype(float), np.log1p(cnt)], 1).astype(np.float32)
cols = ["rep_log_pos", "rep_is_last", "rep_log_count100"]
import lightgbm as lgb
def run(label, Xm, nm):
    p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
    m = lgb.train(p, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 8000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(Xm[B], num_iteration=m.best_iteration)
    hb = hl_row[B]; light = (hb >= 1) & (hb <= 20)
    print(f"  {label:32s} half-B {auc(pb, y[B]):.5f}  eval-weighted {wauc(pb, y[B], W_EVAL[B]):.5f}  "
          f"1-20 known clicks {auc(pb[light], y[B][light]):.5f}  ({m.best_iteration} trees)", flush=True)
    return pb
pa = run("bundle + sqrtview", base_X, base_names)
pb_ = run("+ repeat position", np.concatenate([base_X, F], 1), base_names + cols)
evb = ev[B]; _, inv = np.unique(evb, return_inverse=True); ne_ = inv.max() + 1
rs = np.random.default_rng(0); dp, dw = [], []
for _ in range(200):
    wt = np.bincount(rs.integers(0, ne_, ne_), minlength=ne_)[inv].astype(float)
    dp.append(wauc(pb_, y[B], wt) - wauc(pa, y[B], wt)); dw.append(wauc(pb_, y[B], wt * W_EVAL[B]) - wauc(pa, y[B], wt * W_EVAL[B]))
for nm_, dd_ in (("plain", dp), ("eval-weighted", dw)):
    dd_ = np.array(dd_)
    print(f"  {nm_:14s} diff {dd_.mean():+.5f}  95% [{np.percentile(dd_, 2.5):+.5f}, {np.percentile(dd_, 97.5):+.5f}]", flush=True)
