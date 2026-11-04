# Run from the repository root (torch only; no LightGBM in this process):
#   EXPORT=data/cache/lab_export_lean_d6.npz <lab/extra_member.py ... SETLEAN>   (writes the feature matrix once)
#   .venv/bin/python experiments/lab/choice_model.py data/cache/lab_export_lean_d6.npz --out data/cache/lab_preds_choice_d6.npz
# A NEURAL CHOICE MODEL (2026-10-30). Every member so far scores rows one by one (binary) or ranks rows across random event
# groups (lambdarank); the simulator picks at most one of 10 candidates per event, with a strong per-user click propensity.
# Here an event is one softmax over 11 options: each candidate's score s_k = MLP(its features) and a "no click" score
# s_0 = MLP(event-constant features: user propensity, activity, history length, ...). P(click k) = softmax(s)_k, so an
# event's total click probability depends on its whole slate, as in the generative process. Features: the LEAN set
# (lab/bundle_args.sh SETLEAN), Gaussian-rank transformed on the training rows. Fit days 2..T-1, early stopping on half A of
# day T, AUC on half B (plain / evaluation-weighted / first events), predictions saved in the lab/paired_ci.py format; also
# the equal rank average with a LightGBM predictions file (--blend).
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import ndtri
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument("export")
ap.add_argument("--out", required=True)
ap.add_argument("--blend", help="lab/extra_member.py --save-preds file of the LightGBM run on the same split")
ap.add_argument("--blend-key", default="base (= the best member)")
ap.add_argument("--hidden", default="256,128")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--seeds", type=int, default=1, help="train this many seeds (seed, seed+1, ...) and average their probabilities")
ap.add_argument("--eval-weight", type=float, default=0.0,
                help="weight each training event by (evaluation share / its day's share of its known-clicks group)^w")
ap.add_argument("--emb", type=int, default=0,
                help="also learn item embeddings of this size end to end: candidate score += tau x cos(mean embedding of the "
                     "user's known clicks, candidate embedding)")
ap.add_argument("--add", action="append", default=[],
                help="cache .npz (mf_fit: one row per fit row, any columns) appended to the inputs (2026-11-03: spectral view)")
ap.add_argument("--device", default="cpu", help="cpu or cuda (2026-11-01: the vast.ai box's GPU)")
ap.add_argument("--attn", action="store_true",
                help="slate context: one self-attention layer over the 10 candidate embeddings, and the no-click score also "
                     "sees the mean candidate embedding")
ap.add_argument("--pair", type=float, default=0.0,
                help="2026-11-03: add this x a cross-event pairwise logistic loss on the row log-probabilities (every clicked "
                     "row vs every unclicked row of the minibatch, any event): a surrogate of the GLOBAL AUC, whose pairs are "
                     "99.997%% cross-event; the softmax alone only ranks inside an event plus a no-click level")
ap.add_argument("--rows-b", default="", help="lab predictions file whose rows_B define the test day and half B")
a = ap.parse_args()
torch.manual_seed(a.seed); torch.set_num_threads(8)
Z = np.load(a.export)
X, y, A, B = Z["X"], Z["y"].astype(np.float32), Z["A"], Z["B"]
if a.add:                                    # rows of the export = whole events in order: row = 10 x event + position
    _rows = Z["ev"] * 10 + np.arange(len(Z["ev"])) % 10
    assert np.all(_rows // 10 == Z["ev"])
    X = np.concatenate([X] + [np.load(f)["mf_fit"][_rows].astype(X.dtype) for f in a.add], 1)
if a.rows_b:                                 # 2026-11-04: the half split of any test day, from a lab predictions file (same rows as the other members)
    _rb = np.load(a.rows_b)["rows_B"]; B = np.zeros(len(y), bool); B[_rb] = True
    A = (Z["d"] == Z["d"][_rb[0]]) & ~B
W_EVAL, FIRST, W_FIRST = Z["W_EVAL"], Z["FIRST"], Z["W_FIRST"]
tr = ~(A | B) & (Z["d"] < Z["d"][A].min())
n_ev = len(y) // 10
print(f"{X.shape[0]:,} rows, {X.shape[1]} features; train rows {tr.sum():,}", flush=True)

# Gaussian-rank transform fitted on (a sample of) the training rows; NaN -> 0 plus nothing else (rare)
rng = np.random.default_rng(0)
smp = rng.choice(np.flatnonzero(tr), min(400_000, tr.sum()), replace=False)
qs = np.linspace(0.0005, 0.9995, 2000)
Xt = np.ascontiguousarray(X, dtype=np.float32)          # transformed in place (memory: the full set is 6.6 GB)
for j in range(X.shape[1]):
    col = X[:, j].astype(np.float64)
    ref = col[smp]; ref = ref[~np.isnan(ref)]
    if len(ref) == 0 or ref.min() == ref.max():
        Xt[:, j] = 0; continue
    q = np.quantile(ref, qs)
    u = np.interp(col, q, qs, left=qs[0], right=qs[-1])
    v = ndtri(u); v[np.isnan(col)] = 0
    Xt[:, j] = v
# event-constant columns (identical over the 10 rows of every event in a sample)
E = Xt.reshape(n_ev, 10, -1)
se = rng.choice(n_ev, 20000, replace=False)
const = np.all(np.ptp(E[se], axis=1) < 1e-6, axis=0)
print(f"event-constant columns: {const.sum()}", flush=True)
Ec = torch.from_numpy(np.ascontiguousarray(E[:, 0, const]))
Et = torch.from_numpy(E)
lab = y.reshape(n_ev, 10)
target = torch.from_numpy(np.where(lab.sum(1) > 0, lab.argmax(1), 10).astype(np.int64))
DEV = torch.device(a.device)
Ec, Et, target = Ec.to(DEV), Et.to(DEV), target.to(DEV)
ev_tr = np.flatnonzero(tr.reshape(n_ev, 10)[:, 0]); ev_A = np.flatnonzero(A.reshape(n_ev, 10)[:, 0])
ev_B = np.flatnonzero(B.reshape(n_ev, 10)[:, 0])

if a.emb:                                    # item ids of each event's candidates and the user's known clicks (numpy only)
    import importlib.util
    _spec = importlib.util.spec_from_file_location("e13", "experiments/13_retune_7days/retune.py")
    e13 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e13)
    fit = e13.load_base("data/cache/13_base_c7_9c47a117c762.npz")[0]
    fev = Z["ev"][::10]
    IT = torch.from_numpy(fit["item"].reshape(-1, 10)[fev].astype(np.int64))
    Hh, hn = e13.long_history(fit, fit, 50)
    HH = torch.from_numpy(Hh[fev].astype(np.int64)); HM = torch.from_numpy((np.arange(50)[None] < hn[fev][:, None]).astype(np.float32))
    IT, HH, HM = IT.to(DEV), HH.to(DEV), HM.to(DEV)
    N_ITEMS = int(fit["item"].max()) + 1
    del fit
h = [int(x) for x in a.hidden.split(",")]
def mlp(i, hs, out=1):
    layers = []
    for k in hs:
        layers += [torch.nn.Linear(i, k), torch.nn.ReLU(), torch.nn.Dropout(a.drop)]; i = k
    return torch.nn.Sequential(*layers, torch.nn.Linear(i, out))
class Choice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if a.attn:
            D = h[-1]
            self.enc = mlp(Et.shape[2], h[:-1], D)               # candidate embedding
            self.att = torch.nn.MultiheadAttention(D, 4, dropout=a.drop, batch_first=True)
            self.norm = torch.nn.LayerNorm(D)
            self.head = mlp(2 * D, [64])
            self.none = mlp(Ec.shape[1] + D, [64])
        else:
            self.cand = mlp(Et.shape[2], h)
            self.none = mlp(Ec.shape[1], [64])
        if a.emb:
            self.E = torch.nn.Embedding(N_ITEMS, a.emb); torch.nn.init.normal_(self.E.weight, std=0.1)
            self.tau = torch.nn.Parameter(torch.tensor(1.0))
    def forward(self, ev):
        if a.attn:
            e = torch.relu(self.enc(Et[ev]))                      # (n, 10, D)
            c, _ = self.att(e, e, e)
            c = self.norm(c)
            s = self.head(torch.cat([e, c], 2)).squeeze(2)
            s0 = self.none(torch.cat([Ec[ev], e.mean(1)], 1))
        else:
            s = self.cand(Et[ev]).squeeze(2)                     # (n, 10)
            s0 = self.none(Ec[ev])                               # (n, 1)
        if a.emb:
            uh = (self.E(HH[ev]) * HM[ev].unsqueeze(2)).sum(1)
            uh = uh / uh.norm(dim=1, keepdim=True).clamp(min=1e-6)
            vc = self.E(IT[ev]); vc = vc / vc.norm(dim=2, keepdim=True).clamp(min=1e-6)
            s = s + self.tau * (uh.unsqueeze(1) * vc).sum(2)
        return torch.cat([s, s0], 1)

def auc(s, y_):
    r = rankdata(s); pos = y_ > 0
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
def wauc(s, y_, w):
    o = np.argsort(s, kind="mergesort"); s, y_, w = s[o], y_[o], w[o]
    new = np.r_[True, s[1:] != s[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, w * y_); wn = np.bincount(g, w * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
@torch.no_grad()
def predict(model, evs):
    model.eval()
    out = [torch.softmax(model(torch.from_numpy(evs[s:s + 8192]).to(DEV)), 1)[:, :10].cpu().numpy() for s in range(0, len(evs), 8192)]
    model.train()
    return np.concatenate(out).ravel()

ya = lab[ev_A].ravel()
EVW = torch.ones(n_ev)
if a.eval_weight:
    EVW = torch.from_numpy((W_EVAL.reshape(n_ev, 10)[:, 0] ** a.eval_weight).astype(np.float32))
    EVW = EVW / EVW[torch.from_numpy(ev_tr)].mean()
EVW = EVW.to(DEV)
pb_seeds = []
for seed in range(a.seed, a.seed + a.seeds):
    torch.manual_seed(seed); rs = np.random.default_rng(seed)
    model = Choice().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best, best_state, bad = -1, None, 0
    for ep in range(a.epochs):
        perm = rs.permutation(ev_tr)
        for s in range(0, len(perm), 512):
            ev = torch.from_numpy(perm[s:s + 512]).to(DEV)
            out = model(ev)
            loss = (F.cross_entropy(out, target[ev], reduction="none") * EVW[ev]).mean()
            if a.pair:
                lp = torch.log_softmax(out, 1)[:, :10].reshape(-1)
                pos = torch.zeros_like(lp, dtype=torch.bool)
                hit = target[ev] < 10
                pos.view(-1, 10)[hit] = F.one_hot(target[ev][hit], 10).bool()
                d = lp[pos].unsqueeze(1) - lp[~pos].unsqueeze(0)
                loss = loss + a.pair * F.softplus(-d).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sc = auc(predict(model, ev_A), ya)
        print(f"  seed {seed} epoch {ep + 1}: half-A AUC {sc:.5f}", flush=True)
        if sc > best:
            best, bad = sc, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    model.load_state_dict(best_state)
    pb_seeds.append(predict(model, ev_B))
    print(f"  seed {seed}: best half-A {best:.5f}; half-B of this seed {auc(pb_seeds[-1], y[B]):.5f}", flush=True)
pb = np.mean(pb_seeds, 0); yb = y[B]
rowsB = np.flatnonzero(B); fb = FIRST[B]
def report(name, p):
    print(f"  {name:28s} half-B {auc(p, yb):.5f}  eval-weighted {wauc(p, yb, W_EVAL[B]):.5f}  "
          f"first {wauc(p[fb], yb[fb], W_FIRST[B][fb]) if fb.any() else float('nan'):.5f}", flush=True)
report("choice model", pb)
SAVED = {"rows_B": rowsB, "base (= the best member)|B": pb}
if a.blend:
    L = np.load(a.blend)
    assert np.array_equal(L["rows_B"], rowsB)
    pl = L[f"{a.blend_key}|B"]
    report("LightGBM", pl)
    print(f"  rank corr choice vs LightGBM {np.corrcoef(rankdata(pb), rankdata(pl))[0, 1]:.4f}")
    for w in (0.3, 0.5):
        pbl = (1 - w) * rankdata(pl) / len(pl) + w * rankdata(pb) / len(pb)
        report(f"rank blend, choice weight {w}", pbl)
        SAVED[f"blend{w}|B"] = pbl
np.savez(a.out, **SAVED)
print("wrote", a.out)
