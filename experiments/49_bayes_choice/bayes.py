# Run from the repository root (torch only):
#   python experiments/49_bayes_choice/bayes.py --cutoff 3 --end 4 --cats 9 --out data/cache/49/x.npz
# 2026-11-04: BAYESIAN CHOICE MODEL. Every model so far is a point estimate; ~140 impressions / 8 clicks per item leave the item
# vectors uncertain, and the mean of the probability over the posterior is not the probability at the mode. One event:
#   u_i = x_u . y_i + b_i + a_u[cat_i] + r [i already clicked by u]      P(i | event) = exp(u_i) / (exp(o_u) + sum_j exp(u_j))
# Exposure: the ordinary candidates of a user (re-shows of their clicks removed) are categorical draws,
#   q(c | u) ~ g_c exp(kappa a_u[c]),   so what a user is SHOWN informs a_u, which also enters the click utility.
# History-only clicks (warm-up, no impression rows): weight-w pseudo-events, the click + 9 alternatives redrawn from the
# impression distribution every sweep, no outside option.
# Sampler, one sweep: (1) users: exact softmax likelihood, joint (x, a, o) position-dependent Newton/MALA proposal with MH
# correction; (2) events: z_e ~ Gamma(w, D_e), which makes items conditionally independent, since D^-w = int z^(w-1) e^(-zD) dz;
# (3) items: joint (y, b) Newton proposal + MH under the augmented likelihood; (4) exp(r) by a conjugate Gamma step;
# (5) hyperparameters: Normal-Wishart on y, Normal-InvGamma on b and o, InvGamma on the variance of a.
# Legal data at cutoff C: events with day < C (labels), the target events' visible history and slate. Geometry and counts come
# from the prefix only. Predictions: mean over retained states of P(item) and P(no click).
import argparse, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.stats import invwishart
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--cutoff", type=int, required=True, help="day index from 2025-01-20; 7 = official evaluation")
ap.add_argument("--end", type=int, default=99, help="targets are first events with day < end")
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--cats", type=int, default=8, help="0 = no category term")
ap.add_argument("--cat-rank", type=int, default=0, help="spectral rank of the partition, 0 = cats - 1")
ap.add_argument("--coupling", type=float, default=1.0, help="kappa; 0 = exposure does not inform a_u")
ap.add_argument("--sweeps", type=int, default=300)
ap.add_argument("--burn", type=int, default=150)
ap.add_argument("--thin", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pw", type=float, default=1.0, help="weight of history-only pseudo-events; 0 = dropped")
ap.add_argument("--norm-strength", type=float, default=0.0, help="soft prior strength/2 (|x|^2 / r^2 - 1)^2")
ap.add_argument("--radius", type=float, default=1.0, help="r; the scale of y is free, so r only sets units")
ap.add_argument("--xprec", type=float, default=0.0, help="Gaussian prior precision of x; 0 = K / r^2")
ap.add_argument("--eps", type=float, default=0.9, help="proposal scale on the Laplace covariance")
ap.add_argument("--map", type=int, default=0, help="first sweeps: greedy Newton (no noise, mean z), a MAP start for the chain")
ap.add_argument("--hyper-start", type=int, default=0, help="hold the y prior fixed at sd --y0 for this many sweeps")
ap.add_argument("--y0", type=float, default=1.0)
ap.add_argument("--sd-a", type=float, default=0.316, help="fixed prior sd of a_u")
ap.add_argument("--sb", type=float, default=1.0, help="scale on the fixed prior sd of b_i")
ap.add_argument("--so", type=float, default=1.0, help="scale on the fixed prior sd of o_u")
ap.add_argument("--all-day", action="store_true", help="targets = every event of day cutoff (features for member training), not first events")
ap.add_argument("--dtype", default="float64")
ap.add_argument("--device", default="cuda")
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--save-state", action="store_true", help="also store posterior means of y, b, x, a, o")
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = torch.device(a.device); DT = getattr(torch, a.dtype)
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
T0 = 1737331200; t_start = time.time()
K, NC = a.K, a.cats


def log(*s): print(f"[{time.time() - t_start:7.1f}s]", *s, flush=True)


# ------------------------------------------------------------------ data at the cutoff
Z = np.load("data/cache/dataset_raw.npz")
NU, NI = int(Z["n_users"]), int(Z["n_items"])
tr_user, tr_ts = Z["tr_user"].astype(np.int64), Z["tr_ts"]
tr_item = Z["tr_item"].astype(np.int64).reshape(-1, 10)
tr_lab = Z["tr_label"].reshape(-1, 10)
day = (tr_ts - T0) // 86400
fit = np.flatnonzero(day < a.cutoff)
if a.cutoff >= 7:
    t_user, t_ts = Z["te_user"].astype(np.int64), Z["te_ts"]
    t_item = Z["te_item"].astype(np.int64).reshape(-1, 10)
    t_hist, t_hts, t_hlen = Z["te_hist"], Z["te_hist_ts"], Z["te_hlen"]
    tgt_idx = np.arange(len(t_user)); t_lab = None
else:
    late = np.flatnonzero(day >= a.cutoff)
    _, fi = np.unique(tr_user[late], return_index=True)
    tgt_idx = np.sort(late[fi]); tgt_idx = tgt_idx[day[tgt_idx] < a.end]
    if a.all_day: tgt_idx = np.flatnonzero(day == a.cutoff)
    t_user, t_ts, t_item = tr_user[tgt_idx], tr_ts[tgt_idx], tr_item[tgt_idx]
    t_hist, t_hts, t_hlen = Z["tr_hist"][tgt_idx], Z["tr_hist_ts"][tgt_idx], Z["tr_hlen"][tgt_idx]
    t_lab = tr_lab[tgt_idx]
log(f"cutoff {a.cutoff}: {len(fit):,} fit events, {len(tgt_idx):,} targets ({len(tgt_idx) * 10:,} rows)")


def hist_triples(users, hist, hts, hlen):
    v = np.arange(hist.shape[1])[None, :] < hlen[:, None]
    return np.repeat(users, v.sum(1)), hist[v].astype(np.int64), hts[v]


hu1, hi1, ht1 = hist_triples(tr_user[fit], Z["tr_hist"][fit], Z["tr_hist_ts"][fit], Z["tr_hlen"][fit])
hu2, hi2, ht2 = hist_triples(t_user, t_hist, t_hts, t_hlen)
hu, hi, ht = np.r_[hu1, hu2], np.r_[hi1, hi2], np.r_[ht1, ht2]
tri = np.unique(np.stack([hu, hi, ht], 1), axis=0)
warm = tri[tri[:, 2] < T0]                                     # clicks with no impression rows (pseudo-events)
fc = fit[tr_lab[fit].sum(1) > 0]
fc_pos = tr_lab[fc].argmax(1)
click_u, click_i, click_t = tr_user[fc], tr_item[fc, fc_pos], tr_ts[fc]
log(f"labelled clicks {len(fc):,}; history-only warm-up clicks {len(warm):,}")

# first known click time of each (user, item), for the repeat flag and for removing re-shows
src = tri if a.all_day else warm                               # all-day targets: earlier same-day clicks are in their visible history
ck = np.r_[click_u, src[:, 0]] * NI + np.r_[click_i, src[:, 1]]
ct = np.r_[click_t, src[:, 2]]
o = np.lexsort((ct, ck)); ck, ct = ck[o], ct[o]
f1 = np.r_[True, ck[1:] != ck[:-1]]; pk, pt = ck[f1], ct[f1]


def repeat_flag(users, items, ts):
    key = users[:, None] * NI + items
    q = np.minimum(np.searchsorted(pk, key), len(pk) - 1)
    return (pk[q] == key) & (pt[q] < ts[:, None])


f_item, f_user, f_ts = tr_item[fit], tr_user[fit], tr_ts[fit]
f_rep, t_rep = repeat_flag(f_user, f_item, f_ts), repeat_flag(t_user, t_item, t_ts)
f_ch = np.where(tr_lab[fit].sum(1) > 0, tr_lab[fit].argmax(1), -1)

# ------------------------------------------------------------------ item partition from the exposure prefix
if NC:
    R = a.cat_rank or NC - 1
    m = ~f_rep.ravel()
    M = sp.csr_matrix((np.ones(m.sum(), np.float32), (np.repeat(f_user, 10)[m], f_item.ravel()[m])), shape=(NU, NI))
    du, di = np.asarray(M.sum(1)).ravel(), np.asarray(M.sum(0)).ravel()
    Mn = sp.diags(1 / np.sqrt(np.maximum(du, 1))) @ M @ sp.diags(1 / np.sqrt(np.maximum(di, 1)))
    _, S, Vt = svds(Mn.astype(np.float64), k=R + 1); oo = np.argsort(-S); S, V = S[oo], Vt[oo].T
    E = V[:, 1:R + 1] / np.sqrt(np.maximum(di, 1))[:, None]
    good = np.flatnonzero(di >= 50 * max(a.cutoff, 1) / 7)
    mu = E[good].mean(0); Lw = np.linalg.inv(np.linalg.cholesky(np.cov((E[good] - mu).T))).T
    Eu = (E - mu) @ Lw; Eu[di == 0] = 0
    Eu /= np.maximum(np.linalg.norm(Eu, axis=1, keepdims=True), 1e-12)
    r0 = np.random.default_rng(0); best = -1
    for s in range(8):
        C = Eu[r0.choice(good, NC, replace=False)]
        for _ in range(60):
            lab = np.argmax(Eu[good] @ C.T, 1)
            C = np.stack([Eu[good][lab == j].mean(0) if (lab == j).any() else C[j] for j in range(NC)])
            C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        v = np.max(Eu[good] @ C.T, 1).mean()
        if v > best: best, Cb = v, C
    cat = np.argmax(Eu @ Cb.T, 1)
    log(f"partition k={NC} rank={R}: singular {np.round(S[:R + 2 if R + 2 < len(S) else len(S)], 3)}, cos {best:.4f}, sizes {np.bincount(cat, minlength=NC)}")
    # counts: prefix + the target's own slate (visible)
    exp_cnt = np.zeros((NU, NC))
    np.add.at(exp_cnt, (np.repeat(f_user, 10)[m], cat[f_item.ravel()[m]]), 1)
    gshare = exp_cnt.sum(0) / exp_cnt.sum()
    mt = ~t_rep.ravel()
    np.add.at(exp_cnt, (np.repeat(t_user, 10)[mt], cat[t_item.ravel()[mt]]), 1)
else:
    cat = np.zeros(NI, np.int64); exp_cnt = np.zeros((NU, 1)); gshare = np.ones(1)
NCc = NC

# ------------------------------------------------------------------ tensors: real + pseudo events, sorted by user
n_real, n_ps = len(fit), (len(warm) if a.pw > 0 else 0)
slate = np.zeros((n_real + n_ps, 10), np.int64); slate[:n_real] = f_item
rep = np.zeros((n_real + n_ps, 10), bool); rep[:n_real] = f_rep
ev_u = np.r_[f_user, warm[:n_ps, 0]]
choice = np.r_[f_ch, np.zeros(n_ps, np.int64)]
w_ev = np.r_[np.ones(n_real), np.full(n_ps, a.pw)]
outside = np.r_[np.ones(n_real, bool), np.zeros(n_ps, bool)]
if n_ps: slate[n_real:, 0] = warm[:, 1]
o = np.argsort(ev_u, kind="stable")
T = lambda x, d=None: torch.as_tensor(x, device=dev, dtype=d)
slate, rep, ev_u, choice, w_ev, outside = T(slate[o]), T(rep[o]), T(ev_u[o]), T(choice[o]), T(w_ev[o], DT), T(outside[o])
is_ps = ~outside
ps_rows = torch.nonzero(is_ps).squeeze(1)
imp_pool = T(f_item.ravel())
N = len(ev_u)
cat_t = T(cat); expc = T(exp_cnt, DT); logg = T(np.log(gshare), DT); expn = expc.sum(1)
tg_item, tg_user, tg_rep = T(t_item), T(t_user), T(t_rep)
y_lab = None if (t_lab is None or a.cutoff >= 5) else t_lab.ravel().astype(np.float64)   # Jan25 stays unseen until the candidate is frozen

# ------------------------------------------------------------------ ALS start
rows = np.r_[click_u, warm[:, 0]]; cols = np.r_[click_i, warm[:, 1]]
Cm = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(NU, NI)); Cm.data[:] = 1
Dn = sp.diags(1 / np.sqrt(np.maximum(np.asarray(Cm.sum(1)).ravel(), 1))) @ Cm @ sp.diags(1 / np.sqrt(np.maximum(np.asarray(Cm.sum(0)).ravel(), 1)))
Us, Ss, Vs = svds(Dn, k=K + 1); oo = np.argsort(-Ss); Vs = Vs[oo][1:].T; Us = Us[:, oo][:, 1:]
prof = None
if NC:                                                         # clicked-item profile in exposure space, a stacker score
    Pu = (Cm @ Eu)[t_user]; Pu /= np.maximum(np.linalg.norm(Pu, axis=1, keepdims=True), 1e-12)
    prof = np.einsum("nd,njd->nj", Pu, Eu[t_item]).astype(np.float32)
Y0 = Vs * np.sqrt(NI) * 0.5
lam = 2.0
for _ in range(5):                                             # ALS on the click matrix
    X0 = (Cm @ Y0) @ np.linalg.inv(Y0.T @ Y0 / 50 + lam * np.eye(K)) / 50
    Y0 = (Cm.T @ X0) @ np.linalg.inv(X0.T @ X0 / 50 + lam * np.eye(K)) / 50
sx = np.sqrt((X0 ** 2).sum(1).mean()) + 1e-12
X0, Y0 = X0 / sx * a.radius * 0.7, Y0 * sx
sy = Y0.std() + 1e-12
Y0 = Y0 / sy * 1.0
X = T(X0, DT).contiguous(); Y = T(Y0, DT).contiguous()
imp_cnt = np.bincount(f_item.ravel(), minlength=NI); clk_cnt = np.bincount(click_i, minlength=NI)
ctr = (clk_cnt + 0.6) / (imp_cnt + 10)
b = T(np.log(ctr / ctr.mean()) * 0.5 - 2.0, DT)
A = torch.zeros(NU, NCc, device=dev, dtype=DT)
ur = (np.bincount(f_user, weights=(f_ch >= 0), minlength=NU) + 0.6) / (np.bincount(f_user, minlength=NU) + 1)
O = T(-np.log(np.clip(ur, .05, .98) / (1 - np.clip(ur, .05, .98))) + 0.0, DT)
r_rep = torch.tensor(1.4, device=dev, dtype=DT)
mu_y = torch.zeros(K, device=dev, dtype=DT); Py = torch.eye(K, device=dev, dtype=DT) / a.y0 ** 2
mu_b, s2_b = b.mean(), b.var()
mu_o, s2_o = O.mean(), O.var()
s2_a = torch.tensor(a.sd_a ** 2, device=dev, dtype=DT); s2_b = s2_b * a.sb ** 2; s2_o = s2_o * a.so ** 2
xprec = a.xprec or K / a.radius ** 2
D = K + NCc + 1                                                # user block: x, a, o
tri_i, tri_j = torch.triu_indices(K + 1, K + 1, device=dev)
CH = 400_000 if K <= 12 else 80_000                            # events per chunk (item Hessian memory grows with K^2)


def redraw():
    if n_ps:
        slate[ps_rows, 1:] = imp_pool[torch.randint(len(imp_pool), (n_ps, 9), device=dev)]


def event_scores(sl, rp, uu, Xc, Ac):
    Yr = Y[sl]
    s = (Yr * Xc[uu][:, None, :]).sum(-1) + b[sl] + r_rep * rp
    if NC: s = s + Ac[uu[:, None], cat_t[sl]]
    return s, Yr


def user_eval(Xc, Ac, Oc, hess=True):
    """Log posterior, gradient and Fisher Hessian of every user's (x, a, o) block under the exact softmax."""
    ll = torch.zeros(NU, device=dev, dtype=DT); g = torch.zeros(NU, D, device=dev, dtype=DT)
    H = torch.zeros(NU, D * D, device=dev, dtype=DT) if hess else None
    for st in range(0, N, CH):
        sl = slice(st, st + CH); S_, rp, uu, ch, ww, ou = slate[sl], rep[sl], ev_u[sl], choice[sl], w_ev[sl], outside[sl]
        s, Yr = event_scores(S_, rp, uu, Xc, Ac)
        s0 = torch.where(ou, Oc[uu], torch.full_like(ww, -1e30))
        al = torch.cat([s, s0[:, None]], 1)
        lse = torch.logsumexp(al, 1)
        p = (al - lse[:, None]).exp()
        n = len(uu)
        sel = torch.where(ch >= 0, ch, torch.full_like(ch, 10))
        ll.index_add_(0, uu, ww * (al.gather(1, sel[:, None]).squeeze(1) - lse))
        F = torch.zeros(n, 11, D, device=dev, dtype=DT)
        F[:, :10, :K] = Yr
        if NC: F[:, :10, K:K + NC].scatter_(2, cat_t[S_][:, :, None], 1.0)
        F[:, 10, D - 1] = 1.0
        m = torch.einsum("nj,njd->nd", p, F)
        fc = F.gather(1, sel[:, None, None].expand(-1, 1, D)).squeeze(1)
        g.index_add_(0, uu, ww[:, None] * (fc - m))
        if hess:
            Hh = torch.bmm((F * p[:, :, None]).transpose(1, 2), F) - m[:, :, None] * m[:, None, :]
            H.index_add_(0, uu, (ww[:, None, None] * Hh).reshape(n, -1))
    # priors and the exposure term
    nrm = (Xc ** 2).sum(1)
    ll = ll - 0.5 * xprec * nrm - 0.5 * (Oc - mu_o) ** 2 / s2_o - 0.5 * (Ac ** 2).sum(1) / s2_a
    g[:, :K] -= xprec * Xc
    g[:, D - 1] -= (Oc - mu_o) / s2_o
    if NC: g[:, K:K + NC] -= Ac / s2_a
    if a.norm_strength > 0:
        r2 = a.radius ** 2; t = nrm / r2 - 1
        ll = ll - 0.5 * a.norm_strength * t ** 2
        g[:, :K] -= (a.norm_strength * t * 2 / r2)[:, None] * Xc
    if hess:
        H = H.reshape(NU, D, D)
        idx = torch.arange(D, device=dev)
        pr = torch.zeros(D, device=dev, dtype=DT); pr[:K] = xprec; pr[D - 1] = 1 / s2_o
        if NC: pr[K:K + NC] = 1 / s2_a
        H[:, idx, idx] += pr
        if a.norm_strength > 0:
            H[:, :K, :K] += (4 * a.norm_strength / r2 ** 2) * Xc[:, :, None] * Xc[:, None, :]
            H[:, idx[:K], idx[:K]] += (2 * a.norm_strength / r2 * t.clamp_min(0))[:, None]
    if NC and a.coupling > 0:
        k = a.coupling
        lq = logg[None, :] + k * Ac
        lz = torch.logsumexp(lq, 1); q = (lq - lz[:, None]).exp()
        ll = ll + k * (expc * Ac).sum(1) - expn * lz
        g[:, K:K + NC] += k * (expc - expn[:, None] * q)
        if hess:
            Hq = torch.diag_embed(q) - q[:, :, None] * q[:, None, :]
            H[:, K:K + NC, K:K + NC] += (k * k * expn)[:, None, None] * Hq
    return ll, g, H


def newton_mh(th, lp, g, H, evalf):
    """Position-dependent Gaussian proposal around one Newton step, with the MH correction, for all blocks in parallel."""
    n, d = th.shape
    eye = 1e-6 * torch.eye(d, device=dev, dtype=DT)

    def chol(M):
        M = torch.nan_to_num(M, nan=0.0, posinf=1e12, neginf=-1e12) + eye
        L_, info = torch.linalg.cholesky_ex(M)
        bad = info > 0
        if bad.any():
            M = M.clone(); M[bad] = 1e6 * torch.eye(d, device=dev, dtype=DT); L_, info = torch.linalg.cholesky_ex(M)
        return L_, bad
    L, _ = chol(H)
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    mean = th + torch.cholesky_solve(g[:, :, None], L).squeeze(2)
    xi = torch.randn_like(th)
    new = mean + (0.0 if MAP else a.eps) * torch.linalg.solve_triangular(L.transpose(1, 2), xi[:, :, None], upper=True).squeeze(2)
    lp2, g2, H2 = evalf(new)
    if MAP:
        acc = (lp2 >= lp) & torch.isfinite(lp2)
        return torch.where(acc[:, None], new, th), acc
    L2, bad2 = chol(H2)
    g2 = torch.nan_to_num(g2, nan=0.0, posinf=0.0, neginf=0.0)
    mean2 = new + torch.cholesky_solve(g2[:, :, None], L2).squeeze(2)
    ld1 = torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1); ld2 = torch.log(torch.diagonal(L2, dim1=1, dim2=2)).sum(1)
    q_fwd = ld1 - 0.5 * (xi ** 2).sum(1) / 1.0
    dv = th - mean2
    q_rev = ld2 - 0.5 * (torch.bmm(L2.transpose(1, 2), dv[:, :, None]).squeeze(2) ** 2).sum(1) / a.eps ** 2
    acc = torch.log(torch.rand(n, device=dev, dtype=DT)) < (lp2 - lp + q_rev - q_fwd)
    acc &= torch.isfinite(lp2) & ~bad2
    return torch.where(acc[:, None], new, th), acc


def update_users():
    global X, A, O
    th = torch.cat([X, A, O[:, None]], 1)
    lp, g, H = user_eval(X, A, O)

    def ev(t): return user_eval(t[:, :K].contiguous(), t[:, K:K + NCc].contiguous(), t[:, D - 1].contiguous())
    th, acc = newton_mh(th, lp, g, H, ev)
    X, A, O = th[:, :K].contiguous(), th[:, K:K + NCc].contiguous(), th[:, D - 1].contiguous()
    if NC:                                                     # a shift of a_u is a shift of o_u: same likelihood, centre it
        c = A.mean(1); A = A - c[:, None]; O = O - c
    return acc.to(DT).mean().item()


def sample_z():
    z = torch.empty(N, device=dev, dtype=DT)
    for st in range(0, N, CH):
        sl = slice(st, st + CH); uu, ou = ev_u[sl], outside[sl]
        s, _ = event_scores(slate[sl], rep[sl], uu, X, A)
        dn = s.exp().sum(1) + torch.where(ou, O[uu].exp(), torch.zeros_like(O[uu]))
        z[sl] = (w_ev[sl] if MAP else torch._standard_gamma(w_ev[sl])) / dn
    return z


def item_eval(Yc, bc, z, hess=True):
    """Log posterior, gradient and Hessian of every item's (y, b) block given z (items independent)."""
    d = K + 1
    ll = torch.zeros(NI, device=dev, dtype=DT); g = torch.zeros(NI, d, device=dev, dtype=DT)
    Ht = torch.zeros(NI, len(tri_i), device=dev, dtype=DT) if hess else None
    for st in range(0, N, CH):
        sl = slice(st, st + CH); S_, uu, ch, ww = slate[sl], ev_u[sl], choice[sl], w_ev[sl]
        Xu = X[uu]
        s = (Yc[S_] * Xu[:, None, :]).sum(-1) + bc[S_] + r_rep * rep[sl]
        if NC: s = s + A[uu[:, None], cat_t[S_]]
        lam_ = z[sl][:, None] * s.exp()                              # (n, 10)
        it = S_.reshape(-1); lf = lam_.reshape(-1)
        Gx = torch.cat([Xu, torch.ones(len(uu), 1, device=dev, dtype=DT)], 1)    # (n, d)
        Gr = Gx[:, None, :].expand(-1, 10, -1).reshape(-1, d)
        ll.index_add_(0, it, -lf)
        g.index_add_(0, it, -lf[:, None] * Gr)
        if hess:
            Ht.index_add_(0, it, lf[:, None] * (Gr[:, tri_i] * Gr[:, tri_j]))
        c = ch >= 0
        ci = S_[c].gather(1, ch[c][:, None]).squeeze(1)
        sc = s[c].gather(1, ch[c][:, None]).squeeze(1)
        ll.index_add_(0, ci, ww[c] * sc)
        g.index_add_(0, ci, ww[c][:, None] * Gx[c])
    dy = Yc - mu_y
    ll = ll - 0.5 * ((dy @ Py) * dy).sum(1) - 0.5 * (bc - mu_b) ** 2 / s2_b
    g[:, :K] -= dy @ Py
    g[:, K] -= (bc - mu_b) / s2_b
    H = None
    if hess:
        H = torch.zeros(NI, d, d, device=dev, dtype=DT)
        H[:, tri_i, tri_j] = Ht; H[:, tri_j, tri_i] = Ht
        H[:, :K, :K] += Py
        H[:, K, K] += 1 / s2_b
    return ll, g, H


def update_items(z):
    global Y, b
    th = torch.cat([Y, b[:, None]], 1)
    lp, g, H = item_eval(Y, b, z)

    def ev(t): return item_eval(t[:, :K].contiguous(), t[:, K].contiguous(), z)
    th, acc = newton_mh(th, lp, g, H, ev)
    Y, b = th[:, :K].contiguous(), th[:, K].contiguous()
    return acc.to(DT).mean().item()


def update_repeat(z):
    global r_rep
    rate = torch.zeros((), device=dev, dtype=DT); clicks = torch.zeros((), device=dev, dtype=DT)
    for st in range(0, N, CH):
        sl = slice(st, st + CH); uu, ch, rp = ev_u[sl], choice[sl], rep[sl]
        s, _ = event_scores(slate[sl], rp, uu, X, A)
        rate = rate + (z[sl][:, None] * (s - r_rep).exp() * rp).sum()
        c = ch >= 0
        clicks = clicks + (w_ev[sl][c] * rp[c].gather(1, ch[c][:, None]).squeeze(1)).sum()
    gam = torch.distributions.Gamma(clicks + 1.0, rate + 0.2).sample()
    r_rep = gam.log()


def update_hypers():
    global mu_y, Py, mu_b, s2_b, mu_o, s2_o, s2_a
    Yn = Y.double().cpu().numpy()
    ybar = Yn.mean(0); Sc = (Yn - ybar).T @ (Yn - ybar)
    W = invwishart.rvs(df=K + 2 + NI, scale=np.eye(K) * 0.1 + Sc, random_state=rng)
    W = np.atleast_2d(W)
    mu = rng.multivariate_normal(ybar * NI / (NI + 1), W / (NI + 1))
    mu_y, Py = T(mu, DT), T(np.linalg.inv(W), DT)

    def ng(v):
        n = len(v); m = v.mean(); ss = ((v - m) ** 2).sum()
        s2 = (ss + 0.2) / (2 * torch.distributions.Gamma(torch.tensor((n + 2) / 2.0, dtype=DT, device=dev), torch.tensor(1.0, dtype=DT, device=dev)).sample())
        return m + (s2 / n).sqrt() * torch.randn((), device=dev, dtype=DT), s2
    mu_b, s2_b = ng(b)
    act = torch.zeros(NU, dtype=torch.bool, device=dev); act[ev_u] = True; act[tg_user] = True
    mu_o, s2_o = ng(O[act])
    if NC:
        v = A[act].reshape(-1); n = len(v)
        s2_a = ((v ** 2).sum() + 0.02) / (2 * torch.distributions.Gamma(torch.tensor((n + 2) / 2.0, dtype=DT, device=dev), torch.tensor(1.0, dtype=DT, device=dev)).sample())


def predict():
    s, _ = event_scores(tg_item, tg_rep, tg_user, X, A)
    al = torch.cat([s, O[tg_user][:, None]], 1)
    return torch.softmax(al, 1)


def auc(p):
    from scipy.stats import rankdata
    r = rankdata(p); npos = y_lab.sum(); return (r[y_lab > 0].sum() - npos * (npos + 1) / 2) / (npos * (len(y_lab) - npos))


MAP = False
acc_p = torch.zeros(len(t_user), 11, device=dev, dtype=torch.float64); n_keep = 0
state = {k: 0.0 for k in ("Y", "b", "X", "A", "O")}
log(f"N {N:,} events ({n_ps:,} pseudo), D {D}, xprec {xprec:.2f}, device {dev}, dtype {a.dtype}")
for it in range(a.sweeps):
    t1 = time.time(); MAP = it < a.map
    redraw()
    au = update_users()
    z = sample_z()
    ai = update_items(z)
    z = sample_z()
    update_repeat(z)
    if it >= a.hyper_start: update_hypers()
    if it >= a.burn and (it - a.burn) % a.thin == 0:
        acc_p += predict().double(); n_keep += 1
        if a.save_state:
            for k, v in (("Y", Y), ("b", b), ("X", X), ("A", A), ("O", O)): state[k] = state[k] + v.double()
    if it % a.log_every == 0 or it == a.sweeps - 1:
        msg = (f"sweep {it}: acc u {au:.3f} i {ai:.3f} r {float(r_rep):.3f} |y|sd {float(Y.std()):.3f} |x| {float(X.norm(dim=1).mean()):.3f} sd_b {float(s2_b.sqrt()):.3f} "
               f"mu_o {float(mu_o):.3f} sd_o {float(s2_o.sqrt()):.3f} sd_a {float(s2_a.sqrt()):.3f} {time.time() - t1:.2f}s/sweep")
        if y_lab is not None:
            msg += f"  AUC state {auc(predict()[:, :10].cpu().numpy().ravel()):.5f}"
            if n_keep: msg += f" avg({n_keep}) {auc((acc_p[:, :10] / n_keep).cpu().numpy().ravel()):.5f}"
        log(msg)

P = (acc_p / max(n_keep, 1)).cpu().numpy()
out = dict(tgt_idx=tgt_idx, P=P[:, :10].astype(np.float32), P0=P[:, 10].astype(np.float32), cat=cat.astype(np.int16), n_keep=n_keep)
if t_lab is not None: out["y"] = t_lab.astype(np.int8)
if prof is not None: out["prof"] = prof
out["user"] = t_user.astype(np.int32); out["ts"] = t_ts
if a.save_state:
    for k, v in state.items(): out["post_" + k] = (v / max(n_keep, 1)).float().cpu().numpy()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out, **out)
if y_lab is not None: log(f"final AUC {auc(P[:, :10].ravel()):.5f}")
log("wrote", a.out)
