# Conditional-logit stacker with an outside option (2026-11-04, plan section 4). Library used by run scripts.
#   u_ej = b . f_ej (+ intercept)     P(j | e) = exp(u_ej) / (1 + sum_k exp(u_ek))     outside utility fixed at 0
# Per family, f = [P(item), log(P(item) / P(no click))]; clipped for numerical stability only. Features standardised with
# training-fold statistics. Loss = sum over events of -log P(observed) + l2 / 2 * |b|^2 (intercept free). L-BFGS, convergence checked.
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

EPS = 1e-6


def auc(y, p):
    r = rankdata(p); n = y.sum()
    return (r[y > 0].sum() - n * (n + 1) / 2) / (n * (len(y) - n))


def family_feats(P, P0):
    """P (n, 10), P0 (n,) -> (n, 10, 2)."""
    P = np.clip(P, EPS, 1 - EPS); P0 = np.clip(P0, EPS, 1 - EPS)
    return np.stack([P, np.log(P) - np.log(P0)[:, None]], 2)


def fit(F, y, l2=100.0, maxiter=2000):
    """F (n, 10, d) raw features, y (n, 10) 0/1. Returns dict with mean, sd, beta, intercept, convergence info."""
    mu = F.reshape(-1, F.shape[2]).mean(0); sd = F.reshape(-1, F.shape[2]).std(0) + 1e-9
    G = (F - mu) / sd
    d = G.shape[2]; clicked = y.sum(1) > 0; pos = y.argmax(1)

    def loss(t):
        b, c = t[:d], t[d]
        u = G @ b + c
        m = np.maximum(u.max(1), 0.0)
        e = np.exp(u - m[:, None]); e0 = np.exp(-m)
        Dn = e.sum(1) + e0
        ll = np.where(clicked, u[np.arange(len(u)), pos], 0.0) - (m + np.log(Dn))
        p = e / Dn[:, None]
        r = y - p
        g = np.einsum("nj,njd->d", r, G)
        return -ll.sum() + 0.5 * l2 * (b ** 2).sum(), np.r_[-g + l2 * b, -r.sum()]   # repo convention (lab/final_ensemble.py)

    res = minimize(loss, np.zeros(d + 1), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, maxfun=maxiter * 2, ftol=1e-10, gtol=1e-4))
    return dict(mu=mu, sd=sd, b=res.x[:d], c=res.x[d], ok=bool(res.success), nit=res.nit, gmax=float(np.abs(res.jac).max()), msg=str(res.message))


def predict(m, F):
    u = ((F - m["mu"]) / m["sd"]) @ m["b"] + m["c"]
    mx = np.maximum(u.max(1), 0.0)
    e = np.exp(u - mx[:, None])
    return e / (e.sum(1) + np.exp(-mx))[:, None]


def user_folds(users, n=5, seed=0):
    """The same fold for a user at every date: a hash of the user id, not of the snapshot."""
    r = np.random.default_rng(seed).permutation(100_000) % n   # fixed size: a user keeps its fold at every date
    return r[users]


def cv_compare(F_train, y_train, u_train, F_test, y_test, u_test, cols_a, cols_b, l2=100.0, folds=5):
    """Whole-user folds: test users of a fold are held out of training at all dates. Returns pooled AUCs and per-fold gains of b over a."""
    ft, fe = user_folds(u_train, folds), user_folds(u_test, folds)
    pa, pb = np.zeros(y_test.shape), np.zeros(y_test.shape); gains, valid = [], True
    for k in range(folds):
        tr, te = ft != k, fe == k
        ma = fit(F_train[tr][:, :, cols_a], y_train[tr], l2); mb = fit(F_train[tr][:, :, cols_b], y_train[tr], l2)
        valid &= ma["ok"] and mb["ok"]
        pa[te] = predict(ma, F_test[te][:, :, cols_a]); pb[te] = predict(mb, F_test[te][:, :, cols_b])
        gains.append(auc(y_test[te].ravel(), pb[te].ravel()) - auc(y_test[te].ravel(), pa[te].ravel()))
    return dict(auc_a=auc(y_test.ravel(), pa.ravel()), auc_b=auc(y_test.ravel(), pb.ravel()), gains=np.array(gains), valid=valid, pa=pa, pb=pb)
