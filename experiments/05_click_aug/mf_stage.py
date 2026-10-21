#!/usr/bin/env python3
"""
Matrix factorisation stage of experiment 05 (run by click_aug.py as a separate process,
because torch and lightgbm crash when loaded into the same process on macOS).

Experiment 04's history MF (experiments/04_history_mf/mf_stage.py), trained with an extra
loss on the click log (every click known before the cutoff, incl. the warm-up period):
  - each logged click k of a user is an example: context = the user's previous <= 20
    clicks, positive = the clicked item, negatives = 9 items drawn in proportion to how
    often items are shown as candidates (candidates are popularity-biased);
  - loss: softmax cross-entropy over [positive, 9 negatives] of (b_item + context . q_item),
    added to the impression loss with weight AUX_WEIGHT.
Time split: an MF used for day d only sees impressions of days < d and clicks before day d.

AUX_WEIGHT is chosen with --tune, inside the fit period only (MF trained on days 0-3,
scored on fit day 4).

Usage (normally called by click_aug.py):
    .venv/bin/python experiments/05_click_aug/mf_stage.py --base <base.npz> --out <mf.npz> --cutoff 5
    .venv/bin/python experiments/05_click_aug/mf_stage.py --base <base.npz> --tune 0,0.3,1
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from metrics import roc_auc           # noqa: E402

# experiment 04's MF stage (model + settings), loaded by path: it has this file's name
_spec = importlib.util.spec_from_file_location(
    "mf_stage_04", ROOT / "experiments" / "04_history_mf" / "mf_stage.py")
m4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m4)

EV_SIZE = m4.EV_SIZE
HIST_MAX = 20
TRAIN_START = 1737331200
DAY = 86400
N_NEG = 9
AUX_WEIGHT = 1.0                      # see --tune
SEED = 0
log = m4.log


def click_examples(fit):
    """All logged clicks with their context (previous <= HIST_MAX clicks of the same user).
    Returns (ctx (N, HIST_MAX), ctx_len, target item, click time); clicks with no earlier
    click are dropped (no context)."""
    u, it, ts = fit["clk_user"], fit["clk_item"], fit["clk_ts"]
    order = np.lexsort((it, ts, u))
    u, it, ts = u[order], it[order], ts[order]
    first = np.r_[0, np.flatnonzero(np.diff(u)) + 1]
    pos = np.arange(len(u)) - np.repeat(first, np.diff(np.r_[first, len(u)]))
    ctx = np.zeros((len(u), HIST_MAX), dtype=np.int64)
    for j in range(1, HIST_MAX + 1):
        ok = pos >= j
        ctx[ok, j - 1] = it[np.flatnonzero(ok) - j]
    keep = pos >= 1
    return ctx[keep], np.minimum(pos, HIST_MAX)[keep].astype(np.float32), it[keep], ts[keep]


def train(fit, events, clicks, n_users, n_items, aux_weight):
    """events: fit events to train on; clicks: (ctx, len, item, ts) examples to train on."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = m4.HistoryMF(n_users, n_items, m4.MF_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=m4.MF_LR)
    Y = torch.from_numpy(fit["label"].reshape(-1, EV_SIZE).astype(np.float32))

    # negatives drawn like candidates: in proportion to impressions in the training events
    rows = (events[:, None] * EV_SIZE + np.arange(EV_SIZE)).ravel()
    p = np.bincount(fit["item"][rows], minlength=n_items).astype(np.float64)
    p /= p.sum()
    ctx, ctx_len, target, _ = clicks
    ctx, ctx_len, target = torch.from_numpy(ctx), torch.from_numpy(ctx_len), torch.from_numpy(target)

    steps, t0 = 0, time.time()
    while steps < m4.MF_STEPS:
        perm = rng.permutation(events)
        for s in range(0, len(perm), m4.MF_BATCH_EVENTS):
            ev = perm[s:s + m4.MF_BATCH_EVENTS]
            logit, _, _, _, reg = model(*m4.event_tensors(fit, ev))
            loss = F.binary_cross_entropy_with_logits(logit, Y[torch.from_numpy(ev)])
            total = loss + m4.MF_L2 * reg / logit.numel()
            if aux_weight and len(target):
                k = torch.from_numpy(rng.integers(0, len(target), size=len(ev)))
                neg = torch.from_numpy(rng.choice(n_items, size=(len(ev), N_NEG), p=p))
                pos_item = target[k].unsqueeze(1)
                cand = torch.cat([pos_item, neg], 1)                           # positive first
                hv = model.yh(ctx[k]).sum(1) / ctx_len[k].clamp(min=1).unsqueeze(1)
                q = model.qi(cand)
                a_logit = model.bi(cand).squeeze(2) + (hv.unsqueeze(1) * q).sum(2)
                # a negative that happens to be the clicked item itself is ignored
                same = torch.cat([torch.zeros_like(pos_item, dtype=torch.bool), neg == pos_item], 1)
                a_logit = a_logit.masked_fill(same, -1e9)
                a_loss = F.cross_entropy(a_logit, torch.zeros(len(ev), dtype=torch.long))
                a_reg = (q ** 2).sum() / cand.shape[1] + (hv ** 2).sum()
                total = total + aux_weight * (a_loss + m4.MF_L2 * a_reg / a_logit.numel())
            opt.zero_grad()
            total.backward()
            opt.step()
            steps += 1
            if steps >= m4.MF_STEPS:
                break
    log(f"    {m4.MF_STEPS} steps on {len(events):,} events + {len(target):,} logged clicks "
        f"(weight {aux_weight}) ({time.time()-t0:.0f}s)")
    return model


def clicks_before(clicks, day):
    keep = clicks[3] < TRAIN_START + day * DAY
    return tuple(x[keep] for x in clicks)


def load_base(path):
    fit, test, n_users, n_items = m4.load_base(path)
    z = np.load(path)
    for k in ("clk_user", "clk_item", "clk_ts"):
        fit[k] = z["tr_" + k]
    return fit, test, n_users, n_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out")
    ap.add_argument("--cutoff", type=int)
    ap.add_argument("--tune", help="comma-separated AUX_WEIGHT values to compare on fit day 4")
    a = ap.parse_args()
    torch.set_num_threads(8)

    fit, test, n_users, n_items = load_base(a.base)
    clicks = click_examples(fit)
    ev_day = fit["day"]
    log(f"  {len(clicks[2]):,} logged clicks with context")

    if a.tune:
        train_ev, target = np.flatnonzero(ev_day < 4), np.flatnonzero(ev_day == 4)
        rows = (target[:, None] * EV_SIZE + np.arange(EV_SIZE)).ravel()
        for w in (float(x) for x in a.tune.split(",")):
            model = train(fit, train_ev, clicks_before(clicks, 4), n_users, n_items, w)
            s = m4.score(model, fit, target)
            log(f"=== aux weight {w}: fit day-4 AUC {roc_auc(s[:, 0], fit['label'][rows]):.5f}")
        return

    mf_fit = np.full((len(ev_day) * EV_SIZE, len(m4.MF_FEATURES)), np.nan, dtype=np.float32)
    for d in range(m4.FIRST_MODEL_DAY, a.cutoff + 1):
        train_ev = np.flatnonzero(ev_day < d)
        clk_d = clicks_before(clicks, d)
        if d < a.cutoff:
            target = np.flatnonzero(ev_day == d)
            log(f"  MF trained on days 0-{d-1} -> scoring fit day {d}")
            model = train(fit, train_ev, clk_d, n_users, n_items, AUX_WEIGHT)
            rows = (target[:, None] * EV_SIZE + np.arange(EV_SIZE)).ravel()
            mf_fit[rows] = m4.score(model, fit, target)
            log(f"    AUC on day {d}: {roc_auc(mf_fit[rows, 0], fit['label'][rows]):.5f}")
        else:
            log(f"  MF trained on days 0-{d-1} -> scoring the test events")
            model = train(fit, train_ev, clk_d, n_users, n_items, AUX_WEIGHT)
            mf_test = m4.score(model, test, np.arange(len(test["user"])))
    np.savez(a.out, mf_fit=mf_fit, mf_test=mf_test)


if __name__ == "__main__":
    main()
