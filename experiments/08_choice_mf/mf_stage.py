#!/usr/bin/env python3
"""
Matrix factorisation stage of experiment 08 (run by choice_mf.py as a separate process,
because torch and lightgbm crash when loaded into the same process on macOS).

Change vs experiment 06/07: the impression loss is a choice model instead of one yes/no per
row. An event has at most one click, as if the user picks one of the 10 candidates or
"nothing":
    p(candidate c) = exp(s_c) / (1 + sum_j exp(s_j)),   p(no click) = 1 / (1 + sum_j exp(s_j))
with s = g + b_user + b_item + mean(history item vectors) . q_item as before. The loss is the
cross-entropy over the 11 outcomes, scaled by LOSS_SCALE (per row, like the old loss, so the
tuned l2 and click-log weight keep their meaning). The MF features are the 11-way
log-probabilities log p(c) (and the parts of s), averaged over N_SEEDS seeds. The click-log
loss and the time split are unchanged from experiment 05/06.

Chosen inside the fit period only (MF trained on days 0-3 + clicks before day 4, AUC on
fit day 4; cutoff-5 base features):
    yes/no loss, 1 seed / 3 seeds          0.736 / 0.758
    choice loss, scale 1.0 (unscaled)      0.691
    choice loss, scale 0.05 / 0.1 / 0.2    0.737 / 0.742 / 0.735   (1 seed)
    choice loss, scale 0.1, 3 seeds        0.765

Usage (normally called by choice_mf.py):
    .venv/bin/python experiments/08_choice_mf/mf_stage.py --base <base.npz> --out <mf.npz> --cutoff 5
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]

# experiment 05's MF stage (click log, model of experiment 04), loaded by path (same file name)
_spec = importlib.util.spec_from_file_location(
    "mf_stage_05", ROOT / "experiments" / "05_click_aug" / "mf_stage.py")
m5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m5)
m4 = m5.m4
from metrics import roc_auc           # noqa: E402  (src/ is on sys.path through m5)

EV_SIZE = m4.EV_SIZE
N_SEEDS = 10
LOSS_SCALE = 0.1
log = m5.log


def choice_log_probs(logit):
    """log p(candidate) for the 11-way choice (10 candidates + no click)."""
    zero = torch.zeros(logit.shape[0], 1)
    return logit - torch.logsumexp(torch.cat([zero, logit], 1), 1, keepdim=True)


def train(fit, events, clicks, n_users, n_items, seed):
    """experiment 05's train() with the choice-model impression loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = m4.HistoryMF(n_users, n_items, m4.MF_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=m4.MF_LR)
    Y = torch.from_numpy(fit["label"].reshape(-1, EV_SIZE).astype(np.float32))

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
            y = Y[torch.from_numpy(ev)]
            choice = torch.where(y.sum(1) > 0, y.argmax(1) + 1, torch.zeros(len(ev), dtype=torch.long))
            full = torch.cat([torch.zeros(len(ev), 1), logit], 1)            # column 0 = no click
            total = LOSS_SCALE * F.cross_entropy(full, choice) + m4.MF_L2 * reg / logit.numel()
            if m5.AUX_WEIGHT and len(target):
                k = torch.from_numpy(rng.integers(0, len(target), size=len(ev)))
                neg = torch.from_numpy(rng.choice(n_items, size=(len(ev), m5.N_NEG), p=p))
                pos_item = target[k].unsqueeze(1)
                cand = torch.cat([pos_item, neg], 1)
                hv = model.yh(ctx[k]).sum(1) / ctx_len[k].clamp(min=1).unsqueeze(1)
                q = model.qi(cand)
                a_logit = model.bi(cand).squeeze(2) + (hv.unsqueeze(1) * q).sum(2)
                same = torch.cat([torch.zeros_like(pos_item, dtype=torch.bool), neg == pos_item], 1)
                a_logit = a_logit.masked_fill(same, -1e9)
                a_loss = F.cross_entropy(a_logit, torch.zeros(len(ev), dtype=torch.long))
                a_reg = (q ** 2).sum() / cand.shape[1] + (hv ** 2).sum()
                total = total + m5.AUX_WEIGHT * (a_loss + m4.MF_L2 * a_reg / a_logit.numel())
            opt.zero_grad()
            total.backward()
            opt.step()
            steps += 1
            if steps >= m4.MF_STEPS:
                break
    log(f"    seed {seed}: {m4.MF_STEPS} steps on {len(events):,} events + {len(target):,} logged clicks "
        f"({time.time()-t0:.0f}s)")
    return model


@torch.no_grad()
def score(model, d, events):
    """Per row: [choice log-probability, item bias, user bias, history dot]."""
    model.eval()
    out = []
    for s in range(0, len(events), 4096):
        ev = events[s:s + 4096]
        logit, bi, bu, dot, _ = model(*m4.event_tensors(d, ev))
        out.append(torch.stack([choice_log_probs(logit), bi, bu, dot], 2).reshape(-1, 4).numpy())
    return np.concatenate(out).astype(np.float32)


def train_and_score(fit, events, clicks, n_users, n_items, score_fn):
    out = None
    for seed in range(N_SEEDS):
        s = score_fn(train(fit, events, clicks, n_users, n_items, seed))
        out = s if out is None else out + s
    return out / N_SEEDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", type=int, required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)

    fit, test, n_users, n_items = m5.load_base(a.base)
    clicks = m5.click_examples(fit)
    ev_day = fit["day"]
    log(f"  {len(clicks[2]):,} logged clicks with context; choice loss x{LOSS_SCALE}, "
        f"{N_SEEDS} seeds, l2 {m4.MF_L2}")

    mf_fit = np.full((len(ev_day) * EV_SIZE, len(m4.MF_FEATURES)), np.nan, dtype=np.float32)
    for d in range(m4.FIRST_MODEL_DAY, a.cutoff + 1):
        train_ev = np.flatnonzero(ev_day < d)
        clk_d = m5.clicks_before(clicks, d)
        if d < a.cutoff:
            target = np.flatnonzero(ev_day == d)
            log(f"  MF trained on days 0-{d-1} -> scoring fit day {d}")
            rows = (target[:, None] * EV_SIZE + np.arange(EV_SIZE)).ravel()
            mf_fit[rows] = train_and_score(fit, train_ev, clk_d, n_users, n_items,
                                           lambda m: score(m, fit, target))
            log(f"    AUC on day {d}: {roc_auc(mf_fit[rows, 0], fit['label'][rows]):.5f}")
        else:
            log(f"  MF trained on days 0-{d-1} -> scoring the test events")
            mf_test = train_and_score(fit, train_ev, clk_d, n_users, n_items,
                                      lambda m: score(m, test, np.arange(len(test["user"]))))
    np.savez(a.out, mf_fit=mf_fit, mf_test=mf_test)


if __name__ == "__main__":
    main()
