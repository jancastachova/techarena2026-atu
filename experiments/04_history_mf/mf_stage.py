#!/usr/bin/env python3
"""
Matrix factorisation stage of experiment 04 (run by history_mf.py as a separate process,
because torch and lightgbm crash when loaded into the same process on macOS).

Change vs experiment 02/03: no per-user embedding. The user is represented only by the
items in their click history, so the model learns "users who clicked these items click
that item" and works the same for every user, including new ones:
    logit = g + b_user + b_item + mean(y_h for h in history) . q_item
Trained on whole events (10 candidates per event) with binary cross-entropy.

Chosen inside the fit period only (MF trained on days 0-3, scored on day 4; never on the
validation test events):
    with user embedding   AUC 0.568
    without (l2 0.01)     AUC 0.660
    l2 0.003 / 0.02 / 0.03 / 0.05 / 0.1  ->  0.650 / 0.669 / 0.674 / 0.679 / 0.659
    dim 64 (l2 0.01)      AUC 0.661

Time split as in experiment 03, for a cutoff day C:
  - fit rows of day d (2 <= d < C) are scored by an MF trained on days < d,
  - test events by an MF trained on all fit days (< C).

Usage (normally called by history_mf.py):
    .venv/bin/python experiments/04_history_mf/mf_stage.py --base <base.npz> --out <mf.npz> --cutoff 5
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from metrics import roc_auc           # noqa: E402

EV_SIZE = 10
FIRST_MODEL_DAY = 2
MF_FEATURES = ["mf_logit", "mf_item_bias", "mf_user_bias", "mf_hist_dot"]

MF_DIM = 32
MF_L2 = 5e-2
MF_STEPS = 5000
MF_BATCH_EVENTS = 820          # ~8,200 rows per step
MF_LR = 3e-3
CTR_PRIOR = 0.063
SEED = 0


def log(msg):
    print(msg, file=sys.stderr, flush=True)


class HistoryMF(torch.nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.g = torch.nn.Parameter(torch.tensor(math.log(CTR_PRIOR / (1 - CTR_PRIOR))))
        self.bu = torch.nn.Embedding(n_users, 1)
        self.bi = torch.nn.Embedding(n_items, 1)
        self.qi = torch.nn.Embedding(n_items, dim)
        self.yh = torch.nn.Embedding(n_items, dim, padding_idx=0)
        torch.nn.init.zeros_(self.bu.weight)       # untrained users / items stay neutral
        torch.nn.init.zeros_(self.bi.weight)
        torch.nn.init.normal_(self.qi.weight, std=0.05)
        torch.nn.init.normal_(self.yh.weight, std=0.05)
        with torch.no_grad():
            self.yh.weight[0].zero_()

    def forward(self, u, items, hist, hlen):
        """u, hist, hlen per event; items (events, 10). Returns per-candidate parts."""
        hv = self.yh(hist).sum(1) / hlen.clamp(min=1).unsqueeze(1)      # events, dim
        q = self.qi(items)                                               # events, 10, dim
        bu = self.bu(u).expand(-1, items.shape[1])                       # events, 10
        bi = self.bi(items).squeeze(2)
        dot = (hv.unsqueeze(1) * q).sum(2)
        reg = (q ** 2).sum() / items.shape[1] + (hv ** 2).sum()
        return self.g + bu + bi + dot, bi, bu, dot, reg


def event_tensors(d, ev):
    return (torch.from_numpy(d["user"][ev].astype(np.int64)),
            torch.from_numpy(d["item"].reshape(-1, EV_SIZE)[ev].astype(np.int64)),
            torch.from_numpy(d["hist"][ev].astype(np.int64)),
            torch.from_numpy(d["hlen"][ev].astype(np.float32)))


def train(d, events, n_users, n_items):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = HistoryMF(n_users, n_items, MF_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=MF_LR)
    Y = torch.from_numpy(d["label"].reshape(-1, EV_SIZE).astype(np.float32))
    steps, t0 = 0, time.time()
    while steps < MF_STEPS:
        perm = rng.permutation(events)
        for s in range(0, len(perm), MF_BATCH_EVENTS):
            ev = perm[s:s + MF_BATCH_EVENTS]
            logit, _, _, _, reg = model(*event_tensors(d, ev))
            loss = F.binary_cross_entropy_with_logits(logit, Y[torch.from_numpy(ev)])
            opt.zero_grad()
            (loss + MF_L2 * reg / logit.numel()).backward()
            opt.step()
            steps += 1
            if steps >= MF_STEPS:
                break
    log(f"    {MF_STEPS} steps on {len(events):,} events ({time.time()-t0:.0f}s)")
    return model


@torch.no_grad()
def score(model, d, events):
    model.eval()
    out = []
    for s in range(0, len(events), 4096):
        ev = events[s:s + 4096]
        logit, bi, bu, dot, _ = model(*event_tensors(d, ev))
        out.append(torch.stack([logit, bi, bu, dot], 2).reshape(-1, len(MF_FEATURES)).numpy())
    return np.concatenate(out).astype(np.float32)


def load_base(path):
    z = np.load(path)
    fit = {k[3:]: z[k] for k in ("tr_user", "tr_item", "tr_hist", "tr_hlen", "tr_label", "tr_day")}
    test = {k[3:]: z[k] for k in ("ev_user", "ev_item", "ev_hist", "ev_hlen")}
    return fit, test, int(z["n_users"]), int(z["n_items"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", type=int, required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)

    fit, test, n_users, n_items = load_base(a.base)
    ev_day = fit["day"]
    mf_fit = np.full((len(ev_day) * EV_SIZE, len(MF_FEATURES)), np.nan, dtype=np.float32)
    for d in range(FIRST_MODEL_DAY, a.cutoff + 1):
        train_ev = np.flatnonzero(ev_day < d)
        if d < a.cutoff:
            target = np.flatnonzero(ev_day == d)
            log(f"  MF trained on days 0-{d-1} ({len(train_ev):,} events) -> scoring fit day {d}")
            model = train(fit, train_ev, n_users, n_items)
            rows = (target[:, None] * EV_SIZE + np.arange(EV_SIZE)).ravel()
            mf_fit[rows] = score(model, fit, target)
            log(f"    AUC on day {d}: {roc_auc(mf_fit[rows, 0], fit['label'][rows]):.5f}")
        else:
            log(f"  MF trained on days 0-{d-1} ({len(train_ev):,} events) -> scoring the test events")
            model = train(fit, train_ev, n_users, n_items)
            mf_test = score(model, test, np.arange(len(test["user"])))
    np.savez(a.out, mf_fit=mf_fit, mf_test=mf_test)


if __name__ == "__main__":
    main()
