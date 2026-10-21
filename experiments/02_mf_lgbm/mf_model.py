#!/usr/bin/env python3
"""
Matrix factorisation stage of experiment 02 (run by mf_lgbm.py as a separate process).

PyTorch and LightGBM each bundle their own OpenMP runtime on macOS and crash when
loaded into the same process, so this file imports torch and never lightgbm.

Model, trained on impressions (clicked = 1, shown but not clicked = 0):
    logit = g + b_user + b_item + (p_user + mean(y_h for h in history)) . q_item
The history part gives a user vector even for users never seen in training.

Leak-free scores: rows of day d are scored by an MF trained on days < d only;
evaluation rows by an MF trained on all 7 training days.

Usage (normally called by mf_lgbm.py):
    .venv/bin/python experiments/02_mf_lgbm/mf_model.py --base <base.npz> --out <mf.npz> [--tune]
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from metrics import roc_auc           # noqa: E402

EV_SIZE = 10
FIRST_MODEL_DAY = 2                   # rows of days >= 2 get MF scores
MF_FEATURES = ["mf_logit", "mf_item_bias", "mf_user_dot", "mf_hist_dot"]

# Tuned on day 6 (fit on days 0-5): l2 1e-4..1e-3 memorises (MF AUC ~0.60-0.62),
# l2 >= 0.03 shrinks all embeddings to zero (~0.55), l2 = 0.01 peaks at ~0.704 after
# ~7 epochs. Training length is a fixed number of steps (= 7 epochs on 6 days), so the
# MFs trained on fewer days for the earlier scored days are trained about as far.
MF_DIM = 32
MF_STEPS = 5000
MF_BATCH = 8192
MF_LR = 3e-3
MF_L2 = 1e-2
CTR_PRIOR = 0.063
SEED = 0


def log(msg):
    print(msg, file=sys.stderr, flush=True)


class MF(torch.nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.g = torch.nn.Parameter(torch.tensor(math.log(CTR_PRIOR / (1 - CTR_PRIOR))))
        self.bu = torch.nn.Embedding(n_users, 1)
        self.pu = torch.nn.Embedding(n_users, dim)
        self.bi = torch.nn.Embedding(n_items, 1)
        self.qi = torch.nn.Embedding(n_items, dim)
        self.yh = torch.nn.Embedding(n_items, dim, padding_idx=0)
        for e in (self.bu, self.pu, self.bi):      # untrained users / items stay neutral
            torch.nn.init.zeros_(e.weight)
        torch.nn.init.normal_(self.qi.weight, std=0.05)
        torch.nn.init.normal_(self.yh.weight, std=0.05)
        with torch.no_grad():
            self.yh.weight[0].zero_()

    def parts(self, u, i, hist, hlen):
        hv = self.yh(hist).sum(1) / hlen.clamp(min=1).unsqueeze(1)
        pu, q = self.pu(u), self.qi(i)
        bias = self.g + self.bu(u).squeeze(1) + self.bi(i).squeeze(1)
        return bias, (pu * q).sum(1), (hv * q).sum(1), (pu, q, hv)


def batches(d, rows, rng=None):
    idx = rng.permutation(rows) if rng is not None else rows
    for s in range(0, len(idx), MF_BATCH):
        r = idx[s:s + MF_BATCH]
        e = r // EV_SIZE
        yield (r, torch.from_numpy(d["user"][e].astype(np.int64)),
               torch.from_numpy(d["item"][r].astype(np.int64)),
               torch.from_numpy(d["hist"][e].astype(np.int64)),
               torch.from_numpy(d["hlen"][e].astype(np.float32)))


def train(d, rows, n_users, n_items, eval_fn=None):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = MF(n_users, n_items, MF_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=MF_LR)
    y_all = torch.from_numpy(d["label"].astype(np.float32))
    steps, ep = 0, 0
    while steps < MF_STEPS:
        ep += 1
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for r, u, i, hist, hlen in batches(d, rows, rng):
            bias, du, dh, (pu, q, hv) = model.parts(u, i, hist, hlen)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                bias + du + dh, y_all[torch.from_numpy(r)])
            reg = MF_L2 * ((pu ** 2).sum() + (q ** 2).sum() + (hv ** 2).sum()) / len(r)
            opt.zero_grad()
            (loss + reg).backward()
            opt.step()
            tot += loss.item() * len(r)
            n += len(r)
            steps += 1
            if steps >= MF_STEPS:
                break
        msg = f"    MF epoch {ep} (step {steps}/{MF_STEPS}): log-loss {tot/n:.5f} ({time.time()-t0:.0f}s)"
        if eval_fn is not None:
            msg += f", {eval_fn(model)}"
        log(msg)
    return model


@torch.no_grad()
def score(model, d, rows):
    model.eval()
    out = np.zeros((len(rows), len(MF_FEATURES)), dtype=np.float32)
    pos = 0
    for r, u, i, hist, hlen in batches(d, rows):
        bias, du, dh, _ = model.parts(u, i, hist, hlen)
        out[pos:pos + len(r)] = torch.stack([bias + du + dh, model.bi(i).squeeze(1), du, dh], 1).numpy()
        pos += len(r)
    return out


def load_base(path):
    z = np.load(path)
    tr = {k[3:]: z[k] for k in ("tr_user", "tr_item", "tr_hist", "tr_hlen", "tr_label", "tr_day")}
    ev = {k[3:]: z[k] for k in ("ev_user", "ev_item", "ev_hist", "ev_hlen")}
    return tr, ev, int(z["n_users"]), int(z["n_items"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out")
    ap.add_argument("--tune", action="store_true", help="only report MF day-6 AUC per epoch")
    a = ap.parse_args()
    torch.set_num_threads(8)

    tr, ev, n_users, n_items = load_base(a.base)
    row_day = np.repeat(tr["day"], EV_SIZE)

    if a.tune:
        fit_rows, target = np.flatnonzero(row_day < 6), np.flatnonzero(row_day == 6)
        y = tr["label"][target]
        log(f"MF tuning: dim {MF_DIM}, lr {MF_LR}, l2 {MF_L2}, batch {MF_BATCH}")
        train(tr, fit_rows, n_users, n_items, eval_fn=lambda m: "day-6 AUC " + " / ".join(
            f"{n} {roc_auc(c, y):.5f}" for n, c in zip(MF_FEATURES, score(m, tr, target).T)))
        return

    mf_tr = np.full((len(row_day), len(MF_FEATURES)), np.nan, dtype=np.float32)
    for d in range(FIRST_MODEL_DAY, 8):
        fit_rows = np.flatnonzero(row_day < d)
        log(f"  MF trained on days 0-{d-1} ({len(fit_rows):,} rows) -> scoring "
            f"{'evaluation' if d == 7 else f'day {d}'}")
        if d < 7:
            target = np.flatnonzero(row_day == d)
            y = tr["label"][target]
            model = train(tr, fit_rows, n_users, n_items,
                          eval_fn=lambda m: f"AUC on day {d} {roc_auc(score(m, tr, target)[:, 0], y):.5f}")
            mf_tr[target] = score(model, tr, target)
        else:
            model = train(tr, fit_rows, n_users, n_items)
            mf_ev = score(model, ev, np.arange(len(ev["item"])))
    np.savez(a.out, mf_tr=mf_tr, mf_ev=mf_ev)


if __name__ == "__main__":
    main()
