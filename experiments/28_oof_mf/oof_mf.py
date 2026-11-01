#!/usr/bin/env python3
"""
TechArena CTR -- experiment 28: OUT-OF-FOLD MF scores for the LightGBM training rows, instead of
time-forward ones.

Why: until now fit day d is scored by an MF trained on days < d (AGENTS.md). That is stricter than
the competition rule (no labels of the events being TESTED on) and costs twice:
  - weak training features: day 2 gets a 2-day MF (MF alone 0.781 on day 2 vs 0.819 on day 6),
    while the test rows get a 7-day MF; LightGBM learns how far to trust the MF from weaker MFs;
  - lost rows: days 0-1 have no MF scores and are dropped (LightGBM trains on 5 of 7 days).
Here the fit events before the cutoff C are split into K folds (by event); each fold is scored by an
MF trained on the OTHER folds of all days < C, so every training row gets a near-full-strength MF
and all days are usable. The events at/after C are scored by an MF trained on all days < C, as now.

No label of a scored event reaches its MF: the clicks that are labels of held-out events (matched
exactly on user, timestamp and item; every train label is in the click log) are removed from that
MF's click log, so they are neither click-log targets nor contexts. The held-out events' histories
are inputs of later events, as at test time.

MF as experiment 13 (same model, loss and settings; --seeds seeds averaged).

    --cutoff 6  validation: days 0-5 are the fit period, day 6 plays the test. Writes
                data/cache/28_oofval_k<K>_s<S>.npz with oof (days 0-5 rows), tf (time-forward scores
                of days 2-5, the current scheme, same seeds) and day6 (MF on days 0-5); compared by
                experiments/lab/oof_check.py.
    --cutoff 7  submission features: data/cache/28_oof_c7_k<K>_s<S>.npz with mf_fit (all days 0-6,
                out of fold) and mf_test (MF on days 0-6).

    .venv/bin/python experiments/28_oof_mf/oof_mf.py --cutoff 6 --folds 5 --seeds 3
"""
import argparse, importlib.util, math, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
EV = e13.EV_SIZE


def click_key(u, t, i):
    return (u.astype(np.int64) << 40) | ((t.astype(np.int64) - (e13.TRAIN_START - 90 * e13.DAY)) << 16) | i.astype(np.int64)


def without_clicks_of(fit, events):
    """A copy of fit whose click log lacks the clicks that are labels of `events`."""
    y = fit["label"].reshape(-1, EV)[events]
    has = y.max(1) > 0
    ev = events[has]
    items = fit["item"].reshape(-1, EV)[ev, y[has].argmax(1)]
    drop = np.isin(click_key(fit["clk_user"], fit["clk_ts"], fit["clk_item"]),
                   click_key(fit["user"][ev], fit["ts"][ev], items))
    out = dict(fit)
    for k in ("clk_user", "clk_item", "clk_ts"):
        out[k] = fit[k][~drop]
    return out, int(drop.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
    ap.add_argument("--cutoff", type=int, choices=[6, 7], required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    torch.set_num_threads(8)

    class TiedMF(torch.nn.Module):
        """Experiment 13's MF: logit = g + b_user + b_item + mean(v_h over history) . v_item."""

        def __init__(self, n_users, n_items, dim):
            super().__init__()
            self.g = torch.nn.Parameter(torch.tensor(math.log(e13.CTR_PRIOR / (1 - e13.CTR_PRIOR))))
            self.bu = torch.nn.Embedding(n_users, 1)
            self.bi = torch.nn.Embedding(n_items, 1)
            qi = torch.nn.Embedding(n_items, dim)
            self.yh = torch.nn.Embedding(n_items, dim, padding_idx=0)
            torch.nn.init.zeros_(self.bu.weight)
            torch.nn.init.zeros_(self.bi.weight)
            torch.nn.init.normal_(qi.weight, std=e13.INIT_STD)
            torch.nn.init.normal_(self.yh.weight, std=e13.INIT_STD)
            with torch.no_grad():
                self.yh.weight[0].zero_()
            self.qi = self.yh

        def forward(self, u, items, hist, hlen):
            hv = self.yh(hist).sum(1) / hlen.clamp(min=1).unsqueeze(1)
            q = self.qi(items)
            bu = self.bu(u).expand(-1, items.shape[1])
            bi = self.bi(items).squeeze(2)
            dot = (hv.unsqueeze(1) * q).sum(2)
            reg = (q ** 2).sum() / items.shape[1] + (hv ** 2).sum()
            return self.g + bu + bi + dot, bi, bu, dot, reg

    def tensors(d, ev):
        return (torch.from_numpy(d["user"][ev].astype(np.int64)),
                torch.from_numpy(d["item"].reshape(-1, EV)[ev].astype(np.int64)),
                torch.from_numpy(d["hist"][ev].astype(np.int64)),
                torch.from_numpy(d["hlen"][ev].astype(np.float32)))

    def train(d, events, clicks, seed):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = TiedMF(n_users, n_items, e13.MF_DIM)
        opt = torch.optim.Adam(model.parameters(), lr=e13.MF_LR)
        Y = torch.from_numpy(d["label"].reshape(-1, EV).astype(np.float32))
        rows = (events[:, None] * EV + np.arange(EV)).ravel()
        p = np.bincount(d["item"][rows], minlength=n_items).astype(np.float64)
        p /= p.sum()
        ctx, ctx_len, target, _ = clicks
        ctx, ctx_len, target = torch.from_numpy(ctx), torch.from_numpy(ctx_len), torch.from_numpy(target)
        steps = 0
        while steps < e13.MF_STEPS:
            perm = rng.permutation(events)
            for s in range(0, len(perm), e13.MF_BATCH_EVENTS):
                ev = perm[s:s + e13.MF_BATCH_EVENTS]
                logit_, _, _, _, reg = model(*tensors(d, ev))
                loss = F.binary_cross_entropy_with_logits(logit_, Y[torch.from_numpy(ev)])
                total = loss + e13.MF_L2 * reg / logit_.numel()
                if e13.AUX_WEIGHT and len(target):
                    k = torch.from_numpy(rng.integers(0, len(target), size=len(ev)))
                    neg = torch.from_numpy(rng.choice(n_items, size=(len(ev), e13.N_NEG), p=p))
                    pos_item = target[k].unsqueeze(1)
                    cand = torch.cat([pos_item, neg], 1)
                    hv = model.yh(ctx[k]).sum(1) / ctx_len[k].clamp(min=1).unsqueeze(1)
                    q = model.qi(cand)
                    a_logit = model.bi(cand).squeeze(2) + (hv.unsqueeze(1) * q).sum(2)
                    same = torch.cat([torch.zeros_like(pos_item, dtype=torch.bool), neg == pos_item], 1)
                    a_logit = a_logit.masked_fill(same, -1e9)
                    a_loss = F.cross_entropy(a_logit, torch.zeros(len(ev), dtype=torch.long))
                    a_reg = (q ** 2).sum() / cand.shape[1] + (hv ** 2).sum()
                    total = total + e13.AUX_WEIGHT * (a_loss + e13.MF_L2 * a_reg / a_logit.numel())
                opt.zero_grad()
                total.backward()
                opt.step()
                steps += 1
                if steps >= e13.MF_STEPS:
                    break
        return model

    @torch.no_grad()
    def score(model, d, events):
        model.eval()
        out = []
        for s in range(0, len(events), 4096):
            logit_, bi, bu, dot, _ = model(*tensors(d, events[s:s + 4096]))
            out.append(torch.stack([logit_, bi, bu, dot], 2).reshape(-1, 4).numpy())
        return np.concatenate(out).astype(np.float32)

    def averaged(d_clk, d_score, train_ev, score_ev, upto_day):
        clk = e13.other_clicks_examples(d_clk, upto_day, e13.HIST_LONG)
        acc = None
        for seed in range(a.seeds):
            s = score(train(fit_l, train_ev, clk, seed), d_score, score_ev)
            acc = s if acc is None else acc + s
        return acc / a.seeds

    fit, test, _, n_users, n_items = e13.load_base(a.base)
    fit_l = e13.with_long_history(fit, fit, e13.HIST_LONG)
    test_l = e13.with_long_history(test, fit, e13.HIST_LONG)
    C = a.cutoff
    ev_day = fit["day"]
    n_rows = len(ev_day) * EV
    rows_of = lambda ev: (ev[:, None] * EV + np.arange(EV)).ravel()
    y = fit["label"]
    fit_ev = np.flatnonzero(ev_day < C)
    fold = np.random.default_rng(12345).integers(0, a.folds, size=len(ev_day))
    t0 = time.time()

    oof = np.full((n_rows, 4), np.nan, np.float32)
    for k in range(a.folds):
        held = fit_ev[fold[fit_ev] == k]
        rest = fit_ev[fold[fit_ev] != k]
        fit_k, n_drop = without_clicks_of(fit, held)
        oof[rows_of(held)] = averaged(fit_k, fit_l, rest, held, C)
        r = rows_of(held)
        e13.log(f"fold {k}: {len(held):,} held-out events, {n_drop:,} of their label clicks removed from the "
                f"click log; out-of-fold MF AUC {e13.roc_auc(oof[r, 0], y[r]):.5f} ({time.time() - t0:.0f}s)")
    for d in range(C):
        r = rows_of(np.flatnonzero(ev_day == d))
        e13.log(f"  day {d}: out-of-fold MF AUC {e13.roc_auc(oof[r, 0], y[r]):.5f}")

    if C == 6:
        tf = np.full((n_rows, 4), np.nan, np.float32)
        for d in range(e13.FIRST_MODEL_DAY, C):
            tgt = np.flatnonzero(ev_day == d)
            tf[rows_of(tgt)] = averaged(fit, fit_l, np.flatnonzero(ev_day < d), tgt, d)
            r = rows_of(tgt)
            e13.log(f"  day {d}: time-forward MF AUC {e13.roc_auc(tf[r, 0], y[r]):.5f}")
        d6 = np.flatnonzero(ev_day == 6)
        day6 = np.full((n_rows, 4), np.nan, np.float32)
        day6[rows_of(d6)] = averaged(fit, fit_l, fit_ev, d6, 6)
        r = rows_of(d6)
        e13.log(f"  day 6 (MF on days 0-5): AUC {e13.roc_auc(day6[r, 0], y[r]):.5f}")
        path = ROOT / f"data/cache/28_oofval_k{a.folds}_s{a.seeds}.npz"
        np.savez(path, oof=oof, tf=tf, day6=day6)
    else:
        mf_test = averaged(fit, test_l, fit_ev, np.arange(len(test["user"])), C)
        path = ROOT / f"data/cache/28_oof_c7_k{a.folds}_s{a.seeds}.npz"
        np.savez(path, mf_fit=oof, mf_test=mf_test)
    e13.log(f"wrote {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
