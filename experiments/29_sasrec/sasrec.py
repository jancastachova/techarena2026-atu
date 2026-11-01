#!/usr/bin/env python3
"""
TechArena CTR -- experiment 29: a SASRec-style NEXT-CLICK model as a new view (extra-group cache
for 24_ensemble/ensemble.py, --extra sas=...).

Why: every model so far scores candidates with the MF's losses (impressions + each click against
the user's OTHER clicks). Experiment 18's sequence model is a transformer USER ENCODER trained with
those same losses. None learns to predict the NEXT click from the clicks before it, the objective
of sequential recommenders (SASRec), which trains on every position of every user's click sequence
(~1M targets). DATASET.md / RESULTS.md found no order or recency effects, and the base features
hold first-order transitions, so the expected gain is small; this is the one untried objective.

Model: item embeddings (dim 64) + learned positions (counted from the newest click), a [START]
token before every sequence (so empty histories work), 2 causal transformer layers, score = h .
e_item + b_item. Trained on windows of up to SEQ_LEN clicks with a sampled softmax (shared
negatives drawn by click popularity, log-Q corrected). An event's candidates are scored as the next
click after the user's SEQ_LEN most recent known clicks.

Time-forward as all views: fit day d is scored by a model trained on the clicks before day d, the
test events by one trained on all clicks before the cutoff. Cache columns as the MF caches:
[score, item bias, 0, dot]; days 0-1 NaN.

    .venv/bin/python experiments/29_sasrec/sasrec.py --quick          # days < 6 -> day 6 only
    .venv/bin/python experiments/29_sasrec/sasrec.py --out data/cache/29_sas_c7.npz --seeds 2
"""
import argparse, importlib.util, math, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
EV = e13.EV_SIZE
SEQ_LEN, DIM, LAYERS, HEADS, DROP = 50, 64, 2, 2, 0.2
BATCH, N_NEG, LR, EPOCHS = 256, 512, 1e-3, 15


def windows(fit, day, n_items):
    """Click sequences before `day` cut into windows of SEQ_LEN + 1 (item ids + 1; 0 = padding)."""
    k = fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY
    u, it, ts = fit["clk_user"][k], fit["clk_item"][k].astype(np.int64) + 1, fit["clk_ts"][k]
    o = np.lexsort((it, ts, u)); u, it = u[o], it[o]
    bounds = np.r_[0, np.flatnonzero(np.diff(u)) + 1, len(u)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seq = it[a:b]
        for end in range(len(seq), 1, -SEQ_LEN):
            w = seq[max(0, end - SEQ_LEN - 1):end]
            out.append(np.r_[np.zeros(SEQ_LEN + 1 - len(w), np.int64), w])
    return np.array(out), np.bincount(it, minlength=n_items + 2).astype(np.float64)


def main():
    global EPOCHS, DIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
    ap.add_argument("--quick", action="store_true", help="train on clicks before day 6, score day 6 only")
    ap.add_argument("--out", help="cache path for the full time-forward run")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--dim", type=int, default=DIM)
    a = ap.parse_args()
    EPOCHS, DIM = a.epochs, a.dim
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(8)
    dev = torch.device(a.device)
    fit, test, _, _, n_items = e13.load_base(a.base)
    START = n_items + 1                                  # ids: 0 pad, 1..n_items items, n_items+1 start

    class SAS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(n_items + 2, DIM, padding_idx=0)
            self.pos = torch.nn.Embedding(SEQ_LEN + 1, DIM)
            self.bias = torch.nn.Embedding(n_items + 2, 1)
            torch.nn.init.normal_(self.emb.weight, std=0.02); torch.nn.init.zeros_(self.bias.weight)
            layer = torch.nn.TransformerEncoderLayer(DIM, HEADS, 2 * DIM, DROP, batch_first=True, norm_first=True)
            self.enc = torch.nn.TransformerEncoder(layer, LAYERS)
            self.norm = torch.nn.LayerNorm(DIM)
            self.register_buffer("causal", torch.triu(torch.ones(SEQ_LEN + 1, SEQ_LEN + 1, dtype=torch.bool), 1))

        def forward(self, x):
            """x: (B, SEQ_LEN + 1) = [START] + right-aligned items (0 = pad) -> hidden (B, T, DIM)."""
            T = x.shape[1]
            pos = torch.arange(T, device=x.device).flip(0)          # 0 = newest (last) position
            h = (self.emb(x) * math.sqrt(DIM) + self.pos(pos)[None]) * (x > 0).unsqueeze(2)
            # causal mask only: padding (zero vectors) stays attendable, so no query is fully masked
            return self.norm(self.enc(h, mask=self.causal[:T, :T]))

    def with_start(w):
        """windows of SEQ_LEN + 1 items -> inputs [START] + first SEQ_LEN, targets the SEQ_LEN + 1."""
        x = np.concatenate([np.full((len(w), 1), START), w[:, :-1]], 1)
        first = (w > 0).argmax(1)                                    # START goes right before the first item
        x = np.where(np.arange(SEQ_LEN + 1)[None] < first[:, None], 0, x)
        x[np.arange(len(w)), first] = START
        return x, w

    def train(day, seed):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        w, pop = windows(fit, day, n_items)
        X, Y = with_start(w)
        q = pop[1:n_items + 1] ** 0.75; q /= q.sum()                # negatives over items (ids 1..n)
        logq = torch.from_numpy(np.log(np.r_[1e-12, q, 1e-12])).float().to(dev)
        m = SAS().to(dev); opt = torch.optim.Adam(m.parameters(), lr=LR)
        Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
        t0 = time.time(); steps = 0
        for ep in range(EPOCHS):
            m.train()
            for s in range(0, len(X), BATCH):
                b = slice(s, s + BATCH)
                x, y = Xt[b].to(dev), Yt[b].to(dev)
                h = m(x)                                              # (B, T, D)
                valid = y > 0
                hv, yv = h[valid], y[valid]
                neg = torch.from_numpy(rng.choice(n_items, N_NEG, p=q) + 1).to(dev)
                cand = torch.cat([yv[:, None], neg[None].expand(len(yv), -1)], 1)
                logit = (hv[:, None] * m.emb(cand)).sum(2) + m.bias(cand).squeeze(2) - logq[cand]
                logit = logit.masked_fill(torch.cat([torch.zeros_like(yv[:, None], dtype=torch.bool),
                                                     neg[None] == yv[:, None]], 1), -1e9)
                loss = F.cross_entropy(logit, torch.zeros(len(yv), dtype=torch.long, device=dev))
                opt.zero_grad(); loss.backward(); opt.step(); steps += 1
            perm = rng.permutation(len(X)); Xt, Yt = Xt[perm], Yt[perm]
        e13.log(f"    day {day} seed {seed}: {len(X):,} windows, {EPOCHS} epochs, {steps} steps, last loss "
                f"{loss.item():.3f} ({time.time() - t0:.0f}s)")
        return m

    @torch.no_grad()
    def score(m, d, events):
        m.eval()
        hist, hlen = e13.long_history(d, fit, SEQ_LEN)              # newest first, 0 = padding
        out = []
        for s in range(0, len(events), 4096):
            ev = events[s:s + 4096]
            h_ = hist[ev].astype(np.int64)[:, ::-1] + 1              # oldest first, ids + 1
            h_ = np.where(h_ == 1, 0, h_)                             # padding (item 0 -> 0)
            n = hlen[ev].astype(int)
            x = np.zeros((len(ev), SEQ_LEN + 1), np.int64)
            x[:, 1:] = h_
            x[np.arange(len(ev)), SEQ_LEN - n] = START                # START right before the clicks
            hl = m(torch.from_numpy(x).to(dev))[:, -1]                # state after the newest click
            c = torch.from_numpy(d["item"].reshape(-1, EV)[ev].astype(np.int64) + 1).to(dev)
            dot = (hl[:, None] * m.emb(c)).sum(2); b = m.bias(c).squeeze(2)
            out.append(torch.stack([dot + b, b, torch.zeros_like(b), dot], 2).reshape(-1, 4).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def averaged(day, d, events):
        return np.mean([score(train(day, s), d, events) for s in range(a.seeds)], 0)

    ev_day = fit["day"]
    rows_of = lambda ev: (ev[:, None] * EV + np.arange(EV)).ravel()
    if a.quick:
        tgt = np.flatnonzero(ev_day == 6)
        s = averaged(6, fit, tgt)
        r = rows_of(tgt)
        e13.log(f"day 6 (trained on clicks before day 6): SASRec AUC {e13.roc_auc(s[:, 0], fit['label'][r]):.5f} "
                f"(MF alone 0.8175 with 3 seeds)")
        np.save(ROOT / f"data/cache/29_sas_quick_day6_e{EPOCHS}_d{DIM}.npy", s)
        return
    mf_fit = np.full((len(ev_day) * EV, 4), np.nan, np.float32)
    for d in range(e13.FIRST_MODEL_DAY, e13.OFFICIAL_CUTOFF):
        tgt = np.flatnonzero(ev_day == d)
        mf_fit[rows_of(tgt)] = averaged(d, fit, tgt)
        r = rows_of(tgt)
        e13.log(f"  day {d}: SASRec AUC {e13.roc_auc(mf_fit[r, 0], fit['label'][r]):.5f}")
    mf_test = averaged(e13.OFFICIAL_CUTOFF, test, np.arange(len(test["user"])))
    np.savez(a.out, mf_fit=mf_fit, mf_test=mf_test)
    e13.log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
