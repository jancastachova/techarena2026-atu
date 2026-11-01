#!/usr/bin/env python3
"""
TechArena CTR -- experiment 30: a Mult-VAE (variational autoencoder for collaborative filtering,
Liang et al. 2018) of the users' click SETS, as a new view (extra-group cache for
24_ensemble/ensemble.py, --extra vae=...).

Why: every taste model so far estimates a user as the MEAN of their clicked items' vectors. In
evaluation.csv 61% of events have <= 20 known clicks (16% have 1-5), where a few-item average is
noisy. Mult-VAE reads the whole click set, encodes it to a latent taste with a learned Gaussian
prior and predicts the user's click distribution over all items (multinomial likelihood): with
few clicks the prior pulls the user toward tastes common in the population. It matches the
simulator as DATASET.md describes it (taste = the unordered set of clicks) and is a strong standard
baseline for set-based recommendation. Not tried before (RESULTS.md).

Model: binary click set -> L2-normalised -> dropout 0.5 -> 600 tanh -> (mu, logvar) of dim 200 ->
600 tanh -> logits over all items; loss = -multinomial log-likelihood + beta * KL, beta annealed
to BETA_MAX. Trained on the users' click sets from the clicks before day d (time-forward, as all
views). An event is scored by encoding the user's known clicks before the event (mu, no sampling)
and reading the candidates' logits. Cache columns as the MF caches: [logit, 0, 0, logit].

    .venv/bin/python experiments/30_multvae/multvae.py --quick        # clicks < day 6 -> day 6 only
    .venv/bin/python experiments/30_multvae/multvae.py --out data/cache/30_vae_c7.npz
"""
import argparse, importlib.util, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
EV = e13.EV_SIZE
HID, LAT, DROP, BETA_MAX, EPOCHS, BATCH, LR = 600, 200, 0.5, 0.2, 60, 512, 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="mps", choices=["cpu", "mps"])
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(8)
    dev = torch.device(a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    fit, test, _, _, n_items = e13.load_base(a.base)

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = torch.nn.Linear(n_items, HID); self.mu = torch.nn.Linear(HID, LAT)
            self.lv = torch.nn.Linear(HID, LAT); self.dec1 = torch.nn.Linear(LAT, HID)
            self.dec2 = torch.nn.Linear(HID, n_items)
        def encode(self, x):
            x = F.normalize(x, dim=1)
            h = torch.tanh(self.enc(F.dropout(x, DROP, self.training)))
            return self.mu(h), self.lv(h)
        def decode(self, z):
            return self.dec2(torch.tanh(self.dec1(z)))

    def click_sets(day):
        k = fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY
        u, it = fit["clk_user"][k], fit["clk_item"][k]
        pairs = np.unique(np.stack([u, it], 1), axis=0)
        users = np.unique(pairs[:, 0])
        pos = np.searchsorted(users, pairs[:, 0])
        starts = np.r_[0, np.flatnonzero(np.diff(pos)) + 1, len(pos)]
        return users, pairs[:, 1], starts

    def dense(items_list):
        x = torch.zeros(len(items_list), n_items)
        for i, its in enumerate(items_list):
            if len(its):
                x[i, torch.from_numpy(np.asarray(its, np.int64))] = 1.0
        return x

    def train(day, seed):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        users, items, starts = click_sets(day)
        sets = [items[starts[i]:starts[i + 1]] for i in range(len(users))]
        m = VAE().to(dev); opt = torch.optim.Adam(m.parameters(), lr=LR)
        n_steps = a.epochs * ((len(sets) + BATCH - 1) // BATCH); step = 0; t0 = time.time()
        for ep in range(a.epochs):
            m.train()
            perm = rng.permutation(len(sets))
            for s in range(0, len(perm), BATCH):
                x = dense([sets[i] for i in perm[s:s + BATCH]]).to(dev)
                mu, lv = m.encode(x)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
                logp = F.log_softmax(m.decode(z), 1)
                nll = -(logp * x).sum(1).mean()
                kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
                beta = BETA_MAX * min(1.0, step / (0.6 * n_steps))
                loss = nll + beta * kl
                opt.zero_grad(); loss.backward(); opt.step(); step += 1
        e13.log(f"    day {day} seed {seed}: {len(sets):,} users, {a.epochs} epochs, nll {nll.item():.2f}, "
                f"kl {kl.item():.2f} ({time.time() - t0:.0f}s)")
        return m

    @torch.no_grad()
    def score(m, d, events):
        m.eval()
        hist, hlen = e13.long_history(d, fit, e13.HIST_LONG)
        out = []
        for s in range(0, len(events), 2048):
            ev = events[s:s + 2048]
            x = dense([hist[e, :hlen[e]] for e in ev]).to(dev)
            mu, _ = m.encode(x)
            logp = F.log_softmax(m.decode(mu), 1)
            c = torch.from_numpy(d["item"].reshape(-1, EV)[ev].astype(np.int64)).to(dev)
            lc = torch.gather(logp, 1, c)
            z0 = torch.zeros_like(lc)
            out.append(torch.stack([lc, z0, z0, lc], 2).reshape(-1, 4).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def averaged(day, d, events):
        return np.mean([score(train(day, s), d, events) for s in range(a.seeds)], 0)

    ev_day = fit["day"]
    rows_of = lambda ev: (ev[:, None] * EV + np.arange(EV)).ravel()
    if a.quick:
        tgt = np.flatnonzero(ev_day == 6)
        s = averaged(6, fit, tgt)
        r = rows_of(tgt)
        e13.log(f"day 6 (trained on click sets before day 6): Mult-VAE AUC {e13.roc_auc(s[:, 0], fit['label'][r]):.5f} "
                f"(MF30 alone 0.81921)")
        np.save(ROOT / f"data/cache/30_vae_quick_day6_e{a.epochs}.npy", s)
        return
    mf_fit = np.full((len(ev_day) * EV, 4), np.nan, np.float32)
    for d in range(e13.FIRST_MODEL_DAY, e13.OFFICIAL_CUTOFF):
        tgt = np.flatnonzero(ev_day == d)
        mf_fit[rows_of(tgt)] = averaged(d, fit, tgt)
        r = rows_of(tgt)
        e13.log(f"  day {d}: Mult-VAE AUC {e13.roc_auc(mf_fit[r, 0], fit['label'][r]):.5f}")
    mf_test = averaged(e13.OFFICIAL_CUTOFF, test, np.arange(len(test["user"])))
    np.savez(a.out, mf_fit=mf_fit, mf_test=mf_test)
    e13.log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
