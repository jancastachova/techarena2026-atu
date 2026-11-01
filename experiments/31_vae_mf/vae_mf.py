#!/usr/bin/env python3
"""
TechArena CTR -- experiment 31: experiment 13's MF with a VARIATIONAL user encoder, trained on the
impression loss.

Why: impression-trained models beat click-only ones by far here (RESULTS.md: MF30 0.819 vs Mult-VAE
0.705, SASRec 0.696), but Mult-VAE's idea -- a learned prior over user taste, so a user with few
clicks is pulled toward common tastes instead of trusting a 3-item average -- targets exactly the
users the evaluation weighs most (61% of events have <= 20 known clicks). Here that prior is put
INSIDE the impression-trained MF:
    user vector = mean(v_h over the known clicks)                         (experiment 13)
                + mu(set) + eps * sigma(set)  during training, mu(set) when scoring
mu / log sigma come from a small encoder of the click set (mean of a separate embedding table,
64 tanh); the last layers start at zero, so training starts as experiment 13's MF; a KL penalty
(weight beta) pulls the correction toward zero, most for users whose set says little.
--tune vae:<beta> uses the residual form above; full:<beta> drops the mean anchor (the encoder
alone gives the user vector, as Mult-VAE). Experiments 18 / 21 (encoders WITHOUT a prior) lost to
the plain MF; the prior is the difference tested here. The prior is N(0, s^2) on the MF's vector
scale (--tune vae:<beta>:<s>, default s 0.1): with N(0, 1) every variant collapsed to ~0.56 (the KL
pushed the noise to ~30x the vectors' scale and the model fell back to its biases).

MF settings, losses and steps as experiment 13; the click-log loss uses the plain mean, as in 13.

    .venv/bin/python experiments/31_vae_mf/vae_mf.py --tune base,vae:0.1,vae:1,full:0.1 --seeds 3
"""
import argparse, importlib.util, math, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
EV = e13.EV_SIZE
ENC_DIM, ENC_HID = 32, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
    ap.add_argument("--tune", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(8)
    fit, test, _, n_users, n_items = e13.load_base(a.base)
    fit_l = e13.with_long_history(fit, fit, e13.HIST_LONG)

    class VaeMF(torch.nn.Module):
        def __init__(self, mode, prior_s):
            super().__init__()
            self.mode, self.prior_s = mode, prior_s
            self.g = torch.nn.Parameter(torch.tensor(math.log(e13.CTR_PRIOR / (1 - e13.CTR_PRIOR))))
            self.bu = torch.nn.Embedding(n_users, 1); self.bi = torch.nn.Embedding(n_items, 1)
            qi = torch.nn.Embedding(n_items, e13.MF_DIM)
            self.yh = torch.nn.Embedding(n_items, e13.MF_DIM, padding_idx=0)
            torch.nn.init.zeros_(self.bu.weight); torch.nn.init.zeros_(self.bi.weight)
            torch.nn.init.normal_(qi.weight, std=e13.INIT_STD); torch.nn.init.normal_(self.yh.weight, std=e13.INIT_STD)
            with torch.no_grad():
                self.yh.weight[0].zero_()
            self.qi = self.yh
            if mode != "base":
                self.es = torch.nn.Embedding(n_items, ENC_DIM, padding_idx=0)
                torch.nn.init.normal_(self.es.weight, std=0.05)
                with torch.no_grad():
                    self.es.weight[0].zero_()
                self.h = torch.nn.Linear(ENC_DIM + 1, ENC_HID)
                self.mu = torch.nn.Linear(ENC_HID, e13.MF_DIM); self.lv = torch.nn.Linear(ENC_HID, e13.MF_DIM)
                for l in (self.mu, self.lv):
                    torch.nn.init.zeros_(l.weight); torch.nn.init.zeros_(l.bias)
                with torch.no_grad():
                    self.lv.bias.fill_(2 * math.log(0.1 * prior_s))   # noise starts at a tenth of the prior
            self.kl = torch.tensor(0.0)

        def user_vec(self, hist, hlen):
            n = hlen.clamp(min=1).unsqueeze(1)
            mean = self.yh(hist).sum(1) / n
            if self.mode == "base":
                return mean
            e = self.es(hist).sum(1) / n
            h = torch.tanh(self.h(torch.cat([e, torch.log1p(hlen).unsqueeze(1)], 1)))
            mu, lv = self.mu(h), self.lv(h)
            s2 = self.prior_s ** 2                         # prior N(0, s^2): on the MF's vector scale
            self.kl = (0.5 * (lv.exp() / s2 + mu ** 2 / s2 - 1 - lv + math.log(s2)).sum(1)).mean()
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
            return z if self.mode == "full" else mean + z

        def forward(self, u, items, hist, hlen):
            hv = self.user_vec(hist, hlen)
            q = self.qi(items)
            bu = self.bu(u).expand(-1, items.shape[1]); bi = self.bi(items).squeeze(2)
            dot = (hv.unsqueeze(1) * q).sum(2)
            reg = (q ** 2).sum() / items.shape[1] + (hv ** 2).sum()
            return self.g + bu + bi + dot, reg

    def tensors(d, ev):
        return (torch.from_numpy(d["user"][ev].astype(np.int64)),
                torch.from_numpy(d["item"].reshape(-1, EV)[ev].astype(np.int64)),
                torch.from_numpy(d["hist"][ev].astype(np.int64)),
                torch.from_numpy(d["hlen"][ev].astype(np.float32)))

    def train(events, clicks, seed, mode, beta, prior_s):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        m = VaeMF(mode, prior_s); opt = torch.optim.Adam(m.parameters(), lr=e13.MF_LR)
        Y = torch.from_numpy(fit_l["label"].reshape(-1, EV).astype(np.float32))
        rows = (events[:, None] * EV + np.arange(EV)).ravel()
        p = np.bincount(fit_l["item"][rows], minlength=n_items).astype(np.float64); p /= p.sum()
        ctx, ctx_len, target, _ = clicks
        ctx, ctx_len, target = torch.from_numpy(ctx), torch.from_numpy(ctx_len), torch.from_numpy(target)
        steps = 0
        while steps < e13.MF_STEPS:
            perm = rng.permutation(events)
            for s in range(0, len(perm), e13.MF_BATCH_EVENTS):
                ev = perm[s:s + e13.MF_BATCH_EVENTS]
                m.train()
                logit_, reg = m(*tensors(fit_l, ev))
                total = F.binary_cross_entropy_with_logits(logit_, Y[torch.from_numpy(ev)]) \
                    + e13.MF_L2 * reg / logit_.numel() + beta * m.kl / EV
                k = torch.from_numpy(rng.integers(0, len(target), size=len(ev)))
                neg = torch.from_numpy(rng.choice(n_items, size=(len(ev), e13.N_NEG), p=p))
                pos_item = target[k].unsqueeze(1); cand = torch.cat([pos_item, neg], 1)
                hv = m.yh(ctx[k]).sum(1) / ctx_len[k].clamp(min=1).unsqueeze(1)
                q = m.qi(cand)
                a_logit = m.bi(cand).squeeze(2) + (hv.unsqueeze(1) * q).sum(2)
                same = torch.cat([torch.zeros_like(pos_item, dtype=torch.bool), neg == pos_item], 1)
                a_logit = a_logit.masked_fill(same, -1e9)
                a_loss = F.cross_entropy(a_logit, torch.zeros(len(ev), dtype=torch.long))
                a_reg = (q ** 2).sum() / cand.shape[1] + (hv ** 2).sum()
                total = total + e13.AUX_WEIGHT * (a_loss + e13.MF_L2 * a_reg / a_logit.numel())
                opt.zero_grad(); total.backward(); opt.step(); steps += 1
                if steps >= e13.MF_STEPS:
                    break
        return m

    @torch.no_grad()
    def score(m, events):
        m.eval()
        return np.concatenate([m(*tensors(fit_l, events[s:s + 4096]))[0].reshape(-1).numpy()
                               for s in range(0, len(events), 4096)])

    day = 6
    train_ev, target = np.flatnonzero(fit["day"] < day), np.flatnonzero(fit["day"] == day)
    rows = (target[:, None] * EV + np.arange(EV)).ravel()
    y = fit["label"][rows]
    clk = e13.other_clicks_examples(fit, day, e13.HIST_LONG)
    _, hl = e13.long_history(fit, fit, e13.HIST_LONG)
    h = np.repeat(hl[target], EV)
    for kind in a.tune.split(","):
        parts = kind.split(":")                         # mode[:beta[:prior sd]]
        mode = parts[0]
        beta = float(parts[1]) if len(parts) > 1 else 0.0
        prior_s = float(parts[2]) if len(parts) > 2 else 0.1
        t0 = time.time()
        s = np.mean([score(train(train_ev, clk, seed, mode, beta, prior_s), target) for seed in range(a.seeds)], 0)
        by = "  ".join(f"{lo}-{hi} {e13.roc_auc(s[(h >= lo) & (h <= hi)], y[(h >= lo) & (h <= hi)]):.4f}"
                       for lo, hi in [(1, 5), (6, 20), (21, 100)])
        e13.log(f"=== {kind:12s} {a.seeds} seed(s): days 0-5 -> day 6 MF AUC {e13.roc_auc(s, y):.5f}   "
                f"by known clicks: {by}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
