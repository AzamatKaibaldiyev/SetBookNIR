"""Training: one config + one split + one seed = one model.

    python -m setbooknir.train --config configs/setbooknir_e_maxvit.json --split all --seed 0 --out runs/

Recipe: LAMB (lr 1e-3, weight decay 1e-3, batch 512), MultiStepLR, drop-path, mixup (Beta(alpha, alpha) on half of
the batches; inputs, labels and percentages mixed alike), multilabel soft-margin loss on the presence logits, L1 loss on
the percentages and a sum constraint on both. The auxiliary set and codebook heads are trained with soft cross-entropy
on the mixed one-hot targets. The last epoch is the model: there is no checkpoint selection.

Distillation (config "distill"): frozen teachers run online on every mixed batch, one deterministic pass each, and the
student is trained on (1 - alpha) * presence label loss + alpha * MSE(student logits, teacher target logits), and
(1 - beta) * composition label loss + beta * L1(student %, teacher target %). Targets:
  logits  logit of the mean teacher probability; mean teacher percentages
  stack   per-fibre marginals of the joint set posterior (teachers' presence probabilities and post-hoc set heads);
          1/2 (mean teacher % + mean codebook expectation from the post-hoc codebook heads)
  knob    stack presence targets; the Bayes-decision read-out (lambda = knob_lambda) of the teachers' codebook posterior
Teachers for stack / knob targets need fitted post-hoc heads (python -m setbooknir.stack fit).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import load_split, load_train, output_spaces
from .metrics import compute_metrics, fmt
from .models import build_model
from .optim import FastLamb
from .readout import knob_decision_torch

DEFAULTS = {
    "backbone": "convnext", "size": "atto", "drop_path": 0.2,
    "epochs": 160, "batch_size": 512, "lr": 1e-3, "weight_decay": 1e-3, "milestones": [80, 110, 140, 150], "gamma": 0.3,
    "mixup_alpha": 0.35, "mixup_prob": 0.5, "sum_constraint": 0.1,
    "set_head_weight": 0.0,         # > 0: auxiliary set head with this loss weight
    "codebook_head_weight": 0.0,    # > 0: auxiliary codebook head with this loss weight
    "distill": None,                # {"teachers": [paths, "{seed}" = the student's seed], "alpha", "beta", "targets", "knob_lambda"}
    "compile": True, "deterministic": False,
}


def setup_seed(seed, deterministic=False):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def mixup(x, y, c, alpha, prob):
    if np.random.rand() >= prob:
        return x, y.float(), c
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return (lam * x + (1.0 - lam) * x[perm], lam * y.float() + (1.0 - lam) * y[perm].float(),
            lam * c + (1.0 - lam) * c[perm])


def soft_ce(logits, target):
    return -(target * F.log_softmax(logits, -1)).sum(-1).mean()


def task_loss(z, r, y, c, cfg, w_cls=1.0, w_reg=1.0):
    """Presence (multilabel soft margin) + composition (L1) + sum constraint."""
    loss = w_cls * F.multilabel_soft_margin_loss(z, y) + w_reg * F.l1_loss(r, c)
    if cfg["sum_constraint"]:
        loss = loss + cfg["sum_constraint"] * (torch.abs(r.sum(1) - c.sum(1)).mean() + torch.abs(torch.sigmoid(z).sum(1) - y.sum(1)).mean())
    return loss


def loss_fn(mlc, reg, y_t, c_t, cfg, w_cls=1.0, w_reg=1.0):
    """mlc / y_t columns: [12 presence | set_classes | cb_classes]."""
    S, K = cfg["set_classes"], cfg["cb_classes"]
    loss = task_loss(mlc[:, :12], reg, y_t[:, :12], c_t, cfg, w_cls, w_reg)
    if S:
        loss = loss + cfg["set_head_weight"] * soft_ce(mlc[:, 12:12 + S], y_t[:, 12:12 + S])
    if K:
        loss = loss + cfg["codebook_head_weight"] * soft_ce(mlc[:, 12 + S:], y_t[:, 12 + S:])
    return loss


class Teachers:
    """Frozen teacher ensemble for online distillation."""

    def __init__(self, paths, targets, dev, knob_lambda=25.0, eps=1e-6, log=print):
        self.targets, self.lam, self.eps, self.models = targets, knob_lambda, eps, []
        for p in paths:
            ck = torch.load(p, map_location=dev, weights_only=True)
            m = build_model(ck["config"]).to(dev).eval()
            m.load_state_dict(ck["model"])
            if targets in ("stack", "knob"):
                h = ck.get("heads")
                assert h is not None, f"{p}: no post-hoc heads; run python -m setbooknir.stack fit first"
                m._lp = nn.Linear(h["d_m"], len(ck["sets"])).to(dev); m._lp.load_state_dict(h["set"])
                m._cb = nn.Linear(h["d"], len(ck["codebook"])).to(dev); m._cb.load_state_dict(h["codebook"])
                self.sets, self.codebook = ck["sets"].float().to(dev), ck["codebook"].float().to(dev)
            for q in m.parameters():
                q.requires_grad_(False)
            self.models.append(m)
            log(f"teacher: {p}")

    @torch.no_grad()
    def __call__(self, x):
        P, R, Q, Pcb = 0.0, 0.0, 0.0, 0.0
        for m in self.models:
            f = m.forward_features(x)
            mf, rf = m.mlc_proj(f), m.reg_proj(f)
            P = P + torch.sigmoid(m.head(mf)); R = R + m.head2(rf)
            if self.targets != "logits":
                Q = Q + torch.softmax(m._lp(mf), -1)
                Pcb = Pcb + torch.softmax(m._cb(torch.cat([mf, rf], 1)), -1)
        n, eps = len(self.models), self.eps
        P, R = P / n, R / n
        if self.targets == "logits":
            return torch.logit(P.clamp(eps, 1 - eps)), R
        Q, Pcb, S = Q / n, Pcb / n, self.sets
        Pc = P.clamp(eps, 1 - eps)
        score = torch.log(Q.clamp_min(1e-30)) + torch.log(Pc) @ S.t() + torch.log1p(-Pc) @ (1 - S).t()
        marg = torch.softmax(score, -1) @ S                      # per-fibre marginal of the joint posterior over sets
        comp = 0.5 * (R + Pcb @ self.codebook)
        if self.targets == "knob":
            tv, ti = Pcb.topk(50, -1)
            comp = knob_decision_torch(comp, tv, self.codebook[ti], self.lam)
        return torch.logit(marg.clamp(eps, 1 - eps)), comp


@torch.no_grad()
def predict(model, X, bs=4096):
    """Plain read-out: presence logits (first 12 columns) and percentages, one deterministic pass."""
    model.eval()
    Z, R = [], []
    for i in range(0, len(X), bs):
        z, r = model(X[i:i + bs])
        Z.append(z[:, :12].float().cpu()); R.append(r.float().cpu())
    return torch.cat(Z).numpy(), torch.cat(R).numpy()


def train(cfg, split, seed, out, data=None, log=print):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(seed, cfg["deterministic"])
    X, Y, C = load_train(data)
    tr, va = load_split(split, len(Y))
    sets, set_idx, codebook, cb_idx = output_spaces(Y[tr], C[tr])
    cfg = {**cfg, "set_classes": len(sets) if cfg["set_head_weight"] else 0,
           "cb_classes": len(codebook) if cfg["codebook_head_weight"] else 0}
    Xtr, Ytr, Ctr = (torch.from_numpy(a[tr]).to(dev) for a in (X, Y, C))
    Xva = torch.from_numpy(X[va]).to(dev) if va is not None else None
    n, bs = len(tr), cfg["batch_size"]
    log(f"split {split}: {n:,} training scans, {len(sets)} fibre sets, {len(codebook)} compositions; seed {seed}")

    model = build_model(cfg).to(dev)
    set_t = torch.from_numpy(set_idx).to(dev); cb_t = torch.from_numpy(cb_idx).to(dev)
    if cfg["set_classes"]:
        model.set_table.copy_(torch.from_numpy(sets))
    fmodel = torch.compile(model, dynamic=False) if cfg["compile"] else model
    opt = FastLamb(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    kd = cfg["distill"]
    teachers = None
    if kd:
        teachers = Teachers([p.format(seed=seed) for p in kd["teachers"]], kd["targets"], dev, kd.get("knob_lambda", 25.0), log=log)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=cfg["milestones"], gamma=cfg["gamma"])

    history, t0 = [], time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        te, tot, nb = time.time(), torch.zeros((), device=dev), 0
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            y_in = Ytr[idx]
            if cfg["set_classes"]:
                y_in = torch.cat([y_in.float(), F.one_hot(set_t[idx], len(sets)).float()], 1)
            if cfg["cb_classes"]:
                y_in = torch.cat([y_in.float(), F.one_hot(cb_t[idx], len(codebook)).float()], 1)
            x, y_t, c_t = mixup(Xtr[idx], y_in, Ctr[idx], cfg["mixup_alpha"], cfg["mixup_prob"])
            if teachers is not None:
                zT, rT = teachers(x)
            opt.zero_grad()
            mlc, reg = fmodel(x)
            if teachers is not None:
                al, be = kd["alpha"], kd["beta"]
                loss = loss_fn(mlc, reg, y_t, c_t, cfg, 1 - al, 1 - be) + al * F.mse_loss(mlc[:, :12], zT) + be * F.l1_loss(reg, rT)
            else:
                loss = loss_fn(mlc, reg, y_t, c_t, cfg)
            loss.backward()
            opt.step()
            tot += loss.detach(); nb += 1
        sched.step()
        rec = {"epoch": ep + 1, "loss": float(tot / nb), "seconds": round(time.time() - te, 1)}
        msg = f"epoch {ep + 1:3d}  {rec['seconds']:5.1f}s  loss {rec['loss']:.4f}"
        if Xva is not None:
            rec["val"] = compute_metrics(*predict(model, Xva), Y[va], C[va])
            msg += "  | val " + fmt(rec["val"])
        log(msg)
        history.append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "split": split, "seed": seed,
                "sets": torch.from_numpy(sets.astype(np.int64)), "codebook": torch.from_numpy(codebook),
                "history": history, "minutes": round((time.time() - t0) / 60, 1)}, out)
    log(f"saved {out}")
    return model, history


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="all", choices=["all", "dev"], help="all: every training scan; dev: development split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs", help="output directory (or a .pt path)")
    ap.add_argument("--data", default=None, help="FabricSpectra directory (default: $FABRICSPECTRA_DATA or ./data)")
    a = ap.parse_args()
    user = json.load(open(a.config))
    unknown = set(user) - set(DEFAULTS) - {"name"}
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **user}
    name = user.get("name", os.path.splitext(os.path.basename(a.config))[0])
    out = a.out if a.out.endswith(".pt") else os.path.join(a.out, f"{name}_{a.split}_s{a.seed}.pt")
    train(cfg, a.split, a.seed, out, a.data)


if __name__ == "__main__":
    main()
