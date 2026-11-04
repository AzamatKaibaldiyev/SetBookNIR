"""The post-hoc inference stack: heads fitted on a frozen network, Monte-Carlo stochastic depth at inference.

    python -m setbooknir.stack fit --ckpt runs/setbooknir_e_maxvit_all_s0.pt      # adds the fitted heads to the checkpoint

Heads (on the frozen projections m = mlc_proj and c = reg_proj of the training scans of the checkpoint's split), each a
softmax fitted with soft cross-entropy (AdamW lr 1e-3, weight decay 1e-2, 10 epochs, cosine, batch 512, feature and target
mixup with the training recipe's alpha and probability):
  set          linear set head over the training fibre sets, on m (one deterministic pass)
  set_noisy    the same, fitted on m from 4 drop-path passes (a random pass per scan and batch)
  codebook     linear codebook head over the distinct training compositions, on [m, c]
  codebook_mlp Linear-GELU-LayerNorm-Linear codebook head on [m, c]

Variants of the stack (setbooknir.stack.infer):
  final  set_noisy; codebook posterior = 1/2 (trained codebook head + codebook_mlp)   (SetBookNIR-E and -S)
  basic  set; codebook posterior = codebook (linear)                                 (networks without a trained codebook head)
Distillation teachers use the linear heads, one deterministic pass.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_split, load_train
from .models import build_model, mc_mode

FIT = dict(epochs=10, lr=1e-3, weight_decay=1e-2, batch_size=512, noisy_passes=4)


def load(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    return ck, model.to(dev).eval()


@torch.no_grad()
def features(model, X, passes=0, bs=4096, dev="cuda"):
    """[m, c] features of X (numpy (N, 2, 200)) for each pass: one deterministic pass (passes = 0) or `passes` drop-path passes."""
    T = max(passes, 1)
    Fs = [[] for _ in range(T)]
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(dev)
        for t in range(T):
            mc_mode(model, passes > 0)
            f = model.forward_features(x)
            Fs[t].append(torch.cat([model.mlc_proj(f), model.reg_proj(f)], 1))
    model.eval()
    return [torch.cat(f) for f in Fs]


def fit_head(Ftr, n_out, target_fn, mix_alpha, mix_prob, deep=False):
    """Ftr (N, d) features, or (P, N, d) feature passes (a random pass per scan and batch)."""
    dev, d = Ftr.device, Ftr.shape[-1]
    head = (torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.LayerNorm(d), torch.nn.Linear(d, n_out)) if deep
            else torch.nn.Linear(d, n_out)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=FIT["lr"], weight_decay=FIT["weight_decay"])
    n, bs = Ftr.shape[-2], FIT["batch_size"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FIT["epochs"] * ((n + bs - 1) // bs))
    for _ in range(FIT["epochs"]):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            f = Ftr[idx] if Ftr.dim() == 2 else Ftr[torch.randint(0, Ftr.shape[0], (len(idx),), device=dev), idx]
            t = target_fn(idx)
            if np.random.rand() < mix_prob:
                lam = float(np.random.beta(mix_alpha, mix_alpha))
                p = torch.randperm(len(idx), device=dev)
                f, t = lam * f + (1 - lam) * f[p], lam * t + (1 - lam) * t[p]
            loss = -(t * F.log_softmax(head(f), -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return head.eval()


def fit(path, data=None, out=None):
    """Fit the four post-hoc heads of a checkpoint on its training scans and store them in the checkpoint (or `out`)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    ck, model = load(path, dev)
    cfg, seed = ck["config"], ck["seed"]
    X, Y, C = load_train(data)
    tr, _ = load_split(ck["split"], len(Y))
    sets, codebook = ck["sets"].numpy(), ck["codebook"].numpy()
    w2 = 2 ** np.arange(12)
    lut = np.full(4096, -1, np.int64); lut[sets @ w2] = np.arange(len(sets))
    set_t = torch.from_numpy(lut[Y[tr] @ w2]).to(dev)
    cb_idx = np.unique(np.round(C[tr], 1), axis=0, return_inverse=True)[1].reshape(-1)
    cb_t = torch.from_numpy(cb_idx).to(dev)
    assert int(set_t.min()) >= 0 and int(cb_t.max()) < len(codebook)
    a, p = cfg["mixup_alpha"], cfg["mixup_prob"]
    on_cb = lambda i: F.one_hot(cb_t[i], len(codebook)).float()  # noqa: E731
    on_set = lambda i: F.one_hot(set_t[i], len(sets)).float()  # noqa: E731

    torch.manual_seed(seed); np.random.seed(seed)
    Ftr = features(model, X[tr], dev=dev)[0]
    d_m = model.head.in_features
    h_cb = fit_head(Ftr, len(codebook), on_cb, a, p)
    h_set = fit_head(Ftr[:, :d_m], len(sets), on_set, a, p)
    state = (torch.get_rng_state(), torch.cuda.get_rng_state() if dev.type == "cuda" else None, np.random.get_state())
    h_mlp = fit_head(Ftr, len(codebook), on_cb, a, p, deep=True)
    _restore(state)
    del Ftr
    state = (torch.get_rng_state(), torch.cuda.get_rng_state() if dev.type == "cuda" else None, np.random.get_state())
    Fn = torch.stack([f[:, :d_m] for f in features(model, X[tr], FIT["noisy_passes"], dev=dev)])
    h_noisy = fit_head(Fn, len(sets), on_set, a, p)
    _restore(state)
    cpu = lambda h: {k: v.cpu() for k, v in h.state_dict().items()}  # noqa: E731
    ck["heads"] = {"d_m": d_m, "d": h_cb.in_features, "set": cpu(h_set), "set_noisy": cpu(h_noisy), "codebook": cpu(h_cb),
                   "codebook_mlp": cpu(h_mlp)}
    torch.save(ck, out or path)
    print(f"heads fitted on {len(tr):,} training scans -> {out or path}")


def _restore(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1])
    np.random.set_state(state[2])


def build_heads(ck, dev):
    h = ck["heads"]
    n_set, n_cb, d_m, d = len(ck["sets"]), len(ck["codebook"]), h["d_m"], h["d"]
    heads = {"set": torch.nn.Linear(d_m, n_set), "set_noisy": torch.nn.Linear(d_m, n_set), "codebook": torch.nn.Linear(d, n_cb),
             "codebook_mlp": torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.LayerNorm(d), torch.nn.Linear(d, n_cb))}
    heads = {k: m for k, m in heads.items() if k in h}       # teachers carry only the linear heads used for distillation
    for k, m in heads.items():
        m.load_state_dict(h[k]); m.to(dev).eval()
    return heads


@torch.no_grad()
def infer(path, X, passes=16, variant="final", topk=50, bs=4096, seed=None):
    """Outputs of one checkpoint on scans X (numpy (N, 2, 200)): plain pass (P0, R0) and Monte-Carlo averages over `passes`
    drop-path passes of the fibre probabilities P, percentages R, set posterior Q and codebook posterior (expectation E_cb
    and its top-k entries)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    ck, model = load(path, dev)
    assert "heads" in ck, f"{path}: no post-hoc heads; run python -m setbooknir.stack fit --ckpt {path}"
    heads = build_heads(ck, dev)
    trained_cb = getattr(model, "cb_classes", 0) == len(ck["codebook"])
    if variant == "final":
        assert trained_cb, "the final stack averages the trained codebook head with codebook_mlp: use --variant basic"
    CB = ck["codebook"].float().to(dev)
    torch.manual_seed(ck["seed"] if seed is None else seed)
    out = {k: [] for k in ("P0", "R0", "P", "R", "Q", "E_cb", "cb_top_i", "cb_top_p")}
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(dev)
        mc_mode(model, False)
        z0, r0 = model(x)
        out["P0"].append(torch.sigmoid(z0[:, :12])); out["R0"].append(r0)
        P = R = Q = Pcb = 0.0
        for _ in range(passes):
            mc_mode(model, True)
            f = model.forward_features(x)
            mf, rf = model.mlc_proj(f), model.reg_proj(f)
            fc = torch.cat([mf, rf], 1)
            P = P + torch.sigmoid(model.head(mf)); R = R + model.head2(rf)
            if variant == "final":
                Q = Q + torch.softmax(heads["set_noisy"](mf), -1)
                Pcb = Pcb + 0.5 * (torch.softmax(model.head4(rf), -1) + torch.softmax(heads["codebook_mlp"](fc), -1))
            else:
                Q = Q + torch.softmax(heads["set"](mf), -1)
                Pcb = Pcb + torch.softmax(heads["codebook"](fc), -1)
        P, R, Q, Pcb = P / passes, R / passes, Q / passes, Pcb / passes
        P = torch.sigmoid(torch.logit(P.clamp(1e-6, 1 - 1e-6)))
        tv, ti = Pcb.topk(topk, -1)
        for k, v in (("P", P), ("R", R), ("Q", Q.half()), ("E_cb", Pcb @ CB), ("cb_top_p", tv.half()), ("cb_top_i", ti.short())):
            out[k].append(v)
    model.eval()
    res = {k: torch.cat(v).cpu().numpy() for k, v in out.items()}
    res.update(sets=ck["sets"].numpy(), codebook=ck["codebook"].numpy(), n_codebook=len(ck["codebook"]))
    return res


def main():
    ap = argparse.ArgumentParser(description="post-hoc inference stack")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit the post-hoc heads of a checkpoint on its training scans")
    f.add_argument("--ckpt", nargs="+", required=True)
    f.add_argument("--data", default=None)
    a = ap.parse_args()
    for p in a.ckpt:
        fit(p, a.data)


if __name__ == "__main__":
    main()
