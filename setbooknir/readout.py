"""Decisions on top of the (Monte-Carlo averaged) head outputs of one model or an ensemble.

Joint set decoding: the fibre set s in S maximising  log q(s|x) + sum_j [s_j log p_j + (1 - s_j) log(1 - p_j)],
the mode of the posterior over sets (the Bayes decision for exact-match accuracy).

Bayes-decision composition read-out (knob lambda): per scan, among the stack mean and the top-50 entries of the codebook
posterior pi (renormalised), the candidate u minimising  E_pi[MSE(u)] - lambda * 12/100 * P_pi[all 12 within +-1 pp of u].
lambda = 0 is the MSE-optimal mean; lambda -> inf is the most probable entry (MAP).
"""
import numpy as np
import torch


def joint_decode(P, Q, sets):
    """P (N, 12) fibre probabilities, Q (N, S) set posterior, sets (S, 12) -> (N, 12) decisions."""
    S = sets.astype(float)
    sc = np.log(np.clip(Q, 1e-30, None)) + np.log(np.clip(P, 1e-9, 1)) @ S.T + np.log(np.clip(1 - P, 1e-9, 1)) @ (1 - S).T
    return sets[sc.argmax(1)]


def sparse_posterior(members, K=50):
    """Codebook posterior of one model or the average of several, from each model's top-K entries (indices, probabilities)."""
    if len(members) == 1:
        return members[0]["cb_top_i"].astype(np.int64), members[0]["cb_top_p"].astype(np.float64)
    N, n_cb = len(members[0]["cb_top_i"]), members[0]["n_codebook"]
    score = np.zeros((N, n_cb), np.float32)
    for d in members:
        k = d["cb_top_i"].shape[1]
        np.add.at(score, (np.repeat(np.arange(N), k), d["cb_top_i"].astype(np.int64).ravel()), d["cb_top_p"].astype(np.float32).ravel())
    idx = np.argpartition(-score, K, 1)[:, :K]
    return idx, np.take_along_axis(score, idx, 1).astype(np.float64)


def knob_decide(mean, idx, p, cbk, lams):
    """mean (N, 12); idx, p (N, K) codebook entries and probabilities; cbk (n_cb, 12). Returns {lambda: (N, 12), "MAP": (N, 12)}."""
    p = p / p.sum(1, keepdims=True); Y = cbk[idx].astype(np.float64)
    out = {l: np.zeros_like(mean) for l in lams}; out["MAP"] = Y[np.arange(len(Y)), p.argmax(1)]
    for i in range(0, len(mean), 1500):
        y, pp, m = Y[i:i + 1500], p[i:i + 1500], mean[i:i + 1500]
        cand = np.concatenate([m[:, None, :], y], 1)
        diff = cand[:, :, None, :] - y[:, None, :, :]
        emse = (pp[:, None, :] * (diff ** 2).mean(-1)).sum(-1)
        phit = (pp[:, None, :] * (np.abs(diff).max(-1) <= 1.0)).sum(-1)
        for l in lams:
            j = np.argmin(emse - l * phit / 100.0 * 12, 1)
            out[l][i:i + 1500] = cand[np.arange(len(j)), j]
    return out


def knob_decision_torch(mean, p, Y, lam):
    """The same decision on the GPU (distillation targets). mean (B, 12); p (B, K) top-K probabilities; Y (B, K, 12)."""
    p = p / p.sum(1, keepdim=True)
    cand = torch.cat([mean[:, None, :], Y], 1)
    diff = cand[:, :, None, :] - Y[:, None, :, :]
    emse = (p[:, None, :] * (diff ** 2).mean(-1)).sum(-1)
    phit = (p[:, None, :] * (diff.abs().amax(-1) <= 1.0).to(p.dtype)).sum(-1)
    j = torch.argmin(emse - lam * phit / 100.0 * 12, 1)
    return cand[torch.arange(len(j), device=j.device), j]


def readouts(members, lams=(25,)):
    """members: outputs of setbooknir.stack.infer for the models of one ensemble (or one model), on the same scans.
    Returns {name: (presence decisions, percentages)}, each row adding one step:
      plain              the networks' own heads, one deterministic pass
      MC                 Monte-Carlo averages: presence p > 0.5, regression head
      MC + set decoding  joint set decoding; composition: MC-averaged regression head
      stack mean         composition 1/2 (regression + codebook expectation), the MSE-optimal read-out
      stack knob <l>     Bayes-decision composition read-out with lambda = l
      stack MAP          most probable codebook entry
    """
    m = lambda k: np.mean([d[k].astype(np.float64) for d in members], 0)  # noqa: E731
    sets, cbk = members[0]["sets"], members[0]["codebook"]
    pres = joint_decode(m("P"), m("Q"), sets)
    mean = 0.5 * (m("R") + m("E_cb"))
    out = {"plain": ((m("P0") > 0.5).astype(int), m("R0")), "MC": ((m("P") > 0.5).astype(int), m("R")),
           "MC + set decoding": (pres, m("R")), "stack mean": (pres, mean)}
    idx, p = sparse_posterior(members)
    dec = knob_decide(mean, idx, p, cbk, tuple(lams))
    for l in lams:
        out[f"stack knob {l:g}"] = (pres, dec[l])
    out["stack MAP"] = (pres, dec["MAP"])
    return out
