"""Evaluate one model or an ensemble with the post-hoc stack.

    python -m setbooknir.evaluate --ckpt runs/setbooknir_e_{internimage,maxvit,overlock}_all_s0.pt          # test set
    python -m setbooknir.evaluate --ckpt runs/configA_dev_s0.pt --on dev --variant basic                   # development split

Checkpoints listed together form one ensemble: their stack outputs are averaged before the decisions. Every read-out
row of setbooknir.readout.readouts is printed; the post-hoc heads must have been fitted (python -m setbooknir.stack fit).
"""
import argparse

import numpy as np
import torch

from .data import load_split, load_test, load_train
from .metrics import compute_metrics, fmt
from .readout import readouts
from .stack import infer


def main():
    ap = argparse.ArgumentParser(description="evaluate a model or an ensemble with the post-hoc stack")
    ap.add_argument("--ckpt", nargs="+", required=True, help="one checkpoint, or several to ensemble")
    ap.add_argument("--on", default="test", choices=["test", "dev"], help="test set, or the development split's held-out fabrics")
    ap.add_argument("--variant", default="final", choices=["final", "basic"], help="see setbooknir.stack")
    ap.add_argument("--passes", type=int, default=16, help="Monte-Carlo drop-path passes")
    ap.add_argument("--lam", type=float, nargs="+", default=[25.0], help="knob values of the Bayes-decision read-out")
    ap.add_argument("--data", default=None)
    ap.add_argument("--save", default=None, help="save the per-scan decisions of every read-out to this .npz")
    a = ap.parse_args()
    if a.on == "test":
        X, Y, C = load_test(a.data)
    else:
        X, Y, C = load_train(a.data)
        _, va = load_split("dev", len(Y))
        for p in a.ckpt:
            assert torch.load(p, map_location="cpu", weights_only=True)["split"] == "dev", f"{p} was not trained on the dev split"
        X, Y, C = X[va], Y[va], C[va]
    members = [infer(p, X, a.passes, a.variant) for p in a.ckpt]
    R = readouts(members, a.lam)
    print(f"{len(a.ckpt)} model(s), {a.on}: {len(Y):,} scans, {a.passes} passes, stack variant {a.variant}")
    for k, (pres, comp) in R.items():
        print(f"  {k:20s} " + fmt(compute_metrics(pres, comp, Y, C)))
    if a.save:
        np.savez_compressed(a.save, **{f"{k}/presence": v[0].astype(np.int8) for k, v in R.items()},
                            **{f"{k}/composition": v[1].astype(np.float32) for k, v in R.items()})


if __name__ == "__main__":
    main()
