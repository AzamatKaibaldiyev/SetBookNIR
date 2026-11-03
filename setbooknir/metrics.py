"""Task metrics of the FabricSpectra benchmark (decision threshold 0.5).

acc      exact-match accuracy over the 12 presence labels
hamming  Hamming loss
mse      mean squared error of the percentages, over all N x 12 entries
g{k}     tolerance accuracy: fraction of scans with all 12 percentages within k percentage points
"""
import numpy as np

GAMMAS = (1, 2, 3, 5)


def compute_metrics(pres, comp, y, c):
    """pres: (N, 12) presence decisions in {0, 1}, or logits (positive = present, i.e. sigmoid > 0.5);
    comp, c: (N, 12) percentages; y: (N, 12) labels in {0, 1}."""
    pred = (np.asarray(pres) > 0).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    comp, c = np.asarray(comp, np.float64), np.asarray(c, np.float64)
    err = np.abs(comp - c)
    out = {"acc": float((pred == y).all(1).mean()), "hamming": float((pred != y).mean()), "mse": float(((comp - c) ** 2).mean())}
    for g in GAMMAS:
        out[f"g{g}"] = float((err <= g).all(1).mean())
    return out


def fmt(m):
    return (f"Acc {100 * m['acc']:.2f}  HL {100 * m['hamming']:.3f}  MSE {m['mse']:.2f}  "
            + "  ".join(f"g±{g} {100 * m[f'g{g}']:.2f}" for g in GAMMAS))
