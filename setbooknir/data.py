"""FabricSpectra data and splits.

The data directory holds the six files of the official release (Harvard Dataverse, doi:10.7910/DVN/QZRWLK):
train_data.npy, train_multi_label_classification_label.npy, train_composition_label.npy and the same three for test.
Set it with --data or the FABRICSPECTRA_DATA environment variable (default: ./data).

Splits:
  all  -- every training scan (the final models of the paper).
  dev  -- the development split: 90% of the fabrics for training, 10% held out for validation (splits/val_idx.npy).
          Fabrics are recovered from runs of identical consecutive compositions (splits/groups.npy) and the held-out
          fabrics are stratified by fibre set, so no fabric has scans on both sides. Every design choice is made here.
"""
import hashlib
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(ROOT, "splits")
FIBRES = ["SP", "CA", "L", "R", "M", "T", "P", "W", "N", "A", "S", "C"]
DEV_FINGERPRINT = "4dbaa91f196e76be"


def data_dir(path=None):
    return path or os.environ.get("FABRICSPECTRA_DATA", os.path.join(ROOT, "data"))


def _load(path, part):
    names = ("data", "multi_label_classification_label", "composition_label")
    files = [os.path.join(data_dir(path), f"{part}_{n}.npy") for n in names]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"missing {missing}; download FabricSpectra (doi:10.7910/DVN/QZRWLK) and pass --data")
    Y = np.load(files[1]).astype(np.int64)
    X = np.load(files[0]).astype(np.float32).reshape(len(Y), 2, 200)
    return X, Y, np.load(files[2]).astype(np.float32)


def load_train(path=None):
    """X (N, 2, 200) float32 [absorbance, reflectance], Y (N, 12) int64 presence, C (N, 12) float32 percentages."""
    return _load(path, "train")


def load_test(path=None):
    return _load(path, "test")


def fingerprint(tr, va):
    h = hashlib.sha256()
    h.update(np.asarray(tr, dtype=np.int64).tobytes()); h.update(b"|")
    h.update(np.asarray(va, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def load_split(name, n_train=293148):
    """(train indices, validation indices or None) into the training arrays."""
    if name == "all":
        return np.arange(n_train), None
    if name != "dev":
        raise ValueError(f"unknown split {name!r}: all | dev")
    tr = np.load(os.path.join(SPLIT_DIR, "train_full_idx.npy"))
    va = np.load(os.path.join(SPLIT_DIR, "val_idx.npy"))
    if fingerprint(tr, va) != DEV_FINGERPRINT:
        raise RuntimeError("development split files do not match their fingerprint")
    return tr, va


def load_groups():
    """Fabric id of every training scan (for fabric-level bootstrap intervals)."""
    return np.load(os.path.join(SPLIT_DIR, "groups.npy"))


def output_spaces(Y, C):
    """The two discrete output spaces, from the training scans of a split: the fibre sets (S, 12) and the distinct
    compositions rounded to 0.1 pp (K, 12), with every scan's index into each."""
    sets, set_idx = np.unique(Y, axis=0, return_inverse=True)
    codebook, cb_idx = np.unique(np.round(C, 1), axis=0, return_inverse=True)
    return sets, set_idx.reshape(-1), codebook.astype(np.float32), cb_idx.reshape(-1)
