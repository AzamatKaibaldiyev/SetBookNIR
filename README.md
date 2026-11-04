# SetBookNIR

Code and trained models for **Set-Level Decoding and Bayes-Decision Read-Outs for Joint Multi-Label Classification and
Composition Estimation from NIR Textile Spectra**.

Each near-infrared spectrum of a textile has two targets: which of 12 fibres it contains, and in what percentages. The
benchmark scores them strictly: exact-match accuracy over the 12 labels, and tolerance accuracy (all 12 percentages
within ±γ points). SetBookNIR makes these decisions at the level of the whole output:

- **Auxiliary heads.** During training, a *set head* over the 281 fibre sets seen in training and a *codebook head* over the
  3,341 distinct training compositions shape the features.
- **Post-hoc inference stack** (on the frozen network, fitted on training data only):
  - Monte-Carlo stochastic depth (16 drop-path passes, averaged).
  - Joint set decoding: the most probable fibre set under the set head and the per-fibre probabilities, which is the
    Bayes decision for exact-match accuracy.
  - A codebook posterior over the training compositions.
- **Bayes-decision read-out.** Per scan, it picks the composition with the lowest expected loss under that posterior. A
  single knob λ moves it from the MSE-optimal mean (λ = 0) to the most probable composition (λ → ∞).
- **Distillation.** The stack's decisions move into a single-pass ConvNeXtV2-atto student.

**SetBookNIR-E** is the three-backbone ensemble (1-D InternImage-, MaxViT- and OverLoCK-atto) with the stack.
**SetBookNIR-S** is the distilled single-pass student.

## Results (FabricSpectra test set)

All rows: trained on the full training set, last epoch of a fixed schedule, test set scored once, mean over 3 seeds
(± sd for Acc). HL in %, MSE in pp², γ±k in %.

| Model | Params | Acc ↑ | HL ↓ | MSE ↓ | γ±1 ↑ | γ±5 ↑ |
|---|---|---|---|---|---|---|
| H-MT-FCN (best published, single run) | – | 77.81 | 2.7 | 55.75 | 35.52 | 64.14 |
| ConvNeXtV2-A, re-run (+ final LayerNorm, drop-path 0.1) | 3.5 M | 75.43 ± 0.11 | 3.022 | 63.91 | 37.86 | 66.14 |
| **SetBookNIR-S**, single pass | 3.3 M | 78.78 ± 0.02 | 2.615 | 55.57 | 34.08 | 67.22 |
| SetBookNIR-S + own stack | 3.3 M | 78.86 ± 0.04 | 2.615 | 54.40 | 31.71 | 66.42 |
| **SetBookNIR-E** (λ = 25) | 25 M | 79.14 ± 0.07 | 2.605 | 53.35 | 47.10 | 65.57 |

Every design choice and λ was selected on a fabric-level development split (below), never on the test set.

## Installation

```bash
pip install -r requirements.txt      # torch, numpy, timm
```

Tested with Python 3.11, PyTorch 2.11 (CUDA 13.0), numpy 2.4 and timm 1.0, on one laptop GPU (RTX PRO 4000, 16 GB).

## Data

Download FabricSpectra (Kim & Chi, ICASSP 2026) from Harvard Dataverse,
[doi:10.7910/DVN/QZRWLK](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QZRWLK).
Put the six `.npy` files (`train_data.npy`, `train_multi_label_classification_label.npy`, `train_composition_label.npy`
and the three `test_*` files) in `data/`, or point `FABRICSPECTRA_DATA` (or `--data`) to their folder.

`splits/` holds the development split: `val_idx.npy` lists 10% of the fabrics (30,328 scans), stratified by fibre set;
`train_full_idx.npy` lists the rest. The dataset has about 20 near-identical replicate scans per fabric, so the split is
made by fabric, never by scan. `groups.npy` gives the fabric of every training scan. Fabrics were recovered from runs of
identical consecutive compositions (14,192 groups).

## Trained models

The [v1.0 release](https://github.com/AzamatKaibaldiyev/SetBookNIR/releases/tag/v1.0) holds the 21 checkpoints of the
paper (3 seeds each, 15-80 MB per file) with their fitted post-hoc heads. Put them in `runs/`:

```bash
mkdir -p runs
for f in setbooknir_e_{internimage,maxvit,overlock}_all_s{0,1,2}.pt setbooknir_s_all_s{0,1,2}.pt \
         teacher_{internimage,maxvit,overlock}_all_s{0,1,2}.pt; do
  curl -L -o runs/$f https://github.com/AzamatKaibaldiyev/SetBookNIR/releases/download/v1.0/$f
done
```

| Files | Model |
|---|---|
| `setbooknir_e_{internimage,maxvit,overlock}_all_s{0,1,2}.pt` | SetBookNIR-E members (set and codebook heads) |
| `setbooknir_s_all_s{0,1,2}.pt` | SetBookNIR-S (ConvNeXtV2-atto with a codebook head) |
| `teacher_{internimage,maxvit,overlock}_all_s{0,1,2}.pt` | the student's teachers (codebook head only), with the linear post-hoc heads used to compute its targets |

## Evaluation

```bash
# SetBookNIR-E, seed 0 (repeat for s1, s2): the three members form one ensemble -> row "stack knob 25"
python -m setbooknir.evaluate --ckpt runs/setbooknir_e_{internimage,maxvit,overlock}_all_s0.pt

# SetBookNIR-S, seed 0: row "plain" = single pass; row "MC + set decoding" = + own stack
python -m setbooknir.evaluate --ckpt runs/setbooknir_s_all_s0.pt
```

Each command prints every read-out step: `plain`, then `MC`, `MC + set decoding`, `stack mean`, `stack knob 25` and
`stack MAP`. Use `--lam 0 5 25 100` for points on the knob's frontier, and `--save` to keep the per-scan decisions.
With the released checkpoints, these commands give (mean over seeds 0-2) Acc 79.16 / MSE 53.36 for SetBookNIR-E, and
78.78 / 55.57 (single pass) and 78.84 / 54.40 (own stack) for SetBookNIR-S: Acc and MSE within 0.03 of the table. The small
differences come from the random Monte-Carlo passes and from the convolution kernels cuDNN selects (TF32).

## Training from scratch

```bash
# SetBookNIR-E members (0.7-2.3 h each on the laptop GPU above), then their post-hoc heads (a few minutes each)
for b in internimage maxvit overlock; do for s in 0 1 2; do
  python -m setbooknir.train --config configs/setbooknir_e_$b.json --split all --seed $s
  python -m setbooknir.stack fit --ckpt runs/setbooknir_e_${b}_all_s$s.pt
done; done

# SetBookNIR-S: teachers with the codebook head, their linear heads, then the student (2.3 h); student seed s uses teacher seed s
for b in internimage maxvit overlock; do for s in 0 1 2; do
  python -m setbooknir.train --config configs/teacher_$b.json --split all --seed $s
  python -m setbooknir.stack fit --ckpt runs/teacher_${b}_all_s$s.pt
done; done
for s in 0 1 2; do
  python -m setbooknir.train --config configs/setbooknir_s.json --split all --seed $s
  python -m setbooknir.stack fit --ckpt runs/setbooknir_s_all_s$s.pt
done
```

`--split dev` trains on the development split instead and reports validation metrics after every epoch. Use it for model
selection, then evaluate with `--on dev`. Example, the stack on Config A (ConvNeXtV2-atto without auxiliary heads; the
stack ablation of the paper):

```bash
python -m setbooknir.train --config configs/config_a.json --split dev --seed 0
python -m setbooknir.stack fit --ckpt runs/config_a_dev_s0.pt
python -m setbooknir.evaluate --ckpt runs/config_a_dev_s0.pt --on dev --variant basic
```

`--variant basic` uses the linear set and codebook heads, for networks without a trained codebook head. The default
`final` variant uses the set head fitted on drop-path features and averages the trained codebook head with a fitted
two-layer one.

## Code

| File | Content |
|---|---|
| `setbooknir/models/` | ConvNeXtV2-1D and the 1-D InternImage, MaxViT and OverLoCK backbones; set and codebook heads |
| `setbooknir/train.py` | training recipe, auxiliary heads, online distillation (`stack`, `knob` or `logits` targets) |
| `setbooknir/stack.py` | post-hoc heads (`fit`) and Monte-Carlo inference (`infer`) |
| `setbooknir/readout.py` | joint set decoding, Bayes-decision read-out, ensembling |
| `setbooknir/metrics.py` | the benchmark's metrics |
| `tests/run_tests.py` | CPU tests on synthetic data |

## Acknowledgements

The backbones are 1-D adaptations of ConvNeXt V2 (Woo et al., CVPR 2023), InternImage (Wang et al., CVPR 2023), MaxViT
(Tu et al., ECCV 2022) and OverLoCK (Lou & Yu, CVPR 2025); see the original projects for their licences. The dataset is
FabricSpectra (Kim & Chi, ICASSP 2026).

## Citation

To be added.
