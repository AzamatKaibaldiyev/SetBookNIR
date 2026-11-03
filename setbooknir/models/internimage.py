"""
InternImage1d — backbone only.

1D port of InternImage (OpenGVLab, CVPR 2023, arXiv:2211.05778) — "Exploring
Large-Scale Vision Foundation Models with Deformable Convolutions".

The token mixer is DCNv3, a DEFORMABLE convolution: instead of sampling a fixed
k-point grid, every output position PREDICTS ITS OWN sampling offsets from the
input — adaptive, data-dependent receptive fields. For NIR spectra this means a
block can learn to sample where the informative wavelengths are, rather than at
fixed distances.

DCNv3's three signatures (all kept in this 1D port):
  1) SEPARATED projections: a shared value projection (pointwise) with
     per-group deformable sampling (depthwise-style), keeping params low.
  2) MULTI-GROUP mechanism: G groups, each with its OWN offsets and
     modulation — the conv analogue of multi-head attention.
  3) SOFTMAX-normalized modulation over the K sample points (v2 used
     sigmoid) — the change that stabilized large-scale training.

1D sampling: for output position p, group g, sample point k:
    pos = p + (k - K//2) + Δp_{g,k}(x)         (Δ predicted, continuous)
    y_g(p) = Σ_k softmax_k(m)_{g,k} · value_g( pos )   (linear interpolation)
Offset/mask layers are ZERO-INITIALIZED, so at init DCNv3 behaves as a plain
uniform K-point local average — deformation is learned from there.

Block = pre-LN transformer skeleton with DCNv3 replacing attention:
    x = x + γ1·DCNv3(LN(x));   x = x + γ2·MLP(LN(x))

Stem: two k3/s2 convs (/4) + LN; k3/s2 + LN downsample between stages.
For a 200-point spectrum: 50 -> 25 -> 13 -> 7 tokens.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath


# Mean wavelength (nm) of each of the 200 sample points (±1nm per-sample error;
# original 228-point 900-1700nm grid with noisy ends trimmed). NOTE: spacing is
# NON-uniform (~4.3nm at the blue end -> ~3.1nm at the red end), which is why
# the "wavelength" positional-encoding mode exists.
WAVELENGTHS_NM = np.array([
    961.2, 965.48, 969.32, 973.26, 977.35, 981.18, 985.16, 989.06, 992.88,
    996.7, 1000.87, 1004.75, 1008.85, 1012.69, 1016.61, 1020.41, 1024.31,
    1028.23, 1032.01, 1036.02, 1039.85, 1043.77, 1047.65, 1051.41, 1055.29,
    1059.28, 1063.14, 1066.88, 1070.88, 1074.61, 1078.56, 1082.32, 1086.03,
    1089.74, 1093.55, 1097.25, 1101.26, 1105.32, 1109.02, 1112.85, 1116.69,
    1120.48, 1124.14, 1127.81, 1131.47, 1135.12, 1139.23, 1142.95, 1146.68,
    1150.53, 1154.39, 1158.06, 1161.68, 1165.3, 1168.92, 1172.86, 1176.66,
    1180.46, 1184.06, 1187.77, 1191.37, 1195.09, 1198.67, 1202.28, 1206.07,
    1209.89, 1213.64, 1217.2, 1220.92, 1224.53, 1228.08, 1231.73, 1235.28,
    1239.03, 1242.8, 1246.5, 1250.02, 1253.59, 1257.28, 1260.86, 1264.36,
    1267.85, 1271.63, 1275.23, 1278.79, 1282.41, 1286.17, 1289.65, 1293.18,
    1296.76, 1300.21, 1303.95, 1307.4, 1310.99, 1314.53, 1317.97, 1321.45,
    1324.98, 1328.4, 1332.16, 1335.85, 1339.26, 1342.85, 1346.25, 1349.75,
    1353.18, 1356.58, 1360.03, 1363.51, 1367.16, 1370.64, 1374.2, 1377.56,
    1380.92, 1384.42, 1387.86, 1391.21, 1394.78, 1398.33, 1401.7, 1405.1,
    1408.53, 1411.91, 1415.26, 1418.79, 1422.1, 1425.53, 1429.03, 1432.31,
    1435.68, 1438.96, 1442.38, 1445.9, 1449.17, 1452.53, 1456.05, 1459.56,
    1462.81, 1466.13, 1469.37, 1472.61, 1475.89, 1479.26, 1482.58, 1486.06,
    1489.28, 1492.55, 1495.83, 1499.34, 1502.54, 1505.8, 1509.03, 1512.39,
    1515.86, 1519.07, 1522.31, 1525.54, 1528.7, 1531.86, 1535.01, 1538.2,
    1541.53, 1544.86, 1548.31, 1551.53, 1554.88, 1558.0, 1561.12, 1564.24,
    1567.39, 1570.55, 1573.84, 1576.94, 1580.24, 1583.5, 1586.65, 1589.76,
    1592.83, 1596.04, 1599.23, 1602.48, 1605.63, 1608.68, 1611.88, 1615.09,
    1618.13, 1621.28, 1624.41, 1627.47, 1630.68, 1633.69, 1636.71, 1639.87,
    1643.03, 1646.09, 1649.18, 1652.23, 1655.36, 1658.63, 1661.64,
], dtype=np.float32)
WL_MIN_NM, WL_MAX_NM = 900.0, 1700.0   # normalization range for encodings


# ==================== LAYERS ====================

class LayerNormCF1d(nn.Module):
    """LayerNorm over channels for channels-first (B, C, L) tensors."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None] * x + self.bias[:, None]


class GRN1d(nn.Module):
    """ConvNeXtV2 GRN for channels-last (B, L, C)."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class DCNv31d(nn.Module):
    """Deformable Convolution v3, 1D. Operates on channels-last (B, L, C).

    An "offset network" (depthwise conv k3 + LN + GELU) predicts, per position
    and per group, K continuous sampling offsets and K modulation logits
    (softmax over K). Values are gathered with linear interpolation at
    p + (k - K//2) + Δ, weighted by the modulation, then output-projected.
    Offset and mask layers are zero-initialized (regular local average at init).
    """
    def __init__(self, dim, groups, kernel_size=3, offset_scale=1.0,
                 proj_drop=0.):
        super().__init__()
        assert dim % groups == 0, f"dim {dim} not divisible by groups {groups}"
        self.dim, self.G, self.K = dim, groups, kernel_size
        self.gc = dim // groups
        self.offset_scale = offset_scale

        self.value_proj = nn.Linear(dim, dim)
        self.offset_net = nn.Sequential(          # shared feature for Δ and m
            nn.Conv1d(dim, dim, 3, 1, 1, groups=dim),
        )
        self.offset_norm = nn.LayerNorm(dim)
        self.offset = nn.Linear(dim, groups * kernel_size)
        self.mask = nn.Linear(dim, groups * kernel_size)
        self.output_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # base grid: (K,) integer offsets around the center
        grid = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        self.register_buffer("grid", grid, persistent=False)

        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.mask.weight)
        nn.init.zeros_(self.mask.bias)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        G, K, gc = self.G, self.K, self.gc

        value = self.value_proj(x)                                # (B, L, C)
        feat = self.offset_net(x.transpose(1, 2)).transpose(1, 2)  # dw conv
        feat = F.gelu(self.offset_norm(feat))
        offsets = self.offset(feat).view(B, L, G, K) * self.offset_scale
        mask = self.mask(feat).view(B, L, G, K).softmax(dim=-1)   # (B, L, G, K)

        # sampling positions: p + grid_k + Δ  → clamp to valid range
        base = torch.arange(L, device=x.device, dtype=x.dtype)    # (L,)
        pos = base.view(1, L, 1, 1) + self.grid.view(1, 1, 1, K) + offsets
        pos = pos.clamp(0, L - 1)                                 # (B, L, G, K)

        # gather per group with linear interpolation
        v = value.view(B, L, G, gc).permute(0, 2, 1, 3)           # (B, G, L, gc)
        v = v.reshape(B * G, L, gc)
        p = pos.permute(0, 2, 1, 3).reshape(B * G, L * K)         # (B*G, L*K)
        i0 = p.floor()
        w = (p - i0).unsqueeze(-1)                                # (B*G, L*K, 1)
        i0 = i0.long().clamp(0, L - 1)
        i1 = (i0 + 1).clamp(max=L - 1)
        g0 = v.gather(1, i0.unsqueeze(-1).expand(-1, -1, gc))     # (B*G, L*K, gc)
        g1 = v.gather(1, i1.unsqueeze(-1).expand(-1, -1, gc))
        sampled = g0 * (1 - w) + g1 * w                           # (B*G, L*K, gc)
        sampled = sampled.view(B, G, L, K, gc)

        m = mask.permute(0, 2, 1, 3).unsqueeze(-1)                # (B, G, L, K, 1)
        out = (sampled * m).sum(dim=3)                            # (B, G, L, gc)
        out = out.permute(0, 2, 1, 3).reshape(B, L, C)
        return self.proj_drop(self.output_proj(out))


class InternImageBlock1d(nn.Module):
    """InternImage basic block: pre-LN, DCNv3 mixer + MLP, both residual
    (optionally LayerScaled). Operates on channels-last (B, L, C)."""
    def __init__(self, dim, groups, dcn_kernel=3, offset_scale=1.0,
                 mlp_ratio=4, drop=0., drop_path=0., use_grn=False,
                 layer_scale=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.dcn = DCNv31d(dim, groups, dcn_kernel, offset_scale, drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.grn = GRN1d(hidden) if use_grn else nn.Identity()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if layer_scale > 0:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))
        else:
            self.gamma1 = self.gamma2 = None

    def forward(self, x):
        a = self.dcn(self.norm1(x))
        if self.gamma1 is not None:
            a = a * self.gamma1
        x = x + self.drop_path(a)
        f = self.fc2(self.drop(self.grn(self.act(self.fc1(self.norm2(x))))))
        if self.gamma2 is not None:
            f = f * self.gamma2
        return x + self.drop_path(f)


class PositionalEncoding1d(nn.Module):
    """Optional ABSOLUTE positional encoding after the stem (InternImage's
    classification model uses none — convs carry position).
    kind: none | learned | sincos | wavelength. Operates on (B, C, L)."""
    def __init__(self, dim, kind="none", stem_wavelengths=None):
        super().__init__()
        self.kind = kind
        self.dim = dim
        if kind == "learned":
            ref_len = len(stem_wavelengths) if stem_wavelengths is not None else 50
            self.pe = nn.Parameter(torch.zeros(1, dim, ref_len))
            trunc_normal_(self.pe, std=.02)
        elif kind == "wavelength":
            if stem_wavelengths is None:
                raise ValueError("wavelength encoding needs stem_wavelengths")
            wl = torch.as_tensor(stem_wavelengths, dtype=torch.float32)
            pos = (wl - WL_MIN_NM) / (WL_MAX_NM - WL_MIN_NM) * len(wl)
            self.register_buffer("pe", self._sincos(pos, dim).unsqueeze(0))
        elif kind not in ("none", "sincos"):
            raise ValueError(f"Unknown pos_encoding '{kind}'")

    @staticmethod
    def _sincos(pos, dim):
        half = (dim + 1) // 2
        div = torch.exp(torch.arange(half, dtype=torch.float32)
                        * (-np.log(10000.0) / max(1, half)))
        ang = pos[None, :] * div[:, None]
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=0)
        return pe[:dim]

    def forward(self, x):
        if self.kind == "none":
            return x
        L = x.shape[-1]
        if self.kind == "learned":
            pe = self.pe
            if pe.shape[-1] != L:
                pe = F.interpolate(pe, size=L, mode="linear", align_corners=False)
            return x + pe
        if self.kind == "wavelength" and self.pe.shape[-1] == L:
            return x + self.pe
        pos = torch.arange(L, dtype=torch.float32, device=x.device)
        return x + self._sincos(pos, self.dim).to(x.dtype).unsqueeze(0)


# ==================== MAIN MODEL ====================

class InternImage1d(nn.Module):
    """
    InternImage for 1D spectra with dual outputs (multi-label classification +
    composition regression).

    Structure: two-conv stem (/4) + LN -> 4 stages of DCNv3 blocks with
    (conv k3/s2 + LN) downsampling between stages. Groups per stage =
    dims[i] / group_dim (multi-group deformable sampling).

    For L=200 the stage lengths are 50 -> 25 -> 13 -> 7.

    Args:
        in_chans / num_classes / depths / dims : as usual.
        group_dim (int): channels per DCN group (paper: 16). Default: 16.
        dcn_kernel (int): sample points K per group. Default: 3.
        offset_scale (float): multiplier on predicted offsets. Default: 1.0.
        mlp_ratio (int): MLP expansion. Default: 4.
        use_grn (bool): GRN in the MLPs. Default: False.
        layer_scale (float): LayerScale (paper T/S: off). Default: 0.
        drop_path_rate / dropout / head_dropout : regularization.
        pos_encoding (str): none (paper default) | learned | sincos | wavelength.
        head_init_scale (float): output-head init scaling.
        use_multiscale (bool): aggregate pooled features from all 4 stages.
    """
    def __init__(self, in_chans=2, num_classes=12,
                 depths=[2, 2, 6, 2], dims=[40, 80, 160, 320],
                 group_dim=16, dcn_kernel=3, offset_scale=1.0, mlp_ratio=4,
                 use_grn=False, layer_scale=0., drop_path_rate=0., dropout=0.,
                 head_dropout=0., pos_encoding="none", head_init_scale=1.,
                 use_multiscale=False):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.use_multiscale = use_multiscale

        def groups_for(dim):
            g = max(1, dim // group_dim)
            while dim % g != 0:
                g -= 1
            return g

        # Stem: conv k3/s2 + LN + GELU + conv k3/s2 + LN  (/4).
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0] // 2 if dims[0] >= 4 else dims[0],
                      3, 2, 1),
            LayerNormCF1d(dims[0] // 2 if dims[0] >= 4 else dims[0]),
            nn.GELU(),
            nn.Conv1d(dims[0] // 2 if dims[0] >= 4 else dims[0], dims[0],
                      3, 2, 1),
            LayerNormCF1d(dims[0]),
        )
        stem_wl = WAVELENGTHS_NM.reshape(-1, 4).mean(axis=1)   # (50,)
        self.pos_enc = PositionalEncoding1d(dims[0], kind=pos_encoding,
                                            stem_wavelengths=stem_wl)

        dp = [x.tolist() for x in
              torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(4):
            if i > 0:
                self.downsamples.append(nn.Sequential(
                    nn.Conv1d(dims[i - 1], dims[i], 3, 2, 1),
                    LayerNormCF1d(dims[i]),
                ))
            else:
                self.downsamples.append(nn.Identity())
            self.stages.append(nn.ModuleList([
                InternImageBlock1d(
                    dims[i], groups_for(dims[i]), dcn_kernel, offset_scale,
                    mlp_ratio, dropout, dp[i][j], use_grn, layer_scale)
                for j in range(depths[i])
            ]))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        if use_multiscale:
            self.ms_pools = nn.ModuleList([nn.AdaptiveAvgPool1d(1) for _ in range(4)])
            self.ms_proj = nn.ModuleList([nn.Linear(dims[i], dims[-1]) for i in range(4)])

        self.mlc_proj = nn.Sequential(
            nn.Linear(dims[-1], dims[-1] // 2), nn.GELU(),
            nn.LayerNorm(dims[-1] // 2), nn.Dropout(head_dropout))
        self.reg_proj = nn.Sequential(
            nn.Linear(dims[-1], dims[-1] // 2), nn.GELU(),
            nn.LayerNorm(dims[-1] // 2), nn.Dropout(head_dropout))
        self.head  = nn.Linear(dims[-1] // 2, num_classes)   # classification (mlc)
        self.head2 = nn.Linear(dims[-1] // 2, num_classes)   # regression (reg)

        self.apply(self._init_weights)
        # re-apply DCN zero-inits (self.apply above overwrites them)
        for m in self.modules():
            if isinstance(m, DCNv31d):
                nn.init.zeros_(m.offset.weight)
                nn.init.zeros_(m.offset.bias)
                nn.init.zeros_(m.mask.weight)
                nn.init.zeros_(m.mask.bias)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        self.head2.weight.data.mul_(head_init_scale)
        self.head2.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.pos_enc(x)
        feats = []
        for i in range(4):
            x = self.downsamples[i](x)
            t = x.transpose(1, 2)                     # (B, L, C) for blocks
            for blk in self.stages[i]:
                t = blk(t)
            x = t.transpose(1, 2)                     # (B, C, L)
            if self.use_multiscale:
                feats.append(self.ms_pools[i](x).flatten(1))
        if self.use_multiscale:
            agg = sum(proj(f) for proj, f in zip(self.ms_proj, feats))
            return self.norm(agg)
        return self.norm(self.pooling(x).flatten(1))

    def forward(self, x):
        x = self.forward_features(x)
        mlc = self.head(self.mlc_proj(x))
        reg = self.head2(self.reg_proj(x))
        return mlc, reg


# ==================== MODEL VARIANTS ====================
# Familiar size names for cross-backbone comparability; 'internimage_t'
# mirrors the official InternImage-T shape (depths 4/4/18/4, dims 64..512,
# group_dim 16).

MODEL_REGISTRY = {
    "deep_zepto": dict(
        depths=[2, 2, 6, 2],
        dims=[32, 64, 128, 256]
    ),
    "shallow_atto": dict(
        depths=[2, 2, 4, 2],
        dims=[40, 80, 160, 320]
    ),
    "atto":   dict(depths=[2, 2, 6, 2],  dims=[40, 80, 160, 320]),
    "femto":  dict(depths=[2, 2, 6, 2],  dims=[48, 96, 192, 384]),
    "pico":   dict(depths=[2, 2, 6, 2],  dims=[64, 128, 256, 512]),
    "nano":   dict(depths=[2, 2, 8, 2],  dims=[80, 160, 320, 640]),
    "tiny":   dict(depths=[3, 3, 9, 3],  dims=[96, 192, 384, 768]),
    "base":   dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
    "large":  dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
    "huge":   dict(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816]),
    "internimage_t": dict(depths=[4, 4, 18, 4], dims=[64, 128, 256, 512]),
}


# ==================== BUILDERS / HELPERS ====================

def build_model(model_size, in_chans, num_classes, drop_path_rate,
                use_multiscale=False, group_dim=16, dcn_kernel=3,
                offset_scale=1.0, mlp_ratio=4, use_grn=False, layer_scale=0.,
                dropout=0., head_dropout=0., pos_encoding="none"):
    if model_size not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model size '{model_size}'. "
                         f"Choose from: {list(MODEL_REGISTRY.keys())}")
    cfg = MODEL_REGISTRY[model_size]
    return InternImage1d(
        in_chans=in_chans, num_classes=num_classes,
        depths=cfg["depths"], dims=cfg["dims"],
        group_dim=group_dim, dcn_kernel=dcn_kernel, offset_scale=offset_scale,
        mlp_ratio=mlp_ratio, use_grn=use_grn, layer_scale=layer_scale,
        drop_path_rate=drop_path_rate, dropout=dropout,
        head_dropout=head_dropout, pos_encoding=pos_encoding,
        use_multiscale=use_multiscale,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
