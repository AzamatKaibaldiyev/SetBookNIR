"""
MaxViT-1D backbone.

A 1D adaptation of MaxViT for time-series / spectra with dual outputs
(classification + regression). Drop-in replacement for the previous
ConvNeXtV2 model: same constructor keys, same `forward_features` / dual-head
interface, same `use_multiscale` behaviour.

Architecture
------------
MaxViT = MBConv (local conv inductive bias) + block attention (local windows)
+ grid attention (global, dilated windows). Every MaxViT block does:

    x -> MBConv1d -> block-attention -> grid-attention

The two attention flavours differ only in how the 1D sequence is partitioned
into windows before self-attention:
  * block : contiguous windows of size W          -> captures LOCAL structure
  * grid  : W tokens spaced (L/W) apart           -> captures GLOBAL structure

Both are O(L * W) instead of O(L^2), so they stay cheap on long spectra.

Partitioning pads the sequence up to a multiple of W and masks the padded
tokens in the attention softmax, so the model accepts ANY input length.

Down-sampling: a stride-2 stem (/2) followed by a stride-2 MBConv at the
start of each of the 4 stages (/2 each) -> total /32. For a length-200 input
the stage resolutions are 100 -> 50 -> 25 -> 13 -> 7.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath

# ==================== CONSTANTS ============================================

MAXVIT_WINDOW   = 7    # block == grid window size (tokens per attention window)
MAXVIT_HEAD_DIM = 32   # channels per attention head


# ==================== BUILDING BLOCKS ======================================

class SqueezeExcite1d(nn.Module):
    """Channel squeeze-and-excitation for 1D feature maps (B, C, L)."""
    def __init__(self, dim, rd_ratio=0.25):
        super().__init__()
        rd = max(1, int(round(dim * rd_ratio)))
        self.fc1 = nn.Conv1d(dim, rd, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv1d(rd, dim, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)          # global avg pool over length
        s = self.fc2(self.act(self.fc1(s)))
        return x * self.gate(s)


class MBConv1d(nn.Module):
    """MobileNetV2 inverted-bottleneck block (1D) with SE and stochastic depth.

    pre-BN -> 1x1 expand -> depthwise(k=3, optional stride 2) -> SE -> 1x1 project
    Residual add; when the block down-samples or changes width the shortcut is
    pooled + 1x1-projected to match. This is the convolutional half of a MaxViT
    block and is where inter-stage down-sampling happens (stride=2).
    """
    def __init__(self, in_dim, out_dim, stride=1, expand_ratio=4,
                 se_ratio=0.25, drop_path=0.):
        super().__init__()
        mid = int(out_dim * expand_ratio)
        self.use_res = (stride == 1 and in_dim == out_dim)

        self.pre_norm = nn.BatchNorm1d(in_dim)
        self.expand = nn.Sequential(
            nn.Conv1d(in_dim, mid, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
        )
        self.dwconv = nn.Sequential(
            nn.Conv1d(mid, mid, kernel_size=3, stride=stride, padding=1,
                      groups=mid, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
        )
        self.se = SqueezeExcite1d(mid, se_ratio)
        self.project = nn.Conv1d(mid, out_dim, kernel_size=1, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if not self.use_res:
            layers = []
            if stride > 1:
                layers.append(nn.AvgPool1d(kernel_size=stride, stride=stride,
                                           ceil_mode=True))
            layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False))
            self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        shortcut = x if self.use_res else self.shortcut(x)
        x = self.pre_norm(x)
        x = self.expand(x)
        x = self.dwconv(x)
        x = self.se(x)
        x = self.project(x)
        return shortcut + self.drop_path(x)


class RelPosAttention1d(nn.Module):
    """Multi-head self-attention over a fixed-length window with a learnable
    1D relative-position bias. Operates on (num_windows*B, W, C)."""
    def __init__(self, dim, window_size, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        # (2W-1, heads) relative-position bias table, indexed by (i - j).
        self.rel_bias = nn.Parameter(torch.zeros(2 * window_size - 1, num_heads))
        coords = torch.arange(window_size)
        rel_index = coords[:, None] - coords[None, :] + (window_size - 1)  # (W, W)
        self.register_buffer("rel_index", rel_index, persistent=False)
        trunc_normal_(self.rel_bias, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        # x: (Bn, W, C);  attn_bias: (Bn, W) additive key mask (0 / -inf) or None
        Bn, W, C = x.shape
        qkv = self.qkv(x).reshape(Bn, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)               # (3, Bn, heads, W, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (Bn, heads, W, W)

        bias = self.rel_bias[self.rel_index.reshape(-1)].reshape(W, W, -1)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)   # (1, heads, W, W)

        if attn_bias is not None:
            attn = attn + attn_bias[:, None, None, :]      # mask padded keys

        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(Bn, W, C)
        return self.proj_drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, dim, ratio=4, drop=0.):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class AttentionUnit1d(nn.Module):
    """One pre-LN (attention + MLP) transformer unit over a 1D partition.

    mode='block' -> contiguous windows (local);  mode='grid' -> dilated windows
    (global). Handles padding to a multiple of the window size and masks the
    padded tokens, so any sequence length is accepted.
    """
    def __init__(self, dim, window_size, num_heads, mode, mlp_ratio=4,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        assert mode in ("block", "grid")
        self.mode = mode
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelPosAttention1d(dim, window_size, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _partition(self, x, B, nW, W):
        # x: (B, Lp, C) -> (B*nW, W, C)
        C = x.shape[-1]
        if self.mode == "block":
            return x.view(B, nW, W, C).reshape(B * nW, W, C)
        return x.view(B, W, nW, C).permute(0, 2, 1, 3).reshape(B * nW, W, C)

    def _merge(self, x, B, nW, W):
        # x: (B*nW, W, C) -> (B, Lp, C)
        C = x.shape[-1]
        if self.mode == "block":
            return x.view(B, nW, W, C).reshape(B, nW * W, C)
        return x.view(B, nW, W, C).permute(0, 2, 1, 3).reshape(B, W * nW, C)

    def forward(self, x):
        # x: (B, C, L) -> (B, C, L)
        B, C, L = x.shape
        x = x.transpose(1, 2)                       # (B, L, C)

        W = self.window_size
        pad = (W - L % W) % W
        if pad:
            x = F.pad(x, (0, 0, 0, pad))            # pad tokens at the tail
        Lp = L + pad
        nW = Lp // W

        xw = self._partition(x, B, nW, W)          # (B*nW, W, C)

        attn_bias = None
        if pad:
            valid = x.new_ones(B, Lp)
            valid[:, L:] = 0.0
            vw = self._partition(valid.unsqueeze(-1), B, nW, W).squeeze(-1)  # (B*nW, W)
            attn_bias = torch.zeros_like(vw)
            attn_bias[vw == 0] = float("-inf")

        xw = xw + self.drop_path(self.attn(self.norm1(xw), attn_bias))
        xw = xw + self.drop_path(self.mlp(self.norm2(xw)))

        x = self._merge(xw, B, nW, W)              # (B, Lp, C)
        if pad:
            x = x[:, :L, :]
        return x.transpose(1, 2)                    # (B, C, L)


class MaxViTBlock1d(nn.Module):
    """Full MaxViT block for 1D: MBConv -> block-attention -> grid-attention."""
    def __init__(self, in_dim, out_dim, window_size, num_heads, stride=1,
                 mlp_ratio=4, expand_ratio=4, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.mbconv = MBConv1d(in_dim, out_dim, stride=stride,
                               expand_ratio=expand_ratio, drop_path=drop_path)
        self.block_attn = AttentionUnit1d(out_dim, window_size, num_heads, "block",
                                          mlp_ratio, drop, attn_drop, drop_path)
        self.grid_attn = AttentionUnit1d(out_dim, window_size, num_heads, "grid",
                                         mlp_ratio, drop, attn_drop, drop_path)

    def forward(self, x):
        x = self.mbconv(x)
        x = self.block_attn(x)
        x = self.grid_attn(x)
        return x


# ==================== MAIN BACKBONE ========================================

class MaxViT1d(nn.Module):
    """
    MaxViT for 1D time-series / spectra with dual outputs (classification + regression).

    Args:
        in_chans (int): Number of input channels. Default: 2
        num_classes (int): Output units for both heads. Default: 12
        depths (list): Number of MaxViT blocks per stage. Default: [2, 2, 6, 2]
        dims (list): Feature width per stage. Default: [40, 80, 160, 320]
        window_size (int): Tokens per (block/grid) attention window. Default: 7
        head_dim (int): Channels per attention head. Default: 32
        mlp_ratio (int): Expansion in the attention MLPs. Default: 4
        expand_ratio (int): Expansion inside MBConv. Default: 4
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling for the two output heads. Default: 1.
        use_multiscale (bool): Aggregate pooled features from all 4 stages.
    """
    def __init__(self, in_chans=2, num_classes=12,
                 depths=[2, 2, 6, 2], dims=[40, 80, 160, 320],
                 window_size=MAXVIT_WINDOW, head_dim=MAXVIT_HEAD_DIM,
                 mlp_ratio=4, expand_ratio=4,
                 drop_path_rate=0., head_init_scale=1., use_multiscale=False):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.use_multiscale = use_multiscale

        # Stem: conv (stride 2, /2) + conv (stride 1).
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(dims[0]),
            nn.GELU(),
            nn.Conv1d(dims[0], dims[0], kernel_size=3, stride=1, padding=1),
        )

        def heads_for(dim):
            h = max(1, dim // head_dim)
            while dim % h != 0:          # guarantee dim divisible by #heads
                h -= 1
            return h

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        in_d = dims[0]
        for i in range(4):
            out_d = dims[i]
            blocks = []
            for j in range(depths[i]):
                stride = 2 if j == 0 else 1     # down-sample at the start of each stage
                blocks.append(MaxViTBlock1d(
                    in_d if j == 0 else out_d, out_d,
                    window_size=window_size, num_heads=heads_for(out_d),
                    stride=stride, mlp_ratio=mlp_ratio, expand_ratio=expand_ratio,
                    drop_path=dp_rates[cur + j],
                ))
            self.stages.append(nn.Sequential(*blocks))
            in_d = out_d
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Multi-scale aggregation with separate pooling for each stage.
        if use_multiscale:
            self.ms_pools = nn.ModuleList([nn.AdaptiveAvgPool1d(1) for _ in range(4)])
            self.ms_proj = nn.ModuleList([nn.Linear(dims[i], dims[-1]) for i in range(4)])

        self.mlc_proj = nn.Sequential(
            nn.Linear(dims[-1], dims[-1] // 2),
            nn.GELU(),
            nn.LayerNorm(dims[-1] // 2)
        )
        self.reg_proj = nn.Sequential(
            nn.Linear(dims[-1], dims[-1] // 2),
            nn.GELU(),
            nn.LayerNorm(dims[-1] // 2)
        )

        self.head  = nn.Linear(dims[-1] // 2, num_classes)  # Classification head (mlc)
        self.head2 = nn.Linear(dims[-1] // 2, num_classes)  # Regression head (reg)

        self.apply(self._init_weights)
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
        feats = []
        for i in range(4):
            x = self.stages[i](x)
            if self.use_multiscale:
                feats.append(self.ms_pools[i](x).flatten(1))   # (B, dims[i])
        if self.use_multiscale:
            agg = sum(proj(f) for proj, f in zip(self.ms_proj, feats))
            x = self.norm(agg)
        else:
            x = self.pooling(x).flatten(1)
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        mlc_features = self.mlc_proj(x)
        reg_features = self.reg_proj(x)
        mlc = self.head(mlc_features)
        reg = self.head2(reg_features)
        return mlc, reg


# ==================== MODEL VARIANTS / REGISTRY ============================
#
# Same size names as the previous ConvNeXtV2 registry so existing --model
# choices keep working. Note a MaxViT block is heavier than a ConvNeXt block
# (it adds two windowed-attention sub-blocks), so a given size has more
# parameters / FLOPs here than under the old backbone.

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
}


def build_model(model_size, in_chans, num_classes, drop_path_rate,
                use_multiscale=False, window_size=MAXVIT_WINDOW,
                head_dim=MAXVIT_HEAD_DIM):
    """Build a MaxViT1d from a registry name."""
    if model_size not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model size '{model_size}'. "
                         f"Choose from: {list(MODEL_REGISTRY.keys())}")
    cfg = MODEL_REGISTRY[model_size]
    return MaxViT1d(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=cfg["depths"],
        dims=cfg["dims"],
        window_size=window_size,
        head_dim=head_dim,
        drop_path_rate=drop_path_rate,
        use_multiscale=use_multiscale,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)