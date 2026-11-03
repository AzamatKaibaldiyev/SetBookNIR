"""
OverLoCK (arXiv:2502.20087, CVPR 2025) adapted to 1D spectra.

Faithful port of the official 2D implementation (LMMMEng/OverLoCK) with all
spatial ops reduced from (H, W) to (L,). Dual-head output:
multi-label classification + composition regression.

Usage:
    from setbooknir.models.overlock import build_model, count_parameters
    model = build_model("atto", in_chans=2, num_classes=12, drop_path_rate=0.2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath


# ==================== BASIC LAYERS ====================

class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class ResDWConv1D(nn.Conv1d):
    """Depthwise convolution with residual connection (official ResDWConv)."""
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size,
                         padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        return x + super()._conv_forward(x, self.weight, self.bias)


class LayerScale1D(nn.Module):
    """Official LayerScale: per-channel scale (conv-style, with bias).
    Operates channels_first (B, C, L)."""
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) * init_value)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


class GRN1D(nn.Module):
    """GRN (Global Response Normalization), official OverLoCK NCHW variant
    reduced to NCL: response norm over the spatial dim, calibrated across
    channels, multiplicative form (gamma * Nx + 1) * x + beta."""
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        return (self.gamma * Nx + 1) * x


class SEModule1D(nn.Module):
    """Official SEModule: reduction 8 (min width 16), GELU inner, Sigmoid out."""
    def __init__(self, dim, red=8):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(inner_dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.proj(x)


# ==================== DILATED REPARAM BLOCK ====================

def _fuse_bn_1d(conv_weight, conv_bias, bn):
    """Fold a BatchNorm1d into a conv weight/bias. Returns (w, b)."""
    if conv_bias is None:
        conv_bias = 0
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1)
    return conv_weight * t, bn.bias + (conv_bias - bn.running_mean) * bn.weight / std


def _dilated_to_dense_1d(weight, dilation):
    """Convert a dilated dw kernel (C,1,k) to its equivalent dense sparse
    kernel of size (k-1)*dilation + 1 by zero-insertion."""
    if dilation == 1:
        return weight
    C, _, k = weight.shape
    eq = torch.zeros(C, 1, (k - 1) * dilation + 1,
                     device=weight.device, dtype=weight.dtype)
    eq[:, :, ::dilation] = weight
    return eq


class DilatedReparamBlock1D(nn.Module):
    """Dilated Reparam Block (UniRepLKNet) in 1D, with the branch table used
    by the official OverLoCK code. A large dw conv trained in parallel with
    small (possibly dilated) dw convs + BNs; merge_dilated_branches() folds
    everything into the single large kernel after training."""
    _BRANCHES = {
        19: ([5, 7, 9, 9, 3, 3, 3], [1, 1, 1, 2, 4, 5, 7]),
        17: ([5, 7, 9, 3, 3, 3],    [1, 1, 2, 4, 5, 7]),
        15: ([5, 7, 7, 3, 3, 3],    [1, 1, 2, 3, 5, 7]),
        13: ([5, 7, 7, 3, 3, 3],    [1, 1, 2, 3, 4, 5]),
        11: ([5, 7, 5, 3, 3, 3],    [1, 1, 2, 3, 4, 5]),
        9:  ([5, 7, 5, 3, 3],       [1, 1, 2, 3, 4]),
        7:  ([5, 3, 3, 3],          [1, 1, 2, 3]),
        5:  ([3, 3],                [1, 2]),
    }

    def __init__(self, channels, kernel_size):
        super().__init__()
        if kernel_size not in self._BRANCHES:
            raise ValueError(f"Dilated Reparam Block requires kernel_size in "
                             f"{sorted(self._BRANCHES)}, got {kernel_size}")
        self.channels = channels
        self.kernel_size = kernel_size
        self.kernel_sizes, self.dilates = self._BRANCHES[kernel_size]
        self.lk_origin = nn.Conv1d(channels, channels, kernel_size,
                                   padding=kernel_size // 2, groups=channels, bias=False)
        self.origin_bn = nn.BatchNorm1d(channels)
        for k, r in zip(self.kernel_sizes, self.dilates):
            assert (k - 1) * r + 1 <= kernel_size
            self.add_module(f'dil_conv_k{k}_r{r}',
                            nn.Conv1d(channels, channels, k,
                                      padding=(r * (k - 1) + 1) // 2, dilation=r,
                                      groups=channels, bias=False))
            self.add_module(f'dil_bn_k{k}_r{r}', nn.BatchNorm1d(channels))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):          # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, f'dil_conv_k{k}_r{r}')
            bn = getattr(self, f'dil_bn_k{k}_r{r}')
            out = out + bn(conv(x))
        return out

    @torch.no_grad()
    def merge_dilated_branches(self):
        if not hasattr(self, 'origin_bn'):
            return
        w, b = _fuse_bn_1d(self.lk_origin.weight, self.lk_origin.bias, self.origin_bn)
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, f'dil_conv_k{k}_r{r}')
            bn = getattr(self, f'dil_bn_k{k}_r{r}')
            bw, bb = _fuse_bn_1d(conv.weight, conv.bias, bn)
            bw = _dilated_to_dense_1d(bw, r)
            pad = self.kernel_size // 2 - bw.shape[2] // 2
            w = w + F.pad(bw, (pad, pad))
            b = b + bb
        merged = nn.Conv1d(self.channels, self.channels, self.kernel_size,
                           padding=self.kernel_size // 2, groups=self.channels, bias=True)
        merged.weight.data.copy_(w)
        merged.bias.data.copy_(b)
        self.lk_origin = merged
        self.__delattr__('origin_bn')
        for k, r in zip(self.kernel_sizes, self.dilates):
            self.__delattr__(f'dil_conv_k{k}_r{r}')
            self.__delattr__(f'dil_bn_k{k}_r{r}')


# ==================== STEM / DOWNSAMPLE ====================

def _stem_1d(in_chans, embed_dim):
    """Official 4-conv stem (total stride 4), BN + GELU."""
    return nn.Sequential(
        nn.Conv1d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm1d(embed_dim // 2),
        nn.GELU(),
        nn.Conv1d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm1d(embed_dim // 2),
        nn.GELU(),
        nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm1d(embed_dim),
        nn.GELU(),
        nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm1d(embed_dim),
    )


def _downsample_1d(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm1d(out_dim),
    )


class CTXDownsample1D(nn.Module):
    """Downsamples the Focus-Net feature and its slim context stream together
    between the stage-3 and stage-4 parts of Focus-Net."""
    def __init__(self, dim, h_dim):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Conv1d(dim, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(h_dim),
        )
        self.h_proj = nn.Sequential(
            nn.Conv1d(h_dim // 4, h_dim // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(h_dim // 4),
        )

    def forward(self, x, ctx):
        return self.x_proj(x), self.h_proj(ctx)


# ==================== BLOCKS ====================

class RepConvBlock1D(nn.Module):
    """OverLoCK's static block (used in Base-Net and Overview-Net):

        x -> ResDWConv3 -> [ LN -> DilatedReparam(K) -> BN -> SE
                             -> 1x1 (mlp) -> GELU -> ResDWConv3 -> GRN -> 1x1
                             -> DropPath ] + residual (LayerScale on identity
                             when res_scale, as in the official res_scale=True)
    """
    def __init__(self, dim, kernel_size=7, mlp_ratio=4, ls_init_value=None,
                 res_scale=True, drop_path=0.):
        super().__init__()
        self.res_scale = res_scale
        mlp_dim = int(dim * mlp_ratio)
        self.dwconv = ResDWConv1D(dim, kernel_size=3)
        self.proj = nn.Sequential(
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            DilatedReparamBlock1D(dim, kernel_size=kernel_size),
            nn.BatchNorm1d(dim),
            SEModule1D(dim),
            nn.Conv1d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv1D(mlp_dim, kernel_size=3),
            GRN1D(mlp_dim),
            nn.Conv1d(mlp_dim, dim, kernel_size=1),
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),
        )
        self.ls = LayerScale1D(dim, init_value=ls_init_value) \
            if ls_init_value is not None else nn.Identity()

    def forward(self, x):
        x = self.dwconv(x)
        if self.res_scale:
            x = self.ls(x) + self.proj(x)
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))
        return x


class DynamicConvBlock1D(nn.Module):
    """OverLoCK's Focus-Net block with ContMix (Context-Mixing Dynamic
    Convolution), ported to 1D.

    The block carries two streams: the main feature x (dim) and a slim
    top-down context stream h_x (ctx_dim // 4) coming from the Overview-Net.
    ContMix generates *input-dependent* depthwise kernels per position: an
    affinity map between local queries (from x) and `pool_size` pooled
    context tokens (from h_x) is linearly projected to dynamic kernel
    weights; half of the heads apply a small kernel (smk_size), half a large
    one (kernel_size), each with NATTEN-style relative position bias and
    border-clamped windows. A large-kernel DilatedReparam branch acts as a
    static positional encoding (lepe), followed by SE, a SiLU gate, and a
    ConvFFN -- exactly mirroring the official block.
    """
    def __init__(self, dim, ctx_dim, kernel_size=7, smk_size=5, num_heads=2,
                 pool_size=7, mlp_ratio=4, ls_init_value=None, res_scale=True,
                 drop_path=0., is_first=False, is_last=False):
        super().__init__()
        ctx_dim = ctx_dim // 4
        out_dim = dim + ctx_dim
        mlp_dim = int(dim * mlp_ratio)
        self.kernel_size = kernel_size
        self.smk_size = smk_size
        self.pool_size = pool_size
        self.res_scale = res_scale
        self.num_heads = num_heads * 2          # official: doubled internally
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.is_first = is_first
        self.is_last = is_last
        assert dim % (2 * self.num_heads) == 0, \
            f"dim {dim} must be divisible by 4 * num_heads ({4 * num_heads})"

        if not is_first:
            self.x_scale = LayerScale1D(ctx_dim, init_value=1)
            self.h_scale = LayerScale1D(ctx_dim, init_value=1)

        self.dwconv1 = ResDWConv1D(out_dim, kernel_size=3)
        self.norm1 = LayerNorm(out_dim, eps=1e-6, data_format="channels_first")

        self.fusion = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Conv1d(out_dim, dim, kernel_size=1),
            GRN1D(dim),
        )

        self.weight_query = nn.Sequential(
            nn.Conv1d(dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim // 2),
        )
        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool1d(pool_size),
            nn.Conv1d(ctx_dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim // 2),
        )
        # Projects the pool_size context affinities of each (head, position)
        # to the dynamic small + large kernel weights (official uses Conv2d
        # over the (heads, positions) plane with affinities as channels).
        self.weight_proj = nn.Conv2d(pool_size, smk_size + kernel_size, kernel_size=1)

        self.dyconv_proj = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
        )

        self.lepe = nn.Sequential(
            DilatedReparamBlock1D(dim, kernel_size=kernel_size),
            nn.BatchNorm1d(dim),
        )
        self.se_layer = SEModule1D(dim)
        self.gate = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, out_dim, kernel_size=1),
        )

        self.dwconv2 = ResDWConv1D(out_dim, kernel_size=3)
        self.norm2 = LayerNorm(out_dim, eps=1e-6, data_format="channels_first")
        self.mlp = nn.Sequential(
            nn.Conv1d(out_dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv1D(mlp_dim, kernel_size=3),
            GRN1D(mlp_dim),
            nn.Conv1d(mlp_dim, out_dim, kernel_size=1),
        )

        self.ls1 = LayerScale1D(out_dim, init_value=ls_init_value) \
            if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale1D(out_dim, init_value=ls_init_value) \
            if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # NATTEN-style relative position biases (1D: 2K-1 offsets)
        self.rpb1 = nn.Parameter(torch.zeros(self.num_heads, 2 * smk_size - 1))
        self.rpb2 = nn.Parameter(torch.zeros(self.num_heads, 2 * kernel_size - 1))

    @staticmethod
    def _window_index(L, K, device):
        """Border-clamped sliding windows (NATTEN semantics, 1D): each query i
        attends to K contiguous positions starting at clamp(i - K//2, 0, L-K).
        Returns (idx (L, K), rel (L, K)) with rel = j - i in [-(K-1), K-1]."""
        pos = torch.arange(L, device=device)
        start = (pos - K // 2).clamp(0, L - K)
        idx = start[:, None] + torch.arange(K, device=device)[None, :]
        rel = idx - pos[:, None]
        return idx, rel

    def _dyn_aggregate(self, attn, value, idx):
        # attn: (B, g, L, K), value: (B, g, L, c), idx: (L, K)
        v = value[:, :, idx]                                  # (B, g, L, K, c)
        return torch.einsum('bglk,bglkc->bglc', attn, v)

    def _forward_inner(self, x, h_x, h_r):
        L_in = x.shape[-1]
        B, C, L = x.shape
        C_h = h_x.shape[1]
        g = self.num_heads

        # Context-stream update: running ctx + static top-down reference
        if not self.is_first:
            h_x = self.x_scale(h_x) + self.h_scale(h_r)

        x_f = torch.cat([x, h_x], dim=1)
        x_f = self.dwconv1(x_f)
        identity = x_f
        x_f = self.norm1(x_f)
        x = self.fusion(x_f)
        gate = self.gate(x)
        lepe = self.lepe(x)

        # If the signal is shorter than the large kernel, temporarily resize
        # (official pad path)
        is_pad = L < self.kernel_size
        if is_pad:
            x = F.interpolate(x, size=self.kernel_size, mode='linear', align_corners=False)
            x_f = F.interpolate(x_f, size=self.kernel_size, mode='linear', align_corners=False)
            L = self.kernel_size

        # ── ContMix: dynamic kernels from query-context affinity ──
        query, key = torch.split(x_f, [C, C_h], dim=1)
        query = self.weight_query(query) * self.scale         # (B, C/2, L)
        key = self.weight_key(key)                            # (B, C/2, P)
        query = query.reshape(B, g, -1, L)
        key = key.reshape(B, g, -1, self.pool_size)
        weight = torch.einsum('bgcl,bgcp->bglp', query, key)  # (B, g, L, P)
        weight = weight.permute(0, 3, 1, 2).contiguous()      # (B, P, g, L)
        weight = self.weight_proj(weight)                     # (B, smk+K, g, L)
        weight = weight.permute(0, 2, 3, 1)                   # (B, g, L, smk+K)

        attn1, attn2 = torch.split(weight, [self.smk_size, self.kernel_size], dim=-1)
        idx1, rel1 = self._window_index(L, self.smk_size, x.device)
        idx2, rel2 = self._window_index(L, self.kernel_size, x.device)
        attn1 = attn1 + self.rpb1[:, rel1 + self.smk_size - 1].unsqueeze(0)
        attn2 = attn2 + self.rpb2[:, rel2 + self.kernel_size - 1].unsqueeze(0)
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)

        # Half of the channels/heads get the small dynamic kernel, half the
        # large one (official m=2 split)
        value = x.reshape(B, 2, g, -1, L).permute(1, 0, 2, 4, 3)   # (2, B, g, L, c)
        x1 = self._dyn_aggregate(attn1, value[0], idx1)
        x2 = self._dyn_aggregate(attn2, value[1], idx2)
        x = torch.cat([x1, x2], dim=1)                        # (B, 2g, L, c)
        x = x.permute(0, 1, 3, 2).reshape(B, C, L)

        if is_pad:
            x = F.adaptive_avg_pool1d(x, L_in)

        x = self.dyconv_proj(x)
        x = x + lepe
        x = self.se_layer(x)
        x = gate * x
        x = self.proj(x)

        if self.res_scale:
            x = self.ls1(identity) + self.drop_path(x)
        else:
            x = identity + self.drop_path(self.ls1(x))

        x = self.dwconv2(x)
        if self.res_scale:
            x = self.ls2(x) + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        if self.is_last:
            return (x, None)
        l_x, h_x = torch.split(x, [C, C_h], dim=1)
        return (l_x, h_x)

    def forward(self, x, h_x, h_r):
        return self._forward_inner(x, h_x, h_r)


# ==================== MAIN MODEL ====================

class OverLoCK1D(nn.Module):
    """
    OverLoCK (Overview-first-Look-Closely-next ConvNet with Context-Mixing
    Dynamic Kernels, arXiv:2502.20087) adapted to 1D spectra, with dual
    outputs (multi-label classification + composition regression).

    Biologically-inspired top-down attention in three sub-networks:
      * Base-Net (stages 1-2 + first part of stage 3): low/mid-level encoding
        with static RepConv blocks.
      * Overview-Net (stage 4 'overview'): a quick low-resolution pass
        producing the coarse context prior.
      * Focus-Net (deep parts of stages 3-4): 'looks closely' with
        DynamicConvBlocks whose ContMix kernels are generated from the
        Overview context, which is injected and refreshed at every block.

    Differences vs. official (documented): the auxiliary deep-supervision
    classification head on the Overview features is omitted so the two-head
    (mlc + reg) training loop stays unchanged; the final projection feature
    feeds both heads.

    Args:
        depth (list): Blocks per stage for Base-Net (stages 1-3) and
            Overview-Net (stage 4).
        sub_depth (list): Focus-Net blocks at stage-3 and stage-4 resolution.
        dims (list): Feature dimension at each stage.
        kernel_size (list): Large-kernel size per stage (official [17,15,13,7]).
        sub_num_heads (list): ContMix heads for the two Focus-Net parts.
        mlp_ratio / sub_mlp_ratio: FFN ratios (official [4,4,4,4] / [3,3]).
        smk_size (int): Small dynamic kernel size (official 5).
        pool_size (int): Pooled context tokens for ContMix (official 7).
        projection (int): Width of the pre-head projection (official 1024).
        ls_init_value (list): LayerScale init per stage (official [None,None,1,1]).
    """
    def __init__(self, in_chans=2, num_classes=12,
                 depth=[2, 2, 2, 1], sub_depth=[4, 1],
                 dims=[40, 80, 160, 320],
                 kernel_size=[17, 15, 13, 7],
                 mlp_ratio=[4, 4, 4, 4], sub_mlp_ratio=[3, 3],
                 sub_num_heads=[4, 8],
                 ls_init_value=[None, None, 1, 1],
                 res_scale=True, smk_size=5, pool_size=7, projection=1024,
                 drop_path_rate=0., head_init_scale=1., use_multiscale=False):
        super().__init__()
        self.depth = depth
        self.sub_depth = sub_depth
        self.use_multiscale = use_multiscale
        fusion_dim = dims[-1] + dims[-1] // 4

        self.patch_embed1 = _stem_1d(in_chans, dims[0])
        self.patch_embed2 = _downsample_1d(dims[0], dims[1])
        self.patch_embed3 = _downsample_1d(dims[1], dims[2])
        self.patch_embed4 = _downsample_1d(dims[2], dims[3])
        self.high_level_proj = nn.Conv1d(dims[-1], dims[-1] // 4, kernel_size=1)
        self.patch_embedx = CTXDownsample1D(dims[2], dims[3])

        dpr = [d.item() for d in torch.linspace(0, drop_path_rate, sum(depth) + sum(sub_depth))]

        def _rep_stage(stage, n, offset):
            return nn.ModuleList([
                RepConvBlock1D(dim=dims[stage], kernel_size=kernel_size[stage],
                               mlp_ratio=mlp_ratio[stage],
                               ls_init_value=ls_init_value[stage],
                               res_scale=res_scale, drop_path=dpr[offset + i])
                for i in range(n)])

        self.blocks1 = _rep_stage(0, depth[0], 0)
        self.blocks2 = _rep_stage(1, depth[1], depth[0])
        self.blocks3 = _rep_stage(2, depth[2], sum(depth[:2]))
        self.blocks4 = _rep_stage(3, depth[3], sum(depth[:3]))

        self.sub_blocks3 = nn.ModuleList([
            DynamicConvBlock1D(dim=dims[2], ctx_dim=dims[-1],
                               kernel_size=kernel_size[2], smk_size=smk_size,
                               num_heads=sub_num_heads[0], pool_size=pool_size,
                               mlp_ratio=sub_mlp_ratio[0],
                               ls_init_value=ls_init_value[2],
                               res_scale=res_scale, drop_path=dpr[sum(depth) + i],
                               is_first=(i == 0))
            for i in range(sub_depth[0])])

        self.sub_blocks4 = nn.ModuleList([
            DynamicConvBlock1D(dim=dims[3], ctx_dim=dims[-1],
                               kernel_size=kernel_size[3], smk_size=smk_size,
                               num_heads=sub_num_heads[1], pool_size=pool_size,
                               mlp_ratio=sub_mlp_ratio[1],
                               ls_init_value=ls_init_value[3],
                               res_scale=res_scale,
                               drop_path=dpr[sum(depth) + sub_depth[0] + i],
                               is_first=False, is_last=(i == sub_depth[1] - 1))
            for i in range(sub_depth[1])])

        # Pre-head projection (official main head: 1x1 -> BN -> SiLU -> pool)
        self.head_proj = nn.Sequential(
            nn.Conv1d(fusion_dim, projection, kernel_size=1, bias=False),
            nn.BatchNorm1d(projection),
            nn.SiLU(),
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(projection, eps=1e-6)

        # Multi-scale aggregation (adapted: stage-1/2 outputs, Focus-Net
        # stage-3 output, and the projected fused feature)
        if use_multiscale:
            self.ms_pools = nn.ModuleList([nn.AdaptiveAvgPool1d(1) for _ in range(4)])
            ms_dims = [dims[0], dims[1], dims[2], projection]
            self.ms_proj = nn.ModuleList([nn.Linear(d, projection) for d in ms_dims])

        self.mlc_proj = nn.Sequential(
            nn.Linear(projection, projection // 2),
            nn.GELU(),
            nn.LayerNorm(projection // 2)
        )
        self.reg_proj = nn.Sequential(
            nn.Linear(projection, projection // 2),
            nn.GELU(),
            nn.LayerNorm(projection // 2)
        )

        self.head  = nn.Linear(projection // 2, num_classes) # Classification head (mlc)
        self.head2 = nn.Linear(projection // 2, num_classes) # Regression head (reg)

        self.apply(self._init_weights)
        # rpb / LayerScale / GRN hold raw Parameters, untouched by apply().
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        self.head2.weight.data.mul_(head_init_scale)
        self.head2.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def switch_to_deploy(self):
        """Merge all Dilated Reparam Blocks (official reparam())."""
        for m in self.modules():
            if isinstance(m, DilatedReparamBlock1D):
                m.merge_dilated_branches()

    def reparam(self):
        self.switch_to_deploy()

    def forward_features(self, x):
        feats = []
        # ── Base-Net (stages 1-2) ──
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.use_multiscale:
            feats.append(self.ms_pools[0](x).flatten(1))
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        if self.use_multiscale:
            feats.append(self.ms_pools[1](x).flatten(1))

        # ── Base-Net stage 3 + Overview-Net (stage 4 context prior) ──
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        ctx = self.patch_embed4(x)
        for blk in self.blocks4:
            ctx = blk(ctx)

        # ── Focus-Net with top-down context guidance ──
        ctx_ori = self.high_level_proj(ctx)
        ctx_up = F.interpolate(ctx_ori, size=x.shape[-1], mode='linear', align_corners=False)
        for idx, blk in enumerate(self.sub_blocks3):
            if idx == 0:
                ctx = ctx_up
            x, ctx = blk(x, ctx, ctx_up)
        if self.use_multiscale:
            feats.append(self.ms_pools[2](x).flatten(1))

        x, ctx = self.patch_embedx(x, ctx)
        for blk in self.sub_blocks4:
            x, ctx = blk(x, ctx, ctx_ori)

        x = self.head_proj(x)                       # (B, projection, L)
        x = self.pooling(x).flatten(1)              # (B, projection)
        if self.use_multiscale:
            feats.append(x)
            x = sum(proj(f) for proj, f in zip(self.ms_proj, feats))
        return self.norm(x)

    def forward(self, x):
        x = self.forward_features(x)
        mlc_features = self.mlc_proj(x)
        reg_features = self.reg_proj(x)
        mlc = self.head(mlc_features)
        reg = self.head2(reg_features)
        return mlc, reg


# ==================== MODEL VARIANTS ====================

# Two families of sizes:
#   * "overlock_*" -- the OFFICIAL OverLoCK-XT/T/S/B layouts (depth/sub_depth/
#     dims/heads exactly as released; kernels [17,15,13,7], mlp [4,4,4,4],
#     sub_mlp [3,3], projection 1024).
#   * the remaining names -- the ADJUSTED (spectra-scale) family reusing the
#     stage widths of the previous registry. Since OverLoCK splits stages 3/4
#     between Base/Overview-Net (static blocks) and Focus-Net (dynamic
#     blocks), the old stage-3/4 depths are split accordingly (~1/3 static +
#     ~2/3 dynamic in stage 3, and 1 overview + the rest dynamic in stage 4,
#     following the official ratios). ContMix head counts are chosen so that
#     dims[2] % (4*heads3) == 0 and dims[3] % (4*heads4) == 0; the pre-head
#     projection is min(1024, 2*dims[3]) to keep small models small.
MODEL_REGISTRY = {
    # ── Official OverLoCK sizes ──────────────────────────────────────────
    "overlock_xt": dict(depth=[2, 2, 3, 2], sub_depth=[6, 2],
                        dims=[56, 112, 256, 336], sub_num_heads=[4, 6],  projection=1024),
    "overlock_t":  dict(depth=[4, 4, 6, 2], sub_depth=[12, 2],
                        dims=[64, 128, 256, 512], sub_num_heads=[4, 8],  projection=1024),
    "overlock_s":  dict(depth=[6, 6, 8, 3], sub_depth=[16, 3],
                        dims=[64, 128, 320, 512], sub_num_heads=[8, 16], projection=1024),
    "overlock_b":  dict(depth=[8, 8, 10, 4], sub_depth=[20, 4],
                        dims=[80, 160, 384, 576], sub_num_heads=[6, 9],  projection=1024),
    # ── Adjusted (spectra-scale) sizes ───────────────────────────────────
    "deep_zepto": dict(depth=[2, 2, 2, 1], sub_depth=[4, 1],
                       dims=[32, 64, 128, 256],     sub_num_heads=[4, 8],  projection=512),
    "shallow_atto": dict(depth=[2, 2, 1, 1], sub_depth=[3, 1],
                       dims=[40, 80, 160, 320],     sub_num_heads=[4, 8],  projection=640),
    "atto":   dict(depth=[2, 2, 2, 1], sub_depth=[4, 1],
                   dims=[40, 80, 160, 320],         sub_num_heads=[4, 8],  projection=640),
    "femto":  dict(depth=[2, 2, 2, 1], sub_depth=[4, 1],
                   dims=[48, 96, 192, 384],         sub_num_heads=[4, 8],  projection=768),
    "pico":   dict(depth=[2, 2, 2, 1], sub_depth=[4, 1],
                   dims=[64, 128, 256, 512],        sub_num_heads=[4, 8],  projection=1024),
    "nano":   dict(depth=[2, 2, 3, 1], sub_depth=[5, 1],
                   dims=[80, 160, 320, 640],        sub_num_heads=[4, 8],  projection=1024),
    "tiny":   dict(depth=[3, 3, 3, 1], sub_depth=[6, 2],
                   dims=[96, 192, 384, 768],        sub_num_heads=[4, 8],  projection=1024),
    "base":   dict(depth=[3, 3, 9, 1], sub_depth=[18, 2],
                   dims=[128, 256, 512, 1024],      sub_num_heads=[8, 16], projection=1024),
    "large":  dict(depth=[3, 3, 9, 1], sub_depth=[18, 2],
                   dims=[192, 384, 768, 1536],      sub_num_heads=[8, 16], projection=1024),
    "huge":   dict(depth=[3, 3, 9, 1], sub_depth=[18, 2],
                   dims=[352, 704, 1408, 2816],     sub_num_heads=[8, 16], projection=1024),
}


def build_model(model_size, in_chans, num_classes, drop_path_rate, use_multiscale=False):
    if model_size not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model size '{model_size}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    cfg = MODEL_REGISTRY[model_size]
    return OverLoCK1D(
        in_chans=in_chans,
        num_classes=num_classes,
        depth=cfg["depth"],
        sub_depth=cfg["sub_depth"],
        dims=cfg["dims"],
        sub_num_heads=cfg["sub_num_heads"],
        projection=cfg["projection"],
        drop_path_rate=drop_path_rate,
        use_multiscale=use_multiscale,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "OverLoCK1D",
    "RepConvBlock1D",
    "DynamicConvBlock1D",
    "DilatedReparamBlock1D",
    "LayerNorm",
    "GRN1D",
    "SEModule1D",
    "LayerScale1D",
    "ResDWConv1D",
    "CTXDownsample1D",
    "MODEL_REGISTRY",
    "build_model",
    "count_parameters",
]