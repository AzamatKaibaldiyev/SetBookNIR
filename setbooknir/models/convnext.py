"""ConvNeXtV2-1D with dual outputs (presence logits, composition percentages) and the optional auxiliary heads.

The trunk is the 1-D ConvNeXtV2 of the FabricSpectra reference code. The pooled feature feeds two projection heads
(Linear-GELU-LayerNorm): m (mlc_proj) for presence and c (reg_proj) for composition. Auxiliary linear softmax heads:
  head3: set head over the training fibre sets, on m
  head4: codebook head over the distinct training compositions, on c
Their logits are appended to the presence logits: [12 presence | set_classes | cb_classes].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath


class LayerNorm(nn.Module):
    """LayerNorm over channels_last (B, L, C) or channels_first (B, C, L) tensors."""
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
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None] * x + self.bias[:, None]


class GRN(nn.Module):
    """Global Response Normalization for (B, L, C) tensors."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXtV2 block: depthwise conv, LayerNorm, pointwise MLP with GRN, stochastic depth on the residual branch."""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        y = self.dwconv(x).permute(0, 2, 1)
        y = self.pwconv2(self.grn(self.act(self.pwconv1(self.norm(y))))).permute(0, 2, 1)
        return x + self.drop_path(y)


class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=2, num_classes=12, depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), drop_path_rate=0.,
                 set_classes=0, cb_classes=0):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=4),
                                                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                                        nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2)))
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[Block(dims[i], dp_rates[cur + j]) for j in range(depths[i])]))
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        hid = dims[-1] // 2
        self.mlc_proj = nn.Sequential(nn.Linear(dims[-1], hid), nn.GELU(), nn.LayerNorm(hid))
        self.reg_proj = nn.Sequential(nn.Linear(dims[-1], hid), nn.GELU(), nn.LayerNorm(hid))
        self.head = nn.Linear(hid, num_classes)        # presence logits z
        self.head2 = nn.Linear(hid, num_classes)       # composition percentages r
        self.apply(self._init_weights)
        self.cb_classes = int(cb_classes)
        if self.cb_classes:
            self.head4 = nn.Linear(hid, self.cb_classes)
            trunc_normal_(self.head4.weight, std=.02); nn.init.zeros_(self.head4.bias)
        self.set_classes = int(set_classes)
        if self.set_classes:
            self.head3 = nn.Linear(hid, self.set_classes)
            trunc_normal_(self.head3.weight, std=.02); nn.init.zeros_(self.head3.bias)
            self.register_buffer("set_table", torch.zeros(self.set_classes, num_classes, dtype=torch.long))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.stages[i](self.downsample_layers[i](x))
        return self.norm(self.pooling(x).flatten(1))

    def forward(self, x):
        f = self.forward_features(x)
        mf, rf = self.mlc_proj(f), self.reg_proj(f)
        mlc = [self.head(mf)] + ([self.head3(mf)] if self.set_classes else []) + ([self.head4(rf)] if self.cb_classes else [])
        return torch.cat(mlc, 1), self.head2(rf)


SIZES = {"atto": dict(depths=(2, 2, 6, 2), dims=(40, 80, 160, 320)),
         "femto": dict(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384)),
         "nano": dict(depths=(2, 2, 8, 2), dims=(80, 160, 320, 640)),
         "tiny": dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))}


def build_convnext(size, drop_path_rate, set_classes=0, cb_classes=0):
    return ConvNeXtV2(2, 12, drop_path_rate=drop_path_rate, set_classes=set_classes, cb_classes=cb_classes, **SIZES[size])
