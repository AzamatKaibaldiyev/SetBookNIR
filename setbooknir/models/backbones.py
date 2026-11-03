"""1-D InternImage, MaxViT and OverLoCK (internimage.py, maxvit.py, overlock.py) with the same interface as the ConvNeXtV2:
forward_features -> mlc_proj / reg_proj -> head (presence) / head2 (composition). The adapter adds the optional set head
(head3, on the presence projection) and codebook head (head4, on the composition projection), logits laid out as in
the ConvNeXtV2: [12 presence | set_classes | cb_classes].
"""
import importlib
import types

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

BACKBONES = ("internimage", "maxvit", "overlock")


def _forward(self, x):
    f = self.forward_features(x)
    mf, rf = self.mlc_proj(f), self.reg_proj(f)
    mlc = [self.head(mf)] + ([self.head3(mf)] if self.set_classes else []) + ([self.head4(rf)] if self.cb_classes else [])
    return torch.cat(mlc, 1), self.head2(rf)


def build_backbone(name, size, drop_path_rate, set_classes=0, cb_classes=0):
    assert name in BACKBONES, name
    m = importlib.import_module(f"setbooknir.models.{name}").build_model(size, 2, 12, drop_path_rate)
    m.cb_classes, m.set_classes = int(cb_classes), int(set_classes)
    if m.cb_classes:
        m.head4 = nn.Linear(m.head2.in_features, m.cb_classes)
        trunc_normal_(m.head4.weight, std=.02); nn.init.zeros_(m.head4.bias)
    if m.set_classes:
        m.head3 = nn.Linear(m.head.in_features, m.set_classes)
        trunc_normal_(m.head3.weight, std=.02); nn.init.zeros_(m.head3.bias)
        m.register_buffer("set_table", torch.zeros(m.set_classes, 12, dtype=torch.long))
    if m.cb_classes or m.set_classes:
        m.forward = types.MethodType(_forward, m)
    return m
