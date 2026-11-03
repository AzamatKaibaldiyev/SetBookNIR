"""Model construction from a run configuration."""
from .backbones import BACKBONES, build_backbone
from .convnext import build_convnext


def build_model(cfg):
    """cfg: backbone (convnext | internimage | maxvit | overlock), size, drop_path, set_classes, cb_classes."""
    args = (cfg["drop_path"], cfg.get("set_classes", 0), cfg.get("cb_classes", 0))
    if cfg["backbone"] == "convnext":
        return build_convnext(cfg["size"], *args)
    assert cfg["backbone"] in BACKBONES, cfg["backbone"]
    return build_backbone(cfg["backbone"], cfg["size"], *args)


def mc_mode(model, on):
    """Monte-Carlo stochastic depth: eval mode everywhere (BatchNorm uses its running statistics, Dropout is off), except
    the DropPath layers, which sample paths when `on`."""
    model.eval()
    if on:
        for m in model.modules():
            if type(m).__name__ == "DropPath":
                m.train(True)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
