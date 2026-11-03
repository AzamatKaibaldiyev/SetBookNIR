"""LAMB (You et al. 2020) without host synchronisation: the update rule of torch_optimizer.Lamb (v0.3.0 defaults: betas
(0.9, 0.999), eps 1e-6, clamp_value 10, adam=False, debias=False), computed on-device with _foreach kernels.
"""
import torch


class FastLamb(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, clamp_value=10.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.clamp_value = clamp_value

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            ps = [p for p in group["params"] if p.grad is not None]
            if not ps:
                continue
            grads = [p.grad for p in ps]
            for p in ps:
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st["step"] += 1
            m = [self.state[p]["exp_avg"] for p in ps]
            v = [self.state[p]["exp_avg_sq"] for p in ps]
            b1, b2 = group["betas"]
            torch._foreach_mul_(m, b1); torch._foreach_add_(m, grads, alpha=1 - b1)
            torch._foreach_mul_(v, b2); torch._foreach_addcmul_(v, grads, grads, value=1 - b2)
            denom = torch._foreach_sqrt(v); torch._foreach_add_(denom, group["eps"])
            upd = torch._foreach_div(m, denom)
            if group["weight_decay"] != 0:
                torch._foreach_add_(upd, ps, alpha=group["weight_decay"])
            wn = torch.stack(torch._foreach_norm(ps)).clamp(0, self.clamp_value)
            an = torch.stack(torch._foreach_norm(upd))
            trust = torch.where((wn == 0) | (an == 0), torch.ones_like(wn), wn / an)
            scale = (-group["lr"]) * trust          # debias=False -> step_size = lr
            torch._foreach_mul_(upd, list(scale.unbind()))
            torch._foreach_add_(ps, upd)
        return loss
