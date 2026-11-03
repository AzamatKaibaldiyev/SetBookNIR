"""CPU tests on synthetic data (no dataset or GPU needed):  python tests/run_tests.py"""
import itertools
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setbooknir import readout, stack, train  # noqa: E402
from setbooknir.metrics import compute_metrics  # noqa: E402
from setbooknir.models import build_model, count_parameters  # noqa: E402
from setbooknir.optim import FastLamb  # noqa: E402


def test_parameter_counts():
    """Backbone sizes as reported in the paper (without auxiliary heads)."""
    for bb, size, m in (("convnext", "atto", 3.3), ("internimage", "atto", 4.2), ("maxvit", "atto", 15.0), ("overlock", "atto", 5.5)):
        n = count_parameters(build_model({"backbone": bb, "size": size, "drop_path": 0.2})) / 1e6
        assert abs(n - m) < 0.15 * m, (bb, n)


def test_output_layout():
    x = torch.randn(4, 2, 200)
    for bb in ("convnext", "internimage", "maxvit", "overlock"):
        z, r = build_model({"backbone": bb, "size": "atto", "drop_path": 0.2, "set_classes": 7, "cb_classes": 9}).eval()(x)
        assert z.shape == (4, 12 + 7 + 9) and r.shape == (4, 12), bb


def test_joint_decode_is_posterior_mode():
    rng = np.random.default_rng(0)
    sets = np.array([s for s in itertools.product([0, 1], repeat=4) if any(s)])[:9]
    P, Q = rng.uniform(0.01, 0.99, (50, 4)), rng.dirichlet(np.ones(len(sets)), 50)
    lik = np.prod(np.where(sets[None], P[:, None], 1 - P[:, None]), -1)          # (N, S)
    assert np.array_equal(readout.joint_decode(P, Q, sets), sets[(Q * lik).argmax(1)])


def test_knob_numpy_equals_torch():
    rng = np.random.default_rng(1)
    cbk = rng.dirichlet(np.ones(12), 200).astype(np.float32) * 100
    idx = np.stack([rng.choice(200, 50, replace=False) for _ in range(40)])
    p = rng.dirichlet(np.ones(50) * 0.3, 40)
    mean = (p[:, :, None] * cbk[idx]).sum(1)
    a = readout.knob_decide(mean, idx, p, cbk, (0, 25))
    for lam in (0, 25):
        b = readout.knob_decision_torch(torch.tensor(mean), torch.tensor(p), torch.tensor(cbk[idx].astype(np.float64)), lam).numpy()
        assert np.allclose(a[lam], b)
    assert np.allclose(a[0], mean)            # lambda = 0: the posterior mean is the MSE-optimal candidate


def test_fastlamb_matches_reference():
    def ref_step(ps, st, lr=1e-2, wd=1e-3, b1=0.9, b2=0.999, eps=1e-6):
        for p in ps:
            s = st.setdefault(p, {"m": torch.zeros_like(p), "v": torch.zeros_like(p)})
            s["m"].mul_(b1).add_(p.grad, alpha=1 - b1); s["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            u = s["m"] / (s["v"].sqrt() + eps) + wd * p.data
            wn, un = p.data.norm().clamp(0, 10), u.norm()
            p.data.add_(u, alpha=-lr * (1.0 if wn == 0 or un == 0 else float(wn / un)))
    torch.manual_seed(0)
    a = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.GELU(), torch.nn.Linear(7, 3)); b = [p.detach().clone().requires_grad_() for p in a.parameters()]
    opt, st, x = FastLamb(a.parameters(), lr=1e-2, weight_decay=1e-3), {}, torch.randn(16, 5)
    for _ in range(5):
        opt.zero_grad(); a(x).pow(2).mean().backward(); opt.step()
        for p in b:
            p.grad = None
        torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(x, b[0], b[1])), b[2], b[3]).pow(2).mean().backward()
        ref_step(b, st)
    assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(a.parameters(), b))


def _fake_data(d, n=640, nt=200):
    rng = np.random.default_rng(0)
    sets = np.array([s for s in itertools.product([0, 1], repeat=12) if 0 < sum(s) <= 3])[rng.choice(298, 12, replace=False)]
    for part, N in (("train", n), ("test", nt)):
        y = sets[rng.integers(0, len(sets), N)]
        c = np.round(y * rng.choice([20.0, 30.0, 50.0], (N, 12)), 0); c = np.round(100 * c / c.sum(1, keepdims=True), 1)
        c[:, 0] += 100 - c.sum(1)
        x = np.stack([y @ rng.normal(size=(12, 200)) + 0.1 * rng.normal(size=(N, 200)), rng.normal(size=(N, 200))], 1)
        np.save(os.path.join(d, f"{part}_data.npy"), x); np.save(os.path.join(d, f"{part}_multi_label_classification_label.npy"), y.astype(np.int32))
        np.save(os.path.join(d, f"{part}_composition_label.npy"), c.astype(np.float32))


def test_pipeline_end_to_end():
    """train (set + codebook heads) -> fit the stack -> infer -> read-outs; then a distilled student with knob targets."""
    d = tempfile.mkdtemp()
    _fake_data(d)
    small = {**train.DEFAULTS, "epochs": 2, "milestones": [1], "batch_size": 128, "compile": False}
    quiet = lambda *a: None  # noqa: E731
    t = os.path.join(d, "teacher_s0.pt")
    train.train({**small, "set_head_weight": 0.1, "codebook_head_weight": 0.1}, "all", 0, t, d, log=quiet)
    stack.FIT["epochs"] = 1
    stack.fit(t, d)
    X, Y, C = (np.load(os.path.join(d, f"test_{k}.npy")) for k in ("data", "multi_label_classification_label", "composition_label"))
    X = X.astype(np.float32)
    outs = [stack.infer(t, X, passes=4, variant=v) for v in ("final", "basic")]
    for o in outs:
        for name, (pres, comp) in readout.readouts([o], (0, 25)).items():
            m = compute_metrics(pres, comp, Y, C)
            assert 0 <= m["acc"] <= 1 and np.isfinite(m["mse"]), name
    s = os.path.join(d, "student_s0.pt")
    kd = {"teachers": [os.path.join(d, "teacher_s{seed}.pt")], "alpha": 0.8, "beta": 0.8, "targets": "knob", "knob_lambda": 25.0}
    _, hist = train.train({**small, "codebook_head_weight": 0.1, "distill": kd}, "all", 0, s, d, log=quiet)
    assert np.isfinite(hist[-1]["loss"])
    shutil.rmtree(d)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in tests:
        f(); print(f"ok  {f.__name__}")
    print(f"{len(tests)} tests passed")
