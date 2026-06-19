"""Seam 1 (R1 correctness gate): full-step gradient diff.

From a shared config + fixed seed + identical init + one fixed batch, run a
single forward+backward on both backends and assert *all* parameter gradients
agree within an fp32-scale tolerance. The torch fp64 twin is the oracle; the MLX
twin is the trainer-side reference.

This single seam covers the base model, GQA, and MTP. A genuine backward-pass bug
diverges far beyond fp32 rounding noise, so this tolerance catches real errors
while passing on floating-point noise. (The ANE backend plugs into this same
harness once its runtime config wiring lands — deferred; needs the hardware.)
"""

import numpy as np
import pytest

from lilbro.configs import Config
from lilbro.mlx_ref import init_params
from lilbro.mlx_ref import model as mlx_twin
from lilbro.mlx_ref import twin_torch as torch_twin

# fp32-scale: measured worst-case noise is ~1e-6; 1e-4 leaves ~2 orders of
# headroom yet is far below the divergence a real backward bug produces.
GRAD_RTOL = 1e-4

CONFIGS = {
    "base_mha": Config(name="base_mha", dim=64, n_layers=2, n_heads=4, head_dim=16,
                       seq=32, vocab=256, hidden=128, seed=0),
    "gqa": Config(name="gqa", dim=64, n_layers=2, n_heads=8, kv_heads=2, head_dim=16,
                  seq=24, vocab=256, hidden=128, seed=1),
    "mtp": Config(name="mtp", dim=64, n_layers=2, n_heads=4, head_dim=16,
                  seq=24, vocab=256, hidden=128, mtp_depth=2, seed=2),
    "gqa_mtp": Config(name="gqa_mtp", dim=64, n_layers=2, n_heads=8, kv_heads=4,
                      head_dim=16, seq=20, vocab=256, hidden=128, mtp_depth=2, seed=3),
}


def _fixed_batch(cfg: Config, b: int = 2):
    rng = np.random.default_rng(cfg.seed + 100)
    x = rng.integers(0, cfg.vocab, size=(b, cfg.seq))
    y = rng.integers(0, cfg.vocab, size=(b, cfg.seq))
    return x, y


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-8))


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CONFIGS))
def test_grad_diff_all_params(name):
    cfg = CONFIGS[name]
    params = init_params(cfg)
    x, y = _fixed_batch(cfg)

    loss_mlx, g_mlx = mlx_twin.loss_and_grads(params, x, y, cfg)
    loss_torch, g_torch = torch_twin.loss_and_grads(params, x, y, cfg)

    # Losses agree first (cheap sanity before per-param grads).
    assert loss_mlx == pytest.approx(loss_torch, rel=1e-4, abs=1e-4)

    assert set(g_mlx) == set(g_torch)
    worst, worst_name = 0.0, None
    for pname in g_torch:
        r = _rel(g_mlx[pname], g_torch[pname])
        if r > worst:
            worst, worst_name = r, pname
    assert worst < GRAD_RTOL, f"{name}: worst grad rel {worst:.2e} @ {worst_name}"


@pytest.mark.slow
def test_grad_diff_catches_injected_bug():
    """A deliberately corrupted gradient must fail the gate — proves the harness
    has teeth (it isn't trivially passing)."""
    cfg = CONFIGS["base_mha"]
    params = init_params(cfg)
    x, y = _fixed_batch(cfg)
    _, g_mlx = mlx_twin.loss_and_grads(params, x, y, cfg)
    _, g_torch = torch_twin.loss_and_grads(params, x, y, cfg)

    g_buggy = {k: v.copy() for k, v in g_mlx.items()}
    g_buggy["layer.0.wq"] *= 1.05  # 5% error in one matrix's gradient
    worst = max(_rel(g_buggy[k], g_torch[k]) for k in g_torch)
    assert worst > GRAD_RTOL
