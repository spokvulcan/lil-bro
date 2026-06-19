"""Seam 2 (R0 gate): overfit one batch.

Train a tiny config on a single repeated batch and assert the loss collapses
toward ~0. Black-box, no oracle: it exercises the *full* loop — forward,
backward, optimizer update, weight write-back — including the Muon path. This is
the behavioral gate that must be green before any ablation number is trusted.

Runs on the MLX twin (the trainer-side backend). The same gate runs against the
ANE backend once its runtime-config wiring lands (deferred; needs the hardware).
"""

from dataclasses import replace

import numpy as np
import pytest

from lilbro.configs import Config
from lilbro.mlx_ref import init_params, train_steps
from lilbro.mlx_ref import model as mlx_twin

R0 = Config(name="r0_overfit", dim=64, n_layers=1, n_heads=4, head_dim=16,
            seq=64, vocab=256, hidden=128, seed=0)


def _one_batch(cfg: Config):
    rng = np.random.default_rng(1234)
    x = rng.integers(0, cfg.vocab, size=(1, cfg.seq))
    y = rng.integers(0, cfg.vocab, size=(1, cfg.seq))
    return x, y


@pytest.mark.slow
@pytest.mark.parametrize("optimizer,lr,steps", [
    ("adamw", 1e-3, 400),
    ("muon", 1e-2, 300),
])
def test_overfit_collapses(optimizer, lr, steps):
    cfg = replace(R0, optimizer=optimizer, lr=lr)
    params = init_params(cfg)
    x, y = _one_batch(cfg)
    hist = train_steps(mlx_twin.loss_and_grads, params, x, y, cfg, steps=steps)
    assert hist[0] > 4.0                  # starts near ln(256) ~ 5.5
    assert hist[-1] < 0.1, f"{optimizer}: loss only fell to {hist[-1]:.4f}"
    assert hist[-1] < hist[0] / 20        # collapsed by >20x


@pytest.mark.slow
def test_overfit_with_mtp():
    """The combined (main + MTP) loss also collapses, confirming MTP is wired
    into the loop correctly (right targets, right shapes)."""
    cfg = replace(R0, optimizer="adamw", lr=1e-3, mtp_depth=1)
    params = init_params(cfg)
    x, y = _one_batch(cfg)
    hist = train_steps(mlx_twin.loss_and_grads, params, x, y, cfg, steps=500)
    assert hist[-1] < 0.2, f"MTP combined loss only fell to {hist[-1]:.4f}"
