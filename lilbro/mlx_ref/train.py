"""Framework-agnostic training loop.

Master weights live in numpy; each step asks a backend's ``loss_and_grads`` for
gradients (MLX twin or torch oracle) and applies the shared CPU optimizer. The
same loop drives the R0 overfit gate on either backend.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from lilbro.configs import Config
from .optim import make_optimizer, optimizer_step

LossAndGrads = Callable[[dict, np.ndarray, np.ndarray, Config], tuple]


def train_steps(
    loss_and_grads: LossAndGrads,
    params: dict,
    x: np.ndarray,
    targets: np.ndarray,
    cfg: Config,
    steps: int,
    log_every: int = 0,
) -> list[float]:
    """Run ``steps`` optimizer updates on a fixed (x, targets) batch in place.

    Returns the per-step loss history. With a single repeated batch this is the
    R0 overfit gate: the history must collapse toward ~0.
    """
    opt = make_optimizer(cfg)
    history: list[float] = []
    for s in range(steps):
        loss, grads = loss_and_grads(params, x, targets, cfg)
        optimizer_step(opt, params, grads)
        history.append(loss)
        if log_every and (s % log_every == 0 or s == steps - 1):
            print(f"step {s:4d}  loss {loss:.5f}")
    return history
