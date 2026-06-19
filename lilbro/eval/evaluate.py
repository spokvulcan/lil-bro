"""Validation-loss evaluation — the signal behind the headline tokens-to-target.

Evaluated on a *fixed* set of batches (see ``TokenStream.fixed_val_batches``) so
the number is comparable step-to-step and run-to-run.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from lilbro.configs import Config

LossOnly = Callable[[dict, np.ndarray, np.ndarray, Config], float]


def val_loss(loss_only: LossOnly, params: dict, batches, cfg: Config) -> float:
    """Mean loss over the fixed val batch set (token-weighted equally per batch)."""
    if not batches:
        raise ValueError("no validation batches")
    total = 0.0
    for x, y in batches:
        total += loss_only(params, x, y, cfg)
    return total / len(batches)
