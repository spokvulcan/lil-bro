"""Shared CPU-side optimizers: AdamW and Muon.

Both twins (MLX, torch) compute gradients with their own autograd engine, then
hand the gradients to *this* numpy optimizer. Running the optimizer in one shared
place guarantees the Muon/AdamW update is byte-for-byte identical across trainers
(and mirrors the ANE design, where Muon is a CPU-side update).

Muon (Newton-Schulz orthogonalization) applies to 2D weight matrices; the
embedding/LM head and all norm vectors stay on AdamW (see ``params.is_muon_param``).
"""

from __future__ import annotations

import numpy as np

from .params import is_muon_param, param_spec
from lilbro.configs import Config


def _newton_schulz(G: np.ndarray, steps: int = 5, eps: float = 1e-7,
                   v4_hybrid: bool = False) -> np.ndarray:
    """Orthogonalize G via quintic Newton-Schulz in float64.

    The prior Muon path uses Keller Jordan's coefficients for every iteration.
    The trainer's default V4 path runs 10 iterations, with the final two using
    the stabilizing (2, -1.5, 0.5) triple. This mirrors ``cpu_ops.h``.
    """
    X = G.astype(np.float64).copy()
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    X /= (np.linalg.norm(X) + eps)
    for i in range(steps):
        if v4_hybrid and i >= steps - 2:
            a, b, c = 2.0, -1.5, 0.5
        else:
            a, b, c = 3.4445, -4.7750, 2.0315
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class AdamW:
    def __init__(self, lr: float, betas=(0.9, 0.95), eps: float = 1e-8, wd: float = 0.0):
        self.lr, self.b1, self.b2 = lr, betas[0], betas[1]
        self.eps, self.wd = eps, wd
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """One in-place update over all params. Owns the bias-correction clock."""
        self.t += 1
        for name, w in params.items():
            params[name] = self.step_param(name, w, grads[name])

    def step_param(self, name: str, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        assert self.t >= 1, "AdamW.step_param requires t>=1; call via step()"
        if name not in self.m:
            self.m[name] = np.zeros_like(w)
            self.v[name] = np.zeros_like(w)
        m, v = self.m[name], self.v[name]
        m[:] = self.b1 * m + (1 - self.b1) * g
        v[:] = self.b2 * v + (1 - self.b2) * g * g
        mh = m / (1 - self.b1 ** self.t)
        vh = v / (1 - self.b2 ** self.t)
        return w - self.lr * (mh / (np.sqrt(vh) + self.eps) + self.wd * w)


class Muon:
    """Muon for 2D matrices with SGD-momentum + Newton-Schulz; AdamW fallback for
    every parameter Muon does not own (embedding, norms)."""

    def __init__(self, cfg: Config, lr: float, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int | None = None,
                 variant: str = "v4"):
        self.cfg = cfg
        self.lr = lr
        self.mu = momentum
        self.nesterov = nesterov
        if variant not in {"v4", "prior"}:
            raise ValueError(f"unknown Muon variant {variant!r}; use 'v4' or 'prior'")
        self.variant = variant
        self.ns_steps = (10 if variant == "v4" else 5) if ns_steps is None else ns_steps
        self.buf: dict[str, np.ndarray] = {}
        self._kind = {n: k for n, k in param_spec(cfg)}
        # AdamW handles the non-Muon parameters with the same base lr.
        self.adamw = AdamW(lr, wd=cfg.weight_decay)

    def _owns(self, name: str) -> bool:
        return is_muon_param(name, self._kind[name])

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """One in-place update; Muon for owned 2D matrices, AdamW for the rest.
        Bumps the shared AdamW clock once so the fallback's bias correction is right."""
        self.adamw.t += 1
        for name, w in params.items():
            params[name] = self.step_param(name, w, grads[name])

    def step_param(self, name: str, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        if not self._owns(name):
            return self.adamw.step_param(name, w, g)
        if name not in self.buf:
            self.buf[name] = np.zeros_like(w)
        buf = self.buf[name]
        buf[:] = self.mu * buf + g
        d = g + self.mu * buf if self.nesterov else buf
        u = _newton_schulz(d, steps=self.ns_steps, v4_hybrid=(self.variant == "v4"))
        # NOTE: the ANE-side Muon MUST replicate this update exactly — momentum,
        # nesterov, coefficient schedule, scale, and weight decay.
        if self.variant == "v4":
            scale = max(w.shape) ** 0.5 * 0.18
            return w * (1.0 - self.lr * self.cfg.weight_decay) - self.lr * u * scale
        scale = max(1.0, w.shape[0] / w.shape[1]) ** 0.5
        return w - self.lr * u * scale


def make_optimizer(cfg: Config):
    """Build the optimizer named by ``cfg.optimizer``. AdamW needs ``t`` bumped
    each step; Muon delegates non-owned params to an inner AdamW that shares it."""
    if cfg.optimizer == "muon":
        return Muon(cfg, cfg.lr, momentum=0.95, nesterov=True, variant="v4")
    return AdamW(cfg.lr, wd=cfg.weight_decay)


def optimizer_step(opt, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
    """Apply one in-place update to ``params`` given ``grads`` (both keyed by name).
    Each optimizer owns its own bias-correction clock (see ``.step``)."""
    opt.step(params, grads)
