"""Shared parameter set for the dense + MTP model.

A single numpy (float64) init that *both* the MLX twin and the torch oracle load
verbatim, so the R1 gradient diff compares identical models from identical
weights. Parameters are a flat ``{name: ndarray}`` dict so grads can be compared
by name and the optimizer can route per-parameter (Muon vs AdamW).

Weight layout is PyTorch ``nn.Linear`` convention: ``[out, in]``, applied as
``x @ W.T`` (no biases anywhere). The token embedding is **tied** to the LM head.
"""

from __future__ import annotations

import numpy as np

from lilbro.configs import Config

# MTP loss weight (shared constant; identical across both trainers).
MTP_LAMBDA = 0.3
_INIT_STD = 0.02


def _block_param_names(prefix: str) -> list[tuple[str, str]]:
    """(name, kind) for one transformer block. kind in {matrix, vector}."""
    return [
        (f"{prefix}.wq", "wq"), (f"{prefix}.wk", "wk"), (f"{prefix}.wv", "wv"),
        (f"{prefix}.wo", "wo"), (f"{prefix}.w1", "w1"), (f"{prefix}.w2", "w2"),
        (f"{prefix}.w3", "w3"),
        (f"{prefix}.rms_att", "vec"), (f"{prefix}.rms_ffn", "vec"),
    ]


def _shape_for(kind: str, cfg: Config) -> tuple[int, ...]:
    d, hd, q, kv = cfg.dim, cfg.head_dim, cfg.q_dim, cfg.kv_dim
    return {
        "wq": (q, d), "wk": (kv, d), "wv": (kv, d), "wo": (d, q),
        "w1": (cfg.hidden, d), "w2": (d, cfg.hidden), "w3": (cfg.hidden, d),
        "vec": (d,),
        "proj": (d, 2 * d),
        "embed": (cfg.vocab, d),
    }[kind]


def param_spec(cfg: Config) -> list[tuple[str, str]]:
    """Ordered (name, kind) list — defines deterministic init draw order."""
    spec: list[tuple[str, str]] = [("embed", "embed")]
    for l in range(cfg.n_layers):
        spec += _block_param_names(f"layer.{l}")
    spec.append(("rms_final", "vec"))
    for d in range(1, cfg.mtp_depth + 1):
        spec.append((f"mtp.{d}.rms_h", "vec"))
        spec.append((f"mtp.{d}.rms_e", "vec"))
        spec.append((f"mtp.{d}.proj", "proj"))
        spec += _block_param_names(f"mtp.{d}")
    return spec


def param_shapes(cfg: Config) -> dict[str, tuple[int, ...]]:
    """Ordered ``{name: shape}`` map, the single source of per-parameter shape.

    Same order as ``param_spec``/``init_params``; shared by the twins and the
    ANE bridge so the flat on-disk layout cannot drift from the model."""
    return {name: _shape_for(kind, cfg) for name, kind in param_spec(cfg)}


# Which parameters get the Muon optimizer: 2D weight matrices, EXCEPT the tied
# embedding/LM head. Norms (vectors) and the embedding stay on AdamW (per PRD).
def is_muon_param(name: str, kind: str) -> bool:
    if kind in ("vec", "embed"):
        return False
    return True  # wq/wk/wv/wo/w1/w2/w3/proj — all 2D


def init_params(cfg: Config, seed: int | None = None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    out: dict[str, np.ndarray] = {}
    for name, kind in param_spec(cfg):
        shape = _shape_for(kind, cfg)
        if kind == "vec":
            out[name] = np.ones(shape, dtype=np.float64)
        else:
            out[name] = (rng.standard_normal(shape) * _INIT_STD).astype(np.float64)
    return out
