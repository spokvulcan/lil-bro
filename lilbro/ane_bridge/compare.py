"""Gradient agreement metrics for the R1 gate (fp16 ANE vs fp64 oracle).

The ANE computes gradients through fp16 matmul kernels; the oracle is fp64. So
element-wise max-abs tolerance (the fp32 metric used for MLX-vs-torch) is the
wrong yardstick — a correct fp16 gradient differs from fp64 by ~1e-3 per op and
more after accumulation. What actually separates "correct" from "silently wrong"
gradients here is **direction** and **magnitude**:

  - cosine similarity per tensor — does ∇_ANE point the same way as ∇_oracle?
    A real backward bug (wrong transpose, missing term, bad mask) tanks this far
    below fp16 rounding; uniform fp16 noise leaves it ≈1.
  - relative L2 (``‖∇_ANE − ∇_oracle‖ / ‖∇_oracle‖``) — is the magnitude right?

Reporting per-parameter (not just an aggregate) is deliberate: a single
divergent tensor is the signature of a forward/backward mismatch in one op,
which an averaged score would hide.
"""

from __future__ import annotations

import numpy as np


def _flat(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def tensor_metrics(ane: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
    """Direction + magnitude agreement of one gradient tensor vs the oracle."""
    a, b = _flat(ane), _flat(oracle)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    diff = float(np.linalg.norm(a - b))
    denom = na * nb
    cosine = float(a @ b / denom) if denom > 0 else (1.0 if diff == 0 else 0.0)
    rel_l2 = diff / nb if nb > 0 else (0.0 if diff == 0 else float("inf"))
    max_abs = float(np.abs(a - b).max()) if a.size else 0.0
    return {
        "cosine": cosine,
        "rel_l2": rel_l2,
        "max_abs": max_abs,
        "ane_norm": na,
        "oracle_norm": nb,
    }


def grad_metrics(g_ane: dict[str, np.ndarray],
                 g_oracle: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Per-parameter metrics. Requires identical key sets (same model both sides)."""
    if set(g_ane) != set(g_oracle):
        only_a = sorted(set(g_ane) - set(g_oracle))
        only_o = sorted(set(g_oracle) - set(g_ane))
        raise ValueError(f"grad key mismatch: ane-only={only_a} oracle-only={only_o}")
    return {name: tensor_metrics(g_ane[name], g_oracle[name]) for name in g_oracle}


def summarize(metrics: dict[str, dict[str, float]]) -> dict[str, object]:
    """Worst-case rollup across parameters (the gate reads these)."""
    worst_cos_name = min(metrics, key=lambda n: metrics[n]["cosine"])
    worst_rel_name = max(metrics, key=lambda n: metrics[n]["rel_l2"])
    return {
        "n_params": len(metrics),
        "min_cosine": metrics[worst_cos_name]["cosine"],
        "min_cosine_param": worst_cos_name,
        "max_rel_l2": metrics[worst_rel_name]["rel_l2"],
        "max_rel_l2_param": worst_rel_name,
        "mean_cosine": float(np.mean([m["cosine"] for m in metrics.values()])),
        "mean_rel_l2": float(np.mean([m["rel_l2"] for m in metrics.values()])),
    }


def gate(metrics: dict[str, dict[str, float]], *,
         min_cosine: float, max_rel_l2: float) -> tuple[bool, list[str]]:
    """Pass/fail against fp16-scale thresholds; returns (passed, offending params)."""
    bad = [n for n, m in metrics.items()
           if m["cosine"] < min_cosine or m["rel_l2"] > max_rel_l2]
    return (len(bad) == 0, sorted(bad))
