"""Optimizer step-diff gate — the ANE's AdamW/Muon update vs the numpy twin.

Validates the hardware artifact written by ``lilbro.ane_bridge.step_diff`` in
``results/step_diff_metrics.json``: from a shared init + the ANE's own gradients,
one optimizer step on the ANE must match the ``lilbro.mlx_ref`` optimizer applied
to the same init+grads, comparing the update delta ``w - init``.

R1 (``test_ane_grad_diff``) gates the gradients; this gates the *step* applied to
them, for both optimizers — Muon especially, whose Newton-Schulz update is the
new, bug-prone part. The metrics file is a regenerable, gitignored hardware
artifact: this skips without it.
"""

import json
from pathlib import Path

import pytest

from lilbro.ane_bridge import gate

MIN_COSINE = 0.9990
MAX_REL_L2 = 0.02

METRICS = Path(__file__).resolve().parents[1] / "results" / "step_diff_metrics.json"


def _load():
    if not METRICS.exists():
        pytest.skip(
            "no results/step_diff_metrics.json — run the optimizer step-diff on "
            "Apple Neural Engine hardware first: "
            ".venv/bin/python -m lilbro.ane_bridge.step_diff")
    return json.loads(METRICS.read_text())


def test_thresholds_match_orchestrator():
    data = _load()
    assert data["gate"]["min_cosine"] == MIN_COSINE
    assert data["gate"]["max_rel_l2"] == MAX_REL_L2


def test_both_optimizers_covered():
    data = _load()
    assert set(data["optimizers"]) == {"adamw", "muon"}
    for cfg in data["configs"].values():
        assert {"adamw", "muon"} <= set(cfg)


def test_all_params_pass_gate():
    """Every parameter's update delta matches the twin within the (tight) gate."""
    data = _load()
    for name, opts in data["configs"].items():
        for opt, rec in opts.items():
            passed, bad = gate(rec["per_param"], min_cosine=MIN_COSINE,
                               max_rel_l2=MAX_REL_L2)
            assert passed, f"{name}/{opt}: params outside step-diff gate: {bad}"


def test_muon_is_not_adamw():
    """Muon must produce a genuinely different matrix update than AdamW — i.e. it
    actually orthogonalizes (Newton-Schulz) rather than silently falling back to
    Adam. The 2D weight matrices' update magnitude must differ between optimizers;
    the non-matrix params (norms, embed) stay AdamW in both and must NOT differ."""
    data = _load()
    for name, opts in data["configs"].items():
        a, m = opts["adamw"]["per_param"], opts["muon"]["per_param"]
        matrix = [k for k in a if k.endswith((".wq", ".wk", ".wv", ".wo",
                                              ".w1", ".w2", ".w3"))]
        assert matrix, "expected weight-matrix params"
        # Muon's update on every matrix differs in magnitude from AdamW's.
        for k in matrix:
            assert abs(a[k]["ane_norm"] - m[k]["ane_norm"]) / (a[k]["ane_norm"] + 1e-12) > 0.05, \
                f"{name}: {k} muon update ~= adamw update (Muon not engaged?)"
        # Norms / embedding are AdamW in both modes -> identical update.
        for k in ("rms_final", "embed"):
            assert m[k]["ane_norm"] == pytest.approx(a[k]["ane_norm"], rel=1e-3), \
                f"{name}: {k} should be optimizer-independent (AdamW in both)"
