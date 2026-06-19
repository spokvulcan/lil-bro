"""Gradient-agreement metrics for the R1 gate.

These pin the math that decides "correct fp16 gradient" vs "silently wrong":
identical → cosine 1 / rel_l2 0; pure rescale → cosine 1 but rel_l2 grows;
direction error → cosine drops. The gate must pass fp16 noise and fail a real
backward bug.
"""

import numpy as np
import pytest

from lilbro.ane_bridge import tensor_metrics, grad_metrics, summarize, gate


def test_identical_is_perfect():
    a = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
    m = tensor_metrics(a, a)
    assert m["cosine"] == pytest.approx(1.0)
    assert m["rel_l2"] == pytest.approx(0.0)


def test_pure_rescale_keeps_direction():
    """fp16 magnitude noise = high cosine, nonzero rel_l2 (the expected regime)."""
    b = np.random.default_rng(1).standard_normal(64)
    a = b * 1.02  # 2% uniform magnitude error
    m = tensor_metrics(a, b)
    assert m["cosine"] == pytest.approx(1.0, abs=1e-9)
    assert m["rel_l2"] == pytest.approx(0.02, rel=1e-6)


def test_small_perturbation_high_cosine():
    rng = np.random.default_rng(2)
    b = rng.standard_normal(1000)
    a = b + 0.01 * rng.standard_normal(1000) * np.linalg.norm(b) / np.sqrt(1000)
    m = tensor_metrics(a, b)
    assert m["cosine"] > 0.999


def test_direction_error_tanks_cosine():
    """A real backward bug (wrong direction) must crater cosine."""
    rng = np.random.default_rng(3)
    b = rng.standard_normal(256)
    a = rng.standard_normal(256)  # unrelated → cosine ≈ 0
    m = tensor_metrics(a, b)
    assert abs(m["cosine"]) < 0.2


def test_sign_flip_is_negative_cosine():
    b = np.random.default_rng(4).standard_normal(32)
    m = tensor_metrics(-b, b)
    assert m["cosine"] == pytest.approx(-1.0)


def test_zero_oracle_handled():
    z = np.zeros(16)
    assert tensor_metrics(z, z)["rel_l2"] == 0.0
    assert tensor_metrics(np.ones(16), z)["rel_l2"] == float("inf")


def test_grad_metrics_key_mismatch_raises():
    with pytest.raises(ValueError, match="key mismatch"):
        grad_metrics({"a": np.ones(3)}, {"b": np.ones(3)})


def test_gate_and_summary():
    g_oracle = {"w": np.array([1.0, 2.0, 3.0]), "v": np.array([1.0, 0.0])}
    g_ane = {"w": np.array([1.01, 2.0, 2.99]), "v": np.array([1.0, 0.0])}
    m = grad_metrics(g_ane, g_oracle)
    s = summarize(m)
    assert s["n_params"] == 2
    assert s["min_cosine"] > 0.999
    passed, bad = gate(m, min_cosine=0.99, max_rel_l2=0.05)
    assert passed and bad == []
    # A strict gate flags the noisier tensor.
    passed2, bad2 = gate(m, min_cosine=0.999999999, max_rel_l2=1e-6)
    assert not passed2 and "w" in bad2
