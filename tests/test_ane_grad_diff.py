"""R1 correctness gate — ANE gradients vs the torch fp64 oracle.

This validates the *measured* ANE-vs-oracle gradient agreement recorded by the
hardware orchestrator (``lilbro.ane_bridge.r1_gate``) in
``results/r1_metrics.json``. It re-applies the documented fp16-scale gate to the
per-parameter metrics, so it is the authority on the thresholds while the JSON is
the data.

The metrics file is a regenerable hardware artifact (gitignored): on a machine
without an Apple Neural Engine, or before the gate has been run, this test skips
with instructions. The committed human-readable evidence is
``results/r1_grad_diff.md``. The always-on, hardware-independent correctness
gate is ``tests/test_grad_diff.py`` (MLX twin vs torch fp64).
"""

import json
from pathlib import Path

import pytest

from lilbro.ane_bridge import gate

# The R1 gate thresholds (fp16-scale). Must match lilbro/ane_bridge/r1_gate.py
# and the rationale in results/r1_grad_diff.md.
MIN_COSINE = 0.99
MAX_REL_L2 = 0.10

METRICS = Path(__file__).resolve().parents[1] / "results" / "r1_metrics.json"


def _load():
    if not METRICS.exists():
        pytest.skip(
            "no results/r1_metrics.json — run the ANE R1 gate on Apple Neural "
            "Engine hardware first: .venv/bin/python -m lilbro.ane_bridge.r1_gate")
    return json.loads(METRICS.read_text())


def test_r1_metrics_thresholds_match_orchestrator():
    """The recorded gate must be the one this test enforces (no silent drift)."""
    data = _load()
    assert data["gate"]["min_cosine"] == MIN_COSINE
    assert data["gate"]["max_rel_l2"] == MAX_REL_L2


def test_r1_covers_mha_and_gqa():
    """R1 must exercise both the dense MHA path and GQA (gqa_ratio>1) — the GQA
    backward is precisely where the bug R1 caught lived."""
    data = _load()
    cfgs = data["configs"]
    assert any(c["cfg"]["kv_heads"] == c["cfg"]["n_heads"] for c in cfgs.values()), \
        "expected an MHA config (kv_heads == n_heads)"
    assert any(c["cfg"]["kv_heads"] < c["cfg"]["n_heads"] for c in cfgs.values()), \
        "expected a GQA config (kv_heads < n_heads)"


def test_r1_all_params_pass_gate():
    """Every parameter of every config agrees with the fp64 oracle within the
    fp16-scale gate. Re-applied here from per-param metrics (not trusting the
    stored 'passed' flag)."""
    data = _load()
    for name, rec in data["configs"].items():
        passed, bad = gate(rec["per_param"], min_cosine=MIN_COSINE,
                           max_rel_l2=MAX_REL_L2)
        assert passed, f"{name}: params outside R1 gate: {bad}"


def test_r1_losses_agree():
    """Forward sanity: ANE loss tracks the oracle (rules out gross fwd drift)."""
    data = _load()
    for name, rec in data["configs"].items():
        assert rec["loss_ane"] == pytest.approx(rec["loss_oracle"], abs=2e-2), name
