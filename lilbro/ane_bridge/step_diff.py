"""Optimizer step-diff — does the ANE's CPU optimizer match the numpy twin?

R1 proves the ANE's *gradients* are right. This proves the **optimizer update**
applied to those gradients is right — for both AdamW (the control) and **Muon**
(the new CPU-side Newton-Schulz update, the bug-prone part).

Isolation: from a shared init and one fixed batch, the ANE runs one
forward+backward, then applies one optimizer step to *its own* gradients and dumps
the post-step weights (``--dump-grads`` + ``--dump-weights``). Python loads those
same gradients and applies the ``lilbro.mlx_ref`` optimizer to the same init. We
compare the **update delta** ``w - init`` (not ``w`` itself — a small step leaves
``w ≈ init`` and would make any comparison trivially pass). So this tests the
optimizer's direction+magnitude, with the gradient held fixed and identical on
both sides (gradient correctness is R1's job).

Agreement is tight: both sides run Newton-Schulz / Adam on the *same* float32
gradients, NS in float64 on both — the only divergence is the final float32
weight write on the ANE. So the gate is far tighter than R1's fp16 gate.

Run:  ``.venv/bin/python -m lilbro.ane_bridge.step_diff``  (needs an Apple Neural
Engine; writes ``results/step_diff_metrics.json``).
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from lilbro.configs import Config
from lilbro.mlx_ref import init_params
from lilbro.mlx_ref.optim import make_optimizer, optimizer_step
from lilbro.eval import write_token_stream
from . import compare, serialize
from .r1_gate import build_batch
from .run import TRAIN_DIR, build, train_argv

RESULTS = Path(__file__).resolve().parents[2] / "results"

# Step-diff gate: both sides apply the same optimizer math to the same float32
# grads (NS in float64), so they agree to ~float32 weight-write round-off. This
# is much tighter than R1's fp16 gradient gate; it still fails a real optimizer
# bug (wrong NS coeffs, wrong scale, transposed update) by a wide margin.
GATE_MIN_COSINE = 0.9990
GATE_MAX_REL_L2 = 0.02

# Small dense configs (no MTP). Both optimizers gated on each.
STEP_CONFIGS = {
    "sd_base": Config(name="sd_base", dim=64, n_layers=2, n_heads=4, head_dim=16,
                      seq=32, vocab=256, hidden=128, seed=0),
    "sd_gqa2": Config(name="sd_gqa2", dim=64, n_layers=2, n_heads=4, kv_heads=2,
                      head_dim=16, seq=32, vocab=256, hidden=128, seed=2),
}
OPTIMIZERS = ("adamw", "muon")
STEP_LR = 0.1  # large enough that the update delta is well above float noise


def run_ane_step(cfg: Config, params: dict, file_tokens: np.ndarray, opt: str,
                 work: Path) -> tuple[dict, dict]:
    """Run one ANE optimizer step; return (ane_grads, ane_post_weights)."""
    build(cfg)
    init_bin = work / "init.bin"
    grads_bin = work / f"grads_{opt}.bin"
    weights_bin = work / f"weights_{opt}.bin"
    data_bin = work / "batch.bin"
    serialize.write_init(str(init_bin), params, cfg)
    write_token_stream(str(data_bin), file_tokens)

    argv = train_argv(
        cfg, scratch=True, overfit=True, steps=1, accum=1, clip=0.0, warmup=1,
        lr=STEP_LR, wd=0.0, opt=opt, data=str(data_bin), init=str(init_bin),
        dump_grads=str(grads_bin), dump_weights=str(weights_bin))
    proc = subprocess.run(argv, cwd=TRAIN_DIR, capture_output=True, text=True)
    if not weights_bin.exists():
        raise RuntimeError(f"ANE did not dump weights ({opt}):\n{proc.stdout}\n{proc.stderr}")
    return serialize.read_grads(str(grads_bin), cfg), serialize.read_flat(str(weights_bin), cfg)


def python_step(cfg: Config, init32: dict, g_ane: dict, opt: str) -> dict:
    """Apply one ``lilbro.mlx_ref`` optimizer step to the f32-rounded init using
    the ANE's own grads. Mirrors the ANE knobs (lr, wd=0, optimizer)."""
    cfg2 = dataclasses.replace(cfg, optimizer=opt, lr=STEP_LR, weight_decay=0.0)
    params = {k: v.astype(np.float64) for k, v in init32.items()}
    opt_obj = make_optimizer(cfg2)
    optimizer_step(opt_obj, params, {k: g_ane[k].astype(np.float64) for k in g_ane})
    return params


def step_diff_one(cfg: Config, opt: str, work: Path) -> dict:
    params = init_params(cfg)
    init32 = {k: v.astype(np.float32) for k, v in params.items()}  # what the ANE actually loads
    file_tokens, _, _ = build_batch(cfg)

    g_ane, w_ane = run_ane_step(cfg, params, file_tokens, opt, work)
    w_py = python_step(cfg, init32, g_ane, opt)

    # Compare the UPDATE delta (w - init), not w — a small step leaves w ~ init.
    d_ane = {k: w_ane[k] - init32[k] for k in w_ane}
    d_py = {k: w_py[k] - init32[k].astype(np.float64) for k in w_ane}
    metrics = compare.grad_metrics(d_ane, d_py)
    summary = compare.summarize(metrics)
    passed, bad = compare.gate(metrics, min_cosine=GATE_MIN_COSINE,
                               max_rel_l2=GATE_MAX_REL_L2)
    return {"summary": summary, "per_param": metrics,
            "passed": bool(passed), "offending": bad}


def main() -> int:
    records = {}
    all_pass = True
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for name, cfg in STEP_CONFIGS.items():
            records[name] = {}
            for opt in OPTIMIZERS:
                print(f"\n=== step-diff: {name} / {opt} ===")
                rec = step_diff_one(cfg, opt, work)
                records[name][opt] = rec
                s = rec["summary"]
                print(f"  min cosine = {s['min_cosine']:.6f} @ {s['min_cosine_param']}")
                print(f"  max rel_l2 = {s['max_rel_l2']:.5f} @ {s['max_rel_l2_param']}")
                print(f"  gate (cos>={GATE_MIN_COSINE}, rel_l2<={GATE_MAX_REL_L2}): "
                      f"{'PASS' if rec['passed'] else 'FAIL ' + str(rec['offending'])}")
                all_pass &= rec["passed"]

    out = {"gate": {"min_cosine": GATE_MIN_COSINE, "max_rel_l2": GATE_MAX_REL_L2},
           "optimizers": list(OPTIMIZERS), "configs": records}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "step_diff_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'step_diff_metrics.json'}  —  "
          f"{'ALL PASS' if all_pass else 'FAILURES PRESENT'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
