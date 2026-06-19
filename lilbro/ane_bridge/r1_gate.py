"""R1 correctness gate — ANE gradients vs the torch fp64 oracle, on real hardware.

End-to-end driver: for a given dense config it seeds the ANE trainer with the
shared numpy init, runs exactly one forward+backward on one fixed batch, dumps
the raw gradients, and diffs them against the torch fp64 oracle's gradients for
the *same* init and batch. This is the scientifically decisive gate — it proves
the ANE's gradients are right (not merely that loss goes down, which R0 showed).

Why the vocab primer: the ANE's LM head softmaxes over the *compact* vocab (only
token ids present in the data file; see ``cpu_ops.h:vocab_map_build``), while the
oracle softmaxes over the full vocab. To make the two identical we append
``[0,1,…,vocab-1]`` after the batch window so every id is active (``CV==VOCAB``,
identity compaction). ``--overfit`` pins ``pos=0`` (``train.m``), so the batch is
always the first ``seq+1`` tokens and the primer never perturbs it.

Run:  ``.venv/bin/python -m lilbro.ane_bridge.r1_gate``  (needs an Apple Neural
Engine; writes ``results/r1_metrics.json``).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from lilbro.configs import Config, write_c_header
from lilbro.mlx_ref import init_params
from lilbro.mlx_ref import twin_torch as oracle
from lilbro.eval import write_token_stream
from . import compare, serialize

REPO = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO / "training" / "training_dynamic"
RESULTS = REPO / "results"

# The R1 gate, in fp16-appropriate terms (justified by results/r1_grad_diff.md).
# Every parameter's gradient must point essentially the same way as the oracle's
# (cosine) and have magnitude within fp16 reach (rel_l2). Measured clean floor on
# an M3 Max is cosine 0.9979 / rel_l2 0.066 (both at the FFN down-proj w2, the
# largest fp16 matmul); the GQA backward bug R1 found sat at cosine 0.44 /
# rel_l2 >1.0. These thresholds clear the floor with margin yet fail that bug — and
# any structural backward error — by a wide gap. rel_l2 is held to ~1.5x the floor
# so a modest systematic magnitude error can't slip past the (direction-only)
# cosine bound. Element-wise fp32 tolerance (the 1e-4 used MLX-vs-torch) does NOT
# apply: the ANE matmuls are fp16.
GATE_MIN_COSINE = 0.99
GATE_MAX_REL_L2 = 0.10

# Dense configs the ANE trainer can run today (no MTP path on the ANE side).
# Both GQA ratios are gated directly: ratio 2 is the production qwen3_06b case;
# ratio 4 stresses the interleave/block head grouping harder.
R1_CONFIGS = {
    "r1_base": Config(name="r1_base", dim=64, n_layers=2, n_heads=4, head_dim=16,
                      seq=32, vocab=256, hidden=128, seed=0),
    "r1_gqa2": Config(name="r1_gqa2", dim=64, n_layers=2, n_heads=4, kv_heads=2,
                      head_dim=16, seq=32, vocab=256, hidden=128, seed=2),
    "r1_gqa4": Config(name="r1_gqa4", dim=64, n_layers=2, n_heads=8, kv_heads=2,
                      head_dim=16, seq=32, vocab=256, hidden=128, seed=1),
}


def build_batch(cfg: Config, seed: int = 12345):
    """One fixed batch (b=1) + a full-vocab primer. Returns (file_tokens, x, y)."""
    rng = np.random.default_rng(seed)
    window = rng.integers(0, cfg.vocab, size=cfg.seq + 1, dtype=np.uint16)
    primer = np.arange(cfg.vocab, dtype=np.uint16)          # force CV == VOCAB
    file_tokens = np.concatenate([window, primer])
    x = window[: cfg.seq][None, :].astype(np.int64)         # inputs  tokens[0:seq]
    y = window[1 : cfg.seq + 1][None, :].astype(np.int64)   # targets tokens[1:seq+1]
    return file_tokens, x, y


def run_ane_grads(cfg: Config, params: dict, file_tokens: np.ndarray,
                  work: Path) -> tuple[dict, float]:
    """Compile + run the ANE trainer for one batch; return (grads, ane_loss)."""
    model = f"gen_{cfg.name}"
    write_c_header(cfg, str(TRAIN_DIR / "models"))
    subprocess.run(["make", f"MODEL={model}"], cwd=TRAIN_DIR, check=True,
                   capture_output=True, text=True)

    init_bin = work / "init.bin"
    grads_bin = work / "grads.bin"
    data_bin = work / "batch.bin"
    serialize.write_init(str(init_bin), params, cfg)
    write_token_stream(str(data_bin), file_tokens)

    proc = subprocess.run(
        ["./train", "--scratch", "--init", str(init_bin),
         "--dump-grads", str(grads_bin), "--overfit", "--data", str(data_bin),
         "--steps", "1", "--accum", "1", "--clip", "0", "--warmup", "1", "--lr", "0"],
        cwd=TRAIN_DIR, check=True, capture_output=True, text=True)

    if not grads_bin.exists():
        raise RuntimeError(f"ANE did not dump grads:\n{proc.stdout}\n{proc.stderr}")
    m = re.search(r"loss=([0-9.]+)", proc.stdout)
    ane_loss = float(m.group(1)) if m else float("nan")
    return serialize.read_grads(str(grads_bin), cfg), ane_loss


def gate_one(cfg: Config, work: Path) -> dict:
    """Run + score one config. Returns a JSON-able record."""
    params = init_params(cfg)
    file_tokens, x, y = build_batch(cfg)

    g_ane, ane_loss = run_ane_grads(cfg, params, file_tokens, work)
    oracle_loss, g_oracle = oracle.loss_and_grads(params, x, y, cfg)

    # Compare only the dense params the ANE produces (== all of them here).
    g_oracle = {k: g_oracle[k] for k in g_ane}
    metrics = compare.grad_metrics(g_ane, g_oracle)
    summary = compare.summarize(metrics)
    passed, bad = compare.gate(metrics, min_cosine=GATE_MIN_COSINE,
                               max_rel_l2=GATE_MAX_REL_L2)
    return {
        "cfg": {"name": cfg.name, "dim": cfg.dim, "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads, "kv_heads": cfg.kv_heads,
                "head_dim": cfg.head_dim, "seq": cfg.seq, "vocab": cfg.vocab,
                "hidden": cfg.hidden},
        "loss_ane": ane_loss, "loss_oracle": float(oracle_loss),
        "summary": summary, "per_param": metrics,
        "passed": bool(passed), "offending": bad,
    }


def main() -> int:
    records = {}
    all_pass = True
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for name, cfg in R1_CONFIGS.items():
            print(f"\n=== R1: {name} ===")
            rec = gate_one(cfg, work)
            records[name] = rec
            s = rec["summary"]
            print(f"  loss: ane={rec['loss_ane']:.4f} oracle={rec['loss_oracle']:.4f}")
            print(f"  min cosine = {s['min_cosine']:.6f} @ {s['min_cosine_param']}")
            print(f"  max rel_l2 = {s['max_rel_l2']:.4f} @ {s['max_rel_l2_param']}")
            print(f"  gate (cos>={GATE_MIN_COSINE}, rel_l2<={GATE_MAX_REL_L2}): "
                  f"{'PASS' if rec['passed'] else 'FAIL ' + str(rec['offending'])}")
            all_pass &= rec["passed"]

    out = {
        "gate": {"min_cosine": GATE_MIN_COSINE, "max_rel_l2": GATE_MAX_REL_L2},
        "configs": records,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "r1_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'r1_metrics.json'}  —  "
          f"{'ALL PASS' if all_pass else 'FAILURES PRESENT'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
