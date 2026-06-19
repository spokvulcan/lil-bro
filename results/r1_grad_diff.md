# R1 — gradient-diff correctness gate (ANE vs fp64 oracle, real hardware)

**Status: GREEN.** For one fixed batch from a shared init, the Apple Neural
Engine's per-parameter gradients agree with the PyTorch **fp64** oracle to the
fp16 noise floor — across **both** the dense MHA path and **GQA**. This is the
ROADMAP **R1** gate: the scientifically decisive one. R0 showed loss goes down;
R1 proves the gradients are *right*, not silently wrong.

Reaching green required finding and fixing a real backward bug in the ANE trainer
(below) — exactly what this gate exists to catch.

## What R1 found: forward/backward GQA convention mismatch

The ANE's GQA forward and backward tiled K/V by **different conventions**, so for
any `GQA_RATIO > 1` model the K/V gradients were routed to the wrong heads —
silently. Loss still fell (the diagonal of the gradient is roughly right), which
is precisely the "genuinely learning vs silently wrong" failure R1 is built to
separate.

| | tile convention | source |
|---|---|---|
| Forward kernel | **interleaved** — q-head `h` → kv-head `h % KV_HEADS` (`concat(interleave=false)` over `GQA_RATIO` copies) | `mil_dynamic.h:171–181` |
| Backward (before) | **block** — `q_head = kv*GQA_RATIO + r` (→ `h // GQA_RATIO`) | `io.h:330–349` (old) |

Evidence trail: with the twin set either way the GQA grads failed (cosine **0.37**
then **0.44**), while MHA passed and **losses matched** (random-init loss ≈
`ln(vocab)` is insensitive to attention, so it hid the mismatch). The standout was
`wo` (attention output proj): cosine **0.44**, rel_l2 **>1.0** — i.e. `attn_out`
itself differed. Reading the forward kernel showed `cid = bool(false)`
(interleaved); the backward used block. The default **`qwen3_06b` model has
`GQA_RATIO=2`**, so this affected real training, not a corner case.

**Fix:** `gqa_tile_kv` / `gqa_reduce_kv` (`io.h`) now use the interleaved
convention (`q_head = r*KV_HEADS + kv`) to match the forward kernel. The twins use
`repeat` / `concatenate` (interleaved), with comments pinning them to
`mil_dynamic.h`. MHA (`GQA_RATIO=1`) is unaffected by the change.

## Result (M3 Max) — post-fix, all configs PASS

Gate (fp16-scale): **cosine ≥ 0.99** and **rel_l2 ≤ 0.10** for every parameter.

| config | arch | mean cosine | worst cosine | worst rel_l2 | gate |
|---|---|---|---|---|---|
| `r1_base` | MHA (4h) | 0.99977 | 0.99789 `@layer.1.w2` | 0.0657 `@layer.1.w2` | **PASS** |
| `r1_gqa2` | GQA 4h/2kv (ratio **2**, = qwen3_06b) | 0.99977 | 0.99778 `@layer.0.w2` | 0.0671 `@layer.0.w2` | **PASS** |
| `r1_gqa4` | GQA 8h/2kv (ratio 4) | 0.99977 | 0.99789 `@layer.1.w2` | 0.0654 `@layer.1.w2` | **PASS** |

All configs: `dim=64, n_layers=2, head_dim=16, seq=32, vocab=256, hidden=128`.
`r1_gqa2` uses the **same GQA ratio (2) as the default `qwen3_06b`**, so the case
the bug actually shipped in is gated directly, not by inference. After the fix the
GQA profiles are **identical** to MHA — the GQA-specific divergence is gone.
Losses agree (e.g. gqa2 `5.5632` vs `5.5634`). Metrics are deterministic
run-to-run (cosine bit-identical).

### Why these thresholds (not the fp32 `1e-4`)

The MLX-vs-torch gate (`tests/test_grad_diff.py`) uses element-wise `1e-4` because
both are ≥fp32. **The ANE matmuls are fp16**, so the right yardstick is direction
(cosine) + magnitude (relative L2), not element-wise max-abs. The clean floor is
cosine `0.998` / rel_l2 `0.067`, both at the FFN down-projection `w2` (the largest
fp16 matmul — direction essentially perfect, magnitude within fp16 reach). The
GQA bug sat at cosine `0.44` / rel_l2 `>1.0`. The gate clears the floor with
margin yet fails that bug — and any structural backward error — by a wide gap;
`rel_l2 ≤ 0.10` (~1.5× the floor) keeps the magnitude bound tight enough that a
modest systematic scale error can't pass on the cosine bound alone.

The diff is taken on the **raw gradient, before the optimizer step**, so it is
optimizer-agnostic — Muon vs AdamW change the *update*, not the gradient. Muon's
end-to-end correctness is covered by R0 (`results/r0_overfit.md`, loss→0 under
Muon); R1 validates the gradient itself.

## Reproduce

```bash
.venv/bin/python -m lilbro.ane_bridge.r1_gate     # needs Apple Neural Engine
.venv/bin/python -m pytest tests/test_ane_grad_diff.py
```

The orchestrator: emits each config's C header from the shared `lilbro.configs`,
builds, writes the shared numpy init (`lilbro.ane_bridge` flat format) + a fixed
batch, runs `./train --init … --dump-grads … --overfit --steps 1 --accum 1`, reads
the raw gradients back, and diffs them against the torch fp64 oracle. Writes
`results/r1_metrics.json` (gitignored; regenerable).

### Bridge mechanics (how the ANE joins the diff)

- **Shared init in / raw grads out** — `train.m` gained `--init` (load the same
  numpy weights the twins use) and `--dump-grads` (dump gradients at the accum
  unscale point — `loss_scale` cancelled, LM-head folded into the tied embedding,
  **before** clip/Adam — then exit). Flat float32, `param_spec` order; the Python
  end is `lilbro/ane_bridge` (`tests/test_ane_bridge.py` pins the contract).
- **Vocab primer** — the ANE softmaxes over the *compact* vocab (ids present in
  the data); the oracle over the full vocab. R1 appends `[0..vocab-1]` after the
  pinned batch window so every id is active (`CV==VOCAB`, identity compaction),
  making the two LM-head softmaxes identical. `--overfit` pins `pos=0`, so the
  primer never enters the batch.

## Scope / next

- **Covered:** dense MHA + GQA, full parameter set (embed, attn, FFN, norms),
  AdamW path. The ANE trainer has **no MTP path**, so MTP stays twin-only
  (`tests/test_grad_diff.py`) until the ANE MTP op lands; `lilbro.ane_bridge`
  rejects MTP configs rather than mis-encode them.
- **Next:** wire the parameterized runtime config (PRD 1a) and move up the ladder
  (R2) now that both gates — R0 (loss→0) and R1 (grads correct) — are green.
