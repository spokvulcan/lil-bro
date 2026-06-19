# lilbro/configs

Parametric model + run configs shared by **all** trainers (ANE, MLX). Replaces
upstream's compile-time `#define`s in `training/stories_config.h`.

One config = one ladder rung / ablation cell. Fields: `dim`, `n_layers`,
`n_heads`, `seq`, `vocab`, `optimizer` (adamw|muon), `mtp_depth`, `lr`, `batch_tokens`.

### DeepSeek-V4 ablation knobs (PRD #2)

Default-**off** / identity knobs, one per V4 mechanism, so toggling a mechanism is
a config change (a re-emitted header) rather than a source edit:

| Field | Default | Mechanism |
|---|---|---|
| `qk_norm` | `False` | Q/KV RMSNorm before attention scores (#7) |
| `attn_sink` | `False` | learnable per-head attention-sink logit (#8) |
| `swiglu_clamp` | `False` | clamp SwiGLU linear to [-10,10], gate cap 10 (#9) |
| `rope_rotary_dims` | `64` | partial RoPE: rotate `min(head_dim, this)` dims (#10) |
| `n_hc` | `1` | mHC residual-stream width; `0`/`1` = off (#11) |

`rope_rotary_dims=64` is identity wherever `head_dim <= 64` (every current ladder
rung); the derived `rope_rotary_eff = min(head_dim, rope_rotary_dims)` is what
actually rotates. With every knob at its default a config is bit-for-bit the plain
pre-norm transformer — verified by the R0 overfit gate landing on the identical
loss (`results/r0_overfit.md`).

The same config file must drive the ANE trainer and the MLX twin, so the
correctness diff (R1) and the energy comparison (c2) compare *identical* models.

## Built

- `schema.py` — `Config` (frozen dataclass; PRD fields + derived `q_dim`,
  `kv_dim`, `gqa_ratio`, `batch_size`, `res_alpha`), JSON load/save, validation,
  and the ladder presets `r0_overfit` / `r2_small` / `r3_110m`.
- `emit_c.py` — the **ANE consumer**: renders a `models/*.h`-compatible C header
  from a `Config` (the MLX twin is the other consumer, importing `Config`
  directly). `python -m lilbro.configs.emit_c r2_small` prints one.

Wiring the Objective-C trainer to *consume* the emitted header (instead of its
hand-written `models/*.h`) is the deferred ANE-side step. Tests:
`tests/test_configs.py` (round-trip + two-consumer identical-dims, incl. a C
compile check in the session log).
