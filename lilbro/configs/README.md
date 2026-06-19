# lilbro/configs

Parametric model + run configs shared by **all** trainers (ANE, MLX). Replaces
upstream's compile-time `#define`s in `training/stories_config.h`.

One config = one ladder rung / ablation cell. Fields (planned): `dim`, `n_layers`,
`n_heads`, `seq`, `vocab`, `optimizer` (adamw|muon), `mtp_depth`, `lr`, `batch_tokens`.

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
