# lilbro/configs

Parametric model + run configs shared by **all** trainers (ANE, MLX). Replaces
upstream's compile-time `#define`s in `training/stories_config.h`.

One config = one ladder rung / ablation cell. Fields (planned): `dim`, `n_layers`,
`n_heads`, `seq`, `vocab`, `optimizer` (adamw|muon), `mtp_depth`, `lr`, `batch_tokens`.

The same config file must drive the ANE trainer and the MLX twin, so the
correctness diff (R1) and the energy comparison (c2) compare *identical* models.
