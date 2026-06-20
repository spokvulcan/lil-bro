# Scale-ladder R2 cells (see results/scale_ladder.md, ADR 0003). The V4-sweep WINNING
# configs, carried up the model-size ladder (r2_small -> r2_mid -> r3_110m) via MODEL=.
# Each family at its own >=3-point LR grid, bracketing the r2_small optimum (extend a
# grid if a larger rung's optimum lands on an edge). Held-out val vs TRAINING TOKENS
# (tokens = step*256; seq/b/accum held fixed across rungs -> shared token axis).
#
#   name | r2_extra | r0_extra | opt | lr      (run_r2 uses name,r2_extra,opt,lr)
# Builds: optimizer is a RUNTIME flag, so muon & adamw share the no-knob "plain" build;
# only N_HC is a compile flag -> 3 distinct builds (plain / mhc2 / mhc4), gated in
# cells_ladder_r0.zsh. mHC (n_hc>1) auto-stamps the v5 checkpoint; each cell's --ckpt
# is distinct (r2_<name>.bin) so there is no collision.
cells=(
# --- plain Muon (the r2_small winner; optimum ~1e-2) -------------------------------
"muon_lr3e3|NONE|NONE|muon|3e-3"
"muon_lr1e2|NONE|NONE|muon|1e-2"
"muon_lr3e2|NONE|NONE|muon|3e-2"
# --- plain AdamW (well-tuned baseline; optimum ~1e-3) ------------------------------
"adamw_lr3e4|NONE|NONE|adamw|3e-4"
"adamw_lr1e3|NONE|NONE|adamw|1e-3"
"adamw_lr3e3|NONE|NONE|adamw|3e-3"
# --- Muon + mHC x2 (redundancy test at scale) -------------------------------------
"mhc2_lr3e3|-DN_HC=2|-DN_HC=2|muon|3e-3"
"mhc2_lr1e2|-DN_HC=2|-DN_HC=2|muon|1e-2"
"mhc2_lr3e2|-DN_HC=2|-DN_HC=2|muon|3e-2"
# --- Muon + mHC x4 (paper anchor; the headline redundancy question) ---------------
"mhc4_lr3e3|-DN_HC=4|-DN_HC=4|muon|3e-3"
"mhc4_lr1e2|-DN_HC=4|-DN_HC=4|muon|1e-2"
"mhc4_lr3e2|-DN_HC=4|-DN_HC=4|muon|3e-2"
)
