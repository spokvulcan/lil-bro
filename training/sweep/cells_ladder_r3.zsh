# Scale-ladder R2 cells for r3_110m, with LR grids RE-CENTERED on the optima extrapolated
# from the 13M->42M drift (ADR 0003 "extend if optimum on an edge", applied with foresight
# at the expensive ceiling rung to avoid a slow edge-extension round):
#   - adamw optimum drifted 1e-3 (13M) -> 3e-4 (42M); extrapolate r3 ~1e-4 -> {3e-5,1e-4,3e-4}
#   - mHC   optimum drifted 1e-2 (13M) -> 3e-3 (42M); extrapolate r3 ~3e-3 -> {1e-3,3e-3,1e-2}
#   - muon  optimum HELD at 1e-2 across both rungs (scale-robust) -> keep {3e-3,1e-2,3e-2}
# Same families/builds as cells_ladder.zsh; only the LR points move. If a family's r3
# optimum still lands on an edge, extend that family (cells_ext_r3.zsh) before the verdict.
#   name | r2_extra | r0_extra | opt | lr
cells=(
# --- plain Muon (optimum held at 1e-2 across 13M/42M) -----------------------------
"muon_lr3e3|NONE|NONE|muon|3e-3"
"muon_lr1e2|NONE|NONE|muon|1e-2"
"muon_lr3e2|NONE|NONE|muon|3e-2"
# --- plain AdamW (optimum drifting down: 1e-3 -> 3e-4 -> ~1e-4) --------------------
"adamw_lr3e5|NONE|NONE|adamw|3e-5"
"adamw_lr1e4|NONE|NONE|adamw|1e-4"
"adamw_lr3e4|NONE|NONE|adamw|3e-4"
# --- Muon + mHC x2 (optimum drifting down: 1e-2 -> 3e-3) ---------------------------
"mhc2_lr1e3|-DN_HC=2|-DN_HC=2|muon|1e-3"
"mhc2_lr3e3|-DN_HC=2|-DN_HC=2|muon|3e-3"
"mhc2_lr1e2|-DN_HC=2|-DN_HC=2|muon|1e-2"
# --- Muon + mHC x4 (paper anchor; the headline redundancy question) ---------------
"mhc4_lr1e3|-DN_HC=4|-DN_HC=4|muon|1e-3"
"mhc4_lr3e3|-DN_HC=4|-DN_HC=4|muon|3e-3"
"mhc4_lr1e2|-DN_HC=4|-DN_HC=4|muon|1e-2"
)
