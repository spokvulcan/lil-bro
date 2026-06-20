# Stage D — close the LR fairness grid for an honest optimizer verdict.
#  - AdamW was pinned at 3e-4 (same under-tuning risk corrected for Muon): sweep min up.
#  - Confirm Muon's optimum isn't past 1e-2.
#  - Run max at Muon's optimum (1e-2), which Stage C never did.
# Same 800-step/accum-4 budget. Architectures R0-green -> FORCE_R2=1. (see results/v4_sweep.md)
MAXX="-DQK_NORM=1 -DATTN_SINK=1 -DSWIGLU_CLAMP=1 -DROPE_ROTARY_DIMS=16 -DN_HC=4 -DMTP_DEPTH=1"
MAXR0="-DQK_NORM=1 -DATTN_SINK=1 -DSWIGLU_CLAMP=1 -DROPE_ROTARY_DIMS=8 -DN_HC=4 -DMTP_DEPTH=1"
cells=(
"min_lr1e3|NONE|NONE|adamw|1e-3"
"min_lr3e3|NONE|NONE|adamw|3e-3"
"muon_lr3e2|NONE|NONE|muon|3e-2"
"max_lr1e2|$MAXX|$MAXR0|muon|1e-2"
)
