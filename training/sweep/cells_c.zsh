# Stage C — fairness LR sweep for Muon, and max re-run at a fair Muon LR.
# Architectures already R0-green (run with FORCE_R2=1). Same 800-step/accum-4 budget
# as the screen so numbers are directly comparable. (see results/v4_sweep.md)
# name | r2_extra | r0_extra | opt | lr
MAXX="-DQK_NORM=1 -DATTN_SINK=1 -DSWIGLU_CLAMP=1 -DROPE_ROTARY_DIMS=16 -DN_HC=4 -DMTP_DEPTH=1"
MAXR0="-DQK_NORM=1 -DATTN_SINK=1 -DSWIGLU_CLAMP=1 -DROPE_ROTARY_DIMS=8 -DN_HC=4 -DMTP_DEPTH=1"
cells=(
"muon_lr1e3|NONE|NONE|muon|1e-3"
"muon_lr3e3|NONE|NONE|muon|3e-3"
"muon_lr1e2|NONE|NONE|muon|1e-2"
"max_lr1e3|$MAXX|$MAXR0|muon|1e-3"
"max_lr3e3|$MAXX|$MAXR0|muon|3e-3"
)
