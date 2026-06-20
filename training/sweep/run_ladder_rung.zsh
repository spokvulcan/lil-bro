#!/bin/zsh
# Scale-ladder: run ONE rung end-to-end (see ADR 0003, results/scale_ladder.md).
#   1. R0 gate the 3 distinct builds (plain/mhc2/mhc4) at the rung's OWN dims.
#   2. Only if all 3 collapse (<PASS_THRESH), R2 the 12 cells (4 families x 3 LR).
# Per-rung artifacts go to $SWEEP_WORK (default /tmp/ladder/$MODEL) so rungs don't collide.
#
# Usage:   MODEL=r2_small zsh training/sweep/run_ladder_rung.zsh
# Env:     MODEL (rung; default r2_small) · SWEEP_WORK · R2_STEPS · ACCUM · R0_STEPS
set -u
SCRIPT_DIR=${0:A:h}
MODEL=${MODEL:-r2_small}
WORK=${SWEEP_WORK:-/tmp/ladder/$MODEL}
export SWEEP_WORK=$WORK
mkdir -p $WORK
STAMP() { print -r -- "[$(date +%H:%M:%S)] $*" | tee -a $WORK/rung.log }

STAMP "===== LADDER RUNG $MODEL — start (work=$WORK) ====="

# ---- Phase 1: R0 correctness gate, the 3 distinct builds at the rung's dims ----
# Tamed recipe (lr 1e-3, clip 1.0, warmup 20): the hot lr-2e-3 overfit explodes in fp16
# at d768 (ADR 0003). 400 steps reaches loss 0.0000 at d768 with margin.
R0_LR=1e-3 R0_CLIP=1.0 R0_WARMUP=20 R0_STEPS=${R0_STEPS:-400} MODEL=$MODEL \
  CELLS_FILE=$SCRIPT_DIR/cells_ladder_r0.zsh \
  zsh $SCRIPT_DIR/run_r0.zsh

# Gate: every distinct build must collapse, else its R2 numbers are untrustworthy.
typeset -A pass
if [[ -f $WORK/r0_results.tsv ]]; then
  while IFS=$'\t' read -r nm fl vd; do [[ $vd == PASS ]] && pass[$nm]=1; done < $WORK/r0_results.tsv
fi
missing=()
for b in plain mhc2 mhc4; do [[ -z ${pass[$b]:-} ]] && missing+=$b; done
if (( ${#missing} )); then
  STAMP "R0 GATE FAILED for $MODEL — builds not green: ${missing[*]}. Aborting R2."
  column -t -s$'\t' $WORK/r0_results.tsv | tee -a $WORK/rung.log
  print -r -- "RUNG_STATUS=R0_FAIL" > $WORK/STATUS
  exit 1
fi
STAMP "R0 gate GREEN (plain/mhc2/mhc4 all collapsed) — proceeding to R2."

# ---- Phase 2: R2 fast-learning, 12 cells (FORCE_R2: R0 gates builds, not cells) ----
# R2_CELLS overrides the LR grid per rung (default = the r2_small-centered grid). Larger
# rungs pass a re-centered grid (the optima drift down ~3x per ~3x params; ADR 0003) so
# the optimum is bracketed in one pass instead of an edge-extension round.
R2_CELLS=${R2_CELLS:-$SCRIPT_DIR/cells_ladder.zsh}
MODEL=$MODEL FORCE_R2=1 R2_STEPS=${R2_STEPS:-800} ACCUM=${ACCUM:-4} \
  CELLS_FILE=$R2_CELLS R2SUM=$WORK/r2_summary.tsv \
  zsh $SCRIPT_DIR/run_r2.zsh

STAMP "===== LADDER RUNG $MODEL — done. Summary: $WORK/r2_summary.tsv ====="
print -r -- "RUNG_STATUS=DONE" > $WORK/STATUS
